
import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from .fact_tool import FactCatalogue, UnknownBrandError
from .schemas import AcceptedClaim, AgentResult, ClaimDraft


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "v1.txt"

# Retry only after a short, bounded delay.
RETRY_DELAY_SECONDS = 2
MAX_ATTEMPTS = 2
REQUEST_TIMEOUT_MS = 15_000
MAX_MODEL_TURNS = 3


class AgentError(RuntimeError):
    """Raised when the agent cannot produce a valid result."""


class GeminiClaimAgent:
    def __init__(
        self,
        catalogue: FactCatalogue | None = None,
        model: str | None = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise AgentError(
                "GEMINI_API_KEY is missing. "
                "Copy .env.example to .env and add your key."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model or os.getenv(
            "GEMINI_MODEL", "gemini-2.5-flash"
        )
        self.catalogue = catalogue or FactCatalogue()
        self.system_instruction = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        self.tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_brand_facts",
                    description=(
                        "Read-only lookup of all supplied catalogue fact "
                        "rows for one known brand. Use this before "
                        "classifying claims."
                    ),
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "brand_id": types.Schema(
                                type=types.Type.STRING
                            )
                        },
                        required=["brand_id"],
                    ),
                )
            ]
        )

    @staticmethod
    def _is_daily_quota_exhausted(error_text: str) -> bool:
        """
        Identify exhausted daily quotas that will not recover by
        waiting a couple of seconds.
        """
        daily_markers = (
            "GENERATEREQUESTSPERDAY",
            "REQUESTS PER DAY",
            "PER DAY",
            "DAILY QUOTA",
            "DAILY LIMIT",
        )

        if any(marker in error_text for marker in daily_markers):
            return True

        # A generic quota error can also mean a per-minute limit.
        # Only treat it as permanent here when it is not a per-minute
        # or per-minute-token limit.
        generic_quota = (
            "QUOTA EXCEEDED" in error_text
            or "QUOTA_EXCEEDED" in error_text
        )
        per_minute = (
            "PER MINUTE" in error_text
            or "PER_MINUTE" in error_text
            or "REQUESTS PER MINUTE" in error_text
            or "TOKENS PER MINUTE" in error_text
        )

        return generic_quota and not per_minute

    @classmethod
    def _is_transient_error(cls, error_text: str) -> bool:
        """
        Return True for errors that may recover with one short retry.

        Daily quota exhaustion is deliberately excluded.
        """
        if cls._is_daily_quota_exhausted(error_text):
            return False

        transient_markers = (
            "503",
            "UNAVAILABLE",
            "429",
            "RESOURCE_EXHAUSTED",
            "408",
            "504",
            "TIMEOUT",
            "TIMED OUT",
            "INTERNAL SERVER ERROR",
            "CONNECTION RESET",
            "CONNECTION ERROR",
        )

        return any(
            marker in error_text for marker in transient_markers
        )

    def _generate_with_retry(self, contents: list[Any]) -> Any:
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=[self.tool],
            temperature=0,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS
            ),
        )

        for attempt in range(MAX_ATTEMPTS):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}".upper()

                if self._is_daily_quota_exhausted(error_text):
                    raise AgentError(
                        "Gemini daily quota is exhausted. "
                        "The request was not retried because waiting "
                        f"{RETRY_DELAY_SECONDS} seconds will not help. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ) from exc

                should_retry = self._is_transient_error(error_text)
                is_first_attempt = attempt == 0

                if should_retry and is_first_attempt:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue

                raise AgentError(
                    f"Gemini request failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        raise AgentError("Gemini request failed after retry")

    def classify(self, answer: dict[str, str]) -> AgentResult:
        required_fields = (
            "answer_id",
            "brand_id",
            "question",
            "ai_answer",
        )

        if any(key not in answer for key in required_fields):
            raise AgentError(
                "Answer is missing required input fields"
            )

        user_payload = {
            key: answer[key] for key in required_fields
        }

        contents: list[Any] = [
            json.dumps(user_payload, ensure_ascii=False)
        ]

        final_text: str | None = None
        response: Any = None

        # Allow a bounded number of model turns for tool use.
        for _ in range(MAX_MODEL_TURNS):
            response = self._generate_with_retry(contents)

            candidate = (
                response.candidates[0].content
                if response.candidates
                else None
            )

            if candidate is None:
                raise AgentError(
                    "Gemini returned no candidate content"
                )

            # Preserve the model's response in the conversation so
            # the tool call and its result are part of the live flow.
            contents.append(candidate)

            calls = list(response.function_calls or [])

            if calls:
                tool_responses = []

                for call in calls:
                    if call.name != "get_brand_facts":
                        raise AgentError(
                            f"Illegal tool requested: {call.name}"
                        )

                    args = dict(call.args or {})
                    brand_id = args.get("brand_id")

                    if (
                        not isinstance(brand_id, str)
                        or not brand_id.strip()
                    ):
                        raise AgentError(
                            "get_brand_facts requires a "
                            "non-empty brand_id"
                        )

                    try:
                        result = self.catalogue.get_brand_facts(
                            brand_id
                        )
                    except UnknownBrandError as exc:
                        raise AgentError(str(exc)) from exc

                    tool_responses.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": result},
                        )
                    )

                contents.append(
                    types.Content(
                        role="user",
                        parts=tool_responses,
                    )
                )
                continue

            final_text = (response.text or "").strip()

            if final_text:
                break

        if not final_text:
            candidate_info = []

            for candidate in (response.candidates or []):
                content = candidate.content
                parts = content.parts if content else []

                candidate_info.append(
                    {
                        "finish_reason": str(
                            candidate.finish_reason
                        ),
                        "parts": [
                            {"text": part.text}
                            for part in parts
                            if getattr(part, "text", None)
                        ],
                    }
                )

            raise AgentError(
                "Gemini returned empty text. "
                f"Candidate details: {candidate_info}"
            )

        # Support JSON enclosed in Markdown code fences.
        if final_text.startswith("```"):
            lines = final_text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            final_text = "\n".join(lines).strip()

        try:
            parsed = json.loads(final_text)

            if (
                not isinstance(parsed, dict)
                or set(parsed) != {"claims"}
                or not isinstance(parsed["claims"], list)
            ):
                raise ValueError(
                    "Expected a JSON object with only a claims list"
                )

            drafts = [
                ClaimDraft.model_validate(item)
                for item in parsed["claims"]
            ]

        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            raise AgentError(
                f"Invalid model output: {exc}"
            ) from exc

        claims = self._validate_and_locate(
            answer["answer_id"],
            answer["ai_answer"],
            drafts,
        )

        return AgentResult(
            answer_id=answer["answer_id"],
            claims=claims,
        )

    def _validate_and_locate(
        self,
        answer_id: str,
        source: str,
        drafts: list[ClaimDraft],
    ) -> list[AcceptedClaim]:
        allowed_fact_ids = self.catalogue.fact_ids
        located: list[AcceptedClaim] = []
        used_spans: set[tuple[int, int]] = set()
        last_start = -1

        for draft in drafts:
            # Reject any catalogue IDs that do not exist.
            if any(
                fact_id not in allowed_fact_ids
                for fact_id in draft.fact_ids
            ):
                bad_ids = [
                    fact_id
                    for fact_id in draft.fact_ids
                    if fact_id not in allowed_fact_ids
                ]
                raise AgentError(
                    f"Invented or unknown fact IDs: {bad_ids}"
                )

            quote = draft.quote

            if not quote:
                raise AgentError("Claim quote cannot be empty")

            # Find an exact occurrence after the previous claim's
            # starting position.
            start = source.find(quote, last_start + 1)

            while (
                start >= 0
                and (start, start + len(quote)) in used_spans
            ):
                start = source.find(quote, start + 1)

            if start < 0:
                raise AgentError(
                    "Claim quote is not an exact substring of "
                    f"ai_answer: {quote[:100]!r}"
                )

            end = start + len(quote)

            if (
                quote != source[start:end]
                or quote[0].isspace()
                or quote[-1].isspace()
            ):
                raise AgentError(
                    "Quote equality or whitespace validation failed"
                )

            if start < last_start:
                raise AgentError(
                    "Claims are not in increasing source order"
                )

            if located and start < located[-1].end_char:
                raise AgentError("Overlapping claim spans")

            used_spans.add((start, end))
            last_start = start

            located.append(
                AcceptedClaim(
                    claim_id=f"{answer_id}:{start}:{end}",
                    start_char=start,
                    end_char=end,
                    quote=quote,
                    label=draft.label,
                    fact_ids=draft.fact_ids,
                    reason=draft.reason,
                )
            )

        return located
