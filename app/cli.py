
"""Command-line runner for development and check evaluations."""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from .agent import (
    AgentError,
    GeminiClaimAgent,
    SUPPORTED_PROMPT_VERSIONS,
)
from .evaluator import read_csv, score


ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Run factual claim extraction and evaluation."
    )
    parser.add_argument(
        "--split",
        choices=["development", "check"],
        default="development",
    )
    parser.add_argument(
        "--version",
        choices=sorted(SUPPORTED_PROMPT_VERSIONS),
        default="v1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum answers to process; 0 means all scheduled answers.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer")

    answers_path = ROOT / "data" / "answers.csv"
    answers = read_csv(answers_path)

    scheduled = [
        answer
        for answer in answers
        if answer["split"] == args.split
    ]

    if args.limit:
        scheduled = scheduled[:args.limit]

    if not scheduled:
        parser.error(
            f"No answers found for split {args.split!r}"
        )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Preserve the canonical full-run reports. Limited runs get
    # separate filenames so they cannot replace the full evaluation.
    suffix = f"_{args.limit}" if args.limit else ""
    output_path = (
        reports_dir
        / f"{args.version}_{args.split}{suffix}.json"
    )

    metrics_path = (
        reports_dir
        / f"{args.version}_{args.split}_metrics{suffix}.json"
    )

    if output_path.exists():
        parser.error(
            f"Refusing to overwrite existing report: {output_path}. "
            "Move or rename it first if replacement is intentional."
        )

    if args.split == "development" and metrics_path.exists():
        parser.error(
            f"Refusing to overwrite existing metrics: {metrics_path}. "
            "Move or rename it first if replacement is intentional."
        )

    agent = GeminiClaimAgent(prompt_version=args.version)
    outputs = []

    for index, answer in enumerate(scheduled, 1):
        started = time.perf_counter()

        try:
            result = agent.classify(answer)
            row = {
                "answer_id": answer["answer_id"],
                "status": "ok",
                "claims": [
                    claim.model_dump()
                    for claim in result.claims
                ],
                "latency_seconds": round(
                    time.perf_counter() - started, 3
                ),
                "agent_version": args.version,
            }
            print(
                f"[{index}/{len(scheduled)}] "
                f"{answer['answer_id']}: "
                f"{len(result.claims)} claims"
            )

        except Exception as exc:
            row = {
                "answer_id": answer["answer_id"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "claims": [],
                "latency_seconds": round(
                    time.perf_counter() - started, 3
                ),
                "agent_version": args.version,
            }
            print(
                f"[{index}/{len(scheduled)}] "
                f"{answer['answer_id']}: ERROR {row['error']}"
            )

        outputs.append(row)

    report = {
        "split": args.split,
        "agent_version": args.version,
        "prompt_version": agent.prompt_version,
        "model": agent.model,
        "scheduled_answer_ids": [
            answer["answer_id"] for answer in scheduled
        ],
        "results": outputs,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved predictions: {output_path}")

    if args.split == "development":
        gold_path = ROOT / "data" / "development_gold.csv"
        gold = read_csv(gold_path)
        scheduled_ids = {
            answer["answer_id"] for answer in scheduled
        }
        gold_subset = [
            row for row in gold
            if row["answer_id"] in scheduled_ids
        ]

        metrics = score(
            outputs,
            gold_subset,
            [answer["answer_id"] for answer in scheduled],
        )

        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary_keys = [
            "scheduled_answers",
            "G_gold_claims",
            "P_accepted_predictions",
            "M_exact_span_matches",
            "C_exact_span_and_label_correct",
            "extraction_precision",
            "extraction_recall",
            "extraction_f1",
            "joint_precision",
            "joint_recall",
            "joint_f1",
            "macro_f1",
            "answer_errors",
        ]

        print(
            json.dumps(
                {key: metrics[key] for key in summary_keys},
                indent=2,
            )
        )
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
