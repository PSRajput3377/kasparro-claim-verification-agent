# Kasparro FSE I assessment — starter implementation

This is the first runnable milestone: Gemini tool-calling agent, exact quote-to-span validation, deterministic development scorer, and basic tests. FastAPI, PostgreSQL, concurrency/retries, and Langfuse are intentionally not yet wired in; add these after validating the first live run.

## Setup (macOS + VS Code)

```bash
cd kasparro-assessment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your own Google AI Studio API key. Keep `.env` private; it is gitignored.

## Run checks

```bash
pytest -q
```

## First live call (one development answer)

```bash
python -m app.cli --split development --version v1 --limit 1
```

If the result looks valid, run all 40 development answers:

```bash
python -m app.cli --split development --version v1
```

The agent writes predictions under `reports/` and scores development outputs against `development_gold.csv` locally. Gold labels are not included in the model request or tool responses. Check split runs write predictions without quality metrics.

## Important limitations of this milestone

- The first agent prompt is `prompts/v1.txt`; preserve it when creating v2.
- This starter does not yet implement the required three-concurrent-request cap, 15-second timeout/retry policy, FastAPI/PostgreSQL persistence, or Langfuse traces.
- Run only the one-answer live test first to verify your Gemini key and model access. API calls may incur charges.
- Inspect quote spans and errors. Do not claim an improvement until v2 is tested on the same development split.
