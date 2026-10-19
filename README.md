# Kasparro FSE I Assessment — Claim Verification Agent

A Gemini-powered agent that extracts factual product claims from AI-generated answers, retrieves evidence from a supplied product catalogue, classifies claims as supported, contradicted, or unknown, and evaluates predictions using exact character spans.

## Features

- Gemini function/tool calling for catalogue fact retrieval.
- Exact quote validation and Python-computed character offsets.
- Versioned prompts: v1 and v2.
- Deterministic evaluation with extraction F1, joint F1, macro F1, and a confusion matrix.
- FastAPI endpoints for health, claim classification, and result retrieval.
- PostgreSQL persistence using SQLAlchemy.
- Bounded concurrency, request timeouts, and limited retry handling.
- Optional Langfuse tracing.
- Automated tests.

## Setup

Requirements: Python, PostgreSQL, and a Google AI Studio API key for live inference.

Create and activate a virtual environment, install dependencies, and configure a local .env file using .env.example as a template. Set GEMINI_API_KEY, GEMINI_MODEL, and DATABASE_URL. Configure Langfuse credentials only if tracing is needed.

Never commit .env or expose credentials.

## Tests

Run:

    python -m compileall -q app tests
    python -m pytest -q

## CLI evaluation

Run one development answer:

    python -m app.cli --split development --version v1 --limit 1

Run a limited v2 trial:

    python -m app.cli --split development --version v2 --limit 1

Run a full development evaluation only when provider quota is available:

    python -m app.cli --split development --version v1
    python -m app.cli --split development --version v2

Predictions and metrics are saved under reports/. Preserve historical reports and inspect the CLI's overwrite safeguards before rerunning evaluations.

## API

Start the server:

    uvicorn app.api:app --reload

Health endpoint: http://127.0.0.1:8000/health

Interactive API documentation: http://127.0.0.1:8000/docs

Ensure PostgreSQL is running and DATABASE_URL is correct before testing persistence. Use the API documentation for request schemas.

## Data and evaluation

- data/answers.csv: input answers and split assignments.
- data/brand_facts.csv: catalogue facts.
- data/development_gold.csv: development annotations.

Gold labels must not be sent to the model or exposed through the catalogue tool. The check split is reserved for final held-out evaluation.

The evaluator matches claims by answer ID and exact character offsets. Failed model requests are errors, not valid evidence of model quality. Report failed requests alongside metrics.

## Observability and operational limits

Langfuse tracing is optional and requires configured credentials. Concurrency limits within one process do not automatically enforce a global limit across multiple workers.

Gemini quota or invalid API credentials can prevent live inference. Resolve provider configuration before running large evaluations; do not repeatedly retry daily quota failures.

## Evaluation status

The existing v1 development reports are historical artifacts. Previous evaluation attempts encountered provider failures, so their aggregate metrics should not be interpreted as a successful full-dataset benchmark.

Prompt v2 is implemented, but a fair v1-versus-v2 comparison requires successful runs on the same development answers. Keep the check split untouched until prompt selection is complete.
