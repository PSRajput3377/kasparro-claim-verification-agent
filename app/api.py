
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .agent import AgentError, GeminiClaimAgent
from .database import Base, engine, get_db
from .db_models import ClaimRun
from .schemas import AgentResult, AnswerInput

logger = logging.getLogger(__name__)

# Shared by requests handled within this application process.
provider_semaphore = asyncio.Semaphore(3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables when the application starts."""
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Kasparro Claim Verification API",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/claims", response_model=AgentResult)
async def create_claims(
    payload: AnswerInput,
    db: Session = Depends(get_db),
) -> AgentResult:
    run = ClaimRun(
        answer_id=payload.answer_id,
        brand_id=payload.brand_id,
        question=payload.question,
        ai_answer=payload.ai_answer,
        status="running",
    )

    db.add(run)
    db.commit()
    db.refresh(run)

    run_id = run.id

    try:
        # Limit concurrent provider calls in this process.
        async with provider_semaphore:
            agent = GeminiClaimAgent()
            result = await asyncio.to_thread(
                agent.classify,
                payload.model_dump(),
            )

        run.status = "completed"
        run.result_json = result.model_dump(mode="json")
        run.error_message = None
        db.commit()

        return result

    except Exception as exc:
        db.rollback()

        # Re-fetch the record after rollback before marking it failed.
        failed_run = db.get(ClaimRun, run_id)

        if failed_run is not None:
            failed_run.status = "failed"
            failed_run.error_message = (
                f"{type(exc).__name__}: {exc}"
            )[:2000]
            db.commit()

        logger.exception(
            "Claim verification failed for answer_id=%s",
            payload.answer_id,
        )

        status_code = (
            502 if isinstance(exc, AgentError) else 500
        )

        raise HTTPException(
            status_code=status_code,
            detail="Claim verification failed",
        ) from exc


@app.get(
    "/claims/{answer_id}",
    response_model=AgentResult,
)
def get_claims(
    answer_id: str,
    db: Session = Depends(get_db),
) -> AgentResult:
    statement = (
        select(ClaimRun)
        .where(
            ClaimRun.answer_id == answer_id,
            ClaimRun.status == "completed",
        )
        .order_by(ClaimRun.id.desc())
        .limit(1)
    )

    run = db.scalar(statement)

    if run is None or run.result_json is None:
        raise HTTPException(
            status_code=404,
            detail="Completed claim result not found",
        )

    return AgentResult.model_validate(run.result_json)
