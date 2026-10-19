
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ClaimRun(Base):
    __tablename__ = "claim_runs"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    answer_id: Mapped[str] = mapped_column(
        String(128), index=True
    )
    brand_id: Mapped[str] = mapped_column(String(128))
    question: Mapped[str] = mapped_column(Text)
    ai_answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(32), default="completed"
    )
    result_json: Mapped[dict | None] = mapped_column(
    JSON().with_variant(JSONB, "postgresql"),
    nullable=True,
)
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
