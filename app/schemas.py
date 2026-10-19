from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

Label = Literal["supported", "contradicted", "unknown"]

class ClaimDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quote: str = Field(min_length=1)
    label: Label
    fact_ids: list[str] = Field(default_factory=list)
    reason: str = Field(max_length=1000)

class AcceptedClaim(ClaimDraft):
    claim_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

class AgentResult(BaseModel):
    answer_id: str
    claims: list[AcceptedClaim]

class AnswerInput(BaseModel):
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str
