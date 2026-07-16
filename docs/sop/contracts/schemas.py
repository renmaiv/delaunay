"""CONTRACT FILE — copy verbatim to server/schemas.py in task WS-A1.

Single source of truth for all API/internal types.
Mirrored exactly by frontend/src/types.ts (see contracts/types.ts).
Any change to this file requires updating types.ts in the same PR.
"""
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"  # tolerated on input; never scored


class DetectionCategory(str, Enum):
    # user-side (scored by encoders / rules)
    jailbreak_steering = "jailbreak_steering"
    social_engineering = "social_engineering"
    coercive_pressure = "coercive_pressure"
    repair_request = "repair_request"
    # model-side (scored by the LLM judge)
    safety_triggered = "safety_triggered"
    appeasement = "appeasement"
    overcompliant = "overcompliant"
    cot_divergence = "cot_divergence"


USER_CATEGORIES = [
    DetectionCategory.jailbreak_steering,
    DetectionCategory.social_engineering,
    DetectionCategory.coercive_pressure,
    DetectionCategory.repair_request,
]
MODEL_CATEGORIES = [
    DetectionCategory.safety_triggered,
    DetectionCategory.appeasement,
    DetectionCategory.overcompliant,
    DetectionCategory.cot_divergence,
]


class Detection(BaseModel):
    category: DetectionCategory
    score: float = Field(ge=0.0, le=1.0)
    source: Literal["encoder", "judge", "rules"]
    evidence_span: Optional[str] = None  # verbatim quote from the turn
    rationale: Optional[str] = None
    # No score in this system is a calibrated probability yet. The field exists
    # so a later temperature-scaling pass is a data-only change.
    calibrated: bool = False


class Turn(BaseModel):
    index: int  # 0-based position in the conversation
    role: Role
    content: str
    cot: Optional[str] = None  # chain of thought, assistant turns only
    sentiment: Optional[float] = Field(default=None, ge=-1.0, le=1.0)  # user turns only
    detections: List[Detection] = Field(default_factory=list)


class CausalLink(BaseModel):
    from_turn: int  # user turn index (trigger)
    to_turn: int    # assistant turn index (effect); must be > from_turn
    from_category: DetectionCategory
    to_category: DetectionCategory
    score: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None


class AnalysisMeta(BaseModel):
    judge_provider: str  # "anthropic" | "mock"
    judge_model: Optional[str] = None
    encoders_available: Dict[str, bool] = Field(default_factory=dict)  # scorer name -> loaded
    warnings: List[str] = Field(default_factory=list)
    # "auth" when the judge could not run because the server's Anthropic key is
    # missing, invalid, or out of credit — the client may offer BYOK and retry.
    judge_error: Optional[Literal["auth"]] = None


class AnalysisResult(BaseModel):
    conversation_id: str
    model_name: Optional[str] = None  # e.g. "chatgpt 5.0" if detected in file metadata
    summary: str  # 2-3 sentences: what the conversation is about
    overall_sentiment: float = Field(ge=-1.0, le=1.0)
    turns: List[Turn]
    causal_links: List[CausalLink] = Field(default_factory=list)
    meta: AnalysisMeta


AnalysisStatus = Literal["pending", "running", "completed", "failed"]


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None


# ---- internal parse types (not exposed over HTTP) ----

class ParsedTurn(BaseModel):
    role: Role
    content: str
    cot: Optional[str] = None


class ParsedConversation(BaseModel):
    conversation_id: str
    model_name: Optional[str] = None
    turns: List[ParsedTurn]
