from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


class InputType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    URL = "url"


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    PROHIBITED = "PROHIBITED"


class MatchType(str, Enum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    NONE = "none"


# ── Agent 1 I/O ──────────────────────────────────────────────────────────────

class ExtractionInput(BaseModel):
    input_type: InputType
    data: str = Field(..., description="Raw text, base64 image, or URL")


class ExtractionOutput(BaseModel):
    ingredients: list[str] = Field(default_factory=list)
    raw_ingredients: list[str] = Field(default_factory=list)
    confidence_scores: list[float] = Field(default_factory=list)
    extraction_method: str = ""
    notes: str = ""


# ── Agent 2 I/O ──────────────────────────────────────────────────────────────

class WADAMatch(BaseModel):
    ingredient: str
    match_type: MatchType
    wada_category: str = ""
    wada_subcategory: str = ""
    prohibited_in: str = ""      # "in-competition", "out-of-competition", "all"
    source: str = ""
    confidence: float = 0.0


class KnowledgeOutput(BaseModel):
    matches: list[WADAMatch] = Field(default_factory=list)
    wada_list_version: str = ""
    retrieval_timestamp: str = ""


# ── Agent 3 I/O ──────────────────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    thought: str
    action: str
    observation: str


class ComplianceOutput(BaseModel):
    risk_level: RiskLevel
    flagged_ingredients: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    safe_ingredients: list[str] = Field(default_factory=list)


# ── Agent 4 I/O ──────────────────────────────────────────────────────────────

class ExplanationOutput(BaseModel):
    summary: str
    athlete_advice: str
    flagged_details: list[dict] = Field(default_factory=list)
    disclaimer: str = (
        "This tool provides general guidance only. "
        "Always verify with your national anti-doping organization before competition."
    )


# ── Final API response ────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    input_type: InputType
    data: str = Field(..., description="Raw text, base64-encoded image, or URL")
    athlete_sport: Optional[str] = Field(None, description="Sport for sport-specific rules")


class AnalyzeResponse(BaseModel):
    risk_level: RiskLevel
    ingredients: list[str] = Field(default_factory=list)
    flagged: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    athlete_advice: str = ""
    trace: list[dict] = Field(default_factory=list)
    wada_list_version: str = ""
    disclaimer: str = (
        "This tool provides general guidance only. "
        "Always verify with your national anti-doping organization before competition."
    )


class BatchAnalyzeRequest(BaseModel):
    items: list[AnalyzeRequest]


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]
