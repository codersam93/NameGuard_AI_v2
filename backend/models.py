from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, constr


# ---------------------------
# Pydantic models
# ---------------------------


# Allow minimum length 1 to avoid backend validation rejecting very short or
# placeholder names like "NDM" that are important for policy tests.
NameStr = constr(strip_whitespace=True, min_length=1, max_length=200)


class EntityType(str, Enum):
    PRIVATE_LIMITED = "private_limited"
    PUBLIC_LIMITED = "public_limited"
    OPC = "opc_private_limited"
    LLP = "llp"
    SECTION8 = "section8"


class NameEvaluationInput(BaseModel):
    name: NameStr
    priority: int = Field(ge=1, le=3)


class EvaluateNamesRequest(BaseModel):
    names: List[NameEvaluationInput]
    entity_type: Optional[EntityType] = None
    industry: Optional[constr(strip_whitespace=True, max_length=100)] = None


class DecisionLabel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RuleFlag(BaseModel):
    code: str
    description: str
    severity: float = Field(ge=0.0, le=1.0)


class NameEvaluationResult(BaseModel):
    name: str
    priority: int
    acceptance_probability: float = Field(ge=0.0, le=100.0)
    decision_label: DecisionLabel
    rule_flags: List[RuleFlag]
    explanations: List[str]


class EvaluateNamesResponse(BaseModel):
    results: List[NameEvaluationResult]
    evaluated_at: datetime
