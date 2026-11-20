"""The structured verdict every investigation must end with."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecommendedAction(BaseModel):
    """A response step. Anything that changes a system needs a human to approve it."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, max_length=500)
    requires_human_approval: bool = True


class Verdict(BaseModel):
    """Outcome of investigating one alert."""

    model_config = ConfigDict(extra="forbid")

    is_malicious: bool
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    attack_techniques: list[str] = Field(default_factory=list, max_length=10)
    evidence: list[str] = Field(default_factory=list, max_length=20)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list, max_length=10)
    needs_human_review: bool = False
    notes: list[str] = Field(default_factory=list)
