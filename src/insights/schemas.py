"""Pydantic schemas for structured LLM output and the final insight record.

The LLM must return `ClusterAnalysis` exactly; everything else (scores,
provenance, traces) is attached by deterministic code so the boundary
between model output and computed fact stays auditable.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RecommendedAction(BaseModel):
    action: str = Field(description="Concrete next step a product team could take")
    effort: Literal["low", "medium", "high"] = Field(
        description="Rough engineering/product effort"
    )
    rationale: str = Field(description="Why this action addresses the issue")


class ClusterAnalysis(BaseModel):
    """What the LLM returns for one cluster — nothing more."""

    theme_name: str = Field(description="Short product-team-friendly theme label")
    summary: str = Field(description="2-3 sentence neutral summary of the issue")
    root_causes: list[str] = Field(
        description="1-3 hypothesized root causes, phrased as hypotheses"
    )
    severity: int = Field(ge=1, le=5, description="1=cosmetic, 5=blocking/churn risk")
    severity_rationale: str = Field(description="Why this severity was chosen")
    recommended_actions: list[RecommendedAction] = Field(min_length=1, max_length=3)
    evidence_quote_ids: list[str] = Field(
        description="IDs of the provided quotes that best support this analysis"
    )


class LLMTrace(BaseModel):
    provider: str
    model: str
    prompt_version: str
    prompt_hash: str
    timestamp: str
    cached: bool = False
    raw_response: str = ""


class Quote(BaseModel):
    id: str
    text: str
    source: str
    timestamp: Optional[str] = None
    rating: Optional[float] = None
    vader_compound: Optional[float] = None


class ScoreComponent(BaseModel):
    name: str
    raw_value: float
    normalized: float
    weight: float
    contribution: float
    explanation: str


class ConfidenceInfo(BaseModel):
    n_items: int
    n_unique: int
    badge: Literal["ok", "low_sample", "insufficient_evidence"]
    wilson_low: float = Field(description="95% Wilson lower bound on negative share")
    wilson_high: float
    negative_share: float
    cohesion: float
    mixed_theme_warning: bool


class Insight(BaseModel):
    """One fully-traceable insight: LLM analysis + deterministic scoring."""

    cluster_id: int
    analysis: ClusterAnalysis
    priority_score: float
    score_components: list[ScoreComponent]
    confidence: ConfidenceInfo
    per_source_counts: dict[str, int]
    representative_quotes: list[Quote]
    member_ids: list[str]
    citation_check_passed: bool
    citation_check_detail: str
    llm_trace: LLMTrace


class ExecutiveSummary(BaseModel):
    """Final synthesis call output."""

    headline: str = Field(description="One-sentence top takeaway")
    summary: str = Field(description="Short executive summary paragraph")
    cross_theme_observations: list[str] = Field(
        description="Patterns spanning multiple themes, if any"
    )
    suggested_focus: list[str] = Field(description="Top 2-3 areas to focus on first")
