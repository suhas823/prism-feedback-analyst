"""Final synthesis call: executive summary across all scored insights."""

from __future__ import annotations

import pandas as pd

from src.insights.llm_client import LLMClient
from src.insights.prompts import SYNTHESIS_SYSTEM, SYNTHESIS_USER, format_themes_block
from src.insights.schemas import ExecutiveSummary, Insight, LLMTrace


def synthesize(
    client: LLMClient, insights: list[Insight], corpus: pd.DataFrame
) -> tuple[ExecutiveSummary, LLMTrace]:
    themes = [
        {
            "theme_name": ins.analysis.theme_name,
            "priority": ins.priority_score,
            "n_items": ins.confidence.n_items,
            "severity": ins.analysis.severity,
            "badge": ins.confidence.badge,
            "summary": ins.analysis.summary,
        }
        for ins in sorted(insights, key=lambda i: -i.priority_score)
    ]
    source_breakdown = ", ".join(
        f"{k}: {v}" for k, v in corpus["source"].value_counts().items()
    )
    user_prompt = SYNTHESIS_USER.format(
        n_total=len(corpus),
        source_breakdown=source_breakdown,
        themes_block=format_themes_block(themes),
    )
    return client.generate_structured(SYNTHESIS_SYSTEM, user_prompt, ExecutiveSummary)
