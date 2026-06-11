"""Per-cluster LLM analysis: theme, root causes, severity, actions.

One structured call per cluster. The quote sample is STRATIFIED (by source
and severity, near-dups excluded) so the LLM sees the breadth of the
cluster, not just one viral phrasing. Citations are verified afterwards —
an insight whose evidence_quote_ids reference unknown items is flagged,
never silently trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.sentiment import severity_prior
from src.insights.llm_client import LLMClient
from src.insights.prompts import (
    CLUSTER_ANALYST_SYSTEM,
    CLUSTER_ANALYST_USER,
    format_quotes_block,
)
from src.insights.schemas import ClusterAnalysis, LLMTrace, Quote


def select_quote_sample(
    cluster_df: pd.DataFrame,
    centroid_order: np.ndarray,
    max_quotes: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sample: nearest-centroid core + highest-severity + per-source
    coverage. Near-duplicates are excluded from quoting (they still count in
    frequency)."""
    pool = cluster_df
    if "is_near_dup" in pool.columns and (~pool["is_near_dup"]).sum() >= 5:
        pool = pool[~pool["is_near_dup"]]
    # Prefer dense-core members over noise-reassigned stragglers for quoting.
    if "from_noise" in pool.columns and (~pool["from_noise"]).sum() >= 5:
        pool = pool[~pool["from_noise"]]

    chosen: list[str] = []

    # 1. Core: nearest to centroid (most representative of the theme).
    order_ids = [i for i in centroid_order if i in pool.index]
    core_n = max(3, max_quotes // 2)
    chosen.extend(order_ids[:core_n])

    # 2. Edge: highest per-item severity not already chosen.
    sev = pool.apply(severity_prior, axis=1).sort_values(ascending=False)
    for idx in sev.index:
        if len(chosen) >= max_quotes - 2:
            break
        if idx not in chosen:
            chosen.append(idx)

    # 3. Coverage: at least one item per source.
    for source, grp in pool.groupby("source"):
        if not any(i in grp.index for i in chosen):
            chosen.append(grp.index[0])

    chosen = chosen[:max_quotes]
    return pool.loc[[i for i in chosen if i in pool.index]]


def _df_to_quotes(sample: pd.DataFrame) -> list[Quote]:
    quotes = []
    for _, row in sample.iterrows():
        ts = row.get("timestamp")
        quotes.append(
            Quote(
                id=row["id"],
                text=row["text"],
                source=row["source"],
                timestamp=None if pd.isna(ts) else str(ts),
                rating=None if pd.isna(row.get("rating")) else float(row["rating"]),
                vader_compound=float(row.get("vader_compound", 0.0)),
            )
        )
    return quotes


def analyze_cluster(
    client: LLMClient,
    cluster_df: pd.DataFrame,
    centroid_order: np.ndarray,
    max_quotes: int = 15,
) -> tuple[ClusterAnalysis, list[Quote], LLMTrace]:
    sample = select_quote_sample(cluster_df, centroid_order, max_quotes)
    quotes = _df_to_quotes(sample)

    source_breakdown = ", ".join(
        f"{k}: {v}" for k, v in cluster_df["source"].value_counts().items()
    )
    user_prompt = CLUSTER_ANALYST_USER.format(
        n_items=len(cluster_df),
        source_breakdown=source_breakdown,
        n_quotes=len(quotes),
        quotes_block=format_quotes_block([q.model_dump() for q in quotes]),
    )
    analysis, trace = client.generate_structured(
        CLUSTER_ANALYST_SYSTEM, user_prompt, ClusterAnalysis
    )
    return analysis, quotes, trace


def _resolve_citation(cited: str, members: set[str]) -> str | None:
    """Map a cited id to a canonical member id, tolerating the common LLM
    quirk of dropping the 'source:' prefix. Only an UNambiguous match counts."""
    if cited in members:
        return cited
    matches = [
        m for m in members if m.split(":", 1)[-1] == cited or m.endswith(cited)
    ]
    return matches[0] if len(matches) == 1 else None


def verify_citations(
    analysis: ClusterAnalysis, cluster_member_ids: set[str]
) -> tuple[bool, str, list[str]]:
    """Every cited quote ID must resolve to a real member of this cluster.

    Returns (passed, detail, resolved_ids). Resolved ids are canonical; ids
    that could not be resolved are kept as-is so the failure stays visible.
    """
    cited = analysis.evidence_quote_ids
    if not cited:
        return False, "LLM cited no evidence quotes", []

    resolved: list[str] = []
    unknown: list[str] = []
    for cid in cited:
        canonical = _resolve_citation(cid, cluster_member_ids)
        if canonical is None:
            unknown.append(cid)
            resolved.append(cid)
        else:
            resolved.append(canonical)

    if unknown:
        return (
            False,
            f"LLM cited {len(unknown)} unknown quote id(s): {sorted(unknown)[:3]}",
            resolved,
        )
    return True, f"all {len(cited)} cited quote ids verified as cluster members", resolved
