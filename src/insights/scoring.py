"""Transparent prioritization scoring — pure Python, no LLM.

priority = w_F·F + w_S·S + w_R·R + w_D·D   (weights from config.yaml)

  F frequency : log(1+n) / log(1+n_largest)
  S severity  : mix of LLM severity, inverse star rating, |VADER|
  R recency   : mean exponential decay, configurable half-life
  D diversity : normalized Shannon entropy over sources

Every component is returned as a ScoreComponent (raw → normalized → weight →
contribution) so the UI can render the full breakdown. Confidence handling
(Wilson bound, sample-size badges, cohesion warning) lives here too.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import ScoringConfig
from src.insights.schemas import ConfidenceInfo, ScoreComponent

# 95% / 90% z-values; anything else falls back to 1.96.
_Z_VALUES = {0.95: 1.96, 0.90: 1.645, 0.99: 2.576}


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (robust at small n)."""
    if n == 0:
        return 0.0, 1.0
    z = _Z_VALUES.get(round(confidence, 2), 1.96)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, center - margin), min(1.0, center + margin)


def shannon_diversity(counts: dict[str, int], n_source_types: int) -> float:
    """Normalized Shannon entropy: 1.0 = perfectly even split, 0 = one source."""
    total = sum(counts.values())
    if total == 0 or n_source_types <= 1:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)
    return entropy / math.log(n_source_types)


def recency_score(
    timestamps: pd.Series, half_life_days: float, now: datetime | None = None
) -> float:
    """Mean exp(−ln2 · age/half_life); items without timestamps are skipped."""
    ts = pd.to_datetime(timestamps, errors="coerce", utc=True).dropna()
    if ts.empty:
        return 0.5  # unknown age → neutral midpoint, not a penalty
    now = now or datetime.now(timezone.utc)
    age_days = (now - ts).dt.total_seconds() / 86400.0
    return float(np.exp(-math.log(2) * age_days / half_life_days).mean())


def recency_from_ages(age_days: pd.Series, half_life_days: float) -> float:
    """Recency from precomputed per-item ages (days). Used when sources have
    different collection windows: age is measured against each source's own
    window end, so an old dataset isn't unfairly zeroed out."""
    ages = pd.to_numeric(age_days, errors="coerce").dropna()
    if ages.empty:
        return 0.5
    return float(np.exp(-math.log(2) * ages / half_life_days).mean())


def score_cluster(
    cluster_df: pd.DataFrame,
    llm_severity: int,
    n_largest_cluster: int,
    n_source_types: int,
    cohesion: float,
    cfg: ScoringConfig,
    now: datetime | None = None,
) -> tuple[float, list[ScoreComponent], ConfidenceInfo]:
    w = cfg.weights
    n = len(cluster_df)

    # F — frequency
    f_norm = math.log(1 + n) / math.log(1 + max(n_largest_cluster, 2))

    # S — severity (LLM + rating + sentiment mix)
    sev_llm = (llm_severity - 1) / 4.0  # 1..5 → 0..1
    ratings = cluster_df["rating"].dropna()
    rating_neg = float((5.0 - ratings).mean() / 4.0) if not ratings.empty else None
    vader_mag = float(cluster_df["vader_compound"].abs().mean())

    mix = cfg.severity_mix
    if rating_neg is None:
        # No star ratings in this cluster: fold the rating weight into VADER.
        sent_w = mix.sentiment + mix.rating
        s_norm = mix.llm * sev_llm + sent_w * vader_mag
        rating_for_display = 0.0
    else:
        s_norm = mix.llm * sev_llm + mix.rating * rating_neg + mix.sentiment * vader_mag
        rating_for_display = rating_neg

    # R — recency (per-source ages when available; see recency_from_ages)
    if "age_days" in cluster_df.columns:
        r_norm = recency_from_ages(cluster_df["age_days"], cfg.recency_half_life_days)
    else:
        r_norm = recency_score(cluster_df["timestamp"], cfg.recency_half_life_days, now)

    # D — source diversity
    counts = cluster_df["source"].value_counts().to_dict()
    d_norm = shannon_diversity(counts, n_source_types)

    components = [
        ScoreComponent(
            name="frequency",
            raw_value=float(n),
            normalized=round(f_norm, 4),
            weight=w.frequency,
            contribution=round(w.frequency * f_norm, 4),
            explanation=f"{n} items; log-scaled against largest cluster ({n_largest_cluster})",
        ),
        ScoreComponent(
            name="severity",
            raw_value=float(llm_severity),
            normalized=round(s_norm, 4),
            weight=w.severity,
            contribution=round(w.severity * s_norm, 4),
            explanation=(
                f"LLM severity {llm_severity}/5 (w={mix.llm}), inverse rating "
                f"{rating_for_display:.2f} (w={mix.rating}), |sentiment| "
                f"{vader_mag:.2f} (w={mix.sentiment})"
            ),
        ),
        ScoreComponent(
            name="recency",
            raw_value=round(r_norm, 4),
            normalized=round(r_norm, 4),
            weight=w.recency,
            contribution=round(w.recency * r_norm, 4),
            explanation=(
                f"exponential decay, half-life {cfg.recency_half_life_days:.0f} days; "
                "age measured against each source's collection-window end"
            ),
        ),
        ScoreComponent(
            name="source_diversity",
            raw_value=float(len(counts)),
            normalized=round(d_norm, 4),
            weight=w.diversity,
            contribution=round(w.diversity * d_norm, 4),
            explanation=f"sources: {counts} (normalized Shannon entropy)",
        ),
    ]
    priority = round(sum(c.contribution for c in components), 4)

    # ── confidence ───────────────────────────────────────────────────────
    negative = int((cluster_df["vader_compound"] < -0.05).sum())
    wl, wh = wilson_interval(negative, n, cfg.wilson_confidence)
    if n < cfg.min_items_insufficient:
        badge = "insufficient_evidence"
    elif n < cfg.min_items_low_sample:
        badge = "low_sample"
    else:
        badge = "ok"

    n_unique = (
        int((~cluster_df["is_near_dup"]).sum())
        if "is_near_dup" in cluster_df.columns
        else n
    )
    confidence = ConfidenceInfo(
        n_items=n,
        n_unique=n_unique,
        badge=badge,
        wilson_low=round(wl, 4),
        wilson_high=round(wh, 4),
        negative_share=round(negative / n, 4) if n else 0.0,
        cohesion=round(cohesion, 4),
        mixed_theme_warning=cohesion < cfg.low_cohesion_threshold,
    )
    return priority, components, confidence
