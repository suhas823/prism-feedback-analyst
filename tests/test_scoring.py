"""Unit tests for the prioritization scoring — the part teams must trust."""

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ScoringConfig
from src.insights.scoring import (
    recency_from_ages,
    score_cluster,
    shannon_diversity,
    wilson_interval,
)


def make_cluster_df(n=20, source="play_store_review", rating=2.0, vader=-0.5, age=10.0):
    now = datetime.now(timezone.utc)
    return pd.DataFrame(
        {
            "id": [f"{source}:{i}" for i in range(n)],
            "source": [source] * n,
            "rating": [rating] * n,
            "vader_compound": [vader] * n,
            "timestamp": [now - timedelta(days=age)] * n,
            "age_days": [age] * n,
            "is_near_dup": [False] * n,
        }
    )


class TestWilson:
    def test_zero_n(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_small_sample_wide_interval(self):
        lo_small, hi_small = wilson_interval(3, 4)
        lo_big, hi_big = wilson_interval(75, 100)
        # Same proportion (0.75), but the small sample must be less certain.
        assert (hi_small - lo_small) > (hi_big - lo_big)

    def test_bounds_in_unit_interval(self):
        for s, n in [(0, 10), (10, 10), (5, 7), (1, 2)]:
            lo, hi = wilson_interval(s, n)
            assert 0.0 <= lo <= hi <= 1.0


class TestDiversity:
    def test_single_source_is_zero(self):
        assert shannon_diversity({"a": 10}, n_source_types=2) == 0.0

    def test_even_split_is_one(self):
        assert shannon_diversity({"a": 5, "b": 5}, 2) == pytest.approx(1.0)

    def test_skewed_between(self):
        d = shannon_diversity({"a": 9, "b": 1}, 2)
        assert 0.0 < d < 1.0


class TestRecency:
    def test_fresh_items_near_one(self):
        assert recency_from_ages(pd.Series([0.0, 1.0]), 90) > 0.95

    def test_half_life(self):
        assert recency_from_ages(pd.Series([90.0]), 90) == pytest.approx(0.5, abs=0.01)

    def test_missing_ages_neutral(self):
        assert recency_from_ages(pd.Series([np.nan]), 90) == 0.5


class TestScoreCluster:
    cfg = ScoringConfig()

    def _score(self, df, severity=3, n_largest=50, cohesion=0.6):
        return score_cluster(df, severity, n_largest, 2, cohesion, self.cfg)

    def test_priority_in_unit_interval(self):
        p, comps, conf = self._score(make_cluster_df())
        assert 0.0 <= p <= 1.0

    def test_components_sum_to_priority(self):
        p, comps, _ = self._score(make_cluster_df())
        assert p == pytest.approx(sum(c.contribution for c in comps), abs=1e-6)

    def test_higher_severity_higher_priority(self):
        df = make_cluster_df()
        p_low, _, _ = self._score(df, severity=1)
        p_high, _, _ = self._score(df, severity=5)
        assert p_high > p_low

    def test_bigger_cluster_higher_priority(self):
        p_small, _, _ = self._score(make_cluster_df(n=5))
        p_big, _, _ = self._score(make_cluster_df(n=50))
        assert p_big > p_small

    def test_insufficient_evidence_badge(self):
        _, _, conf = self._score(make_cluster_df(n=3))
        assert conf.badge == "insufficient_evidence"

    def test_low_sample_badge(self):
        _, _, conf = self._score(make_cluster_df(n=10))
        assert conf.badge == "low_sample"

    def test_ok_badge(self):
        _, _, conf = self._score(make_cluster_df(n=30))
        assert conf.badge == "ok"

    def test_mixed_theme_warning(self):
        _, _, conf = self._score(make_cluster_df(), cohesion=0.1)
        assert conf.mixed_theme_warning

    def test_no_rating_source_uses_sentiment(self):
        df = make_cluster_df(source="support_ticket")
        df["rating"] = None
        p, comps, _ = self._score(df)
        sev = next(c for c in comps if c.name == "severity")
        assert sev.normalized > 0  # sentiment carried the rating weight
