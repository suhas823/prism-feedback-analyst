"""Per-item sentiment via VADER (lexicon-based, no model download).

`vader_compound` ranges -1 (very negative) to +1 (very positive). It feeds
the severity prior, especially for tickets, which carry no star rating.
"""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def add_sentiment(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    analyzer = _get_analyzer()
    df = df.copy()
    df["vader_compound"] = df[text_col].map(
        lambda t: analyzer.polarity_scores(str(t))["compound"]
    )
    return df


def severity_prior(row: pd.Series) -> float:
    """Per-item severity prior in [0, 1].

    Reviews: low star rating dominates. Tickets: negative sentiment dominates.
    """
    sent_neg = max(0.0, -row.get("vader_compound", 0.0))  # 0..1
    rating = row.get("rating")
    if rating is not None and not pd.isna(rating):
        rating_neg = (5.0 - float(rating)) / 4.0  # 1★→1.0, 5★→0.0
        return 0.7 * rating_neg + 0.3 * sent_neg
    return sent_neg
