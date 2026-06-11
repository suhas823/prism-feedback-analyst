"""Adapter for Google Play app reviews (data/raw/reviews_raw.csv).

Handles the column layouts of the known Kaggle Spotify-review datasets via
candidate-column mapping, so swapping the upstream dataset doesn't break
ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.ingest.base import FeedbackItem, SourceAdapter, hash_author, make_item_id

TEXT_COLS = ["review_text", "content", "Review", "review"]
RATING_COLS = ["review_rating", "score", "Rating", "rating"]
TIME_COLS = ["review_timestamp", "at", "Time_submitted", "date"]
ID_COLS = ["review_id", "reviewId", "id"]
AUTHOR_COLS = ["pseudo_author_id", "author_name", "userName", "user_name"]


def _pick(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


class PlayStoreReviewsAdapter(SourceAdapter):
    source_name = "play_store_review"

    def __init__(self, csv_path: Path, sample_size: int, seed: int = 42):
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.seed = seed

    def load(self) -> Iterable[FeedbackItem]:
        df = pd.read_csv(self.csv_path)

        text_col = _pick(df, TEXT_COLS)
        if text_col is None:
            raise ValueError(
                f"No recognizable review-text column in {self.csv_path.name}; "
                f"columns: {list(df.columns)}"
            )
        rating_col = _pick(df, RATING_COLS)
        time_col = _pick(df, TIME_COLS)
        id_col = _pick(df, ID_COLS)
        author_col = _pick(df, AUTHOR_COLS)

        df = df.dropna(subset=[text_col])
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.seed)

        for idx, row in df.iterrows():
            native_id = str(row[id_col]) if id_col else f"row{idx}"
            rating = None
            if rating_col is not None and pd.notna(row[rating_col]):
                try:
                    rating = float(row[rating_col])
                except (TypeError, ValueError):
                    rating = None
            timestamp = None
            if time_col is not None and pd.notna(row[time_col]):
                ts = pd.to_datetime(row[time_col], errors="coerce", utc=True)
                timestamp = None if pd.isna(ts) else ts.to_pydatetime()
            author = (
                hash_author(str(row[author_col]))
                if author_col is not None and pd.notna(row[author_col])
                else None
            )
            yield FeedbackItem(
                id=make_item_id(self.source_name, native_id),
                source=self.source_name,
                text=str(row[text_col]),
                rating=rating,
                timestamp=timestamp,
                author_hash=author,
                metadata={"dataset_file": self.csv_path.name},
            )
