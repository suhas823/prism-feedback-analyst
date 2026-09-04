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

TEXT_COLS = ["review_text", "content", "Review", "review", "text", "body"]
RATING_COLS = ["review_rating", "score", "Rating", "rating", "stars"]
TIME_COLS = ["review_timestamp", "at", "Time_submitted", "date", "created_at"]
ID_COLS = ["review_id", "reviewId", "id"]
AUTHOR_COLS = ["pseudo_author_id", "author_name", "userName", "user_name"]


def _norm(name: str) -> str:
    """Fold a column name for comparison: 'Star_Rating' -> 'starrating'."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pick(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find the first column matching any candidate.

    Matching is case- and separator-insensitive, so 'Review_Text', 'review text'
    and 'reviewText' all match the candidate 'review_text'. Exact matches win
    over partial ones ('Star_Rating' matches the candidate 'rating').
    """
    normalized = {_norm(c): c for c in df.columns}

    for cand in candidates:
        hit = normalized.get(_norm(cand))
        if hit is not None:
            return hit

    for cand in candidates:
        n_cand = _norm(cand)
        if len(n_cand) < 4:  # too short to match on safely ('at', 'id')
            continue
        for n_col, col in normalized.items():
            if n_cand in n_col or n_col in n_cand:
                return col
    return None


def guess_text_column(df: pd.DataFrame) -> Optional[str]:
    """Best guess at the free-text column: a name match, else the column with
    the longest average text. Falling back to the first column is a trap, since
    that is usually a row number."""
    named = _pick(df, TEXT_COLS)
    if named is not None:
        return named

    best, best_len = None, 0.0
    for col in df.columns:
        if df[col].dtype.kind in "ifbc":  # numeric or boolean, not free text
            continue
        avg = df[col].astype(str).str.len().mean()
        if avg > best_len:
            best, best_len = col, avg
    # Free-text feedback is meaningfully longer than labels or IDs.
    return best if best_len >= 20 else (best or (df.columns[0] if len(df.columns) else None))


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
