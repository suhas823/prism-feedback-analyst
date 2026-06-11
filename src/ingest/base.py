"""Unified feedback schema and the source-adapter interface.

Every data source (app reviews, support tickets, surveys, ...) is loaded by
an adapter that maps its native columns into `FeedbackItem`. Downstream
stages only ever see this one schema, which is what makes the pipeline
multi-source by construction.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
from pydantic import BaseModel, Field


class FeedbackItem(BaseModel):
    id: str
    source: str                       # e.g. "play_store_review", "support_ticket"
    text: str
    rating: Optional[float] = None    # 1-5 stars where the source has them
    timestamp: Optional[datetime] = None
    author_hash: Optional[str] = None # hashed author id — never raw PII
    metadata: dict = Field(default_factory=dict)


def make_item_id(source: str, native_id: str) -> str:
    return f"{source}:{native_id}"


def hash_author(raw_author: str) -> str:
    return hashlib.sha1(raw_author.encode("utf-8")).hexdigest()[:12]


class SourceAdapter(ABC):
    """Loads one feedback source into the unified schema."""

    source_name: str

    @abstractmethod
    def load(self) -> Iterable[FeedbackItem]:
        ...

    def to_dataframe(self) -> pd.DataFrame:
        rows = [item.model_dump() for item in self.load()]
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"Adapter {self.source_name!r} produced no rows")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df


def combine_sources(adapters: list[SourceAdapter]) -> pd.DataFrame:
    """Run every adapter and concatenate into one corpus."""
    frames = [a.to_dataframe() for a in adapters]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    return df
