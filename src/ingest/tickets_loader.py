"""Adapter for support tickets derived from the Customer Support on Twitter
dataset (data/raw/tickets_raw.csv, pre-filtered to inbound @SpotifyCares
conversation starters by scripts/download_data.py).

Each first customer tweet in a thread is treated as one support ticket.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.ingest.base import FeedbackItem, SourceAdapter, hash_author, make_item_id

HANDLE_RE = re.compile(r"@\w+")


class SupportTicketsAdapter(SourceAdapter):
    source_name = "support_ticket"

    def __init__(self, csv_path: Path, sample_size: int, seed: int = 42):
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.seed = seed

    def load(self) -> Iterable[FeedbackItem]:
        df = pd.read_csv(self.csv_path)
        required = {"tweet_id", "author_id", "created_at", "text"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"tickets_raw.csv missing columns: {missing}")

        df = df.dropna(subset=["text"])
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.seed)

        for _, row in df.iterrows():
            # Strip @handles: they are routing noise, not feedback content.
            text = HANDLE_RE.sub("", str(row["text"])).strip()
            ts = pd.to_datetime(row["created_at"], errors="coerce", utc=True)
            yield FeedbackItem(
                id=make_item_id(self.source_name, str(row["tweet_id"])),
                source=self.source_name,
                text=text,
                rating=None,  # tweets carry no star rating
                timestamp=None if pd.isna(ts) else ts.to_pydatetime(),
                author_hash=hash_author(str(row["author_id"])),
                metadata={"dataset_file": self.csv_path.name},
            )
