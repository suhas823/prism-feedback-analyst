"""Duplicate handling.

Exact duplicates (same normalized text) are dropped — they are usually spam
or double-submissions. Near-duplicates (cosine similarity above threshold on
embeddings) are KEPT but flagged `is_near_dup`: they still count toward an
issue's frequency, but representative quotes skip them so one viral phrasing
can't dominate the evidence panel.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.lower().strip().encode("utf-8")).hexdigest()


def drop_exact_dupes(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df["_hash"] = df["text_clean"].map(_text_hash)
    before = len(df)
    df = df.drop_duplicates(subset="_hash").drop(columns="_hash").reset_index(drop=True)
    return df, before - len(df)


def flag_near_dupes(
    df: pd.DataFrame, embeddings: np.ndarray, threshold: float = 0.92
) -> pd.DataFrame:
    """Flags items whose nearest earlier neighbor exceeds `threshold` cosine.

    Embeddings must be L2-normalized (dot product == cosine). Processes in
    blocks to keep memory bounded for corpora of a few thousand items.
    """
    n = len(df)
    if n != len(embeddings):
        raise ValueError(f"df ({n}) and embeddings ({len(embeddings)}) length mismatch")

    is_dup = np.zeros(n, dtype=bool)
    block = 512
    for start in range(0, n, block):
        end = min(start + block, n)
        if start == 0 and end == n and n <= block:
            sims = embeddings @ embeddings.T
            # Only earlier items count as "the original".
            tri = np.tril(sims, k=-1)
            is_dup = tri.max(axis=1, initial=0.0) > threshold
            break
        sims = embeddings[start:end] @ embeddings[:end].T
        for i in range(end - start):
            row = sims[i, : start + i]  # strictly earlier items
            if row.size and row.max() > threshold:
                is_dup[start + i] = True

    df = df.copy()
    df["is_near_dup"] = is_dup
    return df
