"""Sentence embeddings (local, CPU) with disk caching.

The cache key covers the model name and the exact corpus, so a changed
sample or model invalidates it automatically.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _corpus_fingerprint(texts: pd.Series, model_name: str) -> str:
    h = hashlib.sha1(model_name.encode())
    h.update(str(len(texts)).encode())
    # Hash a stable digest of all texts (order matters).
    joined = "\x1f".join(texts.tolist())
    h.update(hashlib.sha1(joined.encode("utf-8", errors="replace")).digest())
    return h.hexdigest()[:16]


def embed_corpus(
    texts: pd.Series,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Returns L2-normalized embeddings, loading from cache when possible."""
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"emb_{_corpus_fingerprint(texts, model_name)}.npy"
        if cache_file.exists():
            return np.load(cache_file)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(
        texts.tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    if cache_file is not None:
        np.save(cache_file, emb)
    return emb
