"""Text normalization and filtering.

Keeps the original text for display (`text`) and adds a cleaned version
(`text_clean`) used for embedding/dedupe. Items that are too short or not
in the target language are dropped — counts are reported for transparency.
"""

from __future__ import annotations

import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
# Keep word chars and basic punctuation; drop emoji and symbol noise.
NOISE_RE = re.compile(r"[^\w\s.,!?'\"()\-:;/$%&+]")


def clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = NOISE_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _is_english(text: str) -> bool:
    from langdetect import LangDetectException, detect

    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_corpus(
    df: pd.DataFrame, min_chars: int = 15, language: str = "en"
) -> tuple[pd.DataFrame, dict]:
    """Returns (cleaned df, stats dict for the run report)."""
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0  # langdetect is stochastic without this

    stats = {"input": len(df)}
    df = df.copy()
    df["text_clean"] = df["text"].astype(str).map(clean_text)

    long_enough = df["text_clean"].str.len() >= min_chars
    stats["dropped_too_short"] = int((~long_enough).sum())
    df = df[long_enough]

    if language == "en":
        is_lang = df["text_clean"].map(_is_english)
        stats["dropped_non_english"] = int((~is_lang).sum())
        df = df[is_lang]

    df = df.reset_index(drop=True)
    stats["output"] = len(df)
    return df, stats
