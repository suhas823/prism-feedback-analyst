"""Tests for duplicate handling."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess.clean import clean_text
from src.preprocess.dedupe import drop_exact_dupes, flag_near_dupes


def _normed(vectors: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class TestExactDupes:
    def test_drops_case_insensitive_dupes(self):
        df = pd.DataFrame(
            {"text_clean": ["App keeps crashing", "app keeps crashing", "Love it"]}
        )
        out, dropped = drop_exact_dupes(df)
        assert dropped == 1
        assert len(out) == 2

    def test_no_dupes_unchanged(self):
        df = pd.DataFrame({"text_clean": ["a b c d e", "f g h i j"]})
        out, dropped = drop_exact_dupes(df)
        assert dropped == 0
        assert len(out) == 2


class TestNearDupes:
    def test_flags_only_later_copies(self):
        # Three near-identical vectors and one distinct: first stays original.
        emb = _normed([[1, 0, 0], [0.999, 0.01, 0], [0.998, 0.02, 0], [0, 1, 0]])
        df = pd.DataFrame({"text_clean": ["a", "b", "c", "d"]})
        out = flag_near_dupes(df, emb, threshold=0.92)
        assert out["is_near_dup"].tolist() == [False, True, True, False]

    def test_distinct_items_not_flagged(self):
        emb = _normed([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        df = pd.DataFrame({"text_clean": ["a", "b", "c"]})
        out = flag_near_dupes(df, emb, threshold=0.92)
        assert not out["is_near_dup"].any()

    def test_length_mismatch_raises(self):
        import pytest

        df = pd.DataFrame({"text_clean": ["a"]})
        with pytest.raises(ValueError):
            flag_near_dupes(df, np.eye(3, dtype=np.float32))


class TestCleanText:
    def test_strips_urls_and_handles(self):
        out = clean_text("@SpotifyCares app broken see https://x.co/abc now")
        assert "http" not in out and "@" not in out
        assert "app broken" in out

    def test_collapses_whitespace(self):
        assert clean_text("a   b\n\nc") == "a b c"


class TestCleanCorpusEmptyPath:
    """Regression: an empty frame after the length filter used to crash with
    'invalid literal for int() with base 10' when computing language stats."""

    @staticmethod
    def _frame(texts):
        return pd.DataFrame(
            {
                "id": [f"x:{i}" for i in range(len(texts))],
                "source": "s",
                "text": texts,
                "rating": np.nan,
                "timestamp": pd.Series(
                    pd.NaT, index=range(len(texts)), dtype="datetime64[ns, UTC]"
                ),
            }
        )

    def test_all_rows_too_short_does_not_crash(self):
        from src.preprocess.clean import clean_corpus

        out, stats = clean_corpus(self._frame(["ok"] * 50), min_chars=15)
        assert out.empty
        assert stats["dropped_too_short"] == 50
        assert stats["dropped_non_english"] == 0
        assert stats["output"] == 0

    def test_normal_corpus_still_filters(self):
        from src.preprocess.clean import clean_corpus

        texts = ["the app crashes constantly on startup"] * 10 + ["ok"] * 3
        out, stats = clean_corpus(self._frame(texts), min_chars=15)
        assert len(out) == 10
        assert stats["dropped_too_short"] == 3
