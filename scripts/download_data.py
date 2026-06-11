"""Download and pre-filter the two public Kaggle datasets into data/raw.

Sources:
  1. Spotify Google Play reviews  -> data/raw/reviews_raw.csv
     primary:  ashishkumarak/spotify-reviews-playstore-daily-update
     fallback: mfaaris/spotify-app-reviews-2022
  2. Customer Support on Twitter (thoughtvector/customer-support-on-twitter)
     filtered to inbound tweets mentioning @SpotifyCares -> data/raw/tickets_raw.csv

Run:  python scripts/download_data.py
Both datasets are public; kagglehub downloads them anonymously (no API key
needed). Raw data stays out of git — this script is the reproducible path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402

REVIEWS_DATASETS = [
    "ashishkumarak/spotify-reviews-playstore-daily-update",
    "mfaaris/spotify-app-reviews-2022",
]
TICKETS_DATASET = "thoughtvector/customer-support-on-twitter"
SUPPORT_HANDLE = "@spotifycares"

# Keep raw reviews manageable: most-recent N rows before ingest sampling.
MAX_RAW_REVIEWS = 30_000


def _find_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)


def download_reviews(raw_dir: Path) -> Path:
    import kagglehub

    last_err: Exception | None = None
    for dataset in REVIEWS_DATASETS:
        try:
            print(f"Downloading reviews dataset: {dataset} ...")
            path = Path(kagglehub.dataset_download(dataset))
            csvs = _find_csvs(path)
            if not csvs:
                raise FileNotFoundError(f"No CSV found in {path}")
            print(f"  using file: {csvs[0].name}")
            df = pd.read_csv(csvs[0])
            # Keep the most recent rows if a timestamp column exists.
            for ts_col in ("review_timestamp", "at", "Time_submitted", "date"):
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                    df = df.sort_values(ts_col, ascending=False)
                    break
            df = df.head(MAX_RAW_REVIEWS)
            out = raw_dir / "reviews_raw.csv"
            df.to_csv(out, index=False)
            print(f"  wrote {len(df):,} rows -> {out}")
            return out
        except Exception as e:  # try the fallback dataset
            print(f"  FAILED ({e}); trying next option")
            last_err = e
    raise RuntimeError(f"All reviews datasets failed: {last_err}")


def download_tickets(raw_dir: Path) -> Path:
    import kagglehub

    print(f"Downloading support tickets dataset: {TICKETS_DATASET} ...")
    print("  (this one is large, ~150MB compressed — be patient)")
    path = Path(kagglehub.dataset_download(TICKETS_DATASET))
    csvs = _find_csvs(path)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {path}")
    twcs = csvs[0]
    print(f"  using file: {twcs.name}; filtering to {SUPPORT_HANDLE} ...")

    # Stream in chunks: the full file is ~3M tweets.
    chunks = []
    for chunk in pd.read_csv(twcs, chunksize=200_000):
        mask = chunk["inbound"].astype(str).str.lower().eq("true") & chunk[
            "text"
        ].str.lower().str.contains(SUPPORT_HANDLE, na=False)
        chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True)
    print(f"  inbound tweets mentioning {SUPPORT_HANDLE}: {len(df):,}")

    # "Ticket" semantics: prefer conversation starters (not replying to anything).
    starters = df[df["in_response_to_tweet_id"].isna()]
    if len(starters) >= 500:
        df = starters
        print(f"  kept conversation starters only: {len(df):,}")
    else:
        print("  too few starters; keeping all inbound mentions")

    out = raw_dir / "tickets_raw.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {len(df):,} rows -> {out}")
    return out


def main() -> None:
    cfg = load_config()
    raw_dir = cfg.paths.raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    download_reviews(raw_dir)
    download_tickets(raw_dir)
    print("\nDone. Next: python -m src.pipeline.run")


if __name__ == "__main__":
    main()
