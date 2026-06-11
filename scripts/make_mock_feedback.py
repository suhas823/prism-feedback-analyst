"""Generate a mock Instagram feedback CSV for demoing the upload feature.

    python scripts/make_mock_feedback.py
    -> data/sample_uploads/instagram_mock.csv  (~160 rows, 7 issue themes)
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

random.seed(42)

THEMES = {
    "story upload fails": (
        [
            "My stories keep failing to upload, stuck at 99% every single time",
            "Can't post stories anymore, it just says upload failed try again",
            "Stories won't upload on wifi or data, been like this for a week",
            "Every story I try to post fails and drains my battery retrying",
            "Upload failed error on every story since the last update, so annoying",
            "Stories stuck on posting forever, I have to force close the app",
        ],
        (1, 3),
    ),
    "reels algorithm complaints": (
        [
            "My reels get zero reach anymore, the algorithm buried my account",
            "Reels feed only shows me the same 5 creators over and over",
            "I keep seeing reels I already watched, the recommendations are broken",
            "Engagement dropped 90 percent, reels are not being shown to my followers",
            "The reels algorithm pushes random content I never interact with",
        ],
        (1, 3),
    ),
    "app crashes on open": (
        [
            "App crashes immediately when I open it on my Samsung",
            "Instagram closes itself after two seconds, reinstalled twice already",
            "Crashes every time I open the camera inside the app",
            "App keeps crashing on Android 14 since yesterday's update",
            "Constant crashing when switching between accounts",
            "Freezes then crashes whenever I open DMs",
        ],
        (1, 2),
    ),
    "ads too frequent": (
        [
            "Every third post is an ad now, the feed is unusable",
            "Way too many ads, more ads than posts from people I follow",
            "Sponsored posts everywhere, I opened the app to see friends not ads",
            "The number of ads is insane lately, considering deleting the app",
        ],
        (1, 3),
    ),
    "account hacked / login issues": (
        [
            "My account was hacked and support never responds to my reports",
            "Locked out of my account, the verification code never arrives",
            "Someone changed my email and Instagram gives me no way to recover",
            "Two factor authentication loop, can't log in for three days",
            "Suspicious login locked my account and the appeal form is broken",
        ],
        (1, 2),
    ),
    "dm notifications delayed": (
        [
            "DM notifications arrive hours late, I miss messages constantly",
            "Not getting message notifications at all unless I open the app",
            "Notifications for DMs are delayed or never show up",
        ],
        (2, 3),
    ),
    "positive feedback": (
        [
            "Love the new editing tools in reels, super smooth",
            "Great app, I use it every day to keep up with friends",
            "The new font options for stories are really nice",
            "Best social app out there, works great on my phone",
        ],
        (4, 5),
    ),
}

FILLERS = [
    "honestly", "seriously", "please fix this", "anyone else?",
    "using latest version", "on my iPhone", "on Android", "", "", "",
]


def main() -> None:
    rows = []
    now = datetime(2026, 6, 10)
    for theme, (texts, (lo, hi)) in THEMES.items():
        # Repeat each phrasing with light variation to build realistic volume.
        for i in range(4 if theme != "positive feedback" else 3):
            for t in texts:
                filler = random.choice(FILLERS)
                text = f"{t} {filler}".strip() if i % 2 else t
                rows.append(
                    {
                        "review_text": text,
                        "rating": random.randint(lo, hi),
                        "date": (now - timedelta(days=random.randint(0, 180))).date(),
                    }
                )
    random.shuffle(rows)
    out_dir = PROJECT_ROOT / "data" / "sample_uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "instagram_mock.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
