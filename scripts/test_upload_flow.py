"""Smoke-test the upload-analysis engine outside Streamlit.

    python scripts/test_upload_flow.py
Runs run_analysis() on the mock Instagram CSV and prints the result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline.workspace import WORKSPACES_DIR, run_analysis  # noqa: E402


def main() -> None:
    raw = pd.read_csv(PROJECT_ROOT / "data" / "sample_uploads" / "instagram_mock.csv")
    slug, n = run_analysis(
        raw,
        text_col="review_text",
        rating_col="rating",
        time_col="date",
        source_name="app review",
        workspace_name="Instagram Mock",
        progress=lambda m: print(f"  > {m}"),
    )
    print(f"\nworkspace: {slug}, insights: {n}")
    data = json.loads((WORKSPACES_DIR / slug / "insights.json").read_text(encoding="utf-8"))
    print("headline:", data["executive_summary"]["headline"])
    for i in data["insights"][:5]:
        a = i["analysis"]
        print(f"  [{i['priority_score']:.2f}] {a['theme_name']} "
              f"(sev {a['severity']}/5, n={i['confidence']['n_items']}, "
              f"citations={'OK' if i['citation_check_passed'] else 'FAIL'})")


if __name__ == "__main__":
    main()
