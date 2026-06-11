"""Shared helpers for the Streamlit app: artifact loading and badge styling.

The dashboard only reads pipeline artifacts — it never calls the LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = PROJECT_ROOT / "data" / "processed"
WORKSPACES = PROJECT_ROOT / "data" / "workspaces"
DEFAULT_WS = "Spotify (demo dataset)"

BADGE_LABELS = {
    "ok": "✅ Solid evidence",
    "low_sample": "⚠️ Low sample",
    "insufficient_evidence": "🚫 Insufficient evidence",
}
SOURCE_LABELS = {
    "play_store_review": "📱 Play Store review",
    "support_ticket": "🎫 Support ticket",
}


def list_workspaces() -> dict[str, Path]:
    """Default dataset + every uploaded workspace that has artifacts."""
    ws: dict[str, Path] = {}
    if (PROCESSED / "insights.json").exists():
        ws[DEFAULT_WS] = PROCESSED
    if WORKSPACES.exists():
        for d in sorted(WORKSPACES.iterdir()):
            if (d / "insights.json").exists():
                ws[d.name] = d
    return ws


def active_workspace_dir() -> Path:
    ws = list_workspaces()
    chosen = st.session_state.get("prism_workspace")
    if chosen in ws:
        return ws[chosen]
    return next(iter(ws.values()), PROCESSED)


@st.cache_data(show_spinner=False)
def load_feedback(dir_str: str) -> pd.DataFrame | None:
    path = Path(dir_str) / "feedback.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_insights(dir_str: str) -> dict | None:
    path = Path(dir_str) / "insights.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_run_meta_from(dir_str: str) -> dict | None:
    path = Path(dir_str) / "run_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_meta() -> dict | None:
    return _load_run_meta_from(str(active_workspace_dir()))


def require_artifacts() -> tuple[pd.DataFrame, dict]:
    ws = list_workspaces()
    if not ws:
        st.warning(
            "No pipeline artifacts found. Run these first:\n\n"
            "```\npython scripts/download_data.py\npython -m src.pipeline.run\n```\n\n"
            "Or upload a CSV on the **New Analysis** page."
        )
        st.stop()
    names = list(ws)
    if len(names) > 1:
        st.sidebar.selectbox("📦 Dataset", names, key="prism_workspace")
    target = active_workspace_dir()
    df = load_feedback(str(target))
    insights = load_insights(str(target))
    if df is None or insights is None:
        st.warning("Selected dataset has no artifacts — re-run its analysis.")
        st.stop()
    return df, insights


def severity_chip(sev: int) -> str:
    colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🔥"}
    return f"{colors.get(sev, '⚪')} {sev}/5"


def insights_to_table(insights: dict) -> pd.DataFrame:
    rows = []
    for ins in insights["insights"]:
        rows.append(
            {
                "Theme": ins["analysis"]["theme_name"],
                "Priority": ins["priority_score"],
                "Severity": ins["analysis"]["severity"],
                "Items": ins["confidence"]["n_items"],
                "Confidence": BADGE_LABELS[ins["confidence"]["badge"]],
                "Sources": len(ins["per_source_counts"]),
                "Citations OK": "✅" if ins["citation_check_passed"] else "❌",
                "cluster_id": ins["cluster_id"],
            }
        )
    return pd.DataFrame(rows)


def export_markdown_report(insights: dict) -> str:
    """Render insights.json into a shareable Markdown report."""
    es = insights["executive_summary"]
    lines = [
        "# Product Feedback Insights Report",
        f"_Generated {insights['generated_at'][:10]} from "
        f"{insights['corpus']['n_items']:,} feedback items_",
        "",
        f"## {es['headline']}",
        "",
        es["summary"],
        "",
    ]
    if es["cross_theme_observations"]:
        lines.append("**Cross-theme observations:**")
        lines.extend(f"- {o}" for o in es["cross_theme_observations"])
        lines.append("")
    lines.append("**Suggested focus:**")
    lines.extend(f"- {f}" for f in es["suggested_focus"])
    lines.append("\n---\n")

    for i, ins in enumerate(insights["insights"], 1):
        a = ins["analysis"]
        c = ins["confidence"]
        lines += [
            f"## {i}. {a['theme_name']}",
            f"**Priority {ins['priority_score']:.2f}** · severity {a['severity']}/5 · "
            f"{c['n_items']} items · {BADGE_LABELS[c['badge']]}",
            "",
            a["summary"],
            "",
            "**Hypothesized root causes:**",
            *[f"- {rc}" for rc in a["root_causes"]],
            "",
            "**Recommended actions:**",
            *[
                f"- **{act['action']}** (effort: {act['effort']}) — {act['rationale']}"
                for act in a["recommended_actions"]
            ],
            "",
            "**Evidence sample:**",
            *[
                f"> {q['text'][:300]} — _{q['source']}_"
                for q in ins["representative_quotes"][:3]
            ],
            "",
        ]
    return "\n".join(lines)
