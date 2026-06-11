"""New Analysis: upload any feedback CSV → full pipeline → new workspace."""

import sys

import pandas as pd
import streamlit as st

import styles as ds
from shared import PROJECT_ROOT, list_workspaces

sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.reviews_loader import RATING_COLS, TEXT_COLS, TIME_COLS, _pick  # noqa: E402
from src.pipeline.workspace import MAX_UPLOAD_ROWS, run_analysis  # noqa: E402

st.set_page_config(page_title="Prism — New Analysis", page_icon="✦", layout="wide")
ds.inject()

st.markdown(
    ds.hero(
        "New ",
        "Analysis",
        "Upload any feedback CSV — app reviews, survey answers, support exports. "
        "Prism maps the columns, runs the full pipeline, and adds the result as "
        "a switchable dataset on every page.",
        badge="Prism · Upload & analyze",
    ),
    unsafe_allow_html=True,
)

GENERIC_TEXT_COLS = TEXT_COLS + ["feedback", "comment", "message", "body", "answer"]

uploaded = st.file_uploader(
    "Feedback CSV (one row per feedback item)", type=["csv"],
    help=f"Up to {MAX_UPLOAD_ROWS:,} rows are analyzed. Only a text column is required — "
    "rating and date columns are optional but improve scoring.",
)

if uploaded is None:
    sample = PROJECT_ROOT / "data" / "sample_uploads" / "instagram_mock.csv"
    if sample.exists():
        st.download_button(
            "⬇ Try it: download a mock Instagram feedback CSV",
            sample.read_bytes(),
            "instagram_mock.csv",
            "text/csv",
        )
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read that CSV: {e}")
    st.stop()

st.markdown(ds.section(f"Preview — {len(raw):,} rows"), unsafe_allow_html=True)
st.dataframe(raw.head(5), use_container_width=True, hide_index=True)

# ── Column mapping (auto-detected, overridable) ──────────────────────────
st.markdown(ds.section("Column mapping"), unsafe_allow_html=True)
cols = list(raw.columns)
NONE = "(none)"

c1, c2, c3 = st.columns(3)
text_guess = _pick(raw, GENERIC_TEXT_COLS) or cols[0]
text_col = c1.selectbox("Feedback text *", cols, index=cols.index(text_guess))

rating_guess = _pick(raw, RATING_COLS)
rating_col = c2.selectbox(
    "Star rating (optional)", [NONE] + cols,
    index=(cols.index(rating_guess) + 1) if rating_guess else 0,
)
time_guess = _pick(raw, TIME_COLS + ["date", "created_at", "Date"])
time_col = c3.selectbox(
    "Date (optional)", [NONE] + cols,
    index=(cols.index(time_guess) + 1) if time_guess else 0,
)

c4, c5 = st.columns(2)
default_name = uploaded.name.rsplit(".", 1)[0].replace("_", " ").title()
ws_name = c4.text_input("Dataset name", value=default_name)
source_name = c5.text_input("Source label", value="app review",
                            help="e.g. app review, survey response, support ticket")

if ws_name.strip() and ws_name in list_workspaces():
    st.warning(f"A dataset named “{ws_name}” exists — analyzing will overwrite it.")

# ── Run ──────────────────────────────────────────────────────────────────
if st.button("✦ Analyze", type="primary", use_container_width=True):
    with st.status("Running the Prism pipeline…", expanded=True) as status:
        log_box = st.empty()
        lines: list[str] = []

        def progress(msg: str) -> None:
            lines.append(f"• {msg}")
            log_box.markdown("\n\n".join(lines[-8:]))

        try:
            slug, n_insights = run_analysis(
                raw,
                text_col=text_col,
                rating_col=None if rating_col == NONE else rating_col,
                time_col=None if time_col == NONE else time_col,
                source_name=source_name.strip() or "uploaded",
                workspace_name=ws_name.strip() or "Uploaded dataset",
                progress=progress,
            )
        except Exception as e:
            status.update(label="Analysis failed", state="error")
            st.error(str(e))
            st.stop()
        status.update(
            label=f"Done — {n_insights} themes analyzed", state="complete", expanded=False
        )

    # Make the new workspace active everywhere and clear stale caches.
    st.cache_data.clear()
    st.session_state["prism_workspace"] = slug
    st.success(f"**{ws_name}** is ready — {n_insights} prioritized themes.")
    st.balloons()
    nav1, nav2 = st.columns(2)
    nav1.page_link("Home.py", label="✦ View insights")
    nav2.page_link("pages/4_Ask_Iris.py", label="💠 Ask Iris about this data")
