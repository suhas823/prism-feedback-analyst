"""Home: hero, KPI cards, executive summary, ranked themes. streamlit run app/Home.py"""

import streamlit as st

import styles as ds
from shared import export_markdown_report, insights_to_table, load_run_meta, require_artifacts

st.set_page_config(
    page_title="Prism — AI Feedback Analyst",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
ds.inject()

df, insights = require_artifacts()
corpus = insights["corpus"]

# ── Hero ─────────────────────────────────────────────────────────────────
st.markdown(
    ds.hero(
        "",
        "Prism",
        "From noise to spectrum — raw reviews and support tickets, clustered "
        "into themes, analyzed for root causes, and prioritized with a fully "
        "transparent score. Every insight links back to its evidence.",
        badge="AI Product Feedback Analyst",
    ),
    unsafe_allow_html=True,
)
st.page_link("pages/4_Ask_Iris.py", label="✦ Ask Iris — chat with your feedback analysis")

# ── KPI row ──────────────────────────────────────────────────────────────
n_low_conf = sum(1 for i in insights["insights"] if i["confidence"]["badge"] != "ok")
citations_ok = sum(1 for i in insights["insights"] if i["citation_check_passed"])
st.markdown(
    ds.kpi_row(
        [
            ("Feedback items", f"{corpus['n_items']:,}", "after cleaning & dedupe"),
            ("Sources", str(len(corpus["per_source"])), "reviews + support tickets"),
            ("Themes found", str(len(insights["insights"])), f"{corpus['n_clusters']} raw clusters"),
            ("Low-confidence flags", str(n_low_conf), "honest uncertainty"),
            ("Citations verified", f"{citations_ok}/{len(insights['insights'])}", "programmatic check"),
        ]
    ),
    unsafe_allow_html=True,
)

# ── Executive summary ────────────────────────────────────────────────────
es = insights["executive_summary"]
focus_html = "".join(ds.cause_item(f) for f in es["suggested_focus"])
st.markdown(
    ds.glass(
        f"<h3 style='margin:0 0 .5rem 0'>{ds.esc(es['headline'])}</h3>"
        f"<p style='color:#9AA3B5;line-height:1.6;margin:0 0 .9rem 0'>{ds.esc(es['summary'])}</p>"
        f"<div class='glass-title'>Suggested focus</div>{focus_html}",
        eyebrow="Executive summary",
    ),
    unsafe_allow_html=True,
)
if es["cross_theme_observations"]:
    with st.expander("Cross-theme observations"):
        for o in es["cross_theme_observations"]:
            st.markdown(f"- {o}")

# ── Filters ──────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Filters")
hide_low = st.sidebar.toggle("Hide low-confidence themes", value=False)
min_items = st.sidebar.slider("Minimum items per theme", 1, 100, 1)
source_filter = st.sidebar.multiselect(
    "Must include source", options=sorted(corpus["per_source"].keys())
)
top_n = st.sidebar.slider("Show top N themes", 5, len(insights["insights"]), 15)

filtered = [i for i in insights["insights"] if i["confidence"]["n_items"] >= min_items]
if hide_low:
    filtered = [i for i in filtered if i["confidence"]["badge"] == "ok"]
for src in source_filter:
    filtered = [i for i in filtered if src in i["per_source_counts"]]

# ── Ranked theme list ────────────────────────────────────────────────────
st.markdown(ds.section(f"Prioritized themes — {len(filtered)} total"), unsafe_allow_html=True)
rows_html = ""
for rank, ins in enumerate(filtered[:top_n], 1):
    a = ins["analysis"]
    c = ins["confidence"]
    n_src = len(ins["per_source_counts"])
    sub = f"{c['n_items']} items · {n_src} source{'s' if n_src > 1 else ''}"
    if not ins["citation_check_passed"]:
        sub += " · ⚠ citation check failed"
    rows_html += ds.theme_row(
        rank,
        a["theme_name"],
        sub,
        ins["priority_score"],
        ds.severity_chip(a["severity"]),
        ds.confidence_chip(c["badge"]),
    )
st.markdown(rows_html, unsafe_allow_html=True)
st.caption("Open **Insight Detail** in the sidebar to drill into any theme — evidence quotes, score breakdown, and the raw LLM trace.")

# ── Exports ──────────────────────────────────────────────────────────────
st.markdown(ds.section("Export"), unsafe_allow_html=True)
table = insights_to_table(insights)
exp1, exp2, _ = st.columns([1, 1, 2])
exp1.download_button(
    "⬇ Insights CSV",
    table.drop(columns="cluster_id").to_csv(index=False).encode("utf-8"),
    "insights.csv",
    "text/csv",
)
exp2.download_button(
    "⬇ Full report (Markdown)",
    export_markdown_report(insights).encode("utf-8"),
    "insights_report.md",
    "text/markdown",
)

meta = load_run_meta()
if meta:
    with st.expander("Run metadata"):
        st.json(meta)

# ── Floating Iris (video avatar → chat page) ─────────────────────────────
st.markdown(
    ds.iris_float(
        "/app/static/iris.mp4", "/Ask_Iris", "Ask Iris anything about feedback"
    ),
    unsafe_allow_html=True,
)
