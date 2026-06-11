"""Insight Detail: full drill-down for one theme — evidence, scores, trace."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import styles as ds
from shared import SOURCE_LABELS, require_artifacts

st.set_page_config(page_title="Prism — Insight Detail", page_icon="✦", layout="wide")
ds.inject()

df, insights = require_artifacts()
if not insights["insights"]:
    st.info("No insights produced — run the pipeline with LLM enabled.")
    st.stop()

st.markdown(
    ds.hero("Insight ", "Detail", "Every claim on this page traces back to real feedback — quotes, counts, score arithmetic, and the raw model trace.", badge="Prism · Drill-down"),
    unsafe_allow_html=True,
)

options = {
    f"#{rank}  {i['analysis']['theme_name']}  ·  priority {i['priority_score']:.2f}": i
    for rank, i in enumerate(insights["insights"], 1)
}
choice = st.selectbox("Theme", list(options.keys()), label_visibility="collapsed")
ins = options[choice]
a = ins["analysis"]
conf = ins["confidence"]

# ── KPI strip ────────────────────────────────────────────────────────────
st.markdown(
    ds.kpi_row(
        [
            ("Priority", f"{ins['priority_score']:.2f}", "of 1.00"),
            ("Severity", f"{a['severity']}/5", "LLM-rated, content-based"),
            ("Items", f"{conf['n_items']}", f"{conf['n_unique']} unique voices"),
            ("Negative share", f"{conf['negative_share']:.0%}",
             f"95% CI {conf['wilson_low']:.0%}–{conf['wilson_high']:.0%}"),
            ("Cohesion", f"{conf['cohesion']:.2f}", "mean pairwise similarity"),
        ]
    ),
    unsafe_allow_html=True,
)

chips = ds.confidence_chip(conf["badge"]) + " " + ds.severity_chip(a["severity"])
if conf["mixed_theme_warning"]:
    chips += " " + ds.chip("Mixed theme — read evidence", "warn")
chips += " " + (
    ds.chip("Citations verified", "ok")
    if ins["citation_check_passed"]
    else ds.chip("Citation check failed", "danger")
)
st.markdown(
    ds.glass(
        f"<h3 style='margin:0 0 .45rem 0'>{ds.esc(a['theme_name'])}</h3>"
        f"<div style='margin-bottom:.7rem'>{chips}</div>"
        f"<p style='color:#9AA3B5;line-height:1.6;margin:0'>{ds.esc(a['summary'])}</p>"
    ),
    unsafe_allow_html=True,
)

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown(ds.section("Hypothesized root causes"), unsafe_allow_html=True)
    st.markdown("".join(ds.cause_item(rc) for rc in a["root_causes"]), unsafe_allow_html=True)

    st.markdown(ds.section("Recommended actions"), unsafe_allow_html=True)
    st.markdown(
        "".join(
            ds.action_card(act["action"], act["effort"], act["rationale"])
            for act in a["recommended_actions"]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(ds.section("Evidence quotes"), unsafe_allow_html=True)
    st.caption(f"{'✅' if ins['citation_check_passed'] else '❌'} {ins['citation_check_detail']} — ⭐ = cited by the model")
    cited = set(a["evidence_quote_ids"])
    quotes_html = ""
    for q in ins["representative_quotes"]:
        src = SOURCE_LABELS.get(q["source"], q["source"])
        meta_bits = [src]
        if q.get("rating") is not None:
            meta_bits.append(f"{q['rating']:.0f}★")
        if q.get("timestamp"):
            meta_bits.append(q["timestamp"][:10])
        quotes_html += ds.quote_card(q["text"], "  ·  ".join(meta_bits), starred=q["id"] in cited)
    st.markdown(quotes_html, unsafe_allow_html=True)

with right:
    st.markdown(ds.section("Why this priority score"), unsafe_allow_html=True)
    comps = ins["score_components"]
    fig = go.Figure(
        go.Bar(
            x=[c["contribution"] for c in comps],
            y=[c["name"].replace("_", " ") for c in comps],
            orientation="h",
            text=[f"{c['normalized']:.2f} × {c['weight']:.2f}" for c in comps],
            textposition="auto",
            marker=dict(
                color=[ds.INDIGO, ds.ROSE, ds.CYAN, ds.AMBER],
                line=dict(width=0),
            ),
        )
    )
    fig.update_layout(**ds.PLOTLY_LAYOUT, height=230, xaxis_title="contribution to priority")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with st.expander("Component arithmetic"):
        for c in comps:
            st.caption(f"**{c['name']}**: {c['explanation']}")

    st.markdown(ds.section("Items per source"), unsafe_allow_html=True)
    src_counts = ins["per_source_counts"]
    fig2 = go.Figure(
        go.Bar(
            x=list(src_counts.values()),
            y=[SOURCE_LABELS.get(s, s) for s in src_counts],
            orientation="h",
            marker=dict(color=[ds.INDIGO, ds.CYAN]),
        )
    )
    fig2.update_layout(**ds.PLOTLY_LAYOUT, height=140)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    members = df[df["id"].isin(ins["member_ids"])].copy()
    if members["timestamp"].notna().any():
        members["month"] = (
            pd.to_datetime(members["timestamp"], utc=True).dt.to_period("M").astype(str)
        )
        trend = members.groupby("month").size().reset_index(name="mentions")
        st.markdown(ds.section("Mentions over time"), unsafe_allow_html=True)
        fig3 = go.Figure(
            go.Scatter(
                x=trend["month"],
                y=trend["mentions"],
                mode="lines",
                fill="tozeroy",
                line=dict(color=ds.CYAN, width=2),
                fillcolor="rgba(34,211,238,.12)",
            )
        )
        fig3.update_layout(**ds.PLOTLY_LAYOUT, height=180)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

st.divider()
with st.expander("🧠 LLM severity rationale"):
    st.write(a["severity_rationale"])
with st.expander("🔍 Raw LLM trace (full provenance)"):
    st.json(ins["llm_trace"])
