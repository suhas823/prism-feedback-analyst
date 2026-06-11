"""Explore Feedback: semantic cluster map + searchable raw corpus."""

import pandas as pd
import plotly.express as px
import streamlit as st

import styles as ds
from shared import require_artifacts

st.set_page_config(page_title="Prism — Explore Feedback", page_icon="✦", layout="wide")
ds.inject()

df, insights = require_artifacts()

st.markdown(
    ds.hero(
        "Explore ",
        "Feedback",
        "Each point is one feedback item, positioned by semantic similarity. "
        "Grey points fit no theme — shown, never hidden.",
        badge="Prism · Corpus map",
    ),
    unsafe_allow_html=True,
)

theme_names = {i["cluster_id"]: i["analysis"]["theme_name"] for i in insights["insights"]}
df = df.copy()
df["theme"] = df["cluster_id"].map(theme_names).fillna("Unclustered")

# ── Cluster map ──────────────────────────────────────────────────────────
top_themes = (
    df[df["theme"] != "Unclustered"]["theme"].value_counts().head(12).index.tolist()
)
plot_df = df.copy()
plot_df["theme_view"] = plot_df["theme"].where(
    plot_df["theme"].isin(top_themes + ["Unclustered"]), "Other themes"
)
plot_df["short_text"] = plot_df["text"].str.slice(0, 110)

fig = px.scatter(
    plot_df,
    x="pca_x",
    y="pca_y",
    color="theme_view",
    hover_data={"short_text": True, "source": True, "pca_x": False, "pca_y": False,
                "theme_view": False},
    opacity=0.65,
    height=560,
    color_discrete_map={"Unclustered": "rgba(140,148,165,.35)", "Other themes": "rgba(167,139,250,.5)"},
)
fig.update_traces(marker=dict(size=5, line=dict(width=0)))
fig.update_layout(
    **{
        **ds.PLOTLY_LAYOUT,
        "legend": dict(title="", orientation="v", font=dict(size=11)),
        "xaxis": dict(visible=False),
        "yaxis": dict(visible=False),
    }
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Raw table ────────────────────────────────────────────────────────────
st.markdown(ds.section("Raw feedback"), unsafe_allow_html=True)
c1, c2, c3 = st.columns([2, 1, 1])
query = c1.text_input("Search text", placeholder="e.g. crash, premium, login…")
source = c2.selectbox("Source", ["all"] + sorted(df["source"].unique().tolist()))
theme = c3.selectbox("Theme", ["all"] + sorted(df["theme"].unique().tolist()))

view = df
if query:
    view = view[view["text"].str.contains(query, case=False, na=False)]
if source != "all":
    view = view[view["source"] == source]
if theme != "all":
    view = view[view["theme"] == theme]

st.markdown(
    ds.chip(f"{len(view):,} of {len(df):,} items", "info"), unsafe_allow_html=True
)
st.dataframe(
    view[["text", "source", "theme", "rating", "vader_compound", "timestamp"]],
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "text": st.column_config.TextColumn("Feedback", width="large"),
        "vader_compound": st.column_config.NumberColumn("Sentiment", format="%.2f"),
        "timestamp": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
    },
)
