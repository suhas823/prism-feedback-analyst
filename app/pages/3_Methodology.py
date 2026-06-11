"""Methodology: the explainability contract, with live config values."""

import streamlit as st

import styles as ds
from shared import load_run_meta

st.set_page_config(page_title="Prism — Methodology", page_icon="✦", layout="wide")
ds.inject()

st.markdown(
    ds.hero(
        "Methodo",
        "logy",
        "The audit trail: how every number on this dashboard is produced, "
        "and the guardrails that keep insights honest.",
        badge="Prism · Explainability",
    ),
    unsafe_allow_html=True,
)

meta = load_run_meta()
cfg = (meta or {}).get("config", {})
w = cfg.get("scoring", {}).get("weights", {})
sm = cfg.get("scoring", {}).get("severity_mix", {})
sc = cfg.get("scoring", {})
cl = cfg.get("clustering", {})

st.markdown(ds.section("Pipeline"), unsafe_allow_html=True)
st.markdown(
    """
```
 Reviews CSV ─┐
              ├─► Unified schema ─► Clean ─► Dedupe ─► Embed (MiniLM, local)
 Tickets CSV ─┘                                            │
                                                           ▼
        Report ◄─ Score ◄─ LLM analysis (per cluster) ◄─ HDBSCAN clustering
```
"""
)

steps = [
    ("Ingest", "Every source is mapped into one FeedbackItem schema (id, source, text, rating, timestamp). Author ids are hashed."),
    ("Clean", "URLs/handles stripped; very short and non-English items dropped — counts reported in run metadata."),
    ("Dedupe", f"Exact duplicates removed; near-duplicates (cosine > {cfg.get('preprocess', {}).get('near_dup_cosine', 0.92)}) flagged: they count toward frequency but are never quoted as evidence."),
    ("Embed", "all-MiniLM-L6-v2 sentence embeddings, computed locally on CPU."),
    ("Cluster", f"PCA({cl.get('pca_components', 15)}) + HDBSCAN (min cluster size {cl.get('min_cluster_size', 8)}) finds dense theme cores; stragglers join the nearest core at cosine ≥ {cl.get('noise_reassign_cosine', 0.55)} (flagged, never quoted). The rest go to a visible Unclustered bucket."),
    ("Analyze", "ONE structured LLM call per cluster with a stratified quote sample. The model returns theme, summary, root-cause hypotheses, severity + rationale, actions — and must cite quote IDs, which are programmatically verified."),
    ("Score", "A transparent formula (below); no LLM involved."),
    ("Synthesize", "One final LLM call writes the executive summary."),
]
st.markdown(
    "".join(
        ds.cause_item(f"{name} — {desc}") for name, desc in steps
    ),
    unsafe_allow_html=True,
)

st.markdown(ds.section("Priority score (live weights)"), unsafe_allow_html=True)
st.markdown(
    ds.glass(
        f"<code style='font-size:1.02rem'>priority = {w.get('frequency', .35)}·F + "
        f"{w.get('severity', .35)}·S + {w.get('recency', .15)}·R + "
        f"{w.get('diversity', .15)}·D</code>"
    ),
    unsafe_allow_html=True,
)
st.markdown(
    f"""
| Component | Meaning | How it's computed |
|---|---|---|
| **F** frequency | How many users hit this | `log(1+n) / log(1+n_largest)` |
| **S** severity | How bad it is | `{sm.get('llm', .5)}·LLM + {sm.get('rating', .3)}·inverse rating + {sm.get('sentiment', .2)}·\\|sentiment\\|` |
| **R** recency | Is it still happening | exp decay, half-life {sc.get('recency_half_life_days', 90):.0f} days, per-source window |
| **D** diversity | Seen across sources? | normalized Shannon entropy |
"""
)

st.markdown(ds.section("Guardrails against misleading conclusions"), unsafe_allow_html=True)
guards = [
    f"Small samples — themes under {sc.get('min_items_insufficient', 5)} items are badged Insufficient evidence; under {sc.get('min_items_low_sample', 15)} Low sample. A 95% Wilson interval shows uncertainty on the negative share.",
    "Emotional bias — severity is judged from content; the prompt instructs the model that emotional language is frustration signal, not technical severity.",
    "One loud user — near-duplicate posts count once for quoting; author IDs are hashed.",
    "Hallucination — the LLM may only cite quote IDs it was given; citations are verified in code and failures flagged in the UI, never hidden.",
    "Skewed sourcing — source diversity is scored, and per-source counts are shown on every insight.",
]
st.markdown("".join(ds.cause_item(g) for g in guards), unsafe_allow_html=True)

st.markdown(ds.section("Limitations"), unsafe_allow_html=True)
lims = [
    "Root causes are hypotheses generated from feedback text — validate against logs/analytics before committing engineering work.",
    "The corpus is a configurable sample of two public datasets; absolute counts are not market-level estimates.",
    "Clustering quality varies; the cohesion metric and mixed-theme warnings say when to read a cluster skeptically.",
]
st.markdown("".join(ds.cause_item(l) for l in lims), unsafe_allow_html=True)

st.markdown(ds.section("Data sources & licensing"), unsafe_allow_html=True)
st.markdown(
    """
| Source | Dataset | Use |
|---|---|---|
| App reviews | Spotify Google Play reviews (Kaggle) | sampled, cleaned |
| Support tickets | [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle, CC BY-NC-SA) | inbound @SpotifyCares thread-starters |

Public datasets used for non-commercial coursework with attribution. Author identifiers are hashed at ingest.
"""
)

if meta:
    with st.expander("Full run configuration"):
        st.json(cfg)
