# ✦ Prism — AI Product Feedback Analyst

**Sprint III** — *From noise to spectrum.* Prism is an AI agent that transforms raw, noisy user feedback (app reviews + support tickets) into **prioritized, explainable, actionable product insights** — and ships with **Iris**, a chat analyst grounded in the generated insights.

> Feedback is noisy, emotional, fragmented, and biased. This agent clusters it into themes, hypothesizes root causes, scores each theme with a fully transparent formula, recommends concrete actions — and lets you audit every claim down to the original quotes.

## How it works

```
 Reviews CSV ─┐
              ├─► Unified schema ─► Clean ─► Dedupe ─► Embed (MiniLM, local)
 Tickets CSV ─┘                                            │
                                                           ▼
        Report ◄─ Score ◄─ LLM analysis (per cluster) ◄─ HDBSCAN clustering
```

**Design principle: classical ML per item, LLM per cluster.** Embeddings, clustering, and sentiment run locally and free. The LLM (Gemini 2.5 Flash free tier) sees only ~30 clusters with stratified quote samples — a full run costs ~35 API calls and re-runs are free (disk-cached).

**Trust features** (the problem's core constraint):
- Transparent priority formula: `0.35·frequency + 0.35·severity + 0.15·recency + 0.15·source-diversity` — every component's contribution is shown in the UI
- Wilson confidence intervals + "Low sample" / "Insufficient evidence" badges guard against small-sample conclusions
- The LLM must cite quote IDs, which are **programmatically verified** against real cluster members; failures are flagged, never hidden
- Full LLM trace (prompt version, model, raw response) stored per insight
- Unclustered items shown in a visible bucket, never silently dropped

## Setup

```powershell
# 1. Python 3.11/3.12 venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install (CPU-only torch first keeps the install small)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. API key (free): https://aistudio.google.com/apikey
copy .env.example .env     # then paste your GEMINI_API_KEY
```

## Run

```powershell
# 1. Download + filter the public datasets (no Kaggle account needed)
python scripts/download_data.py

# 2. Run the pipeline  (use --no-llm to test without an API key)
python -m src.pipeline.run

# 3. Open the dashboard
streamlit run app/Home.py

# Optional: clustering evaluation report
python -m src.eval.clustering_eval

# Tests
pytest
```

## Dashboard pages

| Page | What it shows |
|---|---|
| **New Analysis** | Upload any feedback CSV → column auto-mapping → full pipeline run → new switchable dataset (try `data/sample_uploads/instagram_mock.csv`) |
| **Home** | KPIs, executive summary, priority-ranked theme list, CSV/Markdown export |
| **Insight Detail** | Per-theme: root causes, actions, evidence quotes, score breakdown, Wilson interval, trend, raw LLM trace |
| **Explore Feedback** | Searchable raw corpus + 2-D semantic cluster map |
| **Methodology** | The full explainability contract: pipeline, formula with live weights, guardrails, limitations |
| **Ask Iris** | Chat with the analysis: "top 3 problems?", "what should we fix first?" — answers grounded ONLY in the generated insights, never invented |

## Project layout

```
config/config.yaml      all tunable knobs (weights, thresholds, model ids)
scripts/download_data.py
src/ingest/             FeedbackItem schema + one adapter per source
src/preprocess/         cleaning, exact + near-duplicate handling
src/analysis/           embeddings, HDBSCAN clustering, VADER sentiment
src/insights/           LLM client (throttle/retry/cache), prompts, schemas,
                        per-cluster analyst, transparent scoring, synthesis
src/pipeline/run.py     one-command orchestrator
src/eval/               clustering evaluation
app/                    Streamlit dashboard (reads artifacts only)
tests/                  scoring, dedupe, schema + citation tests
docs/                   architecture, evaluation, insight-quality rubric
```

## Data sources & attribution

- **Spotify Google Play reviews** — Kaggle (`ashishkumarak/spotify-reviews-playstore-daily-update`, fallback `mfaaris/spotify-app-reviews-2022`)
- **Customer Support on Twitter** — Kaggle (`thoughtvector/customer-support-on-twitter`, CC BY-NC-SA), filtered to inbound @SpotifyCares conversation starters

Used for non-commercial coursework with attribution. Author identifiers are hashed at ingest; raw data stays out of git.

## Limitations

Root causes are hypotheses generated from feedback text — validate against logs/analytics before committing engineering work. The corpus is a configurable sample, so absolute counts aren't market estimates. See the **Methodology** page for the complete list of guardrails and caveats.
