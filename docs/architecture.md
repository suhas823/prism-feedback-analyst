# Architecture

## Design principle

**Classical ML per item, LLM per cluster.** Every per-item operation
(embedding, clustering, sentiment, dedupe) is local, free, and deterministic.
The LLM is reserved for what it is uniquely good at — naming themes,
hypothesizing root causes, judging severity, proposing actions — and sees
only cluster-level stratified samples. A full pipeline run makes ~35 API
calls regardless of corpus size, which keeps it inside any free tier and
makes runs reproducible (responses are disk-cached by prompt hash).

## Pipeline stages

| # | Stage | Module | Technique |
|---|-------|--------|-----------|
| 1 | Ingest | `src/ingest/` | Source adapters → unified `FeedbackItem` (pydantic). Author IDs hashed. |
| 2 | Normalize | `src/preprocess/clean.py` | URL/handle/emoji stripping, min-length filter, langdetect English filter |
| 3 | Dedupe | `src/preprocess/dedupe.py` | Exact dupes dropped (SHA-1 of normalized text); near-dups (cosine > 0.92) flagged, kept for frequency, excluded from quoting |
| 4 | Embed | `src/analysis/embed.py` | all-MiniLM-L6-v2, L2-normalized, fingerprint-cached |
| 5 | Cluster | `src/analysis/cluster.py` | PCA(15) → sklearn-native HDBSCAN (min_samples=1) finds dense theme cores; noise soft-assigns to nearest core centroid at cosine ≥ 0.55 (flagged `from_noise`, excluded from evidence quotes); the rest → visible Unclustered bucket; KMeans baseline for eval. Parameters chosen via `scripts/tune_clustering.py` sweep |
| 6 | Signals | `src/analysis/sentiment.py` | VADER compound; severity prior from rating/sentiment |
| 7 | Analyze | `src/insights/cluster_analyst.py` | One structured Gemini call per cluster; stratified quote sample (centroid-core + high-severity + per-source coverage); citation verification |
| 8 | Score | `src/insights/scoring.py` | Transparent formula + Wilson bounds + badges (below) |
| 9 | Synthesize | `src/insights/synthesis.py` | One call: executive summary |
| 10 | Artifacts | `src/pipeline/run.py` | feedback.parquet, clusters.parquet, insights.json, run_meta.json |

## Scoring

```
priority = 0.35·F + 0.35·S + 0.15·R + 0.15·D

F = log(1+n) / log(1+n_largest)
S = 0.5·LLM_severity + 0.3·inverse_rating + 0.2·|VADER|
    (rating weight folds into sentiment for unrated sources)
R = mean exp(−ln2 · age/90d), age relative to each source's collection window
D = normalized Shannon entropy over sources
```

Weights live in `config/config.yaml`. Each component is persisted as
raw value → normalized → weight → contribution, so the UI renders the
exact arithmetic.

## Trust boundary

`insights.json` keeps a hard boundary between **model output** (the
`analysis` object, exactly as the LLM returned it) and **computed fact**
(scores, confidence, counts, citation checks). The raw LLM response is
stored verbatim in `llm_trace`. Nothing the model says is presented as a
measurement; nothing measured is attributed to the model.

## Free-tier survival (`src/insights/llm_client.py`)

- Sliding-window throttle (8 RPM default) under Gemini free-tier limits
- Tenacity exponential backoff on 429/5xx
- Disk cache keyed by (provider, model, prompt version, full prompt) —
  validated **before** caching, so bad responses are never persisted
- Provider swap (Gemini ↔ Groq) is one config/env change
