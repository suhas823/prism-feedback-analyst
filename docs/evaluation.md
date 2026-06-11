# Evaluation Plan

Three layers, from automated to human judgment.

## 1. Clustering quality (automated)

Run `python -m src.eval.clustering_eval` after a pipeline run. It writes
`docs/evaluation_results.md` with:

- **Silhouette score** (higher better) and **Davies-Bouldin** (lower better)
  for HDBSCAN vs a silhouette-swept KMeans baseline, both on the PCA-50
  vectors actually used for clustering.
- **Noise share** — HDBSCAN's refusal rate. A moderate noise share is
  expected and honest; forcing every item into a theme (KMeans) inflates
  apparent coverage at the cost of theme purity.
- **Caveat stated in the report:** HDBSCAN's metrics exclude its noise
  points, which flatters it relative to KMeans. That's why the comparison
  is read alongside the manual check, not alone.

## 2. Manual cluster coherence (human, ~20 min)

The eval script samples 5 items from each of the 10 largest clusters into a
checklist table. Rate each item ✓/✗ for "belongs with this cluster's theme"
and report **coherence = ✓ / total**. Target: ≥ 80%.

## 3. Insight quality (human, rubric)

Score the top-10 insights with `docs/rubric.md` (groundedness,
actionability, root-cause plausibility, evidence sufficiency, clarity,
each 1–5). Two raters if possible.

## 4. Faithfulness (automated, every run)

Every insight's `evidence_quote_ids` are verified against actual cluster
membership at pipeline time. The Home page shows the pass rate
("Citations verified N/M"). Target: 100%; any failure is visibly flagged
on the insight itself.

## Recording results

After each evaluation round, append a dated section to
`evaluation_results.md` with: corpus size, config hash (from
run_meta.json), metric values, coherence %, rubric means, and one
paragraph of qualitative observations.
