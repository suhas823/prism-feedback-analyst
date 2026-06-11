"""Clustering evaluation: HDBSCAN vs KMeans baseline + manual-check sample.

    python -m src.eval.clustering_eval

Writes docs/evaluation_results.md with:
  - silhouette / Davies-Bouldin / noise-share comparison
  - a manual-coherence checklist (5 random items from each of the 10 largest
    clusters) to be rated by a human
Caveat reported in output: silhouette is computed on PCA-50 vectors with
HDBSCAN noise points excluded, which flatters HDBSCAN slightly.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import davies_bouldin_score, silhouette_score  # noqa: E402

from src.analysis.cluster import kmeans_baseline  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    processed = cfg.paths.processed
    feedback_path = processed / "feedback.parquet"
    reduced_path = cfg.paths.interim / "reduced_50.npy"
    if not feedback_path.exists() or not reduced_path.exists():
        raise FileNotFoundError("Run the pipeline first: python -m src.pipeline.run")

    df = pd.read_parquet(feedback_path)
    reduced = np.load(reduced_path)
    labels = df["cluster_id"].to_numpy()

    # ── HDBSCAN metrics (noise excluded — stated caveat) ────────────────
    mask = labels != -1
    hdb_sil = silhouette_score(reduced[mask], labels[mask])
    hdb_db = davies_bouldin_score(reduced[mask], labels[mask])
    noise_share = float((~mask).mean())
    n_clusters = len(set(labels[mask]))

    # ── KMeans baseline ──────────────────────────────────────────────────
    km_labels, km_k, km_sil = kmeans_baseline(
        reduced, tuple(cfg.clustering.kmeans_k_range), cfg.data.random_seed
    )
    km_db = davies_bouldin_score(reduced, km_labels)

    # ── Manual coherence sample ──────────────────────────────────────────
    top_clusters = (
        df[df["cluster_id"] != -1]["cluster_id"].value_counts().head(10).index.tolist()
    )
    rng = np.random.default_rng(cfg.data.random_seed)
    checklist_rows = []
    for cid in top_clusters:
        grp = df[df["cluster_id"] == cid]
        sample = grp.sample(n=min(5, len(grp)), random_state=int(rng.integers(1e6)))
        for _, row in sample.iterrows():
            checklist_rows.append((cid, row["text"][:160].replace("|", "/")))

    # ── Report ───────────────────────────────────────────────────────────
    lines = [
        "# Clustering Evaluation",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} on "
        f"{len(df):,} items ({noise_share:.0%} noise)_",
        "",
        "## HDBSCAN vs KMeans baseline",
        "",
        "| Metric | HDBSCAN | KMeans (best k) |",
        "|---|---|---|",
        f"| clusters | {n_clusters} | {km_k} |",
        f"| silhouette ↑ | {hdb_sil:.3f} | {km_sil:.3f} |",
        f"| Davies-Bouldin ↓ | {hdb_db:.3f} | {km_db:.3f} |",
        f"| noise share | {noise_share:.1%} | 0% (forced assignment) |",
        "",
        "> **Caveat:** metrics use PCA-50 vectors; HDBSCAN noise points are",
        "> excluded from its scores, which flatters HDBSCAN — KMeans must place",
        "> every point. Read alongside the manual coherence check below.",
        "",
        "## Manual coherence check",
        "",
        "For each item, mark ✓ if it belongs with its cluster's theme.",
        "Report coherence = ✓ count / total.",
        "",
        "| Cluster | Item (first 160 chars) | Coherent? |",
        "|---|---|---|",
    ]
    lines += [f"| {cid} | {text} |  |" for cid, text in checklist_rows]

    out = PROJECT_ROOT / "docs" / "evaluation_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"HDBSCAN: {n_clusters} clusters, silhouette {hdb_sil:.3f}, DB {hdb_db:.3f}, "
          f"noise {noise_share:.1%}")
    print(f"KMeans:  k={km_k}, silhouette {km_sil:.3f}, DB {km_db:.3f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
