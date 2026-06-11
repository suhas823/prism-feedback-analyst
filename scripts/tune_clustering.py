"""Parameter sweep for the clustering stage.

    python scripts/tune_clustering.py

Loads the cached embeddings and tries combinations of PCA dimensionality and
HDBSCAN parameters, reporting cluster count and noise share. Used to choose
the defaults in config.yaml; kept for reproducibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.cluster import HDBSCAN  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    emb_files = sorted(cfg.paths.interim.glob("emb_*.npy"))
    if not emb_files:
        raise FileNotFoundError("No cached embeddings; run the pipeline first")
    emb = np.load(emb_files[-1])
    print(f"embeddings: {emb.shape}\n")
    print(f"{'pca':>4} {'mcs':>4} {'ms':>3} | {'clusters':>8} {'noise':>6} {'largest':>8}")

    for pca_dim in (15, 30, 50):
        reduced = PCA(n_components=pca_dim, random_state=42).fit_transform(emb)
        for mcs in (8, 15, 25):
            for ms in (1, 2, 3, 5):
                labels = HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    metric="euclidean",
                ).fit_predict(reduced)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise = float((labels == -1).mean())
                largest = max(
                    ((labels == c).sum() for c in set(labels) if c != -1),
                    default=0,
                )
                print(
                    f"{pca_dim:>4} {mcs:>4} {ms:>3} | "
                    f"{n_clusters:>8} {noise:>6.0%} {largest:>8}"
                )


if __name__ == "__main__":
    main()
