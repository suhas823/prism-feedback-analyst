"""Clustering: PCA(50) -> HDBSCAN (sklearn-native), with a KMeans baseline
for evaluation and per-cluster cohesion metrics.

HDBSCAN label -1 means "noise" — those items go to the visible Unclustered
bucket rather than being silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA


@dataclass
class ClusteringResult:
    labels: np.ndarray                      # per-item cluster id; -1 = noise
    reduced_50: np.ndarray                  # PCA vectors used for clustering
    reduced_2: np.ndarray                   # PCA(2) for the scatter plot
    from_noise: np.ndarray = None           # True where item was soft-assigned
    cohesion: dict[int, float] = field(default_factory=dict)  # mean pairwise cosine
    centroid_order: dict[int, np.ndarray] = field(default_factory=dict)
    # indices of cluster members ordered nearest-to-centroid first


def _soft_assign_noise(
    embeddings: np.ndarray, labels: np.ndarray, min_cosine: float
) -> tuple[np.ndarray, np.ndarray]:
    """Assign noise points to the nearest dense-core centroid when similar
    enough. Short noisy feedback leaves HDBSCAN with large noise shares; the
    dense cores are good theme seeds, so related stragglers join them while
    genuinely diffuse items stay unclustered. Reassigned items are flagged so
    the UI and quote selection can treat them as second-class evidence."""
    cluster_ids = sorted(c for c in set(labels) if c != -1)
    if not cluster_ids:
        return labels, np.zeros(len(labels), dtype=bool)

    centroids = []
    for cid in cluster_ids:
        c = embeddings[labels == cid].mean(axis=0)
        centroids.append(c / (np.linalg.norm(c) + 1e-12))
    centroids = np.stack(centroids)  # (k, dim)

    labels = labels.copy()
    from_noise = np.zeros(len(labels), dtype=bool)
    noise_idx = np.where(labels == -1)[0]
    if noise_idx.size:
        sims = embeddings[noise_idx] @ centroids.T  # (n_noise, k)
        best = sims.argmax(axis=1)
        best_sim = sims[np.arange(len(noise_idx)), best]
        accept = best_sim >= min_cosine
        labels[noise_idx[accept]] = np.array(cluster_ids)[best[accept]]
        from_noise[noise_idx[accept]] = True
    return labels, from_noise


def run_clustering(
    embeddings: np.ndarray,
    pca_components: int = 15,
    min_cluster_size: int = 8,
    min_samples: int = 1,
    noise_reassign_cosine: float = 0.55,
    seed: int = 42,
) -> ClusteringResult:
    n = len(embeddings)
    n_comp = min(pca_components, n - 1, embeddings.shape[1])
    pca50 = PCA(n_components=n_comp, random_state=seed)
    reduced = pca50.fit_transform(embeddings).astype(np.float32)

    pca2 = PCA(n_components=2, random_state=seed)
    reduced2 = pca2.fit_transform(embeddings).astype(np.float32)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)
    labels, from_noise = _soft_assign_noise(embeddings, labels, noise_reassign_cosine)

    result = ClusteringResult(
        labels=labels, reduced_50=reduced, reduced_2=reduced2, from_noise=from_noise
    )

    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        members = embeddings[idx]  # original normalized vectors → dot = cosine
        centroid = members.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        sims_to_centroid = members @ centroid
        result.centroid_order[cid] = idx[np.argsort(-sims_to_centroid)]

        if len(idx) > 1:
            sims = members @ members.T
            off_diag = sims[np.triu_indices(len(idx), k=1)]
            result.cohesion[cid] = float(off_diag.mean())
        else:
            result.cohesion[cid] = 1.0

    return result


def kmeans_baseline(
    reduced_50: np.ndarray, k_range: tuple[int, int] = (8, 30), seed: int = 42
) -> tuple[np.ndarray, int, float]:
    """Silhouette-swept KMeans used only for evaluation comparison."""
    from sklearn.metrics import silhouette_score

    best = (None, -1, -2.0)  # labels, k, score
    lo, hi = k_range
    hi = min(hi, len(reduced_50) - 1)
    for k in range(lo, hi + 1, 2):
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        labels = km.fit_predict(reduced_50)
        score = silhouette_score(reduced_50, labels)
        if score > best[2]:
            best = (labels, k, score)
    return best


def cluster_summary(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster counts and source breakdown, including the noise bucket."""
    out = df.copy()
    out["cluster_id"] = labels
    rows = []
    for cid, grp in out.groupby("cluster_id"):
        rows.append(
            {
                "cluster_id": cid,
                "n_items": len(grp),
                "n_unique": int((~grp.get("is_near_dup", False)).sum()),
                "sources": grp["source"].value_counts().to_dict(),
            }
        )
    return pd.DataFrame(rows).sort_values("n_items", ascending=False).reset_index(drop=True)
