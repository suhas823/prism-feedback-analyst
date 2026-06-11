"""In-app analysis for uploaded CSVs → self-contained workspaces.

A workspace is a directory under data/workspaces/<slug>/ holding the same
artifacts the CLI pipeline writes (feedback.parquet, insights.json,
run_meta.json), so every dashboard page can read it interchangeably with
the default dataset.

Uploads run on the lightweight LLM (separate free-tier bucket) so a live
demo can never exhaust the main analysis model's quota.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from src.analysis.cluster import run_clustering
from src.analysis.embed import embed_corpus
from src.analysis.sentiment import add_sentiment
from src.config import PROJECT_ROOT, load_config
from src.insights.cluster_analyst import analyze_cluster, verify_citations
from src.insights.llm_client import LLMClient
from src.insights.schemas import ExecutiveSummary, Insight
from src.insights.scoring import score_cluster
from src.insights.synthesis import synthesize
from src.pipeline.run import add_source_relative_age
from src.preprocess.clean import clean_corpus
from src.preprocess.dedupe import drop_exact_dupes, flag_near_dupes

WORKSPACES_DIR = PROJECT_ROOT / "data" / "workspaces"

MAX_UPLOAD_ROWS = 5000
MAX_CLUSTERS_FOR_UPLOAD = 12


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "upload"


def _light_client(cfg) -> LLMClient:
    """LLM client on the lightweight model bucket, with a faster throttle
    (the small model's free-tier RPM is much higher)."""
    llm_cfg = cfg.llm.model_copy()
    if llm_cfg.synthesis_model:
        if llm_cfg.provider == "groq":
            llm_cfg.groq_model = llm_cfg.synthesis_model
        else:
            llm_cfg.gemini_model = llm_cfg.synthesis_model
    llm_cfg.requests_per_minute = 20
    return LLMClient(llm_cfg)


def run_analysis(
    raw: pd.DataFrame,
    text_col: str,
    source_name: str,
    workspace_name: str,
    rating_col: str | None = None,
    time_col: str | None = None,
    progress: Callable[[str], None] = lambda msg: None,
) -> tuple[str, int]:
    """Run the full pipeline on an uploaded dataframe.

    Returns (workspace slug, number of insights).
    """
    t0 = time.time()
    cfg = load_config()
    slug = slugify(workspace_name)
    ws_dir = WORKSPACES_DIR / slug
    ws_dir.mkdir(parents=True, exist_ok=True)

    # ── Unified schema ───────────────────────────────────────────────────
    progress("Reading and mapping columns…")
    raw = raw.dropna(subset=[text_col]).head(MAX_UPLOAD_ROWS).reset_index(drop=True)
    source_slug = slugify(source_name).replace("-", "_") or "uploaded"
    df = pd.DataFrame(
        {
            "id": [f"{source_slug}:{i}" for i in raw.index],
            "source": source_slug,
            "text": raw[text_col].astype(str),
            "author_hash": None,
        }
    )
    df["rating"] = (
        pd.to_numeric(raw[rating_col], errors="coerce") if rating_col else np.nan
    )
    df["timestamp"] = (
        pd.to_datetime(raw[time_col], errors="coerce", utc=True)
        if time_col
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    )

    # ── Classical stages ─────────────────────────────────────────────────
    progress(f"Cleaning {len(df):,} items (language filter, noise stripping)…")
    df, clean_stats = clean_corpus(df, cfg.preprocess.min_text_chars, cfg.preprocess.language)
    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} usable items after cleaning — need at least 20. "
            f"(Stats: {clean_stats})"
        )
    df, _ = drop_exact_dupes(df)

    progress("Embedding text (local model)…")
    emb = embed_corpus(
        df["text_clean"], cfg.embedding.model_name, cfg.embedding.batch_size,
        cache_dir=ws_dir,
    )
    df = flag_near_dupes(df, emb, cfg.preprocess.near_dup_cosine)

    progress("Clustering into themes…")
    # Scale cluster size down for small uploads so themes can still form.
    mcs = max(4, min(cfg.clustering.min_cluster_size, len(df) // 25))
    cres = run_clustering(
        emb,
        cfg.clustering.pca_components,
        mcs,
        cfg.clustering.min_samples,
        cfg.clustering.noise_reassign_cosine,
        cfg.data.random_seed,
    )
    df["cluster_id"] = cres.labels
    df["from_noise"] = cres.from_noise
    df["pca_x"] = cres.reduced_2[:, 0]
    df["pca_y"] = cres.reduced_2[:, 1]

    progress("Scoring sentiment…")
    df = add_sentiment(df)
    df = add_source_relative_age(df)

    # ── LLM insight layer (lightweight bucket) ───────────────────────────
    client = _light_client(cfg)
    cluster_ids = sorted(cres.cohesion, key=lambda c: -(cres.labels == c).sum())
    cluster_ids = cluster_ids[:MAX_CLUSTERS_FOR_UPLOAD]
    n_largest = max(((cres.labels == c).sum() for c in cluster_ids), default=1)

    insights: list[Insight] = []
    for i, cid in enumerate(cluster_ids, 1):
        progress(f"Analyzing theme {i}/{len(cluster_ids)} with the LLM…")
        cluster_df = df[df["cluster_id"] == cid]
        try:
            analysis, quotes, trace = analyze_cluster(
                client, cluster_df, cres.centroid_order[cid],
                cfg.llm.max_quotes_per_cluster,
            )
        except Exception:
            continue  # skip a failed cluster; the run stays useful
        ok, detail, resolved = verify_citations(analysis, set(cluster_df["id"]))
        analysis.evidence_quote_ids = resolved
        priority, components, confidence = score_cluster(
            cluster_df, analysis.severity, int(n_largest),
            df["source"].nunique(), cres.cohesion[cid], cfg.scoring,
        )
        insights.append(
            Insight(
                cluster_id=int(cid),
                analysis=analysis,
                priority_score=priority,
                score_components=components,
                confidence=confidence,
                per_source_counts={
                    str(k): int(v)
                    for k, v in cluster_df["source"].value_counts().items()
                },
                representative_quotes=quotes[:8],
                member_ids=cluster_df["id"].tolist(),
                citation_check_passed=ok,
                citation_check_detail=detail,
                llm_trace=trace,
            )
        )
    if not insights:
        raise RuntimeError("No themes could be analyzed (LLM unavailable?)")
    insights.sort(key=lambda x: -x.priority_score)

    progress("Writing executive summary…")
    try:
        exec_summary, exec_trace = synthesize(client, insights, df)
        exec_trace_dump = exec_trace.model_dump()
    except Exception as e:
        top = insights[:3]
        exec_summary = ExecutiveSummary(
            headline=f"Top issue: {top[0].analysis.theme_name}",
            summary="Automatic synthesis unavailable (LLM quota). Top priorities: "
            + "; ".join(t.analysis.theme_name for t in top) + ".",
            cross_theme_observations=[],
            suggested_focus=[t.analysis.theme_name for t in top],
        )
        exec_trace_dump = {"fallback": True, "error": str(e)}

    # ── Artifacts ────────────────────────────────────────────────────────
    progress("Saving workspace…")
    df.to_parquet(ws_dir / "feedback.parquet", index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "n_items": len(df),
            "per_source": {str(k): int(v) for k, v in df["source"].value_counts().items()},
            "n_clusters": len(cres.cohesion),
            "noise_share": round(float((cres.labels == -1).mean()), 4),
        },
        "executive_summary": exec_summary.model_dump(),
        "executive_summary_trace": exec_trace_dump,
        "insights": [ins.model_dump() for ins in insights],
    }
    (ws_dir / "insights.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "workspace": workspace_name,
        "uploaded_source": source_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - t0, 1),
        "config": json.loads(
            cfg.model_dump_json(exclude={"llm": {"gemini_api_key", "groq_api_key"}})
        ),
        "stages": {
            "clean": clean_stats,
            "clustering": {"n_clusters": len(cres.cohesion), "min_cluster_size": mcs},
            "llm": {"provider": client.provider, "model": client.model,
                    "api_calls": client.calls_made, "cache_hits": client.cache_hits},
        },
    }
    (ws_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return slug, len(insights)
