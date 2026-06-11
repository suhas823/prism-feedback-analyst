"""End-to-end pipeline orchestrator.

    python -m src.pipeline.run            # full run
    python -m src.pipeline.run --no-llm   # classical stages only (no API key needed)

Produces in data/processed/:
    feedback.parquet   per-item corpus with cluster ids, sentiment, PCA coords
    clusters.parquet   per-cluster stats
    insights.json      scored, evidence-linked insights + executive summary
    run_meta.json      stage stats, config snapshot, LLM call accounting
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows consoles often default to cp1252, which chokes on unicode in logs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_config  # noqa: E402
from src.ingest.base import combine_sources  # noqa: E402
from src.ingest.reviews_loader import PlayStoreReviewsAdapter  # noqa: E402
from src.ingest.tickets_loader import SupportTicketsAdapter  # noqa: E402
from src.preprocess.clean import clean_corpus  # noqa: E402
from src.preprocess.dedupe import drop_exact_dupes, flag_near_dupes  # noqa: E402
from src.analysis.embed import embed_corpus  # noqa: E402
from src.analysis.cluster import cluster_summary, run_clustering  # noqa: E402
from src.analysis.sentiment import add_sentiment  # noqa: E402
from src.insights.cluster_analyst import analyze_cluster, verify_citations  # noqa: E402
from src.insights.llm_client import LLMClient  # noqa: E402
from src.insights.schemas import ExecutiveSummary, Insight  # noqa: E402
from src.insights.scoring import score_cluster  # noqa: E402
from src.insights.synthesis import synthesize  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_ingest(cfg) -> pd.DataFrame:
    raw = cfg.paths.raw
    adapters = []
    reviews_csv = raw / "reviews_raw.csv"
    tickets_csv = raw / "tickets_raw.csv"
    if reviews_csv.exists():
        adapters.append(
            PlayStoreReviewsAdapter(
                reviews_csv, cfg.data.reviews_sample_size, cfg.data.random_seed
            )
        )
    if tickets_csv.exists():
        adapters.append(
            SupportTicketsAdapter(
                tickets_csv, cfg.data.tickets_sample_size, cfg.data.random_seed
            )
        )
    if not adapters:
        raise FileNotFoundError(
            f"No raw data in {raw}. Run: python scripts/download_data.py"
        )
    df = combine_sources(adapters)
    log(f"ingested {len(df):,} items from {df['source'].nunique()} sources "
        f"({df['source'].value_counts().to_dict()})")
    return df


def add_source_relative_age(df: pd.DataFrame) -> pd.DataFrame:
    """age_days relative to each source's newest item, so sources collected in
    different eras (2017 tweets vs current reviews) compete fairly on recency."""
    df = df.copy()
    df["age_days"] = np.nan
    for source, grp in df.groupby("source"):
        ts = pd.to_datetime(grp["timestamp"], errors="coerce", utc=True)
        ref = ts.max()
        if pd.isna(ref):
            continue
        df.loc[grp.index, "age_days"] = (ref - ts).dt.total_seconds() / 86400.0
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedback → Insight pipeline")
    parser.add_argument("--no-llm", action="store_true",
                        help="run classical stages only (no API key required)")
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config()
    meta: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": json.loads(cfg.model_dump_json(exclude={"llm": {"gemini_api_key", "groq_api_key"}})),
        "stages": {},
    }

    # 1-2. Ingest + clean
    df = stage_ingest(cfg)
    df, clean_stats = clean_corpus(df, cfg.preprocess.min_text_chars, cfg.preprocess.language)
    meta["stages"]["clean"] = clean_stats
    log(f"cleaned: {clean_stats}")

    # 3a. Exact dupes
    df, n_exact = drop_exact_dupes(df)
    meta["stages"]["exact_dupes_dropped"] = n_exact
    log(f"dropped {n_exact} exact duplicates -> {len(df):,} items")

    # 4. Embeddings (cached)
    interim = cfg.paths.interim
    emb = embed_corpus(
        df["text_clean"], cfg.embedding.model_name, cfg.embedding.batch_size, interim
    )
    log(f"embeddings: {emb.shape}")

    # 3b. Near-dupes (needs embeddings)
    df = flag_near_dupes(df, emb, cfg.preprocess.near_dup_cosine)
    meta["stages"]["near_dups_flagged"] = int(df["is_near_dup"].sum())
    log(f"flagged {int(df['is_near_dup'].sum())} near-duplicates")

    # 5. Clustering
    cres = run_clustering(
        emb,
        cfg.clustering.pca_components,
        cfg.clustering.min_cluster_size,
        cfg.clustering.min_samples,
        cfg.clustering.noise_reassign_cosine,
        cfg.data.random_seed,
    )
    df["cluster_id"] = cres.labels
    df["from_noise"] = cres.from_noise
    df["pca_x"] = cres.reduced_2[:, 0]
    df["pca_y"] = cres.reduced_2[:, 1]
    n_clusters = len(cres.cohesion)
    noise_share = float((cres.labels == -1).mean())
    meta["stages"]["clustering"] = {
        "n_clusters": n_clusters,
        "noise_share": round(noise_share, 4),
        "soft_assigned_from_noise": int(cres.from_noise.sum()),
    }
    log(f"HDBSCAN: {n_clusters} clusters, {noise_share:.0%} noise "
        f"({int(cres.from_noise.sum())} soft-assigned)")

    # 6. Per-item signals
    df = add_sentiment(df)
    df = add_source_relative_age(df)

    # Persist per-item artifacts (parquet keeps dtypes; timestamps stay UTC).
    processed = cfg.paths.processed
    processed.mkdir(parents=True, exist_ok=True)
    np.save(interim / "reduced_50.npy", cres.reduced_50)
    df.to_parquet(processed / "feedback.parquet", index=False)
    summary = cluster_summary(df, cres.labels)
    summary["cohesion"] = summary["cluster_id"].map(cres.cohesion).fillna(0.0)
    summary.to_parquet(processed / "clusters.parquet", index=False)
    log(f"wrote feedback.parquet ({len(df):,} rows) and clusters.parquet")

    if args.no_llm:
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["duration_s"] = round(time.time() - t0, 1)
        (processed / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log("--no-llm: stopping after classical stages")
        return

    # 7-9. LLM analysis, scoring, synthesis
    client = LLMClient(cfg.llm)
    log(f"LLM: {client.provider} / {client.model}")

    cluster_ids = sorted(cres.cohesion, key=lambda c: -(cres.labels == c).sum())
    cluster_ids = cluster_ids[: cfg.llm.max_clusters_analyzed]
    n_largest = int(max((cres.labels == c).sum() for c in cluster_ids))
    n_source_types = df["source"].nunique()

    insights: list[Insight] = []
    for i, cid in enumerate(cluster_ids, 1):
        cluster_df = df[df["cluster_id"] == cid]
        log(f"analyzing cluster {cid} ({len(cluster_df)} items) [{i}/{len(cluster_ids)}]")
        try:
            analysis, quotes, trace = analyze_cluster(
                client, cluster_df, cres.centroid_order[cid], cfg.llm.max_quotes_per_cluster
            )
        except Exception as e:
            log(f"  cluster {cid} FAILED after retries: {e} — skipping")
            continue
        ok, detail, resolved_ids = verify_citations(analysis, set(cluster_df["id"]))
        # Normalize citations to canonical ids (raw response stays in llm_trace).
        analysis.evidence_quote_ids = resolved_ids
        priority, components, confidence = score_cluster(
            cluster_df,
            analysis.severity,
            n_largest,
            n_source_types,
            cres.cohesion[cid],
            cfg.scoring,
        )
        insights.append(
            Insight(
                cluster_id=int(cid),
                analysis=analysis,
                priority_score=priority,
                score_components=components,
                confidence=confidence,
                per_source_counts={
                    str(k): int(v) for k, v in cluster_df["source"].value_counts().items()
                },
                representative_quotes=quotes[:8],
                member_ids=cluster_df["id"].tolist(),
                citation_check_passed=ok,
                citation_check_detail=detail,
                llm_trace=trace,
            )
        )

    insights.sort(key=lambda x: -x.priority_score)
    log(f"analyzed {len(insights)} clusters "
        f"({client.calls_made} API calls, {client.cache_hits} cache hits)")

    # Synthesis is nice-to-have: never lose the per-cluster analyses because
    # the final call failed (e.g. daily quota). Fall back to a deterministic
    # summary and mark it so the UI can show provenance honestly.
    synthesis_failed = None
    synth_client = client
    if cfg.llm.synthesis_model:
        synth_cfg = cfg.llm.model_copy()
        if synth_cfg.provider == "groq":
            synth_cfg.groq_model = cfg.llm.synthesis_model
        else:
            synth_cfg.gemini_model = cfg.llm.synthesis_model
        synth_client = LLMClient(synth_cfg)
        log(f"synthesis model override: {synth_client.model}")
    try:
        exec_summary, exec_trace = synthesize(synth_client, insights, df)
        exec_trace_dump = exec_trace.model_dump()
    except Exception as e:
        synthesis_failed = str(e)
        log(f"synthesis FAILED: {e} — writing deterministic fallback summary")
        top = [i for i in insights if i.confidence.badge == "ok"][:3] or insights[:3]
        exec_summary = ExecutiveSummary(
            headline=f"Top issue: {top[0].analysis.theme_name}" if top else "No themes",
            summary=(
                "Automatic synthesis was unavailable for this run (LLM quota). "
                "The ranked themes below are fully analyzed and scored; the top "
                "priorities are: "
                + "; ".join(t.analysis.theme_name for t in top)
                + "."
            ),
            cross_theme_observations=[],
            suggested_focus=[t.analysis.theme_name for t in top],
        )
        exec_trace_dump = {"fallback": True, "error": synthesis_failed}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "n_items": len(df),
            "per_source": {str(k): int(v) for k, v in df["source"].value_counts().items()},
            "n_clusters": n_clusters,
            "noise_share": round(noise_share, 4),
        },
        "executive_summary": exec_summary.model_dump(),
        "executive_summary_trace": exec_trace_dump,
        "insights": [ins.model_dump() for ins in insights],
    }
    (processed / "insights.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta["stages"]["llm"] = {
        "provider": client.provider,
        "model": client.model,
        "clusters_analyzed": len(insights),
        "api_calls": client.calls_made,
        "cache_hits": client.cache_hits,
        "synthesis_fallback": synthesis_failed,
    }
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["duration_s"] = round(time.time() - t0, 1)
    (processed / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"wrote insights.json ({len(insights)} insights) — done in {meta['duration_s']}s")


if __name__ == "__main__":
    main()
