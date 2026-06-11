"""Central configuration: merges config/config.yaml with .env secrets.

Everything tunable lives in config.yaml; only secrets and the provider
override come from the environment. Import `load_config()` anywhere.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class DataConfig(BaseModel):
    reviews_sample_size: int = 2500
    tickets_sample_size: int = 1500
    random_seed: int = 42


class PreprocessConfig(BaseModel):
    min_text_chars: int = 15
    language: str = "en"
    near_dup_cosine: float = 0.92


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64


class ClusteringConfig(BaseModel):
    pca_components: int = 15
    min_cluster_size: int = 8
    min_samples: int = 1
    noise_reassign_cosine: float = 0.55
    kmeans_k_range: tuple[int, int] = (8, 30)


class LLMConfig(BaseModel):
    provider: str = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    requests_per_minute: int = 8
    max_quotes_per_cluster: int = 15
    max_clusters_analyzed: int = 40
    synthesis_model: str = ""  # optional override; "" = same model as analysis
    cache_dir: str = ".cache/llm"
    gemini_api_key: str = ""
    groq_api_key: str = ""


class ScoringWeights(BaseModel):
    frequency: float = 0.35
    severity: float = 0.35
    recency: float = 0.15
    diversity: float = 0.15


class SeverityMix(BaseModel):
    llm: float = 0.5
    rating: float = 0.3
    sentiment: float = 0.2


class ScoringConfig(BaseModel):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    severity_mix: SeverityMix = Field(default_factory=SeverityMix)
    recency_half_life_days: float = 90
    min_items_insufficient: int = 5
    min_items_low_sample: int = 15
    low_cohesion_threshold: float = 0.35
    wilson_confidence: float = 0.95


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"

    @property
    def raw(self) -> Path:
        return PROJECT_ROOT / self.raw_dir

    @property
    def interim(self) -> Path:
        return PROJECT_ROOT / self.interim_dir

    @property
    def processed(self) -> Path:
        return PROJECT_ROOT / self.processed_dir


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    load_dotenv(PROJECT_ROOT / ".env")
    raw: dict = {}
    if CONFIG_PATH.exists():
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    cfg = AppConfig(**raw)
    cfg.llm.provider = os.getenv("LLM_PROVIDER", cfg.llm.provider).lower()
    cfg.llm.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    cfg.llm.groq_api_key = os.getenv("GROQ_API_KEY", "")
    return cfg
