"""Provider-agnostic LLM client: Gemini (primary) or Groq (fallback).

Free-tier survival layer:
- RPM throttle (config `requests_per_minute`)
- tenacity retry with exponential backoff on transient errors
- disk cache keyed by (provider, model, prompt version, full prompt) hash —
  a re-run of an unchanged pipeline costs zero API calls.

Every call returns (parsed pydantic object, LLMTrace) so the exact request
provenance lands in insights.json.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Type, TypeVar

from diskcache import Cache
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import PROJECT_ROOT, LLMConfig
from src.insights.prompts import PROMPT_VERSION
from src.insights.schemas import LLMTrace

T = TypeVar("T", bound=BaseModel)


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    # A *daily* quota exhaustion won't resolve within retry backoff — fail
    # fast so the caller can skip/switch provider instead of burning minutes.
    if "perday" in msg or "per day" in msg:
        return False
    return any(
        token in msg
        for token in ("429", "rate", "quota", "503", "500", "timeout", "temporarily")
    )


class _Throttle:
    """Simple sliding-window RPM limiter (thread-safe for future parallelism)."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(rpm, 1)
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._last + self.min_interval - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.provider = cfg.provider
        self.model = cfg.gemini_model if self.provider == "gemini" else cfg.groq_model
        self.throttle = _Throttle(cfg.requests_per_minute)
        cache_path = Path(cfg.cache_dir)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        self.cache = Cache(str(cache_path))
        self._client = None
        self.calls_made = 0
        self.cache_hits = 0

    # ── public API ───────────────────────────────────────────────────────
    def generate_structured(
        self, system: str, user: str, schema: Type[T]
    ) -> tuple[T, LLMTrace]:
        key = self._cache_key(system, user, schema)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            obj = schema.model_validate_json(cached)
            return obj, self._trace(key, raw=cached, cached=True)

        raw = self._call_with_retry(system, user, schema)
        obj = schema.model_validate_json(raw)  # validate BEFORE caching
        self.cache.set(key, raw)
        self.calls_made += 1
        return obj, self._trace(key, raw=raw, cached=False)

    def generate_text(self, system: str, user: str) -> str:
        """Plain-text completion (used by the chat assistant). Same throttle,
        retry, and cache machinery as structured calls."""
        key = self._cache_key_text(system, user)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.throttle.wait()
        raw = self._text_call(system, user)
        self.cache.set(key, raw)
        self.calls_made += 1
        return raw

    # ── internals ────────────────────────────────────────────────────────
    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _text_call(self, system: str, user: str) -> str:
        if self.provider == "gemini":
            if self._client is None:
                from google import genai

                if not self.cfg.gemini_api_key:
                    raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
                self._client = genai.Client(api_key=self.cfg.gemini_api_key)
            resp = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config={"system_instruction": system, "temperature": 0.3},
            )
            return resp.text
        if self.provider == "groq":
            if self._client is None:
                from groq import Groq

                if not self.cfg.groq_api_key:
                    raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
                self._client = Groq(api_key=self.cfg.groq_api_key)
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content
        raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    def _cache_key_text(self, system: str, user: str) -> str:
        payload = "\x1f".join(
            [self.provider, self.model, PROMPT_VERSION, "text", system, user]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_key(self, system: str, user: str, schema: Type[BaseModel]) -> str:
        payload = "\x1f".join(
            [self.provider, self.model, PROMPT_VERSION, schema.__name__, system, user]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _trace(self, key: str, raw: str, cached: bool) -> LLMTrace:
        return LLMTrace(
            provider=self.provider,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            prompt_hash=key[:16],
            timestamp=datetime.now(timezone.utc).isoformat(),
            cached=cached,
            raw_response=raw,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call_with_retry(self, system: str, user: str, schema: Type[BaseModel]) -> str:
        self.throttle.wait()
        if self.provider == "gemini":
            return self._call_gemini(system, user, schema)
        if self.provider == "groq":
            return self._call_groq(system, user, schema)
        raise ValueError(f"Unknown LLM provider: {self.provider!r}")

    def _call_gemini(self, system: str, user: str, schema: Type[BaseModel]) -> str:
        if self._client is None:
            from google import genai

            if not self.cfg.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
            self._client = genai.Client(api_key=self.cfg.gemini_api_key)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0.2,
            },
        )
        return resp.text

    def _call_groq(self, system: str, user: str, schema: Type[BaseModel]) -> str:
        if self._client is None:
            from groq import Groq

            if not self.cfg.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
            self._client = Groq(api_key=self.cfg.groq_api_key)
        # Groq json_object mode needs the schema spelled out in the prompt.
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        user_with_schema = (
            f"{user}\n\nRespond with a single JSON object matching this JSON "
            f"Schema exactly:\n{schema_json}"
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_with_schema},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return resp.choices[0].message.content
