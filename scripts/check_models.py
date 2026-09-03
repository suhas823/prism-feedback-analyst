"""Check which LLM models the configured provider actually offers.

    python scripts/check_models.py

Free providers retire models frequently (Groq dropped the whole Llama 3.x line
in mid-2026). When the app starts returning `model_not_found`, run this to see
what's available, then update `llm.groq_model` / `llm.synthesis_model` in
config/config.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_config  # noqa: E402

# Models that transcribe audio or classify prompts, not chat completions.
NON_CHAT_HINTS = ("whisper", "orpheus", "prompt-guard", "tts", "embed")


def main() -> None:
    cfg = load_config()
    if cfg.llm.provider != "groq":
        print(f"Provider is {cfg.llm.provider!r}; this script only lists Groq models.")
        return
    if not cfg.llm.groq_api_key:
        print("GROQ_API_KEY is not set (see .env.example)")
        return

    from groq import Groq

    available = sorted(m.id for m in Groq(api_key=cfg.llm.groq_api_key).models.list().data)
    chat = [m for m in available if not any(h in m.lower() for h in NON_CHAT_HINTS)]

    print("Chat models available:")
    for m in chat:
        print(f"  {m}")

    print("\nConfigured:")
    for label, model in (
        ("analysis  (llm.groq_model)", cfg.llm.groq_model),
        ("chat/synth(llm.synthesis_model)", cfg.llm.synthesis_model or "(same as analysis)"),
    ):
        if model.startswith("("):
            print(f"  {label}: {model}")
        else:
            mark = "OK " if model in available else "GONE"
            print(f"  [{mark}] {label}: {model}")

    missing = [
        m for m in (cfg.llm.groq_model, cfg.llm.synthesis_model) if m and m not in available
    ]
    if missing:
        print("\nFix: update config/config.yaml with a model from the list above.")
        sys.exit(1)
    print("\nAll configured models are available.")


if __name__ == "__main__":
    main()
