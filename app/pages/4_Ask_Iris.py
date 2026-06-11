"""Ask Iris — conversational analyst over the generated insights.

Iris answers ONLY from the pipeline's analysis artifacts (insights.json),
so every answer is grounded in the same evidence-linked themes the rest of
the dashboard shows. Uses the lightweight chat model (separate free-tier
token bucket from the main analysis model).
"""

import sys
from pathlib import Path

import streamlit as st

import styles as ds
from shared import PROJECT_ROOT, require_artifacts

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.insights.llm_client import LLMClient  # noqa: E402

st.set_page_config(page_title="Prism — Ask Iris", page_icon="✦", layout="wide")
ds.inject()

df, insights = require_artifacts()

st.markdown(
    ds.hero(
        "Ask ",
        "Iris",
        "Iris is Prism's conversational analyst. She answers from the analyzed "
        "themes only — grounded, no improvising beyond the evidence.",
        badge="Prism · Conversational analyst",
    ),
    unsafe_allow_html=True,
)


# ── Grounding context (built once per artifact load) ─────────────────────
@st.cache_data(show_spinner=False)
def build_context(generated_at: str) -> str:
    es = insights["executive_summary"]
    corpus = insights["corpus"]
    lines = [
        f"Corpus: {corpus['n_items']:,} feedback items "
        f"({', '.join(f'{k}: {v}' for k, v in corpus['per_source'].items())}); "
        f"{len(insights['insights'])} analyzed themes.",
        f"Executive summary: {es['summary']}",
        "",
        "Ranked themes (by priority score 0-1):",
    ]
    for rank, i in enumerate(insights["insights"][:15], 1):
        a = i["analysis"]
        c = i["confidence"]
        lines.append(
            f"{rank}. {a['theme_name']} — priority {i['priority_score']:.2f}, "
            f"severity {a['severity']}/5, {c['n_items']} items, confidence: {c['badge']}"
        )
        lines.append(f"   Summary: {a['summary']}")
        lines.append(f"   Root causes: {' / '.join(a['root_causes'])}")
        lines.append(
            "   Recommended actions: "
            + " / ".join(
                f"{act['action']} (effort {act['effort']})"
                for act in a["recommended_actions"]
            )
        )
    return "\n".join(lines)


IRIS_SYSTEM = """\
You are Iris, the conversational analyst inside Prism, an AI product-feedback
analysis tool. You answer questions from product managers about the analyzed
feedback themes provided below.

Rules:
- Answer ONLY from the provided analysis context. If something isn't in it,
  say so and suggest the Explore Feedback page for raw data.
- Be concise and use markdown. Bold theme names. Mention priority scores and
  item counts when ranking or comparing.
- Root causes are hypotheses — present them that way.
- When asked for recommendations, give the concrete recommended actions from
  the context, not generic advice.
- Never invent numbers, themes, or quotes.

ANALYSIS CONTEXT:
{context}
"""


@st.cache_resource
def get_chat_client() -> LLMClient:
    cfg = load_config()
    chat_cfg = cfg.llm.model_copy()
    # The chat model rides the lightweight bucket (separate free-tier quota).
    if chat_cfg.synthesis_model:
        if chat_cfg.provider == "groq":
            chat_cfg.groq_model = chat_cfg.synthesis_model
        else:
            chat_cfg.gemini_model = chat_cfg.synthesis_model
    return LLMClient(chat_cfg)


def ask_iris(question: str, history: list[dict]) -> str:
    context = build_context(insights["generated_at"])
    system = IRIS_SYSTEM.format(context=context)
    convo = ""
    for turn in history[-6:]:
        speaker = "PM" if turn["role"] == "user" else "Iris"
        convo += f"{speaker}: {turn['content']}\n"
    user = f"{convo}PM: {question}\nIris:"
    return get_chat_client().generate_text(system, user)


# ── Conversation state ───────────────────────────────────────────────────
if "iris_history" not in st.session_state:
    st.session_state.iris_history = []

GREETING = (
    "Hi, I'm **Iris** ✦ — I've read all "
    f"{insights['corpus']['n_items']:,} feedback items so you don't have to. "
    "Ask me about the top problems, root causes, or what to fix first."
)

# Suggested questions (also serve as one-click demo path)
SUGGESTIONS = [
    "What are the top 3 problems?",
    "Give me a summary of the feedback",
    "What should we fix first and why?",
    "Which issues affect both reviews and tickets?",
]

with st.chat_message("assistant", avatar="💠"):
    st.markdown(GREETING)

for turn in st.session_state.iris_history:
    avatar = "💠" if turn["role"] == "assistant" else "👤"
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["content"])

pending = None
if not st.session_state.iris_history:
    st.markdown('<div class="iris-suggest">', unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTIONS))
    for col, q in zip(cols, SUGGESTIONS):
        if col.button(q, key=f"sugg_{q}"):
            pending = q
    st.markdown("</div>", unsafe_allow_html=True)

typed = st.chat_input("Ask Iris about your feedback…")
question = typed or pending

if question:
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="💠"):
        try:
            with st.spinner("Iris is reading the analysis…"):
                answer = ask_iris(question, st.session_state.iris_history)
        except Exception as e:
            answer = (
                "I hit the LLM rate limit just now — give it a minute and ask "
                f"again. (Detail: {e})"
            )
        st.markdown(answer)
    st.session_state.iris_history.append({"role": "user", "content": question})
    st.session_state.iris_history.append({"role": "assistant", "content": answer})
    st.rerun()

st.caption(
    "Iris answers only from the analyzed themes (top 15 by priority). "
    "For raw data, use **Explore Feedback**; for full provenance, **Insight Detail**."
)
