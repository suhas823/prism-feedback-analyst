# Demo Script (~5 minutes)

> Pre-demo checklist: pipeline has been run (artifacts exist), `streamlit run
> app/Home.py` is open in a browser tab, and the LLM cache is warm so a live
> re-run costs zero API calls.

## 1. The problem (30s)

"Product teams drown in feedback — reviews, tickets, surveys. It's noisy,
emotional, and biased. Important signals get missed. We built an agent that
turns that mess into prioritized, *auditable* insights."

## 2. Home page (60s)

- Point at the KPI row: N items, 2 sources, themes found, **citations verified**.
- Read the executive headline aloud.
- Scroll the ranked table: "This ordering isn't a vibe — it's a formula:
  35% frequency, 35% severity, 15% recency, 15% source diversity."
- Toggle **Hide low-confidence** — "the agent tells you when it doesn't have
  enough evidence, instead of bluffing."

## 3. Insight Detail — the trust story (2 min)

Open the #1 theme:

- Summary + root causes: "root causes are labeled *hypotheses* — the agent
  doesn't pretend feedback text proves causality."
- Recommended actions with effort levels: "actions, not summaries."
- **Score breakdown chart**: "every point of the priority score is explained —
  this is the anti-black-box requirement."
- Evidence quotes: "every claim cites real quotes; the ⭐ ones are what the
  LLM cited, and we *programmatically verify* those IDs exist in this cluster.
  If the model hallucinates a citation, it's flagged in red, not hidden."
- Wilson interval: "small skewed samples are the classic feedback trap — we
  show statistical uncertainty instead of point claims."
- Open the **raw LLM trace** expander: "full provenance — model, prompt
  version, raw response."

## 4. Explore page (45s)

- Cluster map: "each dot is one piece of feedback, positioned by meaning.
  Grey = unclustered; we show what the system *couldn't* categorize."
- Search a word ("crash"), filter by source.

## 5. Methodology page (30s)

- "If a PM challenges any number, this page is the audit trail: pipeline,
  formula with the live weights, guardrails, and limitations."

## 6. Close (15s)

"Classical ML does the per-item heavy lifting locally; the LLM only reasons
over clusters — ~35 API calls per run, free tier, fully cached. Cheap,
reproducible, and every insight can defend itself."

## Likely questions

- **"What if the LLM is wrong?"** → severity is only half the severity score;
  citations are verified; rationale is shown verbatim; rubric eval in docs.
- **"Does it scale?"** → per-item work is local and linear; LLM cost grows
  with *themes*, not items.
- **"New sources?"** → one adapter class mapping to `FeedbackItem` (~40 lines).
