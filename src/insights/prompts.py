"""Versioned prompt templates.

Bump PROMPT_VERSION whenever wording changes — it is part of the cache key
and stored in every llm_trace, so cached analyses never silently mix prompt
generations.
"""

PROMPT_VERSION = "v1.2"

CLUSTER_ANALYST_SYSTEM = """\
You are a senior product analyst. You turn clustered user feedback into
honest, actionable product insights. Rules:
- Be specific and concrete; never use marketing language.
- Root causes are HYPOTHESES inferred from the quotes — phrase them that way.
- Do not exaggerate severity. Emotional language in feedback is signal of
  user frustration, not proof of technical severity.
- Recommended actions must be things a product team could actually start
  next sprint, not vague advice like "improve quality".
- Only cite quote IDs that were given to you.
- If the quotes seem to describe multiple unrelated problems, say so in the
  summary and pick the dominant one for the theme.
"""

CLUSTER_ANALYST_USER = """\
A clustering pipeline grouped {n_items} user feedback items (from sources:
{source_breakdown}) into one cluster. Below is a stratified sample of
{n_quotes} quotes from this cluster. Each has an ID, source, optional star
rating, and date.

{quotes_block}

Analyze this cluster and respond with JSON matching the provided schema:
- theme_name: short label a product team would put on a kanban card
- summary: 2-3 neutral sentences describing the issue and who it affects
- root_causes: 1-3 hypothesized root causes (phrased as hypotheses)
- severity: 1-5 (1=cosmetic annoyance, 3=degrades core experience,
  5=blocks usage / drives churn). Judge from content, not emotion.
- severity_rationale: 1-2 sentences justifying the severity
- recommended_actions: 1-3 concrete actions with effort (low/medium/high)
  and rationale
- evidence_quote_ids: the 3-6 quote IDs that best support your analysis
"""

SYNTHESIS_SYSTEM = """\
You are a product analyst writing the executive section of a feedback
insights report. Be concise, neutral, and specific. Do not invent facts
beyond the provided theme analyses.
"""

SYNTHESIS_USER = """\
A feedback-analysis pipeline produced the following prioritized themes from
{n_total} user feedback items across sources ({source_breakdown}).
Themes are listed with priority score (0-1), item count, severity (1-5),
and confidence badge:

{themes_block}

Write the executive summary as JSON matching the schema:
- headline: single-sentence top takeaway
- summary: one short paragraph (3-5 sentences) for a product leader
- cross_theme_observations: 1-3 patterns that span multiple themes
  (empty list if none are genuine)
- suggested_focus: the 2-3 themes to act on first, with a clause on why
"""


def format_quotes_block(quotes: list[dict]) -> str:
    lines = []
    for q in quotes:
        rating = f" | rating: {q['rating']:.0f}/5" if q.get("rating") is not None else ""
        date = f" | {q['timestamp'][:10]}" if q.get("timestamp") else ""
        lines.append(f"[{q['id']}] ({q['source']}{rating}{date}) {q['text']}")
    return "\n".join(lines)


def format_themes_block(themes: list[dict]) -> str:
    lines = []
    for t in themes:
        lines.append(
            f"- {t['theme_name']} (priority {t['priority']:.2f}, n={t['n_items']}, "
            f"severity {t['severity']}/5, confidence: {t['badge']}): {t['summary']}"
        )
    return "\n".join(lines)
