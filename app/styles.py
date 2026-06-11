"""Design system for the dashboard: theme CSS + reusable HTML components.

Dark modern-SaaS look: glass cards, gradient accents, custom typography.
All dynamic text passed into component helpers is HTML-escaped.
"""

from __future__ import annotations

import html

import streamlit as st

# ── Palette ──────────────────────────────────────────────────────────────
INDIGO = "#6366F1"
CYAN = "#22D3EE"
GREEN = "#10B981"
AMBER = "#F59E0B"
ROSE = "#F43F5E"
TEXT_DIM = "#8B94A7"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#A9B2C3", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    colorway=[INDIGO, CYAN, GREEN, AMBER, ROSE, "#A78BFA", "#34D399", "#F472B6"],
    margin=dict(l=10, r=10, t=30, b=10),
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,450..750&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp {
  background:
    radial-gradient(1100px 600px at 85% -10%, rgba(99,102,241,.16), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(34,211,238,.10), transparent 60%),
    #0B0E14;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }

/* Main content container: centered, generous width */
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1240px !important;
  padding: 1.2rem 2.2rem 4rem !important;
  margin: 0 auto;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: rgba(13,17,26,.92);
  border-right: 1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] * { font-size: .92rem; }

h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: 0; font-weight: 600; }

/* ── Hero ── */
.hero { padding: .6rem 0 1.2rem 0; }
.hero-badge {
  display: inline-flex; align-items: center; gap: .45rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .7rem; font-weight: 600; letter-spacing: .16em;
  color: #A5B4FC; text-transform: uppercase;
  border: 1px solid rgba(99,102,241,.35); border-radius: 999px;
  padding: .3rem .8rem; background: rgba(99,102,241,.10);
}
.hero h1 {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 2.9rem; font-weight: 650; margin: .55rem 0 .2rem 0; color: #F1F4FA;
}
.hero h1 .grad {
  background: linear-gradient(90deg, #818CF8, #22D3EE);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: #8B94A7; font-size: 1.02rem; max-width: 720px; margin: 0; }

/* ── KPI cards ── */
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 14px; margin: 1.1rem 0 .4rem 0; }
.kpi {
  background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.08); border-radius: 16px;
  padding: 1rem 1.1rem; position: relative; overflow: hidden;
}
.kpi::before {
  content: ''; position: absolute; inset: 0 0 auto 0; height: 2px;
  background: linear-gradient(90deg, #6366F1, #22D3EE);
  opacity: .8;
}
.kpi-label { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #8B94A7; }
.kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.7rem; font-weight: 700; color: #F1F4FA; margin-top: .15rem; }
.kpi-sub { font-size: .78rem; color: #6C7689; margin-top: .1rem; }

/* ── Glass card ── */
.glass {
  background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.015));
  border: 1px solid rgba(255,255,255,.08); border-radius: 18px;
  padding: 1.3rem 1.5rem; margin: .6rem 0;
}
.glass h3 { margin-top: 0; font-size: 1.28rem; line-height: 1.35; color: #F1F4FA; }
.glass-title { font-family: 'IBM Plex Mono', monospace; font-size: .7rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: #818CF8; margin-bottom: .5rem; }

/* ── Chips ── */
.chip {
  display: inline-flex; align-items: center; gap: .3rem;
  font-size: .74rem; font-weight: 600; border-radius: 999px; padding: .22rem .65rem;
  border: 1px solid; white-space: nowrap;
}
.chip-ok     { color: #34D399; border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.10); }
.chip-warn   { color: #FBBF24; border-color: rgba(251,191,36,.35); background: rgba(251,191,36,.10); }
.chip-danger { color: #FB7185; border-color: rgba(251,113,133,.35); background: rgba(251,113,133,.10); }
.chip-info   { color: #818CF8; border-color: rgba(129,140,248,.35); background: rgba(129,140,248,.10); }
.chip-dim    { color: #8B94A7; border-color: rgba(139,148,167,.30); background: rgba(139,148,167,.08); }

/* ── Ranked theme rows ── */
.trow {
  display: grid; grid-template-columns: 44px 1.6fr 1.2fr 110px 150px;
  align-items: center; gap: 14px;
  background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015));
  border: 1px solid rgba(255,255,255,.07); border-radius: 14px;
  padding: .85rem 1.1rem; margin-bottom: .55rem;
  transition: border-color .15s ease, transform .15s ease;
}
.trow:hover { border-color: rgba(129,140,248,.45); transform: translateY(-1px); }
.trank {
  font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1rem; color: #5B647A;
  width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.07);
}
.tname { font-weight: 600; color: #E6EAF2; font-size: .98rem; line-height: 1.25; }
.tsub { display: block; font-size: .76rem; color: #6C7689; font-weight: 500; margin-top: .15rem; }
.tbar { display: flex; align-items: center; gap: .6rem; }
.tbar-track { flex: 1; height: 7px; border-radius: 99px; background: rgba(255,255,255,.07); overflow: hidden; }
.tbar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #6366F1, #22D3EE); }
.tbar-val { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: .88rem; color: #C7CDDB; min-width: 42px; }

/* ── Quote cards ── */
.quote {
  border-left: 3px solid rgba(129,140,248,.55);
  background: rgba(255,255,255,.03);
  border-radius: 0 12px 12px 0;
  padding: .8rem 1rem; margin-bottom: .6rem;
}
.quote.starred { border-left-color: #22D3EE; background: rgba(34,211,238,.05); }
.quote-text { color: #C7CDDB; font-size: .92rem; line-height: 1.5; }
.quote-meta { font-size: .74rem; color: #6C7689; margin-top: .4rem; }

/* ── Action cards ── */
.action {
  background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
  border-radius: 14px; padding: .9rem 1.1rem; margin-bottom: .6rem;
}
.action-title { font-weight: 600; color: #E6EAF2; font-size: .95rem; display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
.action-rationale { color: #8B94A7; font-size: .84rem; margin-top: .35rem; line-height: 1.45; }

/* ── Root cause list ── */
.cause { display: flex; gap: .7rem; margin-bottom: .65rem; align-items: flex-start; }
.cause-dot { min-width: 8px; height: 8px; border-radius: 99px; background: linear-gradient(90deg,#6366F1,#22D3EE); margin-top: .45rem; }
.cause-text { color: #B6BECF; font-size: .92rem; line-height: 1.5; }

/* Section heading */
.section { font-family: 'Fraunces', Georgia, serif; font-size: 1.18rem; font-weight: 650; color: #F1F4FA; margin: 1.4rem 0 .7rem 0; display: flex; align-items: center; gap: .55rem; }
.section .dot { width: 9px; height: 9px; border-radius: 3px; background: linear-gradient(135deg,#6366F1,#22D3EE); }

/* Streamlit widget polish */
.stDownloadButton button, .stButton button {
  border-radius: 12px !important; border: 1px solid rgba(129,140,248,.4) !important;
  background: rgba(99,102,241,.12) !important; color: #C7D2FE !important; font-weight: 600 !important;
}
.stDownloadButton button:hover, .stButton button:hover { border-color: #818CF8 !important; background: rgba(99,102,241,.22) !important; }
div[data-baseweb="select"] > div { border-radius: 12px !important; background: rgba(255,255,255,.04) !important; border-color: rgba(255,255,255,.1) !important; }
.stTextInput input { border-radius: 12px !important; background: rgba(255,255,255,.04) !important; border-color: rgba(255,255,255,.1) !important; }
[data-testid="stExpander"] { border: 1px solid rgba(255,255,255,.08); border-radius: 14px; background: rgba(255,255,255,.02); }
[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,.08); border-radius: 14px; }
hr { border-color: rgba(255,255,255,.07) !important; }

/* ── Chat (Ask Iris) ── */
[data-testid="stChatMessage"] {
  background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015));
  border: 1px solid rgba(255,255,255,.08); border-radius: 16px;
  padding: 1rem 1.2rem; margin-bottom: .4rem;
}
[data-testid="stChatInput"] { border-radius: 14px; }
[data-testid="stChatInput"] textarea {
  background: rgba(255,255,255,.04) !important; border-radius: 14px !important;
}
.iris-suggest button {
  font-size: .8rem !important; padding: .3rem .8rem !important;
}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


# ── Components (return HTML strings; render with st.markdown(..., unsafe_allow_html=True)) ──

def hero(title_plain: str, title_grad: str, subtitle: str, badge: str = "AI Product Analyst") -> str:
    return f"""
<div class="hero">
  <div class="hero-badge">✦ {esc(badge)}</div>
  <h1>{esc(title_plain)}<span class="grad">{esc(title_grad)}</span></h1>
  <p>{esc(subtitle)}</p>
</div>"""


def kpi_row(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="kpi"><div class="kpi-label">{esc(l)}</div>'
        f'<div class="kpi-value">{esc(v)}</div>'
        f'<div class="kpi-sub">{esc(s)}</div></div>'
        for l, v, s in items
    )
    return f'<div class="kpi-row">{cells}</div>'


def chip(text: str, kind: str = "info") -> str:
    return f'<span class="chip chip-{kind}">{esc(text)}</span>'


CONF_CHIP = {
    "ok": ("Solid evidence", "ok"),
    "low_sample": ("Low sample", "warn"),
    "insufficient_evidence": ("Insufficient evidence", "danger"),
}


def confidence_chip(badge: str) -> str:
    label, kind = CONF_CHIP.get(badge, (badge, "dim"))
    return chip(label, kind)


def severity_chip(sev: int) -> str:
    kind = "ok" if sev <= 2 else ("warn" if sev == 3 else "danger")
    return chip(f"Severity {sev}/5", kind)


def glass(body_html: str, eyebrow: str | None = None) -> str:
    eb = f'<div class="glass-title">{esc(eyebrow)}</div>' if eyebrow else ""
    return f'<div class="glass">{eb}{body_html}</div>'


def section(title: str) -> str:
    return f'<div class="section"><span class="dot"></span>{esc(title)}</div>'


def theme_row(rank: int, name: str, sub: str, priority: float,
              sev_html: str, conf_html: str) -> str:
    pct = max(2, min(100, round(priority * 100)))
    return f"""
<div class="trow">
  <div class="trank">{rank}</div>
  <div><div class="tname">{esc(name)}<span class="tsub">{esc(sub)}</span></div></div>
  <div class="tbar"><div class="tbar-track"><div class="tbar-fill" style="width:{pct}%"></div></div>
       <span class="tbar-val">{priority:.2f}</span></div>
  <div>{sev_html}</div>
  <div>{conf_html}</div>
</div>"""


def quote_card(text: str, meta: str, starred: bool = False) -> str:
    cls = "quote starred" if starred else "quote"
    star = "⭐ " if starred else ""
    return (
        f'<div class="{cls}"><div class="quote-text">“{esc(text)}”</div>'
        f'<div class="quote-meta">{star}{esc(meta)}</div></div>'
    )


def action_card(action: str, effort: str, rationale: str) -> str:
    kind = {"low": "ok", "medium": "warn", "high": "danger"}.get(effort, "dim")
    return (
        f'<div class="action"><div class="action-title">{esc(action)} '
        f'{chip(f"effort: {effort}", kind)}</div>'
        f'<div class="action-rationale">{esc(rationale)}</div></div>'
    )


def cause_item(text: str) -> str:
    return (
        f'<div class="cause"><div class="cause-dot"></div>'
        f'<div class="cause-text">{esc(text)}</div></div>'
    )
