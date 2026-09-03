# Interview walkthrough script (7 minutes)

For live screen-share. Timings assume the interviewer interrupts, which is good.
Let them.

## Before you share your screen

- Live app open and loaded: your `.streamlit.app` URL
- Dataset selector on **Spotify (demo dataset)**
- Second tab open on **Insight Detail**, top theme already selected
- Close Slack, WhatsApp, and anything with notifications
- Have the GitHub repo in a third tab in case they ask to see code

If the hosted app is slow to wake, say "it's on a free tier, it cold-starts"
and keep talking. Don't sit in silence watching a spinner.

## 1. Frame the problem (30 sec, before clicking anything)

> "Product teams collect thousands of reviews and support tickets, and nobody
> reads them. The usual fix is to throw it all at an LLM and ask for a summary.
> That fails for a specific reason: a PM can't act on a summary they can't
> verify. If the tool says 'fix payments first' and can't show why, it gets
> ignored.
>
> So I built Prism around one rule. Every insight has to be able to defend
> itself."

Now share the screen.

## 2. Home (90 sec)

Point at the KPI row.

> "3,904 pieces of real feedback from two sources: Google Play reviews of
> Spotify and support tweets to their help account. It found 40 themes."

Point at **Citations verified 40/40**.

> "I'll come back to this number. It's the one I'm proudest of."

Scroll to the ranked themes.

> "This ordering isn't the model's opinion. It's a formula: 35% frequency,
> 35% severity, 15% recency, 15% source diversity. That last one matters.
> A problem showing up in both reviews and support tickets is more credible
> than one living in a single channel."

Toggle **Hide low-confidence themes** in the sidebar.

> "And it tells you when it doesn't have enough evidence instead of bluffing."

## 3. Insight Detail (2 min) — the important screen

Open the top theme.

> "Account Access Issues. 197 items, and look at the source split: 173 support
> tickets, 24 reviews. People don't leave a review when they're locked out,
> they contact support. You'd miss this if you only read one source."

Root causes:

> "Root causes are labelled hypotheses, deliberately. Feedback text can suggest
> a cause, it can't prove one. Claiming otherwise would be the kind of
> overconfidence that makes PMs distrust these tools."

Recommended actions:

> "Actions with effort estimates, not summaries. That's the difference between
> a report and something you can take to sprint planning."

The score breakdown chart:

> "Every point of that 0.77 is explained. If a PM challenges the ranking,
> there's an answer."

Evidence quotes:

> "Real quotes. The starred ones are what the model cited as evidence, and
> those citations get checked in code against the actual cluster members.
> If it cites something that isn't there, it's flagged red, not hidden.
>
> First run, only 19 of 39 passed. I assumed hallucination. Then I read the
> failures: it was citing real IDs but dropping the source prefix, like a page
> number without the book title. Fixed the resolver, now it's 40 out of 40,
> and genuine hallucinations still fail."

Wilson interval:

> "And this is the interval, not a point estimate. Small samples get wide
> intervals and a warning badge. Twelve angry users shouldn't outrank two
> hundred quiet ones."

## 4. Explore Feedback (45 sec)

> "Every dot is one piece of feedback, positioned by meaning. Grey is
> unclustered, which I show rather than hide. My first clustering run marked
> 64% as noise, so insights would have covered a third of the data. I ran a
> parameter sweep, then let the algorithm find dense cores and assign
> stragglers to the nearest one only if they're actually similar. Noise went
> to 14% and coverage stayed honest."

Search something like "crash" to show it's live data.

## 5. Ask Iris (45 sec)

> "This is Iris. She only sees the generated insights, so she can't invent
> a theme that doesn't exist."

Click **What are the top 3 problems?** and read the answer.

> "Same numbers as the dashboard, because it's the same source of truth."

## 6. Methodology (20 sec) — only if time allows

> "And if anyone challenges a number, this page is the audit trail. Pipeline,
> the formula with live weights, and the limitations written down."

## 7. Close (30 sec)

> "The architecture point I'd make: classical ML runs per item, the LLM runs
> per cluster. Embeddings and clustering happen locally on CPU for free, and
> the model only sees about 40 cluster-level samples. So a run costs 40 API
> calls whether there are 4,000 items or 40,000, it fits in a free tier, and
> it's cached so re-runs cost nothing.
>
> The hard part of this project wasn't calling the model. It was the
> engineering around it that makes the output trustworthy and affordable."

## If they interrupt

Good. Answer, then say "let me show you that" and jump to the page. The script
is a route, not a rail.

## If something breaks live

Say what happened plainly and keep moving. If Iris is rate-limited: "free tier,
daily quota. The analysis is all pre-computed, so everything else works."
Debugging live is a worse look than acknowledging a limit you designed for.

## If they ask "did you use AI to build this?"

Answer honestly. It's 2026, everyone does, and pretending otherwise reads
badly.

> "Yes, I used Claude as a coding partner. I made the design decisions: the
> scoring formula, the per-cluster architecture, the citation check, what to
> do when the evidence is thin. I also debugged the things that actually broke:
> 64% clustering noise, two API providers hitting quota limits mid-run, Groq
> retiring the entire model family my pipeline ran on. Knowing what to build
> and why it's wrong when it breaks is the part that mattered."

Then pivot to a decision you can defend in depth. The scoring weights and the
noise tradeoff are both good ground.
