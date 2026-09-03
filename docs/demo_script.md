# Interview walkthrough (about 6 minutes)

Talking points, not a recital. Read them a few times, then talk normally.
If you memorise sentences you'll sound like you memorised sentences.

## Setup before screen share

App loaded on the Spotify dataset. Second tab on Insight Detail with the top
theme already open. Notifications off. GitHub in a third tab in case they want
code. If the app is slow to wake, say it's on a free tier and keep talking.

## Opening, before you share anything

> "So the problem I started with is that companies collect thousands of reviews
> and support tickets and nobody actually reads them. You can dump it all into
> ChatGPT and get a summary, but I don't think a PM can act on that. If a tool
> tells you to fix payments first and you can't check why, you're just trusting
> it. So the whole thing is built around making every output traceable back to
> real quotes."

## Home

> "This is real data. About four thousand pieces of feedback from two sources,
> Google Play reviews for Spotify and support tweets to their help account. It
> found forty themes."

Point at 40/40 citations. "I'll come back to that one."

> "The ranking isn't the model deciding. It's a formula and you can see the
> weights: frequency and severity at 35% each, then recency, then whether it
> shows up in more than one source. That last one is there because if something
> appears in reviews and in tickets, I trust it more than something sitting in
> one channel."

Toggle hide-low-confidence. "It also tells you when there isn't enough evidence
instead of just ranking it anyway."

## Insight Detail

> "Top theme is account access. 197 items. The split is the interesting part,
> most of it is support tickets and only about 24 reviews. Which makes sense,
> if you're locked out you don't go write a review, you contact support. So if
> I'd only looked at reviews I'd have basically missed it."

Root causes:

> "I labelled these as hypotheses deliberately. Feedback can point at a cause
> but it can't prove one, and I didn't want it stating guesses as fact."

Actions: "These have effort estimates so you could actually take them to
planning."

Score chart: "And you can see where the 0.77 came from, so if someone argues
with the ranking there's something to argue with."

Evidence, and tell this one properly:

> "The starred quotes are the ones the model said it was using. There's a check
> in code that those IDs actually exist in the cluster. First run, only 19 out
> of 39 passed and I assumed it was hallucinating. But I read the failures and
> it was citing real IDs, it was just dropping the source prefix off the front.
> So it was my ID format, not the model making things up. Fixed the matching
> and it's 40 out of 40 now, and actual hallucinations still fail."

Wilson interval:

> "This is a range rather than one number. Small clusters get a wide range and
> a warning badge. I didn't want twelve loud users outranking two hundred quiet
> ones."

## Explore

> "Every dot is one piece of feedback placed by meaning. The grey ones didn't
> fit any theme and I show them rather than hide them. My first version marked
> 64% as noise, which was useless, the insights would've covered a third of the
> data. I swept the parameters and changed the approach so it finds the dense
> groups first and then pulls in nearby stragglers. Got it to 14%."

Search "crash" so they can see it's live.

## Iris

Click a suggested question, read the answer.

> "She only gets the generated insights as context, so she can't invent a theme
> that isn't in there. Same numbers as the dashboard because it's the same file."

## Close

> "The design decision I'd point at is that all the per-item work runs locally.
> Embeddings, clustering, sentiment, none of it touches an API. The model only
> sees cluster-level samples, so it's about forty calls per run whether there's
> four thousand items or forty thousand. That's what keeps it inside a free
> tier and makes re-runs free.
>
> Most of my time went on the parts that make it trustworthy. The AI calls were
> honestly the easy bit."

## When they interrupt

Answer, then say "I can show you" and go to the page. Don't finish your
sentence first.

## When something breaks

Say what it is and move on. Iris rate-limited: "free tier, daily quota. The
analysis is all pre-computed so the rest works." Do not start debugging.

## "Did you use AI to build this?"

Say yes. Everyone does and dodging looks worse.

> "Yeah, Claude wrote a lot of the code. The decisions were mine: the scoring
> weights, running analysis per cluster instead of per item, adding the citation
> check. And most of my time went on things breaking. The clustering was
> useless at first, I hit quota walls on two different providers, and at one
> point Groq deleted the model my whole pipeline was running on. That's where
> the real work was."

Then move to something you can go deep on. The scoring weights and the noise
tradeoff are both good ground.
