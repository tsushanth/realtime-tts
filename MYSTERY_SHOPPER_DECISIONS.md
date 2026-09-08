# Mystery-shopper framework — decision log

Goal: an automated, repeatable framework that places the SAME scripted
customer call against (a) Retell+ElevenLabs and (b) our own engine
(call-loop-poc)+ElevenLabs, captures both conversations, and has an
independent judge (Claude, blind) score and compare them so we get concrete,
repeatable suggestions instead of one-off manual test calls.

## Decision 1: the "shopper" is a real AI agent placing outbound calls, not a fixed script

Why: we already know Retell's flow and ours ask for booking details in a
different order (name+time together vs. separately). A fixed script of
caller lines said at fixed times would desync against whichever backend
doesn't ask in the expected order. An adaptive agent (STT -> LLM with a
customer persona/goal -> TTS) responds to whatever's actually asked,
regardless of backend. User confirmed this tradeoff explicitly (chose "Full
AI agent" over "fixed script with branches").

## Decision 2: reuse call-loop-poc's existing CallSession, don't build a new service

Why: CallSession already supports flow-less operation (a flat systemPrompt,
no node graph) when no `flow` is set in its context message — this is
already how anonymous browser demo calls without a business flow work
today. A "shopper" is just a CallSession with:
  - a customer-persona systemPrompt instead of a business prompt
  - no auto-generated opening line (a shopper calls IN, it should wait for
    the far end to greet first, not speak first)
That second point needs one real code change: today `_runNodeTurn`'s
call-opening logic (push a synthetic "[Call connected — begin the flow.]"
user turn) only fires when `this.flow` is set, so a flow-less CallSession
already stays silent until it hears something — confirmed by reading
server.js, no change needed there. Good — less to build than expected.

## Decision 3: both call legs go through OUR Twilio number and OUR /twilio/voice webhook

Why: regardless of which phone number we dial (our own business number, or
Retell's agent's phone number), Twilio's outbound Calls.json API always
routes the CALLING leg's TwiML through the `Url` we specify — so our
existing /place-test-call + /twilio/voice + TwilioCallAdapter infra works
unchanged for "call a business" in general, not just "call ourselves." The
only new thing is a `mode=shopper` flag so /twilio/voice sets up a shopper
persona instead of looking up a tenant's real flow.

This also means Twilio's own call recording (already wired via the
`record:true` flag added earlier) captures BOTH directions of audio for
either target, since it's one call resource from our account's perspective
either way — we don't need Retell-side recording access at all for the
"ours" leg, and for the Retell leg we cross-check against Retell's own
list-calls/transcript API using phone number + timestamp to find the
matching inbound call on their side (we can't get a Retell call_id directly
since we didn't place the call through Retell's own API).

## Decision 4: judge scores BLIND (Call A / Call B, randomized which is which)

Why: an unblinded judge told "here's ours, here's Retell's" carries an
obvious labeling bias risk (models can pick up on framing). Blind A/B with
the label randomized per run, then de-anonymized only when reporting back
to the user, avoids that.

## Decision 5: safety caps

A shopper call needs a hard max-duration / max-turn cutoff independent of
either backend's own goodbye logic, in case a backend gets stuck in a loop
or never naturally ends the call -- otherwise a bug on either side could
leave a real phone call (and real per-minute billing on both Twilio and
Retell) running indefinitely.
