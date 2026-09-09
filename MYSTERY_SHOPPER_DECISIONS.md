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

## Finding from cycle 1 (real run): goodbye-loop + character break

First real run: both the Retell call (181s) and our own call (80s, but a
stale-looking tail) showed a bizarre pattern — after the real booking
conversation ended, the SHOPPER kept going for another 60-90s, ultimately
hallucinating an near-identical ("I'll stay in character", "share the setup
and I'll get started") meta-conversation about "the roleplay scenario" on
BOTH calls independently. Root cause: neither side proactively hangs up
after saying goodbye (normal phone etiquette — wait for the other party),
so both loop exchanging "bye"/"take care" until the repetitive context
causes the LLM to break character. This is a shopper-design bug, not
something either backend did wrong — confirmed by it happening identically
on both. Fix shipped: shopper hangs up proactively after its second
closing-shaped reply (see the `isClosing` regex + `_shopperClosingCount` in
server.js).

## Finding from cycle 2 (real run): fix works against Retell, not against ourselves

Re-ran both calls after the fix. Retell call: 64s, clean hangup — fix
worked. Our own call: still hit the 181s safety cap. Logs show why: the
shopper's `close()` fired correctly at the right point in the conversation,
but the underlying CallSession kept processing NEW incoming turns and
speaking new replies afterward (the shopper re-asked for the caller's name
a second time, well after "hanging up"). Root cause:
`TwilioCallAdapter.close()` only actually disconnects once its audio queue
drains, but nothing stops `_onUserTurnComplete`/`_generateTurn` from
continuing to run and refilling that queue if the other side keeps talking
— so if the other party never stops, the queue never drains and the real
disconnect never happens. This didn't show up on the Retell call because
Retell's own system independently hung up its side of the real phone line;
it only surfaces when our own close() is the only thing that's supposed to
end the call. NOT YET FIXED — next concrete step: set a `this._closing`
flag when hangup is decided and make `_onUserTurnComplete`/`_generateTurn`
a no-op once it's set, so no new turns are generated after a hangup
decision regardless of what the other side does; consider also having
close() explicitly call Twilio's REST API to force-terminate the Call
resource rather than relying solely on the media-stream WebSocket closing.

## Finding from cycle 3: unrelated infra bug surfaced by the framework itself

Cycle 3 (re-testing the "ours" self-call after the _closing-flag fix) showed
a THIRD, completely different symptom: the shopper never spoke at all after
the business's opening greeting, silent for the full 3-minute safety cap.
Root cause, found in the logs: `ElevenLabs request failed: 401` right as
the business tried to speak its greeting — production's ElevenLabs key was
returning a real 401. Verified directly against ElevenLabs' API (not just
assumed from the log): the OLD key was unauthorized, and even a fresh key
the user provided returned `quota_exceeded` (0/10000 credits) — a real
account-level billing issue, not a code or auth-config bug. Fixed by
setting the new key as the Fly secret (ready to work the instant credits
are added; the user is adding credits separately).

This is worth calling out explicitly: the framework's job is to catch real
quality regressions, and its first three real runs caught (1) a shopper
design bug, (2) a close()-doesn't-stop-new-turns bug, and (3) a completely
unrelated production infra outage — none of which were things anyone was
specifically looking for. That's the framework doing exactly what it's for.

## Full automation validated (mystery-shopper-run.sh)

Per explicit user direction: the Retell number swap/restore no longer asks
for per-run confirmation (user: "it should run whenever.. reuse then put
it back") and the judge runs via the `claude` CLI (OAuth session auth)
instead of the Anthropic SDK + ANTHROPIC_API_KEY (user: "make sure
everything runs off of oauth"), consistent with the repo's own rule
against burning the API key on one-off local analysis.

Closed the concurrent-run transcript-disambiguation gap flagged after
cycle 4: every turn log line now carries `[call <Twilio CallSid>]`, so a
script can grep one specific session's turns out of several interleaved
ones instead of a human trying to disentangle colliding turn counters by
hand. `mystery-shopper-run.sh` uses a `trap ... EXIT` to guarantee the
Retell number gets restored to Audexa DJ even if the script fails
partway through — the swap is the one step whose failure mode (a live
production number stuck pointed at the wrong agent) actually matters.

First fully-unattended run: real, different finding each time (this run:
our flow never resolves a vague time like "afternoon" into a concrete
slot and closes without answering the customer's direct follow-up
question) — plus a fair, balanced result flagging a real Retell-side TTS
stutter, not a one-sided "ours is always worse" outcome. Framework is now
genuinely push-button.

## Decision 5: safety caps

A shopper call needs a hard max-duration / max-turn cutoff independent of
either backend's own goodbye logic, in case a backend gets stuck in a loop
or never naturally ends the call -- otherwise a bug on either side could
leave a real phone call (and real per-minute billing on both Twilio and
Retell) running indefinitely.
