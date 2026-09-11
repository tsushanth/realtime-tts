# Cross-cycle pattern analysis (6 real automated cycles, 2026-09-08)

## Pattern 1: our flow's slot-filling is consistently worse than Retell's — 4/4 relevant cycles

Every cycle that reached a real booking conversation showed the same shape
of gap, phrased differently each time by the judge but structurally
identical:

| Cycle | What ours did | What Retell did |
|---|---|---|
| 3 (manual) | Asked name, then re-asked name after it was given | Asked name+time together, one shot |
| 4 | (same session as 3, reconfirmed) | — |
| 5 | Name in a separate confirm turn, then date "acknowledged" separately, then time asked separately (3 separate slot-asks) | Name+time together, immediate slot offer |
| 6 | Name+time together this time — but never resolved "afternoon" into a concrete time, closed without answering the customer's direct question about the time | Name+time together, offered a specific slot (2 PM), confirmed before closing |

Verdict: this is not a one-off — it's a structural pattern. Ours treats
name/date/time as sequential form fields; Retell's treats them as one
combined ask and always resolves a vague answer ("afternoon") into a
concrete committed slot. This is the single highest-value fix identified
across all cycles — worth doing before anything else, since it showed up
in some form in every single cycle that got far enough to matter.

**Concrete fix**: rewrite the booking node's extraction prompt to (a) ask
for name and date/time in one combined question, (b) never accept a vague
time as final — the LLM must propose/confirm a specific clock time before
transitioning to the goodbye node.

## Pattern 2: our own flow double-closes — 1/4 cycles (cycle 5)

After the flow's own goodbye node fires and the customer says goodbye, the
BUSINESS side (not the shopper) sometimes speaks a second, redundant
closing line ("Is that all set?" then another "Thanks so much for calling
[...] Have a great rest of your day" after the customer already said
goodbye). Lower frequency than Pattern 1, but a real, separate bug in the
production flow, not a framework artifact — this is our own code's
goodbye-node logic, unrelated to the shopper's hangup detection.

**Concrete fix**: once the flow reaches its goodbye node's own closing
line, treat any further customer utterance that's itself goodbye-shaped
as terminal — don't generate a second farewell.

## Pattern 3: the shopper's own goodbye-loop fix is incomplete — 1/4 cycles regressed (cycle 6)

The fix shipped after cycle 2 (hang up after 2 closing-shaped SHOPPER
replies) worked cleanly in cycles 3, 4, and 5, but recurred in cycle 6.
Root cause confirmed from the raw transcript (see conversation): the
detection only watches the shopper's OWN replies, not the other party's.
Here the other party (Retell) also said a clear goodbye ("You're very
welcome. Goodbye, and take care!") right after the shopper's first
goodbye — a much stronger and earlier signal to hang up than waiting for
a second matching phrase from the shopper itself, which never came
because dead-air mis-transcription fed the LLM garbage input instead.

**Concrete fix**: hang up as soon as EITHER party's reply is
closing-shaped following the shopper's own first closing-shaped reply,
not only after two from the shopper. This closes the loophole dead-air
mis-transcription was exploiting.

## Pattern 4: TTS/turn-management stutter on our side — 1/4 cycles (cycle 5)

Retell showed a "You're / You're welcome!" repeated-fragment stutter on
one closing line — flagged by the judge as a real defect on the Retell
side specifically. Only seen once; not enough data to call this a
pattern yet, just logged for completeness. Notably this is evidence the
judge isn't systematically biased toward one side — it dinged Retell
here.

## What did NOT recur (worth noting so it isn't over-fixed)

- The "transfer you to our booking system" phantom-transfer line (flagged
  cycle 1/3/4) did not appear in cycles 5 or 6 — possibly already
  inconsistent/prompt-dependent rather than a fixed, guaranteed defect.
  Worth confirming with a larger sample before spending effort on it
  specifically; Pattern 1 (slot-filling structure) subsumes most of its
  practical impact anyway.

## Stopping point after cycles 7-9 (2026-09-09)

Shipped and confirmed working, in order:
- Combined name+time ask, vague-time resolution, hangup-loophole fix
  (either party's closing-shaped reply now triggers hangup, not just the
  shopper's own two) — cycle 7 confirmed clean, no loop/hallucination.
- Goodbye-node internal redundancy fix (no more double "thanks"/repeated
  name in one closing line) — addressed after cycle 7's finding.
- `record_field` tool: persists a captured field the instant it's given,
  independent of transition_flow, since `collectedData` previously only
  updated at transition time — real root-cause fix for "the model forgot
  a name it was already given," identified from raw logs after cycle 8.

Still open, NOT resolved: cycle 9 showed the model calling `record_field`
for `preferred_time` but not for `name` in the same turn where the caller
gave both — the tool works, but isn't reliably invoked for every field
every time. Cycle 9 also ended with no closing at all, a new failure mode
not seen in cycles 1-8.

Pattern across cycles 7-9: each fix resolved the specific prior failure
and surfaced a *different* new one (goodbye redundancy → name-forgetting
with a truncated fragment → name-forgetting with no closing at all)
rather than converging. This node's system prompt has grown to ~6 stacked
behavioral instructions (combined-ask, vague-time resolution, name/number
read-back, no-closing-language, no-redundant-confirmation, record_field
usage) — instruction overload is a real candidate explanation for
inconsistent tool-calling, distinct from any single bug.

**Decision: stopped here deliberately** rather than keep patching forward
— per user direction, banking the confirmed wins (listed above, all live
in production) and leaving the remaining booking-flow polish for a fresh
pass, likely by simplifying/consolidating the node prompt rather than
adding another targeted instruction on top of it.

## Cycles 10-15 (2026-09-11): applying judge suggestions, real deadlock chain uncovered

Asked to apply cycle-9's judge suggestions and rerun. This surfaced a
chain of FOUR distinct, increasingly deep real bugs, each found only
because the prior fix was validated with another real call rather than
assumed correct:

1. **record_field suppressing spoken output** (cycle 10) — the model
   treated calling record_field as the complete turn action, producing no
   text at all. Fixed the prompt wording; had zero effect (cycle 11) —
   proved this was real Claude Haiku tool-use behavior, not a prompt-
   compliance issue, by confirming `final.content` genuinely had no text
   block (not a missed-streaming-event bug).
2. **Wrong fallback message** (cycle 12) — the generic "Sorry, could you
   say that again?" fallback fired every time record_field succeeded
   silently, falsely implying an error when the field was captured
   correctly. Fixed: only use that apology when NO tool call happened
   either; let a silent-but-successful tool call pass without comment.
3. **CLOSING_SHAPED_RE false positive** (cycle 13) — the regex's bare
   "thanks$" branch matched ordinary conversational replies ("Perfect,
   thanks!"), falsely arming the either-party hangup check and killing a
   call mid-booking. Fixed by requiring real farewell phrases only.
4. **Genuine deadlock** (cycle 14, found via a live diagnostic run after
   explicitly deciding to keep digging rather than stop) — the model
   recorded every required field via record_field, satisfying the node's
   transition condition, but called neither transition_flow nor said
   anything else. With no new caller utterance coming, nothing was left
   to prompt it forward — both sides silently waited until the 3-minute
   safety cap. Fixed at the code level (not prompt wording): once per
   node, if every extract field is captured but no transition happened,
   push a system-note nudge and generate a follow-up turn immediately.

Cycle 15 confirmed the fix: call closed cleanly for the first time since
cycle 9, with real progress found: no stall, no false apology, no
premature hangup, no hallucination. The one remaining issue is a
duplicate name-ask — the SAME category of bug found in cycle 3, milder
than anything found in cycles 10-14, suggesting the deeper/catastrophic
bugs are now cleared and iteration is back to polishing the original,
lower-severity pattern.

**Lesson for next time:** three of these four bugs would have been
invisible without pulling raw logs after every single cycle rather than
trusting the judge's summary alone — the judge correctly identified
symptoms (call didn't close, weird apology, etc.) but the actual root
causes were only findable in `flyctl logs`, cross-referenced against the
code that had just changed.

## Priority order for fixes, given the above

1. **Pattern 1** (slot-filling redesign) — highest frequency, highest
   impact, affects every cycle.
2. **Pattern 3** (shopper hangup loophole) — affects the framework's own
   reliability, not production quality, but blocks trustworthy future
   runs if left as-is (a cycle can silently burn 3 minutes of real call
   time and produce a noisy transcript).
3. **Pattern 2** (double-close in our flow) — real but lower frequency.
4. Pattern 4 — insufficient data, no action yet.
