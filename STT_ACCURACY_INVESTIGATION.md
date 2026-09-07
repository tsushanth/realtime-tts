# STT accuracy investigation — handoff

Trigger: a real test call transcribed the caller's name "Sushant" as "Ashant"
(Deepgram Flux, `flux-general-en`, no custom vocab). Retell's own ASR got the
same class of name right on a comparable call. This is a candidate
contributor to the "night and day" quality gap, separate from the TTS-model
fix and the flow-design gaps already identified (see conversation / git log
around 2026-09-07 for those).

Current config (`call-loop-poc/server.js:278-279`):
```
wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000&eot_threshold=0.7&eot_timeout_ms=5000
```
No `keyterm`/`keywords` boosting, no `smart_format`, no per-tenant vocabulary,
no fallback/second-pass on low confidence. This doc breaks the problem into
independently investigatable tracks so they can be parallelized.

## What we actually know vs. what we're assuming

Known: one real call, one misheard proper noun. That's an n=1 anecdote, not
a measured error rate. Before spending engineering effort on a fix, get a
real WER (word error rate) baseline — otherwise track 2-4 below risk being
solutions to a problem we haven't confirmed is systemic vs. one unlucky
utterance ("Sushant" said quickly on a phone mic is genuinely ambiguous
audio — a human could mishear it too).

## Track 1 — Measure before fixing (do this first, blocks nothing else)

Build a small real-call test set (10-20 calls) covering:
- Proper nouns / names (the actual failure mode observed)
- Numbers (dates, times, phone numbers — high-stakes for a booking flow)
- Noisy backgrounds / phone-mic quality (not clean studio audio)

For each: capture Deepgram's transcript vs. ground truth (what was actually
said) and compute word error rate. This tells us whether we have a systemic
STT problem or a name-specific edge case. Also worth pulling Deepgram's own
per-word confidence scores from the Flux response (if exposed) — low
confidence on exactly the words that get bungled would validate a
confidence-gated re-ask strategy (Track 4) cheaply.

## Track 2 — Model/config knobs within Deepgram (cheapest, no architecture change)

- **`flux-general-en` vs `flux-phonecall-en`** (if it exists) or Deepgram's
  phone-optimized model variant — check Deepgram's current model catalog,
  general-purpose models are tuned on different acoustic conditions than
  8kHz PSTN audio specifically.
- **`keyterm`/`keywords` boosting** — Deepgram supports boosting recognition
  of a supplied word list. For a booking flow this is directly actionable:
  boost common first names, or even the specific caller's name if collected
  via caller ID / CRM lookup before the call starts.
- **`smart_format`** — check if it's on; affects number/date formatting
  which matters a lot for a booking flow's "tomorrow at 1pm" parsing.
- **`eot_threshold=0.7`** — this controls turn-taking confidence, not
  transcription accuracy, but worth confirming it isn't cutting off
  mid-utterance and truncating audio Deepgram would have transcribed
  correctly given more context.
- Sample rate mismatch: we connect to Deepgram at 16kHz
  (`sample_rate=16000`) after upsampling real Twilio 8kHz mu-law input via
  the same naive nearest-neighbor resampler flagged in `twilioAdapter.js`
  for the *output* path. Upsampling doesn't create information the mic
  didn't have, but check whether Deepgram has an 8kHz-native phone model —
  skipping our own upsample and feeding it real 8kHz audio directly might
  avoid resample artifacts entirely, however small.

## Track 3 — Alternative STT vendors (bigger lift, worth benchmarking)

Since TTS backend is already pluggable in this codebase, STT could be made
pluggable the same way and A/B tested:
- **AssemblyAI** — real-time streaming, known strong on phone-call audio,
  has built-in PII/entity detection that could double as name recognition.
- **Speechmatics** — markets itself specifically on accuracy across accents;
  worth checking real-time API pricing/latency vs Deepgram.
- **OpenAI's realtime transcription** (if using a cascaded, not full
  Realtime-API, approach) — check current accuracy benchmarks.
- **Retell's own ASR** — we don't know what they use; if discoverable (their
  docs/changelog sometimes mention vendor), worth checking directly rather
  than guessing.

For any candidate: same audio in, compare WER against the Track 1 test set.
Don't swap vendors on vibes — the same discipline the TTS investigation used
(native mu-law fix confirmed via real evidence, not the first plausible fix)
applies here.

## Track 4 — Architecture-level mitigations (independent of which STT wins)

These don't require the "best" STT vendor, they reduce impact of any STT's
mistakes:
- **Confirmation-by-repeat for high-stakes fields.** Retell-style and most
  production voice bots explicitly repeat back captured names/times as a
  question ("Got it, Sushant — is that right?") rather than assuming
  first-pass transcription is correct. This is a flow-design fix, not an
  STT fix, and it's the single highest-leverage mitigation regardless of
  which vendor we use, since transcription will never be 100%.
- **Confidence-gated re-ask**: if Deepgram exposes per-word/per-utterance
  confidence and it's low on a critical slot (name, date), have the LLM ask
  for spelling or repetition instead of silently accepting a low-confidence
  guess.
- **Per-tenant custom vocabulary**: if a business has a known caller list,
  common local names, or product/service names, feed those into
  keyword-boosting per call rather than one global config.
- **Caller-ID-based name pre-fill**: if the inbound number is in a CRM/prior
  call log, skip asking for the name at all where possible — removes the
  chance of mis-transcribing it.

## Suggested split for parallel investigation

- **Agent A**: Track 1 (measurement/baseline) — needs real test calls or a
  recorded audio corpus, produces the WER numbers everything else should be
  judged against.
- **Agent B**: Track 2 (Deepgram config knobs) — desk research + Deepgram
  docs, cheap to try once Track 1's test set exists.
- **Agent C**: Track 3 (vendor comparison) — desk research now (pricing,
  latency claims, streaming API shape), real comparison blocked on Track 1's
  test set existing.
- **Agent D**: Track 4 (flow-level mitigation, confirmation-by-repeat) — no
  dependency on STT vendor choice, can ship independently and immediately;
  probably the best ROI-per-effort of all four tracks.

Track 4 doesn't need to wait on anything — recommend starting there while
the others gather evidence.
