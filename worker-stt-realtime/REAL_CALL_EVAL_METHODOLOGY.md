# Real-call STT evaluation: data, method and decisions (2026-09-24)

Companion to `REAL_CALL_EVAL.md` (results and gate) and `bench/results/real_summary.json` (aggregate numbers per
engine). Spec: `docs/superpowers/specs/2026-09-24-realtime-stt-real-call-eval-design.md`. Plan:
`docs/superpowers/plans/2026-09-24-realtime-stt-real-call-eval.md` (Tasks 1-6). Branch: `stt-realtime-eval`.
This file contains no call audio, transcripts, phone numbers or call IDs.

## 1. Why we did this

Goal: a cheap, low-latency, CPU-bound streaming English STT service, like Piper is for TTS, to compete with ElevenLabs
Scribe v2 Realtime ($0.39/audio-hour) and Deepgram Flux ($0.39/hr; Nova-3 streaming $0.29/hr; Flux is what
`call-loop-poc` uses). Before this work we had CPU engine benchmarks (zipformer, NeMo FastConformer, Moonshine) but only
on read audiobook speech (LibriTTS) and macOS-`say` synthetic turns, no head-to-head with the clouds, and nothing
deployed. Engine choice decides what Phase 2 (Fly deploy + autoscaler) builds, so we measured on real phone audio first.

## 2. What was compared

| engine | what | notes |
|---|---|---|
| `zip-en-int8` | sherpa-onnx streaming zipformer (2023-06-26), int8, Apache-2.0 | LibriSpeech-only training; server default |
| `nemo-480` | NeMo streaming FastConformer transducer, 480 ms chunks, int8, CC-BY-4.0 | best local candidate in earlier benchmarks |
| `dg-flux` | Deepgram Flux `flux-general-en`, `/v2/listen`, linear16 16 kHz, eot_threshold 0.7, eot_timeout_ms 5000 | cloud |
| `el-scribe` | ElevenLabs `scribe_v2_realtime`, `commit_strategy=vad`, `pcm_16000`, language en | cloud, VAD settings left at defaults |

Software: sherpa-onnx 1.13.8, faster-whisper 1.2.1 (drafts), websockets 17.1, jiwer 4.0.0, numpy 2.5.3. Local engines ran on
an Apple M2 Pro (1 thread per engine), a proxy for the Fly target, not the same silicon.

## 3. Data: what we used and why

**Source.** Calls recorded by the `call-loop-poc` engine (Twilio, 8 kHz), logged in the calldesktech Supabase table
`calldesk_call_logs`: 213 recorded poc-engine calls, 17-21 Sept 2026, median ~2.4 min (~7.5 h). Why these: they are the
real telephone path our product hears (8 kHz codec, real Twilio audio), we own them, and there is a working way to fetch
them. Limits: most far-end speech is synthetic (a test framework's scripted personas), plus calls the system placed to the
owner's cell and calls from the owner's own toll-free line. No real customers are on these calls (owner attestation, and
spot checks of the Twilio call records).

**Which channel.** Recordings are 2-channel. Checked by transcribing each channel of one inbound and one outbound call
against the stored assistant lines: channel 0 is the far end (caller/callee/persona), channel 1 is our agent. We use
channel 0 only because production STT only hears the far end; mixing both would score our own TTS voice as speech the
STT should have heard.

**How audio was fetched.** The Twilio token in `call-loop-poc/.env` was stale (API 401). We fetched through
calldesktech's existing `/recording-audio` proxy (`Bearer` secret from `calldesktech/.env`), requesting the `.wav`
variant (8 kHz PCM stereo, lossless) rather than the stored mp3 URL, then kept channel 0 as mono 16 kHz PCM_16.
Integrity checks: sidecar file recording the channel, duration sanity check against the logged duration, atomic writes.

**Own-call selection.** The plan first selected by phone number. Real data showed the log's phone columns are unreliable
(outbound calls to the owner's cell were logged with the poc line as `to_number`; 133 inbound rows log caller = called
number). We used Twilio's own records (via the Twilio MCP) to identify parties on samples and the owner attested that all
poc-engine calls are their own tests. The binding privacy control for third-party engines is a per-call allowlist
(`data/confirmed_calls.txt`), enforced in code: Deepgram/ElevenLabs runs refuse any clip whose call is not listed, and
fail closed for the `real` set when a clip has no call id.

**Subset.** Segmenting and hand-checking all 213 calls was too much, so we took a stratified 30-call subset (durations
20-200 s, spread across dates): 8 outbound-to-owner, 10 from the owner's toll-free line, 12 test-persona calls
(60.8 min). Why stratified: outbound calls to the owner's cell are the closest thing to human/voicemail speech; a random
draw would have been almost entirely synthetic personas.

**Segmentation.** Silero VAD (via sherpa-onnx) on the far-end channel, utterances of 1.5-15 s (min silence 0.6 s,
padding 0.25 s each side, never longer than 15 s including padding): 150 utterances from 26 calls (4 calls had no usable
speech), 10.9 min of speech.

**References (ground truth).** No human transcripts existed. Each utterance got a draft from faster-whisper
large-v3-turbo; the owner then listened one utterance at a time and corrected or rejected. Outcome:

- 150 segmented -> **110 verified** (21 calls: 72 test-persona, 24 toll-free-line, 14 outbound-to-owner; 7.5 min of
  speech; 1,255 reference words, 1,340 after digits are spelled out by the scorer).
- **51 hand-listened, 59 accepted from the draft without listening** (owner instruction to speed up); each verified
  utterance carries `ref_source` = `listened` or `assumed-from-draft` in the manifest.
- **Excluded 40**: 34 non-English (French, Spanish, Italian, Dutch and others: the drafts had silently translated them
  into English), 1 clipped mid-word at the segment end, 2 garbled, 3 by request. Reason: the evaluation is English-only.
- **33 keyterms** (names, digit strings, a few place/product names) extracted automatically from verified text (the owner
  approved this instead of typing them); three keyterms taken from obviously garbled draft numbers were dropped.

## 4. Method

Same paced replay as `bench/rtbench.py`: each utterance is fed in 40 ms chunks on the wall clock with a 0.3 s lead-in and
1.6 s trailing silence, like a live phone stream. For cloud engines a fresh websocket per utterance, opened before pacing
starts.

- **WER**: jiwer on normalised text (`textnorm.norm`: lowercase, punctuation stripped, digits spelled as words) over the
  final hypothesis.
- **Keyterm recall**: fraction of keyterms present (whole-word match after normalisation) in the final hypothesis.
- **Latency**: `commit` = client signals end of speech exactly at clip end and we time to the final (for Deepgram this
  includes the CloseStream flush); `native` = the engine's own endpointing (Deepgram EndOfTurn, ElevenLabs VAD commit);
  first partial = time to first non-empty hypothesis.
- **CPU**: process CPU-seconds per audio-second (local engines only). Price for local engines is compute-only at the Modal
  CPU rate assuming a fully packed machine; not like-for-like with per-hour cloud pricing.
- **Decision gate** (`bench/report_real.py`, from the spec): the best local engine must have WER <= 1.5x the best cloud
  WER, keyterm recall within 10 points of the best cloud keyterm recall, commit-final median <= 150 ms and <= 0.15
  CPU-seconds per audio-second. Checks are tri-state: PASS / FAIL / INSUFFICIENT (needs n >= 30 utterances, >= 30
  keyterms, equal n across engines, and >= 90% of utterances firing).

## 5. Results

| engine | WER | keyterm recall | first partial ms | commit final ms med/p90 | native final ms med/p90 | cpu-s per audio-s |
|---|---|---|---|---|---|---|
| el-scribe | 2.5% | 93.9% | 2245 | 110/274 | 1517/1606 | n/a |
| dg-flux | 4.0% | 78.8% | 1045 | 168/265 | 580/1127 | n/a |
| nemo-480 | 8.3% | 63.6% | 1288 | 52/85 | n/a | 0.089 |
| zip-en-int8 | 15.4% | 39.4% | 1139 | 37/67 | n/a | 0.063 |

Gate: both local engines FAIL (WER 3.3x and 6.2x the best cloud; keyterm recall 30 and 54 points behind); latency and CPU
checks pass. Robustness: on the 51 hand-listened utterances only, WER is 0.9% / 3.6% / 8.8% / 14.1% (same order), and
dropping filler words changes nothing material. Aggregate JSON for all engines: `bench/results/real_summary.json`.

## 6. Caveats

- Mostly synthetic far-end speech (test personas); only ~14 verified utterances are non-synthetic. Human callers may be
  easier or harder for each engine.
- Small sample (110 utterances, 33 keyterms). The 8.3% vs 2.5% gap is large; 2.5% vs 4.0% is within noise.
- 59 of 110 references are unlistened Whisper drafts. Whisper-written references could favour Whisper-like cloud models;
  the listened-only rerun shows the ranking does not change.
- Latencies are not like-for-like: ElevenLabs endpointing is at its untuned default; local engines have no server-side
  endpointing in this harness (only the commit path); Deepgram's commit path includes a flush. First-partial numbers
  include the harness lead-in.
- Mac CPU, single stream, no concurrency or cold-start test; Fly shared CPU throttles sustained load.

## 7. Process and decisions (the reasoning trail)

1. Assessed what existed (three prior STT workers, none deployed; CPU engines benchmarked only on read/synthetic speech).
2. Wrote a spec and a phased plan: Phase 1 evaluates and decides; Phase 2 (deploy on Fly with the autoscaler, load test,
   speculative end-of-turn, gateway wiring, pricing) waits for the gate.
3. Built the harness with subagent-driven development: one implementer per task, an independent task review after each,
   fix rounds for every Important finding, a final whole-branch review and a consolidated fix wave. 119 tests.
4. Privacy stance: audio local and gitignored; per-call allowlist enforced in code for third parties; secrets read at run
   time and never printed; the gate fails closed; result files with transcripts are gitignored (only aggregates are
   committed).
5. Real-data findings changed the plan (Task 6): stale Twilio token -> use the calldesktech proxy; stereo recordings -> keep
   the far-end channel only; unreliable phone columns -> owner attestation plus the per-call allowlist.
6. Live API behaviour the fake servers could not show: Deepgram Flux sends no EndOfTurn within a short silence and closes
   the socket gracefully after `CloseStream` with close code 1005 ("no status received"), not 1000/1001; the adapter now
   treats 1005/1000/1001 after a successful CloseStream as a clean end of the flush, while every other close (abnormal,
   before CloseStream, error message, timeout) still raises so truncated text is never scored as complete.
7. Reference building was interactive: play an utterance, show the draft, owner corrects/excludes. Language detection
   (Whisper tiny) flagged non-English calls; it was unreliable for individual utterances, so the owner's ear decided.

## 8. Known issues and follow-ups

- Draft generation forces `language="en"`, so non-English speech is translated into English drafts. It should detect the
  language and flag it (`real_calls/segment.py`, `draft_refs`).
- Segment end padding is too short and clips trailing words on some utterances.
- 59 references are unlistened drafts; listening to them would tighten the numbers.
- Needs a larger set with real human speech; concurrent-load and Fly-silicon runs; tuning ElevenLabs/Deepgram endpointing
  for a fair latency comparison.
- Minor deferred review items are listed in the SDD ledger (`.superpowers/sdd/...`, local, gitignored).

## 9. Reproduce

Pre-flight and the exact commands (fetch, segment, review, benchmark matrix, report) are in the plan's controller steps
(Task 2 Step 7, Task 5 Step 6, Task 6 Step 5). In short, from `worker-stt-realtime/bench` with the venv from
`requirements-real.txt`: `python -m real_calls.fetch fetch --include-unmatched --caller-channel 0`,
`python -m real_calls.segment --raw ../data/<subset> --out ../data/real`, fill `data/real/review.tsv`,
`rtbench.py --engine <e> --set real --only-verified --out results/real__<e>__real.json` for each engine, then
`python report_real.py`. Call audio, manifests and result files with transcripts stay local and gitignored.
