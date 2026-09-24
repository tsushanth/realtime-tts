# Realtime STT: real-call evaluation and engine decision (Phase 1)

Date: 2026-09-24. Repo: realtime-tts. Branch: `stt-realtime-eval`.

## Problem

We want a cheap, low-latency, CPU-bound streaming English STT service, the way Piper is for TTS. Competitors:
ElevenLabs Scribe v2 Realtime ($0.39/audio-hour, claims ~150 ms partials) and Deepgram Flux ($0.39/hr; Nova-3
streaming $0.29/hr; Flux is what `call-loop-poc` uses today).

What exists (`worker-stt-realtime/`): sherpa-onnx zipformer (server default), NeMo FastConformer 480 ms, Moonshine,
faster-whisper re-decode, a FastAPI WebSocket server, and `bench/rtbench.py`. Everything was measured on read
audiobook speech (LibriTTS) and macOS-`say` synthetic call turns, on a Modal CPU container. Nothing is deployed
(no Fly/Modal STT app; gateway returns 501 for `mode:"realtime"` because `STT_REALTIME_WORKER_URL` is unset).

What we do not know: accuracy on real phone calls, and how we compare with Deepgram/ElevenLabs on the same audio.
Engine choice (zipformer vs NeMo 480) changes what Phase 2 deploys, so it is decided by data first.

## Scope

Phase 1 (this spec and its plan): build the real-call evaluation, run it, and produce an engine decision.
Phase 2 (a separate plan, written after the decision gate): deploy the chosen engine on Fly with the
fly-autoscaler pattern (`worker-piper-fly/AUTOSCALER.md`), load-test on real Fly silicon, speculative end-of-turn,
gateway realtime wiring, pricing for `audio_seconds` engine `stt-realtime`.

## Data

`calldesktech` Supabase table `calldesk_call_logs`: 213 rows with `recording_url`; the 60 most recent are
`voice_engine='poc'` (call-loop-poc, Twilio recordings, 8 kHz telephone audio, median 124 s, 2026-09-20/21).
Relevant columns: `id, voice_engine, direction, caller_phone, to_number, recording_url, created_at, duration_seconds`.
Recordings need Twilio basic auth (`TWILIO_ACCOUNT_SID`/`TWILIO_AUTH_TOKEN` in `call-loop-poc/.env`).

## Privacy constraints (binding)

1. Audio stays on the local machine, in the gitignored `worker-stt-realtime/data/`. Never committed, never uploaded
   by tooling except as in (3).
2. A call is "own" only if its counterparty number is in `data/own_numbers.txt`, which the user writes after seeing
   masked-number counts. Direction-aware: inbound -> `caller_phone`, outbound -> `to_number`.
3. Third-party engines (Deepgram, ElevenLabs) run only on calls listed in `data/confirmed_calls.txt` (user-confirmed
   own test calls). The harness refuses otherwise (`PermissionError`), enforced in code, not by convention.
4. Secrets are read from env files at run time and never printed or logged.

## Ground truth

No human-verified transcripts exist. Calls are cut into utterances (silero VAD, 1.5-15 s). Each utterance gets a draft
reference from faster-whisper large-v3 (approximate). The user hand-corrects about 15 calls (~30 min of audio),
prioritising names and numbers, and lists `keyterms` (names, digit strings) per utterance. Headline metrics use
`verified: true` utterances only; unverified utterances are reported separately as approximate.

## Candidates

`zip-en-int8`, `nemo-480` (ours, local CPU); `dg-flux` (Deepgram `flux-general-en`, `/v2/listen`, linear16 16 kHz);
`el-scribe` (ElevenLabs `scribe_v2_realtime`, `commit_strategy=vad`, `pcm_16000`).

## Metrics (per candidate, same paced replay as `rtbench.py`)

WER on verified utterances; keyterm recall (names/numbers); first-partial latency; end-of-speech -> final latency
(`commit` and native endpointing); mid-utterance cut-offs; CPU-seconds per audio-second and RSS (ours); price per hour.

## Decision gate (computed by `report_real.py`, applied by a human)

Ship-worthy if the best CPU candidate has: verified WER <= 1.5x the best cloud candidate's WER; keyterm recall within
10 percentage points of the best cloud candidate; `commit` final latency median <= 150 ms; cpu/audio-s <= 0.15.
Otherwise Phase 2 becomes an accuracy investigation (bigger/finetuned model, hotword biasing) before any deploy.

## Non-goals

No deploy, no gateway change, no pricing decision, no non-English, no changes to `server.py` in Phase 1.
Numbers from a Mac CPU are a proxy; Fly numbers come in Phase 2.
