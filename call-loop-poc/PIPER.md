# Piper TTS backend (`ttsBackend: 'piper'`)

Owned, self-hosted TTS for a cheap tier. The engine talks directly to the Piper service (Fly app `piper-tts-sjc`, source
`realtime-tts/worker-piper-fly/server.py`) over its WebSocket `/tts` protocol. Code: `piperTts.js` (client, warm sockets, stream cap),
`piperVoices.js` (voice ids and validation), `_speakPiper` in `server.js`.

## Why direct, not via the gateway

The `kokoro` path uses `TTS_GATEWAY_WS_URL`, and the gateway's plain `/tts` route proxies to the Kokoro worker. Piper in the gateway is
opt-in through the `/tts/authorize` flow (`{engine:"piper"}`, `PIPER_WORKER_URL`), which returns a session token and a different URL; it
is not a voice id on the existing socket, and the call engine does not use that flow. A voice id such as `custom:en-us-ljspeech` is
resolved by the Piper worker itself. So the engine connects to the worker directly with its static bearer token.

## How it works

- Per sentence, via the existing `SentenceChunker` and `_speakHttpTts` (same ordering, barge-in and `[latency]` plumbing as ElevenLabs).
- Telephony calls request `mulaw_8000` (no resampling, same `mulaw8k` frames as ElevenLabs); browser calls request `pcm_24000`.
- One synth stream per call at a time: sentences queue behind each other on the session's warm socket (playback is serial anyway).
  A socket is opened early (prewarm at context time) and reused; idle sockets are closed after `PIPER_IDLE_MS`.
- Barge-in aborts the request: `{"type":"stop"}` is sent and the socket is closed (not reused).
- Any Piper failure before audio was sent (connect error or timeout, server `error`, no audio in time, socket closed, stream cap
  reached) logs `[tts] piper failed, falling back (<code>: <reason>)` and speaks that sentence with the tenant's ElevenLabs voice
  (`ELEVENLABS_API_KEY`, the session's `elevenVoiceId`). Later sentences try Piper again. If Piper fails after audio already went out,
  the sentence is not repeated. Without `ELEVENLABS_API_KEY` the sentence is dropped (`[tts] piper fallback unavailable`).
- Non-English agents: `_applyLanguage` moves a Piper call to the language's backend (needs ElevenLabs), as it does for kokoro.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `PIPER_TTS_TOKEN` | none | Bearer token (the worker's `AUTH_TOKEN`). Required: without it `ttsBackend: 'piper'` is treated as unconfigured and the call keeps the process default backend (same as other backends with missing keys). Set with `fly secrets set`, never commit. |
| `PIPER_TTS_URL` | `wss://piper-tts-sjc.fly.dev/tts` (unverified hostname) | Worker URL. A Fly private-network URL is lower latency if the apps share an org. |
| `PIPER_MAX_STREAMS` | 8 | Open sockets (busy plus warm idle) per engine instance. Overflow falls back to ElevenLabs. |
| `PIPER_CONNECT_TIMEOUT_MS` | 1500 | Connect timeout. |
| `PIPER_FIRST_AUDIO_TIMEOUT_MS` | 4000 | Request to first audio frame. |
| `PIPER_REQUEST_TIMEOUT_MS` | 15000 | Request to `done`. |
| `PIPER_IDLE_MS` | 20000 | Warm idle socket lifetime. |
| `PIPER_VOICE` | `custom:en-us-ljspeech` | Process default voice (validated). |
| `PIPER_COST_PER_1K_CHARS` | 0 | Owned-infra TTS cost per 1000 characters for `[cost]` lines. No vendor rate or margin is in code. |

Voices (`piperVoices.js`): `default`, `custom:en-us-john`, `custom:en-us-ljspeech` (backend default). Only `custom:en-us-ljspeech` is the
licence-cleared voice (see internal-docs `cost-lab/tts-gates/LICENCE.md`); `default` must not be sold, and `john` is not counsel-cleared.
The licence answer is still outstanding, so treat this as engineering readiness, not a clearance to sell.

## Enable for ONE test agent version

1. Deploy is a separate decision (not done here). On the engine app set the secret `PIPER_TTS_TOKEN` (and `PIPER_TTS_URL` if not default).
2. In `calldesk_agent_versions` set, for that single version row only: `tts_backend = 'piper'` and `voice_id = 'custom:en-us-ljspeech'`
   (or leave `voice_id` null for the default). Check first that the `tts_backend` column has no CHECK constraint or enum rejecting
   `piper` (not verified: no DB access here). Everything else stays on its current backend.
3. Alternative without touching the DB: `POST /test-tts-override {number, ttsBackend:'piper'}` (the allowlist now includes `piper`).
4. Expected log lines on a call: `[call-loop] context set - ... ttsBackend: piper`, `[call-loop] version voice applied (piper)` (only when
   a voice_id was set), per sentence `[tts] piper turn N firstAudioMs=.. frames=.. socket=warm|new`, and per turn
   `[latency] [call CA...] turn N {"source":"piper","ttsBackend":"piper","llmTtfbMs":..,"ttsLegMs":..,"responseMs":..}`.
   A fallback shows `[tts] piper failed, falling back (...)` and `"source":"piper-fallback-elevenlabs"` in `[latency]`.

## Live test (one call, Telnyx test number)

Dial the test number whose agent version is the Piper one, say 4 to 5 short things, hang up. Then read the logs
(`fly logs -a call-loop-poc | grep -E "\[latency\]|\[tts\] piper"`). `ttsLegMs` is LLM first token to first audio sent (for Piper this is
the whole first sentence synthesized plus one hop, since sentences are buffered per request like the other HTTP backends); compare with
the same agent on ElevenLabs. `[tts] piper ... firstAudioMs` is the Piper-only request to first frame number. Pass criteria: no
`falling back` lines, ttsLegMs in the low hundreds of ms for ~45-char sentences, audio intelligible on the phone.

## Not verified

- Never run against the real service or a real call; all behaviour is tested only against a local mock built from the server source.
- Default URL hostname and that engine-to-worker latency from `sjc` matches the 200 ms p50 measured inside Fly.
- `tts_backend` DB constraint; whether the token is already set on the engine app; ljspeech presence on production (LICENCE.md says
  the publish script lists it, not independently verified); audio quality of ljspeech over mulaw (gates were run on john).
- Mid-sentence cancel and `at capacity` handling are from reading `server.py`, not observed on the live service.
- Cost lines count characters at the Piper rate even for sentences that fell back to ElevenLabs.

## Concurrency caveat

Measured on one machine (2 threads): first audio p50/p95 about 200/271 ms with 1 stream, 410/542 with 2 simultaneous synths, 820/1041
with 4. More than about 2 simultaneous synths per machine pushes p95 over 450 ms. Each active call needs at most one synth at a time,
and only while a sentence is being generated (a fraction of the call), but bursts coincide. The worker's `MAX_CONNECTIONS` (12) counts
open sockets, including warm idle ones, so keep `PIPER_MAX_STREAMS` times the number of engine instances at or below the fleet's
total. To scale: add Piper machines (the app has 2 `performance-2x`; capacity is roughly 2 concurrent synths per machine for a 450 ms p95),
or move to `performance-4x` (untested), rather than raising `PIPER_MAX_STREAMS`, which only moves queueing into the worker. Lower
`PIPER_MAX_STREAMS` to send overflow to ElevenLabs earlier if latency matters more than cost.
