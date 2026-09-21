# Dubbing MVP: text/audio translated narration

Text/audio only. Explicitly **not** video dubbing — no lip-sync, no video processing, no frame
handling anywhere in this module.

Pipeline: STT (existing worker-stt) → LLM translation with an explicit duration budget in the
prompt → Piper TTS in the target language → silence-aware retiming → bounded (±15%) time-stretch.

Callable two ways:
- CLI (unchanged): `python3 -m dubbing.pipeline --text ... --source-lang en --target-lang de_DE --duration 6.5 --out out.wav`
- **Async job API** (new): `POST /dubbing/jobs` → poll `GET /dubbing/jobs/{id}` → fetch
  `GET /dubbing/jobs/{id}/result`. See "Async job API" below for the full request/response shapes.

## Async job API

Implemented in `dubbing/job_server.py` (submit/poll/fetch logic in `dubbing/jobs.py`, no new
dependencies - stdlib `http.server` + a `ThreadPoolExecutor`). Run it with:

```
DUBBING_JOB_SECRET=<shared secret> python3 -m dubbing.job_server --host 127.0.0.1 --port 8090
```

### Where this API lives (gateway vs. a new Modal app)

Neither `gateway/server.js` nor a newly deployed Modal app. Reasoning:

- **Not `gateway/server.js`**: its `/tts/authorize` and `/stt/authorize` pattern is a *synchronous
  authorize-then-let-the-client-talk-directly-to-the-worker* call - it hands back a token and a
  URL and is done in one round trip. Dubbing is a multi-stage job (STT → translate → TTS →
  retime) that can run for seconds to tens of seconds; forcing that into an "authorize" call means
  either blocking the HTTP response for the whole pipeline (bad: ties up a Node request thread,
  no way to poll, no partial-failure visibility) or reimplementing a job queue in Node on top of a
  pattern that was never designed for one. Just as importantly: every actual step (OpenRouter
  translation in `translate.py`, the gateway/Piper HTTP call in `tts.py`, the worker-stt HTTP call
  in `stt.py`, the ffmpeg subprocess retiming in `retime.py`) is already real, working Python.
  None of it is Node-specific or depends on anything only reachable from the gateway process -
  they're outbound HTTP calls and a local `ffmpeg` subprocess, which Python already does natively.
  Reimplementing that pipeline in JavaScript would duplicate real logic for no benefit.
- **Not a newly deployed Modal app**: `voice-pipeline/intake.py`'s submit → `.spawn()` a
  background function → poll a status file on a Modal Volume pattern is the right *shape*
  (async job submit/poll/fetch) to mirror, and `dubbing/job_app.py` sketches exactly that mapping
  for later. But dubbing's actual work needs no GPU and no Modal-specific CPU container - it's the
  same class of work `intake.py`'s *API* function already does synchronously between spawning the
  real (GPU) training job: outbound HTTP + bookkeeping. Provisioning and deploying a new Modal app
  only to run outbound HTTP calls and shell out to ffmpeg would be paying for infrastructure the
  work doesn't need. `modal` and `fastapi` also aren't installed in this environment, so an actual
  Modal deployment couldn't be tested here, and per this task's explicit "prefer NOT deploying
  anything and just building/testing the code locally" instruction, nothing was deployed.
- **What was actually built**: `dubbing/jobs.py`'s `JobStore` (submit/poll/fetch state machine,
  background execution via a thread pool, status mirrored to disk so a restarted process can still
  answer a poll) plus `dubbing/job_server.py` (stdlib HTTP wrapper around it). This is a small,
  dependency-free, directly-testable async job runner with the *same submit → poll → fetch shape*
  as `intake.py`/`train_job.py`, without requiring a Modal deployment or a Node reimplementation
  of already-working Python. `dubbing/job_app.py` shows how this same `jobs.py` logic could be
  wrapped in a `modal.asgi_app()` mirroring `intake.py`'s structure if/when GPU-backed local Piper
  synthesis (rather than the existing gateway call) is wanted for this pipeline specifically.

### `POST /dubbing/jobs`

Auth: `Authorization: Bearer <DUBBING_JOB_SECRET>` (a single shared secret for whoever is allowed
to submit dubbing jobs - e.g. the product backend - matching `intake.py`'s `INTAKE_SECRET` model,
not gateway's per-customer API keys, since dubbing jobs are submitted server-to-server.)

Request body:
```json
{
  "text": "Thanks for calling, how can I help you today?",
  "source_lang": "en",
  "target_lang": "de_DE",
  "duration_s": 4.5
}
```
`audio_path` (a server-local WAV path) may be sent instead of `text` to exercise the real STT step
via `stt.SttClient` - see the STT caveat below; `duration_s` is optional for `text` mode (skips
duration-fit retiming if omitted). `target_lang` is validated against `voices/catalog.json`
(`catalog.validate_target_language`) before a job is even queued.

Response: `202 Accepted`
```json
{ "job_id": "dub-045219198d13", "status": "queued", "created_at": 1790031354.98, "updated_at": 1790031354.98 }
```

### `GET /dubbing/jobs/{job_id}` (poll)

Same auth. Response `200`:
```json
{
  "job_id": "dub-045219198d13",
  "status": "done",
  "created_at": 1790031354.98,
  "updated_at": 1790031356.74,
  "result": {
    "source_text": "Thanks for calling, how can I help you today?",
    "source_lang": "en",
    "target_lang": "de_DE",
    "translated_text": "Danke für Ihren Anruf, wie kann ich Ihnen helfen?",
    "voice_id": "de-de-mls-m",
    "tts_backend": "NullTTS",
    "fit": { "source_target_s": 4.5, "before_s": 3.27, "stretch_ratio": 0.85, "final_s": 4.26, "within_bound": true, "hit_target": false }
  },
  "result_ready": true
}
```
`status` is one of `queued | running | done | error`; on `error` the body carries `"error": "<type>: <message>"`
instead of `result`. `404` for an unknown job id, `401` if unauthorized/missing bearer token. Local
filesystem paths (`out_path`) are never included in this response.

### `GET /dubbing/jobs/{job_id}/result` (fetch)

Same auth. `200` with `Content-Type: audio/wav` and the dubbed WAV body once `status` is `done`;
`409` if the job isn't done yet; `410` if the result is no longer on disk; `404` for an unknown job.

### What's REAL vs. STUBBED through this API (verified live in this environment)

Ran a real end-to-end request through `dubbing/job_server.py` (not just unit tests) via
`DUBBING_JOB_SECRET=... python3 -m dubbing.job_server --port 8091`, then `POST /dubbing/jobs` with
`{"text": "Thanks for calling, how can I help you today?", "source_lang": "en", "target_lang":
"de_DE", "duration_s": 4.5}`, polled to `done`, and fetched `/result`:

```json
{
  "translated_text": "Danke für Ihren Anruf, wie kann ich Ihnen helfen?",
  "tts_backend": "NullTTS",
  "fit": {"stretch_ratio": 0.85, "final_s": 4.263946, "within_bound": true, "hit_target": false}
}
```
and `/result` returned a real 22.05kHz mono WAV.

- **Translation: REAL.** `translate.OpenRouterTranslator` via `OPENROUTER_API_KEY`, which is
  present and working in this environment (confirmed by the `env` check and the live call above).
- **Retiming/time-stretch: REAL.** `retime.py`'s ffmpeg-based silence detection + bounded
  `atempo` stretch, unchanged from the CLI path, exercised live above.
- **TTS: STILL STUBBED (`NullTTS`).** `TTS_GATEWAY_API_KEY` is not set in this environment (and no
  `ADMIN_SECRET` to mint one via `gateway/keys.js`'s `/admin/keys` either) - checked directly via
  `env`, not assumed. `tts.PiperGatewayTTS` is implemented and unchanged from the earlier MVP
  (mirrors `eval/synth.mjs`'s authorize → POST flow) but remains untested end-to-end for lack of
  a provisioned key. `NullTTS` logs loudly (`[NullTTS] STUB synth: ...`) on every call and is
  never silently mistaken for real speech.
- **STT: STILL STUBBED (bypassed via `text`).** `STT_BASE` / `STT_SECRET_FILE` are not set in this
  environment - checked directly via `env`, not assumed. `stt.SttClient` (used when a job is
  submitted with `audio_path` instead of `text`) is implemented, matches `worker-stt/test_client.py`'s
  mint scheme, and is wired into `pipeline.run()`/`jobs.py` unchanged, but was not exercised live
  here for the same reason as the original CLI MVP: no worker-stt credentials available.
- **To go fully real**: provision `TTS_GATEWAY_API_KEY` (a billing-enabled gateway key - mint via
  `POST /admin/keys` with `ADMIN_SECRET`, which is also not set here) for TTS, and `STT_BASE` +
  `STT_SECRET_FILE` for a real (non-production) `worker-stt` deployment for STT. No code changes
  are needed for either - `tts.get_default_backend()` and `pipeline.run()`'s `audio_path` branch
  already pick the real client automatically once those env vars are set.

### Test-key hygiene for this smoke test

No TTS/STT test key was minted for the live run above (both would require credentials -
`ADMIN_SECRET`/`TTS_GATEWAY_API_KEY`, `STT_SECRET_FILE` - that are not provisioned in this
environment, so there was nothing to mint or revoke). The only live external call made was to
OpenRouter, which uses the ambient `OPENROUTER_API_KEY` and mints no separate session credential.
If/when TTS or STT credentials are provisioned to test those paths for real, follow the same
mint → use → revoke pattern documented below under "Test-key hygiene" - do not leave a minted key
active afterward.

## Files

- `catalog.py` — enumerates target languages from `voices/catalog.json` (the single source of
  truth; nothing here hardcodes a language list) and picks a default voice per language.
- `translate.py` — LLM translation behind a `Translator` interface. `OpenRouterTranslator` is the
  only real backend wired up, `StubTranslator` fails loudly (not silently) when no key is present.
- `jobs.py` — stdlib async job store (submit/poll/fetch state machine + thread pool) that runs
  `pipeline.run()` in the background. No `modal`/`fastapi` dependency.
- `job_server.py` — the callable HTTP API: `POST /dubbing/jobs`, `GET /dubbing/jobs/{id}`,
  `GET /dubbing/jobs/{id}/result`, built on `jobs.py`. See "Async job API" above.
- `job_app.py` — a Modal-app sketch (not deployed) mirroring `voice-pipeline/intake.py`'s
  conventions, wrapping the same `jobs.py` logic in a `modal.asgi_app()`, for if/when GPU-backed
  local Piper synthesis is wanted for this pipeline instead of the existing gateway call.
- `tts.py` — Piper synthesis via this repo's existing gateway (`gateway/server.js` /tts/authorize,
  same call as `eval/synth.mjs`). `NullTTS` is an explicit stub for exercising the rest of the
  pipeline without live TTS credentials.
- `stt.py` — thin client for the existing `worker-stt` service (same mint scheme as
  `worker-stt/test_client.py`). `TextInput` bypasses STT with manually supplied source text, per
  this task's explicit scope-reduction option.
- `retime.py` — silence-aware retiming (ffmpeg `silencedetect`/`atrim`/`concat`, touches only
  inter-sentence pauses) + bounded time-stretch (ffmpeg `atempo`, capped at ±15%).
- `pipeline.py` — orchestrates all of the above; also a CLI (`python3 -m dubbing.pipeline ...`).

## What's real vs. stubbed in *this* environment

Checked directly (`env`, `df -h`, `which`), not assumed:

| Step | Status here | Why |
|---|---|---|
| Translation | **Real, tested live** | `OPENROUTER_API_KEY` is present. Used `openai/gpt-4o-mini` via OpenRouter. |
| STT | **Stubbed (bypassed via `TextInput`)** | No `STT_BASE` / `STT_SECRET_FILE` for the deployed `worker-stt` Modal app in this environment. The real client (`SttClient`) is implemented and matches `worker-stt/test_client.py`'s mint scheme exactly, but untested here for lack of credentials. |
| TTS | **Stubbed (`NullTTS`)** | No `TTS_GATEWAY_API_KEY` (and no `ADMIN_SECRET` to mint one via `gateway/keys.js`'s `/admin/keys`, which would be needed to get one at all). Real client (`PiperGatewayTTS`) mirrors `eval/synth.mjs`'s auth flow but is untested here. Also: local disk had ~140MB free at build time — not enough to install Piper + onnxruntime + a voice model (~60-80MB each) as a local fallback. |
| Retiming + stretch | **Real, tested live** | Pure ffmpeg (already installed), no external creds needed. |

**Time-stretch tool note** (task asked which of librosa/pyrubberband was used, and to say so if
neither): neither. ffmpeg's `atempo` filter was used instead — same category of tool
(pitch-preserving WSOLA-style tempo change), chosen because ffmpeg was already installed and disk
space (~140MB free) was too tight to safely `pip install librosa` (numpy+scipy+numba+librosa) or
`pyrubberband` (needs the separate `rubberband` CLI) without risking filling the disk further.
Swapping in librosa's `time_stretch` behind `retime.time_stretch()`'s existing signature is a
small change if disk space frees up.

**Translation API note**: an LLM *is* configured in this environment (`OPENROUTER_API_KEY`), so
translation was not stubbed — it's a real, tested step. No other LLM provider key
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) was found set.

## What was actually tested end to end

Ran `python3 -m dubbing.pipeline --text "..." --source-lang en --target-lang de_DE --duration 6.5
--out /tmp/out.wav`:

1. Real OpenRouter translation: `"Thanks for calling, how can I help you today? Your order should
   arrive within three to five business days."` → `"Danke für Ihren Anruf! Wie kann ich helfen?
   Ihre Bestellung kommt in drei bis fünf Tagen."` (budgeted to source duration).
2. `NullTTS` stub synth (clearly logged as non-speech placeholder audio) to exercise the audio
   pipeline mechanically.
3. Real silence-aware retiming + bounded stretch: 5.93s stub clip → 6.146s after silence retiming →
   6.496s after a 0.946x tempo pass, landing within 0.15s of the 6.5s target and within the ±15%
   stretch bound.
4. Also verified the stretch bound itself clamps correctly on an extreme case (0.4s clip targeting
   10s: `stretch_ratio` clamps to 0.85, pipeline honestly reports `hit_target: false` rather than
   silently overshooting the bound).
5. Verified `catalog.validate_target_language` rejects a language not in `voices/catalog.json`
   (e.g. `ja_JP`) with a clear error listing what's actually supported.

STT and real Piper TTS were **not** tested end to end here — see the table above for exactly why,
and `stt.py`/`tts.py` for the real (untested) clients that are ready to run once
`STT_BASE`/`STT_SECRET_FILE` and `TTS_GATEWAY_API_KEY` are provisioned.

## Test-key hygiene (when STT/TTS credentials do get provisioned)

Follow `eval/`'s mint → use → revoke pattern, not a standing key:
- TTS: mint via `gateway/keys.js`'s `/admin/keys` (needs `ADMIN_SECRET`, not this repo's problem to
  generate), use for the test run, then `DELETE /admin/keys` to revoke.
- STT: `stt.mint()` tokens are short-TTL (10 min) stateless HMAC session tokens, not stored
  server-side — "revoke" here means minting a short TTL and not reusing it, there's no server-side
  delete for these (unlike the gateway's `/admin/keys` DELETE flow).
- Never a production-named app, never a real end-user-facing key.

## Language scope

Target languages are read live from `voices/catalog.json`, not hardcoded. As of this build:
`de_DE, en_AU, en_CA, en_GB, en_IE, en_IN, en_NZ, en_US, en_ZA, es_ES, es_MX, fr_FR, it_IT, nl_BE,
nl_NL, pl_PL, pt_BR, ru_RU` (run `python3 dubbing/catalog.py` to regenerate). Kokoro (GPU, English
only) is not used as a target-language engine here since it can't cover any non-English target;
Piper is the only engine in this repo that does.
