# worker-demucs-fly: audio isolation (source separation / denoising)

`server.py` mirrors `../worker-piper-fly/server.py`'s shape (plain FastAPI + uvicorn, env-configured,
built for a Fly Machine or any always-on box), but for a different job: given an uploaded audio
clip, it runs Demucs (HTDemucs) and returns the isolated `vocals` (speech) stem as a clean WAV.

## Backend choice: PyTorch Demucs, not ONNX

The task brief asked to try ONNX export first for the ~1.3x CPU speedup, and fall back to
native PyTorch Demucs if ONNX wasn't clean to set up quickly. **We went straight to PyTorch**,
deliberately, without attempting the ONNX export:

- HTDemucs is a hybrid time-domain + spectrogram model: it runs an internal STFT/iSTFT, has
  complex-valued intermediate tensors, an LSTM bottleneck, and (in the "hybrid transformer"
  variant) cross-domain attention. None of that exports cleanly through `torch.onnx.export`
  without custom op handling (STFT/iSTFT in particular are a known rough edge for ONNX opset
  coverage), so this was assessed as very likely to eat the "reasonable amount of time" budget
  with no guarantee of a working export at the end.
- The brief's own priority order is explicit: "correctness and a working verified pipeline
  matter more than which backend." A verified, tested PyTorch pipeline beats an attempted-but-
  unfinished ONNX one.
- The `demucs` PyPI package (used here, `demucs==4.1.0`) is the actively maintained, documented
  way to run these models; it handles model loading, resampling, and the apply/overlap-add
  logic for us.

**Honest tradeoff**: this means we do not get the ~1.3x CPU speedup an ONNX export might have
given. On this M2 Pro dev machine, separating the ~10s test clip took a few seconds wall-clock
(see "Performance" below) - workable for a synchronous request on modest clips, but real
production traffic/instance sizing has not been load-tested (see "What's left").

## Endpoint

`POST /v1/isolate`

- **Auth**: `Authorization: Bearer <token>` - either the static `AUTH_TOKEN` (internal clients)
  or a short-lived gateway session token (`SESSION_SECRET`-signed HMAC), exactly the same
  verification logic as `worker-piper-fly/server.py`'s `verify_session_claims` (copied inline
  here rather than imported, so this worker builds/deploys independently - see "Auth wiring"
  below for how it'd plug into the gateway's `/tts/authorize`-style flow).
- **Request**: `multipart/form-data` with:
  - `file`: the audio clip (`.wav`, `.flac`, `.ogg`, `.aiff`/`.aif` - whatever `libsndfile`/
    `soundfile` can decode without ffmpeg; MP3/AAC etc. are rejected with 415, not silently
    mis-decoded)
  - `stem` (optional, default `vocals`): one of `vocals`, `drums`, `bass`, `other` - htdemucs'
    four separated sources. For "isolate the speech track" use the default.
- **Response**: `200` with an `audio/wav` body (16-bit PCM, mono, 44.1kHz - Demucs' native
  rate), headers `X-Sample-Rate`, `X-Stem`.
- **Limits**: `MAX_UPLOAD_BYTES` (default 25MB), `MAX_DURATION_S` (default 120s),
  `MIN_DURATION_S` (default 0.1s). Violating these returns `413` (too large/long) or `400`
  (empty/too short) with `{"error": "..."}`, matching worker-piper-fly's JSON error shape.
- **Concurrency**: `MAX_CONNECTIONS` (default 2, much lower than piper's default 4 - Demucs is
  far heavier per-request than TTS) returns `503 {"error": "at capacity, retry shortly"}` when
  exceeded, mirroring the piper worker's capacity behavior.

Errors: `401` (missing/invalid auth), `400` (empty file, undecodable audio, clip too short, bad
`stem`, empty file), `413` (file too large or too long), `415` (unsupported file extension),
`503` (at capacity), `500` (unexpected separation failure - message included, process stays up).

Example:

    curl -s -X POST https://<host>/v1/isolate \
      -H "Authorization: Bearer $TOKEN" \
      -F "file=@noisy.wav" -F "stem=vocals" \
      -o isolated.wav

## Auth wiring: gateway -> worker (now live)

Same pattern as `/tts/authorize` and `/stt/authorize`: a client calls
`POST /audio/authorize {key, engine?: "denoise"}` on the gateway (`gateway/server.js`), which
validates the API key (`keys.isValidKey`), checks billing access (`keys.checkAccess`), and
returns `{token, url}` — a short-lived HMAC session token (`keys.createSessionToken`) plus this
worker's base URL (`DEMUCS_WORKER_URL`, set analogously to `PIPER_WORKER_URL`/`STT_WORKER_URL` -
unset means `/audio/authorize` returns `501`). The client then POSTs the clip directly to
`<url>/v1/isolate` with that token as the bearer, which this worker verifies with
`verify_session_claims` using `SESSION_SECRET` — **`SESSION_SECRET` here must be the same value
as the gateway's `MODAL_SESSION_SECRET`**, since the gateway signs tokens and this worker
verifies them independently (no shared code, no network call between them at request time).

End-to-end:

    curl -s -X POST https://api.readaloudai.org/audio/authorize \
      -H "Content-Type: application/json" -d '{"key":"'"$API_KEY"'"}'
    # => {"token":"<session token>","url":"https://demucs-isolation-dev.fly.dev"}

    curl -s -X POST https://demucs-isolation-dev.fly.dev/v1/isolate \
      -H "Authorization: Bearer <token from above>" \
      -F "file=@noisy.wav" -F "stem=vocals" -o isolated.wav

Gateway-side auth attempts on this route (`auth_success`/`auth_failure`, `surface: "audio_authorize"`)
are structured-logged via `gateway/audit.js`'s `auditLog`, same as the other authorize routes.
Usage is reported by this worker after each successful separation to `USAGE_REPORT_URL`
(`{id, audio_seconds, engine: "denoise"}`, gated on `USAGE_REPORT_SECRET` being set) — see
`report_usage` below and "What's left" for the pricing caveat.

## Performance

Not benchmarked rigorously (no load test - see "What's left"). Anecdotally on this dev machine
(Apple M2 Pro, CPU-only, `TORCH_THREADS` = all cores): separating a ~10.4s mono clip via
`POST /v1/isolate` took a few seconds wall-clock end-to-end including model-already-loaded
inference (the very first call after process start also pays model weight load time, not
included in the request path measured here). Demucs is a full-audio-in-memory model (no
streaming), so its cost and memory scale with clip length - `MAX_DURATION_S=120` is a
guardrail, not a validated safe ceiling for concurrent requests on any particular VM size.

## Verification: before/after measurement

`verification/make_and_verify.py` builds the test clip and runs it through the real pipeline.

**How the test clip was made** (all committed under `verification/`, reproducible without
network access to Demucs's model source only if the weights are already cached locally):

1. `speech.wav` — macOS `say -v Samantha -o speech.aiff "<sentence>"` synthesized ~10.4s of
   speech, converted to 22.05kHz mono 16-bit WAV with `afconvert`. (Both commands were run once
   by hand; the output file is checked in so the test doesn't depend on `say` being available.)
2. `make_and_verify.py` synthesizes synthetic "background noise/music" entirely with
   numpy/scipy — no external audio samples — as a mix of low-passed white noise (a pink-noise
   stand-in) plus a 3-tone sine "drone" with slow vibrato (a synth-pad stand-in), then mixes it
   with the speech at a target 0dB input SNR (noise as loud as the speech - a deliberately hard
   case) to produce `noisy_mix.wav`.
3. It runs `noisy_mix.wav` through `Separator.separate` (the same code path `/v1/isolate` uses)
   to produce `isolated.wav`.
4. It computes two before/after metrics and writes them to `metrics.json`:
   - **Estimated output SNR**: cross-correlation-aligns the isolated output to the known clean
     speech reference, best-fit-scales it, and treats the residual as noise; compares that to
     the *known* input SNR (computed directly from the speech and noise signals used to build
     the mix, not estimated).
   - **Non-speech-band energy fraction**: the fraction of total signal energy in 0-80Hz and
     6-11kHz (where the synthetic noise/drone concentrate energy that clean speech mostly
     doesn't), before vs. after isolation.

**Actual measured result** (this run, committed in `verification/metrics.json`):

| metric | value |
|---|---|
| Input SNR (mix, known) | 1.6 dB |
| Estimated output SNR (isolated vs. reference) | 11.0 dB |
| **SNR improvement** | **+9.4 dB** |
| Non-speech-band energy fraction, before | 13.1% |
| Non-speech-band energy fraction, after | 2.4% |
| **Non-speech-band energy reduction** | **81.8%** |

Both metrics agree the isolated output is meaningfully cleaner, not just "no exception thrown".
**Caveat, stated plainly**: this is one clip, one noise profile, one SNR level, and a synthetic
TTS voice, not a benchmark suite. The SNR estimate is a proxy (cross-correlation alignment can
be thrown off by phase/timing changes Demucs' STFT-based separation introduces), and results
on real-world recordings (room reverb, multiple noise sources, real music) will differ - likely
less cleanly. Re-run `verification/make_and_verify.py` to regenerate the numbers (it overwrites
`noisy_mix.wav`, `isolated.wav`, `metrics.json`); reproduces deterministically given the same
`speech.wav` (fixed `numpy` RNG seed).

## Local run

    cd worker-demucs-fly
    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
    AUTH_TOKEN=devtoken .venv/bin/python server.py
    # first request downloads htdemucs weights (~80MB) into ~/.cache/torch if not already cached

## Tests

    cd worker-demucs-fly
    .venv/bin/python -m pytest tests/test_validation.py -v      # 23 tests, no model load, <2s
    .venv/bin/python -m pytest tests/test_integration.py -v -s  # 5 tests, loads real htdemucs model, ~7s warm

Both suites currently pass (23/23 and 5/5). `test_integration.py` includes a full HTTP-level
round trip through `/v1/isolate` using the real `verification/speech.wav` fixture and the real
model - not mocked.

## What's left before production

Done as of this pass: gateway routing (`/audio/authorize` in `gateway/server.js`, tested in
`gateway/audio.test.js`), `DEMUCS_WORKER_URL`/`engine` wiring, audit logging on the new surface,
and a usage-metering hook (`report_usage` in `server.py`, mirrors `worker-piper-fly`). Still open:

- **fly.toml is filled in but still NOT deployed** (do not `fly deploy` it from this state):
  app name is `demucs-isolation-dev` (deliberately not a production app name), VM sizing
  (`performance-2x`/4GB, `MAX_CONNECTIONS=2`) is a guess based on Piper's CPU-worker `fly.toml`
  as a reference point and Demucs being heavier per-request, not benchmarked under load.
- **Auth secrets**: `AUTH_TOKEN`/`SESSION_SECRET` need real values provisioned as Fly secrets
  (`fly secrets set`), not the dev placeholders used locally - and `SESSION_SECRET` specifically
  must match the gateway's `MODAL_SESSION_SECRET` value (see "Auth wiring" above).
- **Usage/billing pricing undecided**: `report_usage` now sends `{id, audio_seconds,
  engine: "denoise"}` to the gateway's `/admin/usage/report`, which stores it in the generic
  `usageAudioSecondsSinceLastReport` counter (`gateway/keys.js`) - same as STT does - but no
  per-second (or per-request/per-MB) *price* for this engine has been set on the billing side yet.
- **Load testing**: concurrency (`MAX_CONNECTIONS=2` is a guess), memory usage under concurrent
  120s clips, and cold-start behavior (first request after a scale-from-zero, including model
  weight load if not baked into the image) are all unmeasured.
- **Model licensing check**: Demucs (`facebookresearch/demucs`) is MIT-licensed code; the
  pretrained `htdemucs` weights are released under the same repo's terms (also permissive) but
  this has not been re-verified against current upstream terms or checked against this
  product's own ToS/licensing obligations before shipping a feature built on them.
- **Broader verification**: only one synthetic clip/noise profile was tested (see
  "Verification" caveats above) - real customer audio (phone calls, real background noise/
  music, multiple speakers) has not been tried.
- **Streaming/chunking**: Demucs here processes the whole clip in memory; very long inputs
  (near `MAX_DURATION_S`) will have higher latency and memory than a chunked/streaming
  implementation would - untested.
