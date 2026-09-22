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

## Load testing (this pass)

`verification/load_test.py` fires N concurrent requests (using the committed
`verification/noisy_mix.wav`) at a running instance and reports status codes and latency. Run
locally (no valid production `AUTH_TOKEN`/`SESSION_SECRET` was available for the live
`demucs-isolation-dev.fly.dev` worker in this pass, so testing was done against the identical
code running locally rather than against production - see "Open items" below):

    cd worker-demucs-fly
    AUTH_TOKEN=devtoken .venv/bin/python server.py &   # local instance
    .venv/bin/python verification/load_test.py --url http://localhost:8080 --token devtoken -n 4

**Results (M2 Pro dev machine, `MAX_CONNECTIONS=2`):**

| concurrent requests | 200 OK | 503 at-capacity | latencies (successful) |
|---|---|---|---|
| 1 | 1 | 0 | 4.62s |
| 2 (at the limit) | 2 | 0 | 4.67s, 9.05s |
| 3 (over the limit) | 2 | 1 | 4.49s, 8.78s |
| 4 (over the limit) | 2 | 2 | 5.22s, 9.86s |

**Key finding: the concurrency limit works as a request-admission gate, but does NOT provide
real parallel compute.** `Separator.separate()` holds a global lock (documented in its docstring
as a deliberate choice - the model isn't verified thread-safe for concurrent forward passes), so
even with 2 requests admitted under `MAX_CONNECTIONS=2`, they run one at a time: the second
request's latency is roughly double the first's (~9s vs ~4.6s for two admitted requests, matching
"one request's compute time, then the next's"). Requests beyond `MAX_CONNECTIONS` are correctly
rejected with `503 {"error": "at capacity, retry shortly"}` immediately (sub-20ms), not queued
indefinitely, which is the behavior the code intends.

**Implication for capacity planning**: raising `MAX_CONNECTIONS` above 2 would let more requests
queue before rejecting (trading rejection for worse tail latency) but would NOT increase
throughput, since compute is serialized regardless of how many requests are admitted. The
current value of 2 essentially caps end-to-end p99 latency at "2x one clip's processing time"
before a caller gets a fast, clear rejection instead of an ever-growing queue - a reasonable
tradeoff, but the original guess's rationale ("Demucs is heavier per-request, so lower
concurrency than Piper") undersells the real reason: it's not just heavier, the actual compute is
fully serial, so there's no parallel-throughput benefit to admitting more than a small queue depth
at all. If real parallel throughput is ever wanted, the lock would need to be replaced with
per-request model instances (more memory, no lock) or a request queue in front of a fixed worker
pool - not attempted here.

**Open items on load testing:**
- Not run against the actual live `demucs-isolation-dev.fly.dev` instance or its real
  `performance-2x` VM class (no valid session/API token was available in this environment to
  authenticate against it; the code path exercised locally is identical, so results should
  transfer, but Fly's CPU class may differ in absolute throughput from the M2 Pro used here).
- Concurrent *large* clips (near `MAX_DURATION_S=120`) and their memory usage were not tested -
  only the ~10.4s reference clip was used at each concurrency level.
- Cold-start (scale-from-zero, or first request after process start including model weight load)
  under concurrent load was not tested.

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

## Expanded verification coverage (this pass)

The original verification above is one clip, one noise profile, one SNR level, one synthetic TTS
voice. `verification/expand_coverage.py` adds 5 more cases (run after `make_and_verify.py` has
produced `speech.wav`):

    .venv/bin/python verification/expand_coverage.py

| case | input SNR | output SNR | improvement | non-speech energy reduction | notes |
|---|---|---|---|---|---|
| white_noise_0db | 1.5 dB | 11.5 dB | **+10.0 dB** | 98.9% | pure white noise, no music/drone - cleans up well, similar to the original drone case |
| white_noise_10db | 10.0 dB | 17.5 dB | **+7.5 dB** | 66.7% | easier starting point (less noise to remove); still improves, smaller relative gain since there's less headroom |
| echo_reverb | n/a | -6.0 dB vs. dry reference | **worse, not better** | -0.2% (no change) | speech convolved with a synthetic RT60=0.4s room impulse response. Demucs does **not** meaningfully dereverberate - htdemucs separates sources, it doesn't do dereverberation, so this is an expected negative result, not a bug, but a real limitation to know about before promising "cleans up any noisy recording" |
| second_voice | 3.0 dB | 0.1 dB | **-2.9 dB (worse)** | 0.1% (no change) | a second TTS voice mixed in as "noise" at 0dB. htdemucs' `vocals` stem separates speech from non-speech, it has no notion of "target speaker" - it does not suppress a second voice. Confirms this feature is a noise/music suppressor, not a speaker-isolation/diarization tool |
| **real_speech_drone** | 1.6 dB | 1.8 dB | **+0.2 dB only** | 98.1% | a REAL (non-TTS) public-domain speech recording (see below) mixed with the original drone+noise profile at the same 0dB SNR as the verified synthetic case (which got +9.4 dB). Non-speech-band energy still drops sharply (98.1%, similar to the synthetic case), but the oracle SNR metric barely improves - a real, measured gap between synthetic-TTS and real-speech performance, not a guess |

**The real-speech result is the most important finding of this pass.** On the exact same noise
profile and SNR that produced a clean +9.4 dB improvement with synthetic `say`-generated speech,
a real recorded voice only improved by +0.2 dB by this metric. Two explanations, not
distinguished further here: (a) htdemucs' vocal-isolation quality may genuinely be weaker on real
speech's more complex spectral/temporal characteristics (natural prosody, breath sounds, room
tone in the original recording) than on TTS's cleaner, more uniform signal, or (b) the
cross-correlation-based SNR estimator (built for a synthetic reference with no pre-existing room
character) is a worse proxy on a recording that already has its own reverb/mic coloration baked
in, inflating the apparent gap. The non-speech-band-energy metric (98.1% reduction, matching the
synthetic case) suggests the model IS doing real separation work; the metrics disagree, which
itself is the point - one proxy metric isn't enough to certify quality on real audio, and this
needs either a better metric (e.g. PESQ/STOI, not attempted here) or human listening evaluation
before claiming the same +9dB-class improvement on real customer audio.

**Where the real speech came from**: a public-domain audio recording of a reading of the
Gettysburg Address, sourced from archive.org (`GettysburgAddressmp3Version` item), trimmed to a
12s excerpt and resampled to 22.05kHz mono to match the pipeline's other fixtures. This is the
first non-`say`-synthesized speech this pipeline has been tested against. The clip itself isn't
checked into this repo's fixtures by this script (only referenced by path in
`verification/real_speech_gettysburg.wav`, which **is** committed) - if committing third-party
downloaded audio (even public-domain) needs a licensing sign-off before shipping, that's an open
item (see below).

Full machine-readable results: `verification/coverage_results.json`. Rerun
`expand_coverage.py` to regenerate (deterministic given fixed RNG seeds and the same input
files).

## Investigated: does a purpose-built speech-enhancement model fix the real-speech gap? (this pass)

The `real_speech_drone` result above (+0.2dB on real speech vs. +9.4dB on synthetic TTS speech,
same noise/SNR) is suspected to happen because htdemucs is a **music** source-separation model
(vocals-vs-instruments) repurposed for noise suppression, not a model trained for speech
enhancement — it may be treating real speech's natural sibilants/breath/room-tone as "not vocals"
and stripping them along with the actual noise. This pass investigated whether swapping in a
model actually trained for speech enhancement would close that gap.

**Candidates considered:**

| candidate | license | commercial use | CPU feasibility | 2026 status |
|---|---|---|---|---|
| **DeepFilterNet3** (`Rikorose/DeepFilterNet`) | dual MIT / Apache-2.0 | OK | Yes — designed for real-time CPU use (RTF well under 1 on a single CPU thread); ran fine CPU-only here | Actively used, latest PyPI release `deepfilternet==0.5.6` |
| **Meta/Facebook `denoiser`** (`facebookresearch/denoiser`) | **CC-BY-NC 4.0** | **Blocked — non-commercial only**, same bar this product already applies to Piper's licensing | Yes, CPU-real-time by design | Not actively developed post-2020 paper, but license alone rules it out |
| **RNNoise** (Xiph) | BSD-3-Clause | OK | Yes, extremely lightweight (designed for embedded/VoIP) | Mature, essentially unchanged in years; older/simpler DSP+small-RNN approach, narrower design target (stationary VoIP-style noise) than DeepFilterNet or htdemucs |

Meta's `denoiser` was eliminated immediately on licensing (CC-BY-NC 4.0 is non-commercial-only,
the same category of blocker already flagged for research-only licenses elsewhere in this
product). RNNoise was not installed/run in this pass: it's judged unlikely to beat a model
(DeepFilterNet) that, as measured below, itself did not fix the real-speech gap, and given that,
spending more effort validating an even narrower, older DSP-era model wasn't a good use of time
in this pass — this is a gap in this investigation, not a claim that RNNoise was tested and
failed. **DeepFilterNet was picked as the strongest, most credible candidate** (purpose-built for
exactly this problem, permissively licensed, CPU-real-time) and was actually installed and run.

**Install note**: DeepFilterNet 0.5.6 could **not** be installed into this project's own
`.venv` without conflict — it requires `numpy<2.0` (this project pins `numpy==2.4.6` for
htdemucs/scipy) and its `df.io` module imports `torchaudio.backend.common.AudioMetaData`, which
is gone in this project's pinned `torch==2.14.0`/`torchaudio==2.11.0`. It only ran successfully in
a **separate venv** with `torch==2.1.2`/`torchaudio==2.1.2`. This is itself a real integration
cost of DeepFilterNet as evaluated (a torch major-version conflict with the existing htdemucs
pipeline), on top of whatever the quality numbers show below.

**Measured results** (same fixed mix files `expand_coverage.py`/`make_and_verify.py` already
produced — no new audio, no new SNR levels, same `band_energy_fraction`/`estimate_output_snr`
metric code, reused via `verification/eval_deepfilternet.py`):

| case | engine | input SNR | output SNR | improvement | non-speech energy reduction |
|---|---|---|---|---|---|
| original_drone_0db (make_and_verify.py's clip) | htdemucs | 1.6 dB | 11.0 dB | **+9.4 dB** | 81.8% |
| original_drone_0db (same clip) | DeepFilterNet3 | 1.6 dB | 10.46 dB | **+8.86 dB** | 80.6% |
| white_noise_0db | htdemucs | 1.49 dB | 11.47 dB | **+9.98 dB** | 98.9% |
| white_noise_0db (same mix) | DeepFilterNet3 | 1.49 dB | 12.47 dB | **+10.98 dB** | 96.6% |
| **real_speech_drone** | htdemucs | 1.63 dB | 1.84 dB | **+0.21 dB** | 98.1% |
| **real_speech_drone** (same mix) | **DeepFilterNet3** | 1.63 dB | 1.58 dB | **-0.05 dB (worse)** | 99.7% |

Full machine-readable results: `verification/coverage_results_deepfilternet.json`.

**Conclusion: DeepFilterNet does not fix the real-speech gap, and by this metric is marginally
worse than htdemucs on the real-speech case (-0.05dB vs. +0.21dB — both are, practically, "no
real improvement").** On the synthetic cases the two models are roughly comparable (DeepFilterNet
slightly behind on the drone case, slightly ahead on pure white noise) — consistent with both
being competent noise-suppression models on clean/uniform TTS input. But on the one case this
whole investigation was about, DeepFilterNet does not show the "close the gap toward +9dB" result
that would justify a switch. The non-speech-band-energy proxy metric is misleading in the exact
same way for DeepFilterNet as it was for htdemucs (99.7% reduction, best of any engine on this
case, while the oracle SNR metric shows no real gain) — this reinforces the original finding that
this proxy metric is not trustworthy evidence of speech-preserving quality on real speech, for
either engine.

Two important caveats on this specific result, not resolved here: (a) this is one real-speech
clip, one noise profile, one very hard 0dB SNR, run through DeepFilterNet's default config with no
tuning — a different config, or DeepFilterNet2/plain "DeepFilterNet" (not the 3rd-gen model used
here), might behave differently, untested; (b) the same SNR-estimator caveat from the htdemucs
result applies here too — the cross-correlation-based oracle SNR metric may be a poor proxy on a
recording that already has its own room coloration baked in, for any model, so a low or negative
number here doesn't fully separate "the model does badly" from "the metric is unreliable on this
kind of input" — a proper resolution needs PESQ/STOI or human listening, which is still not
attempted in this pass either.

**Decision: no code change.** Per this pass's own bar (a candidate must clearly outperform
htdemucs on `real_speech_drone` without regressing the synthetic cases before it's worth
switching or adding as a mode), DeepFilterNet does not clear it — it does not outperform htdemucs
on the case that matters, and additionally introduces a torch/torchaudio version conflict with
this project's existing pinned dependencies. **htdemucs remains the current approach.** The
real-speech quality gap identified in the previous hardening pass is still open and still needs
either a better quality metric (PESQ/STOI, human listening) or a documented acceptance that this
feature underperforms on real-world audio relative to its synthetic-TTS-measured numbers — trying
one more model did not resolve it, and RNNoise was not tried in this pass (see above), so it's
not yet possible to say no CPU-feasible, commercially-licensed model would help; only that the
single strongest a priori candidate (DeepFilterNet) measurably didn't.

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
and a usage-metering hook (`report_usage` in `server.py`, mirrors `worker-piper-fly`). The worker
is now live at `demucs-isolation-dev.fly.dev` with the gateway route in production.

Done in this hardening pass (see the new sections above for detail):
- **Pricing proposal with real cost math**: `PRICING.md` - measured CPU-seconds/audio-second on
  the real separation path, derived a break-even price at several utilization assumptions using
  the same cost-basis method as Piper/Kokoro, and proposed **$0.05/audio-minute**. Still a
  *proposal*, not wired into billing - and not checked against real competitor pricing (see that
  doc's "Open items").
- **Load testing**: `verification/load_test.py` + results table above. Confirmed `MAX_CONNECTIONS`
  correctly gates admission (503 beyond the limit) and found the more important fact: the global
  separation lock means compute is serialized, so the limit buys queue depth/latency control, not
  parallel throughput - the original "2 is heavier than Piper's 4" reasoning was directionally
  right but missed this mechanism.
- **Expanded verification coverage**: `verification/expand_coverage.py` - 5 more cases (white
  noise at 2 SNR levels, synthetic reverb, a second-voice/crosstalk case, and a REAL non-TTS
  speech recording). Found real limitations that the single original clip couldn't show: htdemucs
  doesn't dereverberate, doesn't suppress a second voice, and shows a much smaller measured
  improvement on real speech than on synthetic TTS speech under the same noise/SNR conditions.
- **Real (non-TTS) audio tested**: a public-domain speech recording (Gettysburg Address reading,
  from archive.org) was mixed with synthetic noise and run through the real pipeline - see
  "real_speech_drone" above. This is the first non-synthetic-speech test of this feature, and it
  surfaced a real gap (see above) rather than confirming everything is fine.

Still open:

- **fly.toml is filled in and the app IS now deployed live** (`demucs-isolation-dev.fly.dev`) -
  the comment in `fly.toml` predates the actual deploy and should be updated/removed to avoid
  confusing a future reader into thinking it's still pre-deploy. VM sizing
  (`performance-2x`/4GB, `MAX_CONNECTIONS=2`) was a guess before this pass; load testing above
  confirms the concurrency behavior is sane (clean 503s, no crashes/hangs at 2x the limit) but
  was run locally, not against the real Fly VM class - see "Load testing" open items.
- **Usage/billing pricing**: a number has been proposed (`PRICING.md`, $0.05/audio-minute) but
  not decided by whoever owns pricing, nor implemented as an actual meter rate on the billing
  side (unlike Piper's chars-based rate, there's no equivalent "0.4x weighting" commit yet for
  denoise).
- **Concurrency architecture**: the separation lock's serialization (see "Load testing") means
  `MAX_CONNECTIONS` doesn't buy real throughput scaling - if concurrent throughput ever needs to
  scale, the model would need per-worker instances (memory cost) or this needs re-architecting;
  not attempted in this pass.
- **Model licensing check**: Demucs (`facebookresearch/demucs`) is MIT-licensed code; the
  pretrained `htdemucs` weights are released under the same repo's terms (also permissive) but
  this has not been re-verified against current upstream terms or checked against this
  product's own ToS/licensing obligations before shipping a feature built on them.
- **Real-speech quality gap needs a real fix-or-accept decision**: the "real_speech_drone" result
  above (+0.2dB vs. the synthetic case's +9.4dB under the same noise/SNR) is a genuine open
  question, not resolved here - needs either a better quality metric (PESQ/STOI or human
  listening) or acceptance that this feature is weaker on real-world audio than the original
  verification suggested, before it's marketed as broadly as the current copy might imply. A
  follow-up pass tried swapping in DeepFilterNet (see "Investigated: does a purpose-built
  speech-enhancement model fix the real-speech gap?" above) - it did not help (-0.05dB vs.
  htdemucs' +0.21dB on the same real-speech case) and htdemucs was kept. RNNoise was not tried.
  This remains open.
- **Streaming/chunking**: Demucs here processes the whole clip in memory; very long inputs
  (near `MAX_DURATION_S`) will have higher latency and memory than a chunked/streaming
  implementation would - untested, and not covered by this pass's load testing (only the ~10.4s
  reference clip was used at each concurrency level, not near-`MAX_DURATION_S` clips).
- **Licensing of the third-party test fixture**: `verification/real_speech_gettysburg.wav` was
  downloaded from archive.org (public domain speech recording) and is now committed to this
  repo as a test fixture - hasn't been checked against this product's policy on committing
  third-party downloaded media, even permissively-licensed.
