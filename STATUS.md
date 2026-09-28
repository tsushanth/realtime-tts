# Status (as of 2026-09-27)

Snapshot of where things stand across the two active efforts, for anyone
(human or agent) picking this up cold. See `docs/superpowers/specs/` and
`docs/superpowers/plans/` for the full design/plan documents this
summarizes.

## Direction

This product (realtime-tts / ReadAloudAI) competes with ElevenLabs on
latency and price for self-hosted voice infra — Piper (CPU, ~251ms,
$4/1M chars) and Kokoro (GPU, higher quality/cost) as the core engines.
Today's work had two threads: (1) closing remaining ElevenLabs feature
gaps (audio isolation, dubbing, voice design, speech-to-speech, phone
agents, concurrency, compliance — mostly built as MVPs, several hardened
with real bugs found and fixed), and (2) migrating the ReadAloud AI
Android app off its old Chatterbox/Hetzner backend onto this platform's
real API, plus building a research harness to make future model/quality
investigations (like the ones done ad hoc today) repeatable instead of
one-off scripts.

The honest overall read: TTS latency and price are essentially at parity
with ElevenLabs; quality is close but not quite; the real remaining gaps
are voice/language breadth and the newer verticals, most of which now
have real (if unshipped) MVPs. Music generation was explicitly decided
against — wrong buyer, wrong competency, breaks the no-GPU-serving
economics.

## Voice research harness — merged to main

**What it is:** an on-demand CLI (`python3 -m harness.run_cycle --recipe
tts-core --budget 30`) that discovers untried model-size candidates,
trains them on Modal GPU within a hard budget cap, evaluates them, and
writes a report — infrastructure for repeatable research, not a one-off
investigation each time. Spec: `docs/superpowers/specs/2026-09-22-voice-research-harness-design.md`.
Plan: `docs/superpowers/plans/2026-09-22-voice-research-harness.md`.

**State: merged to `main` at `8c4e331`.** Branch `harness-implementation`
was a no-ff merge (plus one pre-merge atomic-write fix for
`record_tried()`). 34/34 tests passing.

**Still open:**
- `TTSCoreRecipe.evaluate()`'s `_run_eval_pipeline()` is `NotImplementedError`
  by design — training and reporting work end-to-end, but a real
  `--budget 30` run would train real candidates and get real cost numbers,
  just no WER/MOS metrics yet. Three specific blockers are documented in
  that method's own docstring (an owner.json validation gap in
  `worker-piper-fly/server.py`, no single-voice scoping in `eval/synth.mjs`,
  a same-run-reference requirement in `eval/score.py`'s MOS calc).
- **No real (non-`--budget 0`) cycle has been run.** The only run so far
  was a zero-budget smoke test proving the wiring, not real results.

## Android app production-readiness — infra live, app work not started

**What it is:** migrating the ReadAloud AI Android app's normal TTS
reading flow off Chatterbox/Hetzner onto this platform's real API,
production-safely (today's test build proved the pipeline works on one
physical phone but embeds a real API key in source and is not
release-ready). Spec:
`ReadAloudAI/docs/superpowers/specs/2026-09-22-android-realtime-tts-production-readiness-design.md`.
Plan: `ReadAloudAI/docs/superpowers/plans/2026-09-22-android-realtime-tts-production-readiness.md`.

**State: Tasks 1-3 of 8 complete and genuinely live in production.**
Piper voice storage migrated from local machine disk to Tigris
(S3-compatible object storage) — deployed to `piper-tts-sjc`, all 69
tier A/B voices backfilled, verified with a real end-to-end synthesis
call. Branch `sdd-prod-readiness` in both repos (realtime-tts worktree
`worktrees/sdd-prod-readiness-realtime-tts` @ `adc021a`; ReadAloudAI
worktree `worktrees/sdd-prod-readiness-readaloudai`, plan/spec docs only,
no app code yet).

**Pending — Tasks 4-8, not started:**
- **Task 4 — Autoscaling.** The Piper worker still runs on one fixed
  machine (4 concurrent syntheses, globally, across every customer). This
  was the original reason this whole project got flagged as unsafe for
  app-scale traffic — still unresolved.
- **Task 5 — Dedicated API key for the app**, provisioned with a real
  quota tied to the (not-yet-built) autoscaling ceiling.
- **Task 6 — Backend-proxy route** (`/api/realtime-tts/authorize` in
  ReadAloudAI's own Node backend) so the app never holds a raw platform
  key.
- **Task 7 — Real per-voice mapping.** Every voice currently maps to one
  hardcoded Piper voice (`en-us-john`); needs the real 10-voice mapping
  table.
- **Task 8 — Point the app at the backend-proxy, remove the embedded key.**

**The app itself**: `RealtimeTTSService.kt`/`TTSService.kt` (the tested
build) is checkpointed on branch `android-realtime-tts-migration` in
ReadAloudAI (commit `b294606`), never merged to `main`. Still has a real
API key hardcoded in source and is an unsigned debug build — explicitly
not safe to ship. **Not ready for app-store review or any real user.**

## Everything else from today

Merged to `main` already, live where noted: concurrency per-key fairness
fix (`worker-piper-fly`), compliance audit-logging groundwork, phone
agent MVP + hardening (Twilio orchestration, real LLM via OpenRouter,
**deployed live at `call-loop-poc.fly.dev` with number `+12245061194`, active**), dubbing MVP + hardening (real async job
API deployed to `dubbing-worker.fly.dev` with real STT (`stt-worker` Modal) and real TTS
(gateway) wired up — **end-to-end verified working: en→de translation + Piper synthesis + retiming produces valid WAV**), audio isolation MVP +
hardening (deployed live at `demucs-isolation-dev.fly.dev`, gateway-wired
— but see the real-speech quality gap noted below), **voice design MVP (Parler-TTS on Modal, self-serve via `/design-voice`, deployed and verified)**, voice research harness (merged to `main`, 34/34 tests passing,
`--budget 0` smoke-tested only), a 56→69 voice / 9→19 language mining batch (5 tier-A voices
published live, 8 tier-B cataloged-only per this repo's existing
licensing policy).

**Known open finding, not yet acted on:** audio isolation's real-speech
quality gap — synthetic-clip testing showed +9.4dB SNR improvement, but
genuine (non-TTS) speech only improved +0.2dB under identical conditions.
Investigated three ways (model swap to DeepFilterNet: no better; a
genuinely-noisy real recording: inconclusive, not noisy enough to be a
fair test). Live customer-facing docs were corrected to disclose this;
the underlying capability gap itself is still open.
