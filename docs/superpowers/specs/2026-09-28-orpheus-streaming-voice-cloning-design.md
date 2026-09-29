# Low-latency streaming voice cloning (Orpheus): production design

Date: 2026-09-28. Repo: realtime-tts (branch `orpheus-voice-clone-poc`), with gateway/web-facing
pieces in ReadAloudAI. Companion pilot artifacts: `voice-pipeline/orpheus_clone_pilot/` (dataset
extraction, LoRA fine-tune, decode validation, serving benchmark — all already built and run in
this branch's history).

## Problem

The shipped voice-cloning product (`worker-piper-fly` + `VoiceCloningDocs.tsx`, $2.50/voice,
10-60 min of audio, 30-60 min Piper full fine-tune) is not fast or high-fidelity enough for
real-time conversational use. An earlier attempt on Fish Speech measured 1-2s time-to-first-audio,
too slow against a 500ms target for live call agents. This spec productionizes the Orpheus
LoRA-cloning pilot already validated in this branch (real end-to-end pipeline: dataset extraction
from source audio, SNAC tokenization verified against canopylabs' own data-prep notebook, LoRA
fine-tune, merge, vLLM-based streaming serving) as a second, parallel cloning offering.

## Pilot results this design builds on

- Voice cloning quality: a LoRA fine-tune on ~250 clips (~8 min) of a reference voice produced
  audio a human listener confirmed sounded like the source voice.
- Warm-state time-to-first-audio-chunk: median ~552-583ms across three test prompts (p90 similar) —
  close to, not yet at, the 500ms target. Cold container start: ~2.6-3.6s (one-time per container
  boot).
- Two unresolved quality issues, believed to need more training (more epochs, more data), not a
  serving fix: (1) real-time factor of ~1.6-2.7x — generation is slower than the audio it produces,
  (2) unreliable stop-token learning — the same prompt produced 0.34-12.9s of audio across 5 runs
  with an otherwise-correct stop token configured.
- Several real dependency/API bugs found and fixed in the pilot, load-bearing for this design:
  `OrpheusModel.__init__` only accepts `model_name`/`dtype` (no `tokenizer`/`max_model_len`); vLLM
  0.7.3 requires `transformers>=4.48.2` (pinning to 4.46.3, used for training, is a hard conflict —
  training and serving need different pinned versions); vLLM defaults KV cache sizing to the base
  model's `max_position_embeddings` (131072), which doesn't fit an A10G and must be capped via a
  monkey-patched `AsyncEngineArgs`; `orpheus-speech`'s own default `stop_token_ids=[49158]` does not
  match this tokenization scheme's real end-of-speech token (`128258`).

## Scope

In scope: a second, parallel self-serve cloning path (`custom-fast:<id>`), its own training job,
its own low-latency serving path, checkpoint persistence, and honest public documentation of its
current limitations.

Out of scope (explicit non-goals): unifying with the existing `/v1/voices` (Piper) API surface;
any pricing decision (SKU exists as a placeholder, actual numbers TBD); any rollout quality gate
blocking self-serve availability (per product decision, ships now with limitations documented, not
gated); GPU-serving cost optimization beyond scale-to-zero; any change to the existing Piper
cloning product's code, pricing, or behavior.

## Decisions (from design discussion)

1. **Relationship to existing cloning**: parallel offering, not a replacement. Existing Piper
   cloning is untouched.
2. **Checkpoint model**: train once, persist the merged checkpoint permanently (like Piper's
   `custom:<id>`); only the *inference* container is loaded on demand and torn down after idle —
   not a per-request retrain.
3. **Isolation**: new standalone Modal app and code path, reusing only the gateway's existing
   shared primitives (API key validation, the 402/free-tier check, the consent-record shape).
   Chosen over unifying into the existing `/v1/voices` surface because Orpheus quality is unproven
   (see pilot results above) and this codebase's established pattern (the realtime STT eval) is to
   keep unproven work isolated until it clears a bar, merging only once decided.
4. **Rollout**: ships self-serve to all API keys immediately, no quality gate. Limitations are
   documented candidly (same pattern as the existing STT docs' "what we measured" section) and
   improved in place based on real usage, not blocked pre-launch.

## Architecture

```
Customer                Gateway (existing)         orpheus-clone-prod (new, Modal)
   |                          |                              |
   |--POST /v1/orpheus-voices-->  (auth, 402 check)  -------->|  create voice record
   |                          |                              |  status: awaiting_dataset
   |<-------------------------|{id, status}------------------|
   |                          |                              |
   |--PUT .../dataset (zip)-->|------------------------------>|  store to Volume (staging)
   |--POST .../commit-------->|------------------------------>|  spawn training job
   |                          |                              |  status: training
   |--GET .../{id} (poll)---->|------------------------------>|  status: training|ready|failed
   |                          |                              |
   |--POST /v1/orpheus-tts--->|  (auth, billing meter)  ----->|  load checkpoint (cold) or
   |   {text, voice:          |                              |  reuse warm container
   |    "custom-fast:<id>"}   |                              |  stream PCM chunks back
   |<-------------------------|<-----------------------------|
```

### Training job (mirrors `voice-design-dev`'s job pattern)

- `POST /v1/orpheus-voices {speaker_name, attested_by, consent, consent_text_version,
  consent_statement}` -> `{id, status: "awaiting_dataset"}`. Same required consent fields as the
  existing Piper cloning endpoint — the legal requirement (attested right to clone the voice)
  doesn't change with the underlying model.
- `PUT /v1/orpheus-voices/{id}/dataset` (zip of WAV/FLAC/MP3, single speaker) — same format as
  Piper cloning's dataset upload, different recommended duration: **8-20 minutes** (the pilot used
  ~8 min of ~250 clips), not 10-60.
- `POST /v1/orpheus-voices/{id}/dataset/commit` -> spawns the Modal training job
  (`build_dataset.py`'s extraction logic generalized to arbitrary uploaded audio + a forced
  transcription step, since uploaded audio has no existing manifest text — see Task breakdown) and
  sets `status: "training"`.
- Training job (Modal, GPU): transcribe each clip (faster-whisper, matching the STT eval's
  reference-generation approach) to get the `text` half of each pair, SNAC-tokenize
  (`finetune_pilot.py`'s verified tokenization, unchanged), LoRA fine-tune from
  `canopylabs/orpheus-tts-0.1-pretrained` (needs the `hf-token` secret, gated repo), merge, save
  the merged checkpoint to `orpheus-clone-checkpoints` volume at `{id}/merged/`. On completion,
  `status: "ready"`; on any failure, `status: "failed"` with an error message field (mirrors
  `intake.py`/`voice_design_dev.py`'s "roll back and report" pattern).
- `GET /v1/orpheus-voices/{id}` -> current status, polled by the client (~30-90 min expected
  training time given LoRA-on-250-clips took well under an hour in the pilot on an A10G, plus
  transcription and image-cold-start overhead; real number to be measured on the actual endpoint,
  not assumed from the pilot's manual `modal run` timings).
- `DELETE /v1/orpheus-voices/{id}` -> deletes the volume subdirectory and the record.

### Serving (new path, not the existing Piper WebSocket/HTTP)

- `POST /v1/orpheus-tts {text, voice: "custom-fast:<id>"}` — a new endpoint, not folded into the
  existing `/v1/text-to-speech`, because this needs a vLLM-backed engine class, not the Piper/Kokoro
  server.
- Backing Modal `@modal.cls` (`OrpheusCloneEngine`, generalizing `serve_benchmark.py`'s
  `ClonedVoiceEngine`): `@modal.enter()` loads the requested voice's checkpoint from the volume
  (path derived from `voice` param), applies the two fixes already found necessary
  (`AsyncEngineArgs` `max_model_len` patch; correct `stop_token_ids=[128258]`), streams PCM16
  chunks back exactly as `serve_benchmark.py` already measured.
- `scaledown_window` tuned from real usage after launch, starting from the pilot's default (300s)
  as a reasonable initial guess, not a measured optimum.
- Per-request generation timeout given RTF>1 is a known, undiagnosed issue: cap wall-clock
  generation time (recommend 15s, generous relative to typical utterance length) and return `503`
  with a clear message if exceeded — same shape as the existing Piper "at capacity" `503`/`1013`
  convention, applied here to "generation ran away" instead of "no capacity."

### Storage

New Modal Volume `orpheus-clone-checkpoints`, layout:
```
{id}/
  manifest.json   # speaker_name, attested_by, consent fields, consent_text_version,
                   # dataset clip count, created_at, trained_at
  merged/          # HF-format checkpoint (config.json, safetensors, tokenizer files)
```
No TTL/retention policy — training cost is sunk once paid; deletion is explicit only
(`DELETE /v1/orpheus-voices/{id}`), same as the existing Piper voice records.

### Billing

New placeholder SKU for the one-time training charge (distinct from the existing $2.50 Piper SKU —
actual price TBD, a business decision, not built here) plus a per-character or per-second
synthesis meter on `/v1/orpheus-tts`, wired to the same 402/free-tier gateway check already used
by every other billed endpoint. This spec defines the wiring point, not the number.

### Documentation

The developers page copy already added (`ReadAloudAI/web/src/app/developers/page.tsx`,
"Low-latency streaming clone (in development)" section) becomes the permanent, living doc for this
feature — updated as real numbers come in from the shipped endpoints, not deleted once "done." It
already states the RTF and stop-token caveats candidly; once real customer-uploaded-voice numbers
exist they replace the pilot's single-reference-voice numbers.

## Error handling

- Training job failure (transcription failure, SNAC encode failure, training crash, gated-repo
  auth failure) -> `status: "failed"` with a message field; volume state cleaned up (no partial
  `merged/` directory left in a state `GET` would treat as ready).
- Serving: generation timeout -> `503` (see above). Missing/unknown `voice` id -> `400`. Checkpoint
  load failure at container start -> `503`, retryable (the client's existing Piper-capacity retry
  logic already handles a `503`; no new client-side pattern needed).
- Same auth/billing error codes as every other endpoint (`401` invalid key, `402` free tier
  exhausted).

## Testing

- Unit: dataset-extraction-from-upload (transcription integration point), SNAC tokenize/detokenize
  round-trip (already covered by the pilot's tests, carried over), training-job status-transition
  logic, checkpoint path derivation from `voice` id.
- Integration: full job lifecycle (`create -> upload -> commit -> poll -> ready`) against a small
  real audio sample in CI or a manual pre-merge run (real GPU cost, not simulated, matching this
  repo's established practice of never mocking the training/serving GPU path); serving smoke test
  (load a known-good test checkpoint, synthesize, verify non-empty audio, verify the
  `max_model_len`/`stop_token_ids` fixes are present so this doesn't regress silently — both were
  real bugs found by running, not by reading the code).
- No automated quality/latency gate blocks merge or rollout, per the rollout decision above — but
  the serving smoke test should still assert the two known-bug fixes are in place, since those are
  correctness bugs, not quality-tuning items.

## Non-goals (restated)

No unification with `/v1/voices`. No pricing decision. No rollout quality gate. No cost
optimization beyond existing scale-to-zero patterns. No change to existing Piper cloning.
