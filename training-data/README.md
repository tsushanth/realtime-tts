# Training data + fine-tuning for a fast TTS model (Option A/B hybrid)

Context: see `../DECISIONS.md` ("Can we match ElevenLabs' TTS latency?" section) for
the full investigation. This directory started as Option A — training a FastPitch-
style (non-recurrent, TensorRT/torch.compile-friendly) model **from scratch** — but
pivoted to **fine-tuning** a pretrained Matcha-TTS checkpoint on this same corpus
instead, once a from-scratch pilot made clear that's the much cheaper, faster path to
the same architecture family.

**Cost correction (2026-09-16):** earlier cost estimates in this file quoted
$62-195 (and, before that, $500-1,500) for training — those numbers came from
FROM-SCRATCH pretraining research (NeMo's published FastPitch convergence: ~44
GPU-hours; HiFi-GAN vocoder from random init: ~72-120 GPU-hours) and were never
recalculated after the pivot to fine-tuning. **Fine-tuning a pretrained checkpoint is
not the same cost class.** Real measured cost from the Phase 0 pilot below: 300
steps in ~70s on a T4 (~$0.01). Even a generous 50,000-step full fine-tune run,
extrapolated from that same measured rate, is ~3 hours (~$1.77) — not $62-195. Exact
step count needed for full convergence on all 22,011 samples is still an open
question (empirical, answered by scaling this run and watching validation loss), but
the cost ceiling is clearly single-digit dollars, not tens-to-hundreds.

## Decision so far

**Voice engine: Amazon Polly, Generative tier, voice "Joanna."**

Rationale:
- Cost at this data volume (~1.2M characters / ~24 hours, the standard single-speaker
  training baseline) is trivial across every provider considered (~$3-36 total) — not
  a real differentiator.
- Quality is what matters, since training data quality directly caps the trained
  model's quality (it can't systematically sound better than what it's shown).
- Compared Polly Standard / Neural / Generative (two voices: Joanna, Ruth) by ear —
  see `voice-samples/`. All sounded usable; picked Generative/Joanna since the
  Neural-vs-Generative cost gap (~$30 total) isn't worth trading quality for at this
  volume.
- Google Cloud TTS (Neural2/Studio/Chirp 3 HD) and Azure Neural HD were considered as
  likely-comparable-or-better on naturalness per weak/unverified aggregator evidence,
  but blocked on account access (no GCP billing enabled, no Azure account) — not ruled
  out, just not sampled. Revisit if Polly's full dataset doesn't produce a
  satisfactory trained voice.

## Text corpus — done, target hit exactly

Decided to use our own authored assistant-reply content, not caller call transcripts
(reopens the same privacy/consent question documented for the "reuse SecureVox/
ClearVoiceRecorder recordings" idea — see `../DECISIONS.md`) and not a generic open
text corpus (less representative of our actual phone-assistant domain).

**Pipeline:**
1. `corpus_batch_00N.txt` — 8 hand-authored batches (~1,300 sentences total, ~120K
   characters), each covering a structurally distinct register: greetings/
   confirmations, multi-clause/empathy language, troubleshooting Q&A, digit sequences/
   policy/pricing, upsell/closing-checklist phrasing, statistics/specs/filler words,
   security-verification/travel/traffic info, and affirmative/escalation/read-back
   confirmation patterns.
2. `expand_corpus.py` — substitutes varied names/dates/times/dollar-amounts/
   percentages/addresses/digit-sequences into each batch's slots, de-duplicated, up
   to a per-template cap. Every substitution pool and pattern is in the script.
3. `corpus_expanded_00N.txt` — each batch's expanded output; `corpus_expanded_all.txt`
   is all eight combined and de-duplicated (2.2M characters at the current cap —
   deliberately overshoots the target so the next step has real headroom to sample
   from).
4. **`corpus_final.txt` — the actual dataset: 22,011 sentences, exactly 1,200,000
   characters**, randomly sampled from step 3 (seeded, reproducible) down to the
   ~24-hour single-speaker training baseline. This is what gets sent to Polly.

Full Polly Generative synthesis cost for this exact corpus: **~$33** (verified
against AWS's official pricing: $30/1M characters, minus the 100K/month free-tier
allowance for the first 12 months).

## Files

- `voice-samples/polly_standard.mp3`, `polly_neural.mp3`, `polly_generative.mp3`,
  `polly_generative_ruth.mp3` — the comparison samples that informed the Polly
  Generative/Joanna decision. Same test sentence across all four engines/voices.
- `corpus_final.txt` — the dataset to synthesize. Everything else in this directory
  is an intermediate artifact of how it was built.

## Bulk audio generation — done

`generate_corpus_audio.py` ran against all 22,011 lines in `corpus_final.txt`
(resumable, rate-limited via a small thread pool + exponential backoff). First run
hit a transient `ConnectionResetError` from AWS around the 20,000 mark (network
blip, not a code/quota bug) — re-running the same script picked up where it left
off since it skips any output file that already exists and is non-empty. One
residual gap: 4 files had been created as zero-byte placeholders by the interrupted
run and were correctly *not* skipped (size-zero check), so a second re-run filled
them in. **Final state: 22,011/22,011 files, 0 errors, 623MB.**

`manifest.jsonl` accumulated duplicate lines across the two runs (appends, not
overwrites) — deduped by `idx` (keep-last) down to exactly 22,011 records matching
the 22,011 audio files. If this script is ever re-run for a partial fill again,
dedupe the manifest the same way afterward.

Output audio (~623MB) and `manifest.jsonl` are gitignored — real destination is S3,
not git, once this is gigabytes rather than a few small samples.

## Phase 0 pilot — done, real signal, not yet production quality

`pilot_finetune.py`: fine-tunes Matcha-TTS's official pretrained LJSpeech checkpoint
(flow-matching, non-autoregressive — same TensorRT/compiler-friendly architecture
family Option A was chasing) on a 279-sample / 20-validation slice of the real
Polly-Joanna audio already generated. Took 8 rounds of environment debugging
(missing deps, Hydra config incompatibilities incompatible with the `compose()` API
vs. the full `@hydra.main` runtime, matplotlib API drift, Modal return-value transfer
for the trained audio) — all fixed, see git history for `pilot_finetune.py`.

**Result:** pretrained weights loaded with 0 missing/unexpected keys. 300 fine-tuning
steps, loss dropped and stabilized (~1.40-1.47 train, ~1.41-1.6 val). Synthesized 3
held-out sentences — saved in `pilot_output/`. Listened by ear: real voice adaptation
happening, audibly moving toward the training data's voice, still somewhat artificial
at this tiny step count (expected — 300 steps is a smoke test, not a converged run).

**Next:** bulk audio generation is now done (see above) — scale this same
fine-tuning script to the full 22,011-sample corpus, with more steps, and
re-evaluate by ear at each checkpoint before committing to a "final" run — same
incremental-verification discipline used throughout this whole investigation. Also
build the Piper fine-tuning pilot (see "Also still open" below) against the same
full corpus once it's ready, for the CPU-serving/cold-start angle.

## Also still open

- Google Cloud TTS (Neural2/Studio/Chirp 3 HD) and Azure Neural HD were considered as
  likely-comparable-or-better on naturalness per weak/unverified aggregator evidence,
  but blocked on account access (no GCP billing enabled, no Azure account) — not ruled
  out, just not sampled. Revisit if the Matcha-TTS fine-tune doesn't produce a
  satisfactory voice.
- Kokoro was independently flagged (by a peer session's research) as possibly the
  stronger fine-tuning target overall — same architecture already running in our own
  production `call-loop-poc` infra, 82M params, Apache 2.0. Real fine-tuning path
  found (2026-09-16): DIY only — hexgrad never released Kokoro training code, only
  inference weights. The only route is patching Kokoro's weights into the separate
  StyleTTS2 training repo plus a custom checkpoint converter; two small community
  projects (semidark/kikiri-tts, avri-schneider/kokoro-hebrew) did this but there's no
  official/maintained path. Higher debugging risk than Matcha-TTS was. Not yet tried;
  worth a pilot only if Piper's fine-tuned quality plateaus below what's needed.
  **Deeper research (2026-09-17):** confirmed real and independently replicated (not
  vaporware) — checkpoint conversion is mechanically simple (Kokoro's `.pth` is
  already keyed by StyleTTS2 module names), but requires a real, documented patch
  set applied to StyleTTS2 itself (symlink table fix, weight_norm API migration,
  restoring deleted loss-computation tensors, `torch.load` weights_only fix, etc. —
  not a one-line loader change). Both community repos did *cross-lingual* transfer
  (English Kokoro -> German/Hebrew); our use case (a different English voice, same
  language) is actually easier than what they proved, since it avoids their
  documented symbol-table/phoneme-index-mismatch risk entirely and an open issue
  showing this recipe failing to generalize to a new language (Vietnamese) doesn't
  apply to us. Real cost regardless of language: a near-full StyleTTS2 Stage 1 + 2
  training run (tens of GPU-hours, not a lightweight adapter fine-tune — one
  reported case was ~48h for 10 epochs on non-cloud hardware), plus a real data
  rebuild (24kHz + IPA phonemization + speaker-id column, not a reuse of the
  Matcha/Piper `path|text` filelists). Lowest-risk approach if pursued: fork
  kikiri-tts directly rather than reimplementing its patch set from scratch. Given
  the cost is an order of magnitude above anything spent so far (~$0.01-0.67 for
  the Matcha/Piper pilots vs. tens of GPU-hours here), this should get an explicit
  go/no-go check before spending real money, not just be started.
- **New candidate: Piper, specifically for CPU-served always-on serving** (see
  `../DECISIONS.md`, "Fine-tuning path reopens Option A/B cheaply"). MIT, VITS-based,
  CPU-optimized. Official, maintained fine-tuning support via OHF-Voice/piper1-gpl
  (`--ckpt_path` against published pretrained checkpoints, works cross-language, plain
  `filename|text` CSV data format). Lower integration risk than Kokoro's DIY path.
  Directly relevant to the cold-start problem from the original ElevenLabs-parity
  investigation: a good Piper fine-tune could run on an always-on CPU instance instead
  of paying for a warm GPU floor (~$425/mo/T4) to avoid the ~17.5s cold start — CPU
  instances are typically much cheaper to keep resident 24/7 than GPU instances. This
  is the next planned pilot, following the same Phase-0-cheap-first discipline used
  for Matcha-TTS.
- Amazon Polly's terms of service may restrict using its synthesized output to train
  a competing TTS model — flagged by a peer session, not yet verified. Should be
  checked before scaling training spend further, independent of which architecture is
  used.

## Full-corpus fine-tune — done (2026-09-16)

Scaled `pilot_finetune.py`'s approach to the complete 22,011-sample corpus:
`full_finetune.py` (training) + `synthesize_full_ft.py` (inference from the
resulting checkpoint) + `build_filelists.py` (train/val split generation).

**Getting the data there was the hard part, not the training.** Uploading the
~4.2GB of WAV audio directly from this Mac — via `modal volume put` or
`aws s3 cp` on the single tar file — died silently and non-deterministically
every single time (at 7%, 17%, then 1% of the transfer, no error text, no
consistent cutoff point), regardless of destination or sandbox mode. One
attempt did surface a real `SSLV3_ALERT_BAD_RECORD_MAC` TLS error, pointing to
a flaky home network link rather than a bug in either CLI. Fixed by splitting
the tar into 100MB chunks (`split -b 100m`) and uploading with `aws s3 sync`
(skips already-uploaded chunks on rerun, so a mid-job failure only costs
chunks in flight) — all 43 chunks landed cleanly. A Modal function
(`setup_corpus_volume` in `full_finetune.py`) then reassembles and extracts
them **from inside Modal** onto the training volume — cloud-to-cloud S3 read,
completely bypassing this Mac's flaky connection for the actual heavy lift.
Uses a dedicated IAM user (`tts-corpus-modal-reader`) scoped to read-only
access on just this one S3 bucket, not the broader shared account key.

**Training itself was uneventful** — same T4/batch-size-8 config as the pilot
(so the ~$0.01/300-steps rate stays a valid basis for cost), 20,000 steps,
completed in full with no crashes. Final losses: train ~1.31-1.37, val
~1.21-1.33 — comparable to the pilot's range, now trained on the real 22K-
sample corpus instead of a 279-sample smoke test slice.

**Result, `training-data/full_ft_output/sample_0.wav` through `sample_4.wav`**:
5 held-out test sentences (call-center-style phrasing, including one with a
spoken-digit sequence) synthesized with the fully fine-tuned model. This is
the first real checkpoint trained on the full corpus, not just the tiny pilot
slice — listen and judge quality before deciding on next steps (more training
steps if it's still improving, or move to comparing against a fine-tuned
Piper for the CPU-serving angle).

Checkpoints for every 2,000 steps (plus the final one) are on the
`tts-checkpoints` Modal volume at `/full_ft/`, in case an earlier step turns
out to sound better than the final one (possible if the model started
overfitting or drifting late in training — worth comparing a few, not just
assuming later is always better).

**Ear-verified result (2026-09-17): good, real progress.** Listened to all 5
samples — overall quality judged "good," a real, usable improvement over the
279-sample pilot. Two specific notes: (1) speaking pace reads as a little
slow throughout — `synthesize_full_ft.py` used `length_scale=0.95`; worth
trying a lower value (e.g. 0.85-0.9) on the next synthesis pass, no retraining
needed since this is an inference-time parameter, not a training artifact.
(2) `sample_3.wav` ("Is there anything else I can help you with today?") had
one word with noticeably robotic intonation, an isolated blip rather than a
pervasive problem across all 5 samples. Worth rechecking after more training
steps or a length_scale change to see if it's still there before treating it
as a real pattern rather than noise from this specific sentence/seed.

## Piper fine-tuning pilot — done, working (2026-09-17)

`piper_pilot_finetune.py`: real, working fine-tune of OHF-Voice/piper1-gpl's
`en_US-lessac-medium` checkpoint on the same 279-sample pilot slice, 300
steps/5 epochs. `val_mel` decreased cleanly every epoch (0.5647 -> 0.5577 ->
0.4938 -> 0.4649/0.4494 -> 0.4240 across two confirming runs) - a real
convergence signal, not yet ear-verified with synthesized audio (next step).

Took 6 real debugging rounds to get a clean run, all documented inline in the
script itself: the same `weights_only` checkpoint-loading issue as
Matcha-TTS but a different mechanism (Lightning's CLI hardcodes it,
independent of torch's own default) and a different fix (must call
`main()` in-process, not via subprocess, for a monkeypatch to take effect);
`--ckpt_path` strictly validates a checkpoint's saved hyperparameters and
breaks on this older checkpoint's stale `sample_bytes` field -
`--model.warmstart_ckpt` is the correct weights-only mechanism instead;
`espeakbridge` (a CMake/scikit-build native extension) is silently never
built by a plain `pip install -e` at all, needing `--no-build-isolation` +
pre-installed build-system deps + a separate explicit
`setup.py build_ext --inplace` (matching upstream's own `script/dev_build`);
`monotonic_align` is a second, separate Cython extension not wired into
setup.py or CMakeLists.txt at all, always needing its own manual build step;
the same NumPy 2 ABI break as Matcha-TTS, but the pin didn't survive Piper's
own dependency resolution and had to be re-applied after everything else
installed; and Piper's own hardcoded `val_mos` checkpoint-quality callback
hard-crashes under our installed Lightning version instead of the soft-skip
its own source comment promises (dropped that one callback).

**Next:** synthesize audio from the resulting checkpoint and listen (same
step Matcha-TTS went through), then decide whether to scale this to the full
22,011-sample corpus the same way `full_finetune.py` did for Matcha-TTS.

## Real CPU inference benchmarks (2026-09-17)

Both fine-tuned checkpoints (Matcha-TTS's full 22,011-sample run, Piper's
279-sample pilot) benchmarked on a plain 4-vCPU Modal container - no GPU at
all - with the same 5 test sentences used throughout this whole
investigation. Real measured numbers, not the stock-voice Jetson numbers
from earlier in this file:

| Model | RTF | Speed vs. realtime | Model load |
|---|---|---|---|
| Matcha-TTS (full corpus) | 0.53-0.61 | 1.6-1.9x | ~3.1s |
| Piper (pilot) | 0.10-0.11 | ~9-10x | ~1.7s |

Both are genuinely fast enough for real-time serving on plain CPU, with
Piper meaningfully faster (expected - it's purpose-built for CPU/ONNX
serving, where Matcha-TTS's win was specifically "not LSTM-based, so at
least *viable* on CPU/compiler backends" rather than "optimized for CPU").
This is real, direct evidence for the always-on-CPU-instance path discussed
throughout this file as an alternative to paying for a warm GPU floor to
avoid the ~17.5s cold start - both models could plausibly serve from a
cheap, always-on CPU box today, pending real load-testing under concurrent
calls (not measured here - this is single-request latency only).

## Kokoro shortcut: real limitation found, not just undertraining (2026-09-17)

Scaled the Kokoro pilot from 1 to 15 epochs to test whether the pure-noise
output (see below) was simply the from-scratch style/prosody encoders
needing more warmup time, as hypothesized after the first listen. **That
hypothesis turned out to be incomplete.** Validation loss did improve
(0.716 vs. ~0.79-0.81 at 1 epoch) but generated durations were essentially
unchanged (58.0s vs. 57.4s, 67.2s vs. 67.0s, etc. for the same sentences
that should run 3-6s) despite 15x more training. Something more structural
than "needs more steps" is producing pathologically long durations under
this shortcut config - a plausible candidate (not yet verified) is a
hop-length/frame-rate mismatch between our config and what Kokoro's
duration predictor was actually trained against, since duration is
predicted in frames and converted to samples via `hop_length`.

**Given this is a real, unresolved structural issue** rather than something
more training time fixes, and given Matcha-TTS and Piper are both already
validated end to end with good quality *and* strong CPU performance (see
above), further Kokoro debugging is being deprioritized rather than sinking
more GPU-hours chasing this specific issue. The DIY StyleTTS2 shortcut path
remains documented and working up through checkpoint training and
inference - if revisited, start by comparing our config's `hop_length`/
`sr`/frame-rate settings against Kokoro's original training config before
assuming more epochs will help.

## Piper full-corpus fine-tune - done (2026-09-18)

`piper_full_finetune.py` (training) + `synthesize_piper_full.py` (ONNX export +
CPU benchmark from the persisted checkpoint). 21,791 train / 220 val samples,
`max_steps=20000`, batch_size=8, T4, warm-started from `en_US-lessac-medium`.
Checkpoints + voice `config.json` live on the `tts-checkpoints` Modal volume at
`piper_full_ft/lightning_logs/version_1/checkpoints/` (`last.ckpt`).

**Step-count lesson (I got this wrong mid-run):** Piper's VITS training is a GAN
with two optimizers, and Lightning's `max_steps` counts *optimizer updates*, not
batches - so 20,000 steps = ~10,000 batches = ~4.2 epochs of this corpus, not
~8. That is why the run finished in ~2.5h instead of the ~4.6h estimated from
"20,000 steps at 1.2 it/s", and why mid-run progress reports (e.g. "35% done at
epoch 2") were wrong. Matcha-TTS uses a single optimizer, so its 20,000 steps
= ~7 epochs: the two full runs are NOT equal amounts of training. For any
future estimate, divide max_steps by the number of optimizers first.

**Validation:** `val_mel` 0.3538 -> 0.3469 -> 0.3337 -> 0.3343 -> 0.3343
(plateaued over the last epochs; pilot was ~0.42).

**Preprocessing stall (fixed, see `piper_full_finetune.py` docstring):** an
earlier attempt sat 8+ hours on `prepare_data()` with no log output. Fixes:
copy WAVs from the Volume to local disk first (95s for 22,011 files) and patch
in progress logging every 500 utterances (`patch_piper_progress.py`).
Also: use `.spawn()` not a blocking `.remote()` for multi-hour jobs - two
blocking runs were cancelled when the local client process died.

**CPU inference (4 vCPU, no GPU), same 5 test sentences:**

| Model | RTF | Speed vs. realtime | Model load |
|---|---|---|---|
| Piper (full corpus) | 0.043-0.064 | ~16-23x | ~1.3s |
| Piper (pilot) | 0.10-0.11 | ~9-10x | ~1.7s |
| Matcha-TTS (full corpus) | 0.53-0.61 | 1.6-1.9x | ~3.1s |

Samples: `piper_full_output/sample_0.wav` .. `sample_4.wav`. Not yet
ear-verified at the time of writing.
