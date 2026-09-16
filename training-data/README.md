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

## Bulk audio generation — in progress

`generate_corpus_audio.py` running against all 22,011 lines in `corpus_final.txt`
(resumable, rate-limited via a small thread pool + exponential backoff). Output
audio (~650MB) and `manifest.jsonl` are gitignored — real destination is S3, not git,
once this is gigabytes rather than a few small samples.

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

**Next:** scale this same fine-tuning script to the full 22,011-sample corpus once
bulk audio generation finishes, with more steps, and re-evaluate by ear at each
checkpoint before committing to a "final" run — same incremental-verification
discipline used throughout this whole investigation.

## Also still open

- Google Cloud TTS (Neural2/Studio/Chirp 3 HD) and Azure Neural HD were considered as
  likely-comparable-or-better on naturalness per weak/unverified aggregator evidence,
  but blocked on account access (no GCP billing enabled, no Azure account) — not ruled
  out, just not sampled. Revisit if the Matcha-TTS fine-tune doesn't produce a
  satisfactory voice.
- Kokoro was independently flagged (by a peer session's research) as possibly the
  stronger fine-tuning target overall — same architecture already running in our own
  production `call-loop-poc` infra, 82M params, Apache 2.0. Not yet tried; worth a
  parallel pilot if Matcha-TTS's fine-tuned quality plateaus below what's needed.
- Amazon Polly's terms of service may restrict using its synthesized output to train
  a competing TTS model — flagged by a peer session, not yet verified. Should be
  checked before scaling training spend further, independent of which architecture is
  used.
