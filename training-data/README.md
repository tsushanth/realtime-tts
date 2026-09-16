# Training data for a from-scratch fast TTS model (Option A)

Context: see `../DECISIONS.md` ("Can we match ElevenLabs' TTS latency?" section) for
the full investigation. This directory holds artifacts for **Option A** — training a
FastPitch-style (non-recurrent, TensorRT/torch.compile-friendly) model from scratch,
since Kokoro's own training data was never released and the LSTM duration predictor
resists compiler optimization.

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

## Files

- `voice-samples/polly_standard.mp3`, `polly_neural.mp3`, `polly_generative.mp3`,
  `polly_generative_ruth.mp3` — the comparison samples that informed the decision
  above. Same test sentence across all four engines/voices.

## Not yet done

- Text corpus for the full ~1.2M-character generation run: decided to use our own
  authored assistant-reply content (flow scripts / prompt templates), not caller call
  transcripts (which would reopen the same privacy/consent question already
  documented for the "reuse SecureVox/ClearVoiceRecorder recordings" idea) and not a
  generic open text corpus (less representative of our actual use case). Sourcing
  this corpus is in progress.
- Bulk generation pipeline (call Polly at scale, store audio+text pairs — likely S3,
  not git, once this is gigabytes rather than four small samples).
- The actual model training run itself (not started).
