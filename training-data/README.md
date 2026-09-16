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

## Not yet done

- Bulk generation pipeline: call Polly Generative on all 22,011 lines in
  `corpus_final.txt`, save each audio+text pair (likely S3, not git, once this is
  gigabytes of audio rather than four small samples).
- The actual model training run itself (not started).
- Google Cloud TTS (Neural2/Studio/Chirp 3 HD) and Azure Neural HD were considered as
  likely-comparable-or-better on naturalness per weak/unverified aggregator evidence,
  but blocked on account access (no GCP billing enabled, no Azure account) — not ruled
  out, just not sampled. Revisit if Polly's full dataset doesn't produce a
  satisfactory trained voice.
