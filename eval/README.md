# TTS eval framework: us vs ElevenLabs

A small, re-runnable framework comparing our TTS engines (Piper, Kokoro) against ElevenLabs
(Flash v2.5, Multilingual v2) on latency and two automated quality proxies. Not statistically
powered — directional, meant to say where to invest engineering time next, not to produce a
marketing number. Re-run this after any engine change to see if it moved.

## Files

- `testset.json` — versioned test sentences (~20, English/Spanish/German/French, call-center
  categories: greeting, order-status, de-escalation, question, multi-clause). Extend this file
  to grow the eval.
- `synth.mjs` — synthesizes the whole test set through every engine, measuring time-to-first-audio
  the same way `benchmarks/latency_probe.js` does (same vantage, interleaved, one warm connection
  per engine reused across sentences). Writes `results/latency.json` and `audio/<engine>/*.wav`.
- `repair.mjs` — reruns only jobs that errored (e.g. a dropped connection) without repaying for
  engines that already succeeded; also supports `RETARGET_IDS` to re-measure specific jobs with a
  proper warm-up (used to fix reconnect-penalty outliers in the first run — see below).
- `score.py` — quality scoring: intelligibility (word error rate via faster-whisper, transcribing
  each clip and comparing to the known input text) and naturalness (an automatic MOS predictor,
  see caveat below). Writes `results/quality.json`.

## How to re-run

```
cd eval
TTS_GATEWAY_API_KEY=<a billing-enabled key> ELEVENLABS_API_KEY=<key> node synth.mjs
python3 score.py   # needs: pip install faster-whisper jiwer torch torchaudio soundfile
```

## Methodology and honest caveats

- **Latency**: time to first audio byte, warm connection (opened once, reused for every sentence
  in the interleaved run — matches this repo's established methodology), same machine/vantage for
  every engine so network conditions are shared. n=21 per engine (11 for Kokoro, English-only).
- **Intelligibility (WER)**: faster-whisper `base` model (not `small` — this run hit a disk-space
  limit on a shared machine; `base` is somewhat less accurate at transcription itself, which adds
  noise to the WER numbers in both directions, not a bias toward any one engine).
  **Known false-positive pattern, do not over-read English WER from this alone**: Whisper
  transcribes spoken digits as numerals ("6-6-3-5") while our ground-truth text spells them out
  ("six six three five") — this alone accounts for a large chunk of Piper's English WER and is a
  measurement artifact, not an intelligibility problem. A normalizing pass (digits-to-words) would
  fix this; not done here for time, flagged instead.
- **Naturalness (MOS proxy)**: torchaudio's `SQUIM_SUBJECTIVE` model, which needs a reference clip
  (not the same content, just a clean reference). We used one ElevenLabs Multilingual v2 clip per
  language as the shared reference for every engine's score in that language. This makes scores
  comparable to EACH OTHER (same reference), not an absolute MOS truth — and it structurally can't
  be fully fair to ElevenLabs Multilingual itself, since for that engine some clips are scored
  against a same-model, same-voice reference. Treat MOS as directional, most useful for comparing
  Piper/Kokoro against each other or against the general quality band, not as a precise ElevenLabs
  head-to-head number. This is an automated proxy, not a human listener.
- **Cost**: not measured, tabulated from list prices already established elsewhere in this repo
  (see `worker-piper-fly/PUBLIC_DRAFT.md`, `DECISIONS.md`).
- **Spend for this run**: ElevenLabs API calls for ~21 sentences × 2 models ≈ 42 short TTS requests
  (well under $1 at $50-100/1M chars for a few thousand characters total); Piper/Kokoro on our own
  infra; Whisper/SQUIM ran locally, no cloud cost.

See `results/REPORT_DRAFT.md` for the actual measured numbers and the "where to invest next" read.
