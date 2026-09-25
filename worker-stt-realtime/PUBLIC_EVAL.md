# Public human-speech telephone benchmark (2026-09-24)

Follow-up to `REAL_CALL_EVAL.md` / `REAL_CALL_EVAL_METHODOLOGY.md`. Question: is the accuracy gap between our CPU
streaming engines and the cloud engines specific to our (mostly synthetic-caller) test calls, or does it hold on
human speech? Plan: `docs/superpowers/plans/2026-09-24-stt-public-human-speech-eval.md`. Aggregates:
`bench/results/pub_summary.json`; per-set tables: `PUBLIC_EVAL_<set>.md`.

## Data (public, with corpus transcripts; no call data)

| set | what | n | words | notes |
|---|---|---|---|---|
| `pub_fleurs` | FLEURS en-US test, raw | 50 | 1,108 | source audio is very quiet (median peak 0.007, ~ -43 dBFS) |
| `pub_fleurs_norm` | same 50, peak-normalised to 0.7 | 50 | 1,108 | fair "clean read speech" baseline |
| `pub_fleurs_tel` | same 50, telephone-degraded (300-3400 Hz band, 8 kHz, mu-law, 25 dB SNR noise, peak 0.7) | 50 | 1,108 | phone channel on human speech |
| `pub_earnings` | Earnings-22 chunked test, 25 shards x 1 random row group | 50 | 984 | real earnings-call audio, spontaneous, accented; 27 distinct calls |

FLEURS is CC-BY-4.0 (read Wikipedia sentences, many speakers/mics). Earnings-22 is CC-BY-SA-4.0. Used for evaluation only.
Earnings references are raw text (disfluencies like "uh", digit/word mismatches), so its WER is inflated for every engine
and is not comparable to FLEURS. Same engines, versions and paced replay as the real-call evaluation.

## Results: WER (95% CI ~ +/-1.2 to 2.7 points, word-level, ignoring clustering)

| engine | fleurs raw (quiet) | fleurs norm (clean) | fleurs tel (phone) | earnings-22 (real calls) |
|---|---|---|---|---|
| el-scribe | 4.5% | 4.6% (+/-1.2) | 6.6% (+/-1.5) | 13.7% (+/-2.1) |
| dg-flux | 7.7% | 6.9% (+/-1.5) | 12.2% (+/-1.9) | 19.6% (+/-2.5) |
| nemo-480 (ours) | 30.3% | 10.3% (+/-1.8) | 15.6% (+/-2.1) | 25.0% (+/-2.7) |
| zip-en-int8 (ours) | 76.7% | 18.7% | 29.2% | 52.3% |

CPU cost of ours is unchanged: nemo-480 0.084, zipformer 0.083 CPU-seconds per audio-second (dev machine).
Latency numbers from these runs are not comparable (two engine chains ran in parallel on one machine); WER is the result.

## What it shows

1. **Input level matters enormously for our engines.** The same 50 utterances: nemo-480 30.3% raw vs 10.3% peak-normalised,
   zipformer 76.7% vs 18.7%; the cloud engines barely change. Our engines have no gain control. A service built on them
   must normalise/AGC input (the real-call audio was at a healthy level, peak ~0.76, so this did not affect that result).
   (The first run looked paradoxical, local engines better on the phone version than on the "clean" one: the cause was the
   level, found by measuring peaks; the raw set is kept as a quiet-input test.)
2. **The gap holds on human speech, but it is smaller against Deepgram than against ElevenLabs.** nemo-480 vs Deepgram:
   1.5x on clean, 1.3x on phone-degraded, 1.3x on real earnings calls. nemo-480 vs ElevenLabs: 2.2x, 2.4x, 1.8x.
   On the test-persona calls it was 2.1x and 3.3x. So NeMo 480 is roughly in Deepgram Flux's neighbourhood on human speech
   with a proper input level, and clearly behind ElevenLabs Scribe v2 Realtime.
3. **The zipformer is not competitive** (LibriSpeech-only training): 18.7% on clean read speech, 52% on real calls. Drop it.
4. Phone degradation costs every engine ~2-5 points; ElevenLabs is the most robust.

## Caveats

50 utterances / ~1,000 words per set: differences under ~2-3 points (e.g. ElevenLabs vs Deepgram on clean speech) are within
noise. Read speech and earnings calls are not customer phone calls; the degradation is simulated. Earnings references
are raw text. Local numbers are dev-machine CPU, single stream. The quiet-input finding means our earlier LibriTTS
benchmarks (peak-level dependent) should be re-checked for level as well.

## Recommendation for the accuracy investigation

- Cheap and certain: add input normalisation (peak/RMS AGC) in front of the local engine and re-measure.
- Best lever: fine-tune the NeMo streaming model on human speech degraded to phone quality (the tooling in
  `bench/telephony.py` and the TTS-training corpora with transcripts), then re-run this benchmark plus the real-call set.
  A realistic target is Deepgram Flux level (about 1.3x -> 1.0x); ElevenLabs Scribe is the harder target.
- Also worth measuring: NeMo 1040 ms chunk (accuracy vs latency), and any other streaming CPU model that has appeared.
- Get more real human phone speech for the final decision; both evaluations so far are limited (synthetic personas; public
  read/earnings audio).
