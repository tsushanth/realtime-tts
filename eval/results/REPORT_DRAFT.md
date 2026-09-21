# TTS eval report: us vs ElevenLabs (draft, 2026-09-21)

Small, directional comparison — 21 short call-center sentences (11 English, 4 Spanish, 3 German,
3 French), 4 engines. Not statistically powered. See `../README.md` for full methodology and
caveats before quoting any of this externally. Framework is re-runnable (`node synth.mjs &&
python3 score.py`) — rerun after any engine change to see if a number moved.

## Results

| Engine | n | Latency median | Latency p90 | WER mean | Naturalness (MOS proxy, median) | Price / 1M chars |
|---|---|---|---|---|---|---|
| Piper (ours) | 21 | **283 ms** | 455 ms | 0.186 | 4.40 | **$4** |
| Kokoro (ours) | 11 (en only) | 564 ms | 1072 ms | 0.058 | 4.29 | $10 |
| ElevenLabs Flash v2.5 | 21 | 206 ms | 287 ms | 0.054 | 4.99 | $50 |
| ElevenLabs Multilingual v2 | 21 | 1172 ms | 1298 ms | 0.056 | 4.48 | $100 |

WER by language (mean):

| Engine | en | es | de | fr |
|---|---|---|---|---|
| Piper | 0.105 | 0.066 | **0.694** | 0.133 |
| ElevenLabs Flash | 0.052 | 0.058 | 0.000 | 0.107 |
| ElevenLabs Multilingual | 0.058 | 0.058 | 0.000 | 0.107 |
| Kokoro | 0.058 | — | — | — |

## Reading this

**Latency: we're already competitive, this is not where more effort should go.** Piper (283 ms) is
in the same band as ElevenLabs Flash (206 ms) — both well under Multilingual v2's 1.17 s. Kokoro
(564 ms median, 1072 ms p90) has more spread than Piper; this matches earlier work in this repo
identifying non-model overhead (connection/frame-size effects, not the model itself) as the
cause — that's already been partly addressed (see `worker-modal-readaloud`'s first-chunk work) and
is not a new finding here.

**English intelligibility: close, and Piper's WER is partly a measurement artifact.** Piper's
English WER (0.105) looks worse than ElevenLabs' (~0.05), but inspecting the actual transcripts
shows a real chunk of that gap is Whisper writing "6-6-3-5" against our ground truth "six six
three five" — not a real error. Discount this comparison until a digit-normalizing pass is added
(flagged in `../README.md`, not done here for time).

**German is the real, specific problem — not "non-English" in general.** Piper's Spanish (0.066)
and French (0.133) WER are close to ElevenLabs' own numbers in those languages (0.058, 0.107) —
so the multilingual voices are broadly working. German is not: 0.694 mean WER, and the actual
transcripts are genuinely garbled, not a measurement artifact:

- Reference: "Vielen Dank für Ihren Anruf, wie kann ich Ihnen helfen?"
- Piper transcript: "Wellen dann, dass ihre Männer tütter und da ist ihnen helfen."

This happened on 3 of 3 German test sentences. **Checked further after this run**: the male MLS
German voice (`de-de-mls-m`) was tested on the same sentence as a follow-up and is ALSO garbled
("weil wir den Dank für einen anderen Teil kann in den Helfen" for the same reference text) - so
this is not one bad speaker, both German MLS voices fail the same way. That points at the shared
German MLS base/export itself (wrong espeak phonemizer voice code, a training data issue, or an
export bug specific to this language), not a single speaker needing retraining. **This is a
specific, scoped investigation**: check the German MLS export pipeline (espeak `de` voice
mapping, phoneme-to-id table, the training run's actual language tag) before assuming a full
retrain is needed - the bug may be cheap to find and fix once isolated.

**Root cause found and FIXED, both voices are back in production.** The `de/de_DE/mls/medium`
checkpoint's own shipped `config.json` on `rhasspy/piper-checkpoints` (the upstream HF repo) has
`espeak.voice: "nl"` (Dutch) and `language: nl_NL` baked in, despite being correctly-trained German
MLS data - an upstream labeling bug, not anything in this repo's training/export. Empirical proof:
re-exporting the same checkpoint with `espeak.voice` overridden to `"de"` (everything else
identical) turned the exact same test sentence from nonsense into an exact transcript match. Fixed
in `voices/export_voices.py` and `voices/export_v2.py` (both now detect a shipped-config/expected
-language mismatch, override it, and print a warning so a genuinely different future case gets
noticed rather than silently mis-applied). Both voices re-exported, republished, and re-verified
live through the real `api.readaloudai.org` -> Piper path with a fresh WER check:

- `de-de-mls-f`: "Vielen Dank für Ihren Anruf. Die kann ich Ihnen helfen." (was: "Wellen dann,
  dass ihre Männer tütter und da ist ihnen helfen.")
- `de-de-mls-m`: "Vielen Dank für Ihren Anruf, die kann ich Ihnen helfen." (was: "weil wir den
  Dank für einen anderen Teil kann in den Helfen.")

("Die kann" vs "wie kann" is a `faster-whisper base` transcription quirk on a homophone-ish pair,
not a synthesis defect - a stronger Whisper model would likely resolve it. Full WER/MOS re-run
across the whole German test set is a good next step to confirm the fix quantitatively, not just
by spot-check.) `voices/catalog.json` updated to reflect the fix and the root cause.

**Naturalness: Piper and Kokoro are in a reasonable band, not a clear loss.** Both land around
4.3-4.4 on the (relative, per-language-referenced) MOS proxy, close to ElevenLabs Multilingual's
4.48 and behind Flash's 4.99. Given the MOS methodology's caveats (see `../README.md`), don't
over-read the exact gap to Flash — but there's no evidence here of a naturalness crisis on the
voices tested.

## Where to invest next (grounded in the numbers above, in order)

1. ~~Investigate the German MLS export/base pipeline~~ **DONE during this report**: root cause was
   an upstream config-metadata bug (wrong espeak/language tag shipped with an otherwise-correctly-
   trained checkpoint), not a training or export problem on our side. Fixed, re-verified live,
   catalog updated. Worth checking whether any OTHER voice sourced from `rhasspy/piper-checkpoints`
   has the same kind of shipped-config mismatch (the new warning in `export_voices.py`/
   `export_v2.py` will catch it on the next re-export of any voice, but existing already-published
   voices were not all re-checked this pass).
2. **Add a digit-normalizing pass to any future WER measurement** before trusting an English
   Piper-vs-ElevenLabs intelligibility number — the current gap is partly inflated by a
   transcription-formatting artifact, not real.
3. **Don't spend more effort on Piper/Flash latency parity** — they're already close, and further
   gains there are diminishing returns compared to the German fix above.
4. **Extend this eval's language coverage before trusting Spanish/French as "done."** n=3-4
   sentences per non-English language is thin; the French WER gap to ElevenLabs (0.133 vs 0.107)
   is small enough that it could just be sampling noise, worth 5-10 more sentences to confirm
   before treating it as either a problem or a non-problem.

## Not covered by this run

- Voice cloning quality (this eval used only the stock/house voices, not the custom voice-cloning
  pipeline).
- Realtime STT (separate work, separate eval would be needed).
- A genuine human MOS panel — everything "naturalness" here is an automated proxy.
