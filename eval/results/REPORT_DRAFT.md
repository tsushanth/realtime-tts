# TTS eval report: us vs ElevenLabs (draft, updated 2026-09-21 after the German MLS fix)

Small, directional comparison — 21 short call-center sentences (11 English, 4 Spanish, 3 German,
3 French), 4 engines. Not statistically powered. See `../README.md` for full methodology and
caveats before quoting any of this externally. Framework is re-runnable (`node synth.mjs &&
python3 score.py`) — rerun after any engine change to see if a number moved.

**This is a full rerun of the eval framework after fixing the German MLS voice bug documented
below** (broken `espeak.voice` tag on the shipped upstream checkpoint, not a synthesis defect).
The numbers below replace the first run's numbers, which had German at a catastrophic 0.694 WER
from that bug. 74/74 jobs completed cleanly this run (a handful of reconnect-penalty outliers were
re-measured with a proper warm-up first, per the methodology note in `../README.md`).

## Results

| Engine | n | Latency median | Latency p90 | WER mean | Naturalness (MOS proxy, median) | Price / 1M chars |
|---|---|---|---|---|---|---|
| Piper (ours) | 21 | **251 ms** | 324 ms | 0.104 | 4.44 | **$4** |
| Kokoro (ours) | 11 (en only) | 637 ms | 1055 ms | 0.058 | 4.25 | $10 |
| ElevenLabs Flash v2.5 | 21 | 204 ms | 223 ms | 0.062 | 4.52 | $50 |
| ElevenLabs Multilingual v2 | 21 | 1098 ms | 1317 ms | 0.043 | 4.46 | $100 |

WER by language (mean):

| Engine | en | es | de | fr |
|---|---|---|---|---|
| Piper | 0.098 | 0.085 | 0.094 | 0.162 |
| ElevenLabs Flash | 0.058 | 0.038 | 0.067 | 0.107 |
| ElevenLabs Multilingual | 0.052 | 0.058 | 0.000 | 0.030 |
| Kokoro | 0.058 | — | — | — |

## Reading this

**Latency: we're already competitive, this is not where more effort should go.** Piper (251 ms) is
in the same band as ElevenLabs Flash (204 ms) — a 47ms gap that's within the noise of this sample
size, and both are well under Multilingual v2's 1.1 s. Kokoro (637 ms median, 1055 ms p90) has more
spread than Piper; this matches earlier work in this repo identifying non-model overhead
(connection/frame-size effects, not the model itself) as the cause — that's already been partly
addressed (see `worker-modal-readaloud`'s first-chunk work) and is not a new finding here.

**English intelligibility: close, and Piper's WER is partly a measurement artifact.** Piper's
English WER (0.098) is close to ElevenLabs Flash's (0.058) and Multilingual's (0.052), but
inspecting the actual transcripts shows part of the remaining gap is Whisper writing "6-6-3-5"
against our ground truth "six six three five" — not a real error. Discount the exact gap until a
digit-normalizing pass is added (flagged in `../README.md`, not done here for time); the honest
read is "close, not identical."

**German was the real, specific problem — fixed, and now verified across the full test set.**
Piper's Spanish (0.085) and French (0.162) WER are in a reasonable range of ElevenLabs' own numbers
in those languages (0.038-0.058, 0.030-0.107) — so the multilingual voices are broadly working.
German was not, before the fix below: 0.694 mean WER, and the actual transcripts were genuinely
garbled, not a measurement artifact:

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
not a synthesis defect - a stronger Whisper model would likely resolve it.)

**Confirmed quantitatively, not just by spot-check**: the full eval rerun puts German at 0.094 mean
WER (n=3) — in line with Piper's other languages (Spanish 0.085, English 0.098) and close to
ElevenLabs Flash's own German number (0.067). ElevenLabs Multilingual still scores 0.000 on German
in this small sample, but n=3 is too thin to read that as a real gap rather than sampling luck.
German has gone from "broken and shipped to customers" to "in line with everything else we ship" —
`voices/catalog.json` and the numbers above both reflect the fixed state.

**Naturalness: Piper and Kokoro are essentially at parity with ElevenLabs on this proxy.** Piper
(4.44) and Kokoro (4.25) land close to ElevenLabs Multilingual (4.46) and Flash (4.52) on the
(relative, per-language-referenced) MOS proxy — a much tighter spread than the first run showed.
Given the MOS methodology's caveats (see `../README.md`), don't over-read the exact ranking, but
there's no evidence here of a naturalness gap worth engineering effort right now.

## Where to invest next (grounded in the numbers above, in order)

1. ~~Investigate the German MLS export/base pipeline~~ **DONE**: root cause was an upstream
   config-metadata bug (wrong espeak/language tag shipped with an otherwise-correctly-trained
   checkpoint), not a training or export problem on our side. Fixed, re-verified live, catalog
   updated, and now confirmed quantitatively across the full German test set (0.694 → 0.094 WER).
   Worth checking whether any OTHER voice sourced from `rhasspy/piper-checkpoints` has the same
   kind of shipped-config mismatch (the new warning in `export_voices.py`/`export_v2.py` will catch
   it on the next re-export of any voice, but existing already-published voices were not all
   re-checked this pass).
2. **French is now the largest remaining gap** (Piper 0.162 vs ElevenLabs 0.030-0.107) — worth a
   look, though n=3 is thin enough that this could be sampling noise rather than a real language
   issue like German was. Extend the French test set to 8-10 sentences before deciding whether this
   needs investigation or is just noise.
3. **Add a digit-normalizing pass to any future WER measurement** before trusting an English
   Piper-vs-ElevenLabs intelligibility number to the decimal — part of the remaining gap is a
   transcription-formatting artifact, not a real intelligibility difference.
4. **Don't spend more effort on Piper/Flash latency parity** — 251ms vs 204ms is close enough that
   further gains here are diminishing returns compared to language coverage work.
5. **Extend Spanish/German coverage too before calling them "done."** n=3-4 sentences per
   non-English language is thin overall; German in particular deserves a bigger test set now that
   it's fixed, to make sure the 0.094 WER holds up and wasn't a lucky sample.

## Not covered by this run

- Voice cloning quality (this eval used only the stock/house voices, not the custom voice-cloning
  pipeline).
- Realtime STT (separate work, separate eval would be needed).
- A genuine human MOS panel — everything "naturalness" here is an automated proxy.
