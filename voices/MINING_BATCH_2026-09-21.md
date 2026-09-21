# Mining batch: rhasspy/piper-checkpoints, 2026-09-21

Goal: expand language coverage by mining `rhasspy/piper-checkpoints` on Hugging Face for checkpoints not yet in
`voices/catalog.json`. Export-only, catalog-only — nothing here was published to a live registry
(`voices/publish_voice.py` was never invoked).

## Before / after

- Voices: 56 -> 69 (+13)
- Languages: 9 -> 19 (+10 new: cs, da, el, fi, hu, no, ro, sk, uk, vi)

Previously covered: de, en, es, fr, it, nl, pl, pt, ru (unchanged by this batch; no depth added there this round —
this batch prioritized new-language breadth per the task brief).

## Discovery method

Enumerated the top-level language dirs of `datasets/rhasspy/piper-checkpoints` via the HF tree API
(`GET /api/datasets/rhasspy/piper-checkpoints/tree/main`), diffed against the 9 languages already in
`catalog.json`, then fetched each new language's `MODEL_CARD` + `config.json` to classify licence tier (A/B/C per
`voices/LICENSES.md`'s existing rubric) before exporting anything.

44 language dirs exist upstream (`ar bg bn ca cs da de el en es et fi fr he hi hu id it ja ka ko ku lb ml mr ne nl no
pl pt ro ru sk sr sw te th tr uk ur vi zh`); this batch inspected 18 of the 35 not-yet-covered ones (the ones with the
clearest CC0/CC-BY dataset cards) and exported the 13 that passed tier A/B screening. The rest (ar, bg, bn, he, hi, id,
ja, ku, lb, ml, mr, ne, sw, te, th, ur, zh, and Catalan/Korean/Serbian/Turkish/Georgian, which were inspected and
rejected — see below) are left for a future batch.

## Exported voices

All exported via `voices/export_v2.py` (`modal run voices/export_v2.py --only <ids>`), which fetches the ONNX from
`rhasspy/piper-voices`, synthesizes the 5 call-centre sentences (new translations added to `TEXTS` for cs/da/el/fi/hu/
no/ro/sk/uk/vi), measures CPU RTF and median F0, and merges into `catalog.json`. Model weights are **not** committed;
they live in the `house-voices` Modal Volume, same as every other voice in this repo.

| id | language | source checkpoint (piper-voices) | licence | tier | verification |
|---|---|---|---|---|---|
| el-gr-rapunzelina | el_GR | `el/el_GR/rapunzelina/medium` | CC0-1.0 | A | espeak-tag check + audio sanity check |
| uk-ua-ukrainian_tts-lada | uk_UA | `uk/uk_UA/ukrainian_tts/medium` (speaker "lada") | CC0-1.0 | A | espeak-tag check + audio sanity check |
| uk-ua-ukrainian_tts-mykyta | uk_UA | `uk/uk_UA/ukrainian_tts/medium` (speaker "mykyta") | CC0-1.0 | A | espeak-tag check + audio sanity check |
| no-no-nvcc-f | no_NO | `no/no_NO/nvcc/medium` (speaker "KNN") | CC0-1.0 | A | espeak-tag check + audio sanity check |
| no-no-nvcc-m | no_NO | `no/no_NO/nvcc/medium` (speaker "MMN") | CC0-1.0 | A | espeak-tag check + audio sanity check |
| cs-cz-jirka | cs_CZ | `cs/cs_CZ/jirka/medium` | CC0-1.0 | B | espeak-tag check + audio sanity check |
| da-dk-talesyntese | da_DK | `da/da_DK/talesyntese/medium` | CC0-1.0 | B | espeak-tag check + audio sanity check |
| fi-fi-harri | fi_FI | `fi/fi_FI/harri/medium` | CC0-1.0 | B | espeak-tag check + audio sanity check |
| hu-hu-anna | hu_HU | `hu/hu_HU/anna/medium` | CC0-1.0 | B | espeak-tag check + audio sanity check |
| hu-hu-imre | hu_HU | `hu/hu_HU/imre/medium` | CC0-1.0 | B | espeak-tag check + audio sanity check |
| ro-ro-mihai | ro_RO | `ro/ro_RO/mihai/medium` | CC0-1.0 | B | espeak-tag check + audio sanity check |
| sk-sk-lili | sk_SK | `sk/sk_SK/lili/medium` | CC0-1.0 | B | espeak-tag check + audio sanity check |
| vi-vn-vais1000 | vi_VN | `vi/vi_VN/vais1000/medium` | CC-BY-4.0 | B | espeak-tag check + audio sanity check |

Tier B voices carry the same "owner risk decision" flag as the rest of the existing tier-B catalog (Lessac
Blizzard-2013 research-only base lineage) — catalogued, not published, same as before.

### Verification detail

For every voice above:
1. **espeak-tag check (automatic, scripted):** `export_v2.py`'s existing mismatch guard ran for each export and
   printed no WARNING for any of these 13 voices — the shipped `espeak.voice` matched the expected language for
   every checkpoint's directory path. (`no_NO` needed an `EXPECTED_ESPEAK = {"no": "nb"}` override added to the
   script so the legitimate `no_NO -> espeak "nb"` mapping — espeak-ng has no bare `"no"` voice — isn't flagged as
   the known upstream-mislabeling bug; see LICENSES.md.) No instance of the de/de_DE/mls-style Dutch-tag bug was
   found in this batch.
2. **Audio sanity check (this session, no WER framework available):** loaded `sample_1.wav` for all 13 voices and
   checked duration (4.6-7.8s, consistent with the sentence length), RMS energy (2.8k-7.7k on a 16-bit PCM scale —
   nowhere near silence), zero-crossing rate (0.06-0.17, consistent with voiced speech, far from white-noise ZCR
   ~0.5), and the fraction of non-silent 1024-sample frames (all >= 82%, most >90%). All 13 pass as plausible speech,
   not silence or garbage. This is **not** a transcription/WER check — no live gateway/eval key is available in this
   session — so mispronunciation or wrong-language artifacts that still "sound like speech" would not be caught by
   this check alone. Treat these as "sounds like real speech in a plausible register," not "verified correct
   transcript."

## Rejected candidates (tier C, not exported)

Screened but rejected during the licence-tier pass (see `voices/LICENSES.md` for the full table):

| dir | reason |
|---|---|
| `ca/ca_ES/upc_ona` | Dataset CC BY-SA 3.0 ES (ShareAlike) + Lessac lineage |
| `sr/sr_RS/serbski_institut` | Dataset CC BY-NC-SA 4.0 (non-commercial) |
| `tr/tr_TR/dfki` | Dataset CC BY-NC-SA 4.0 (non-commercial) |
| `ko/ko_KR/kss` | Dataset CC BY-NC-SA 4.0 (non-commercial) |
| `ka/ka_GE/natia` | Dataset LICENSE restricts use to individuals/personal use, explicitly prohibits organizations |
| `no/no_NO/talesyntese` | Redundant 2nd Norwegian voice, Lessac lineage; skipped in favour of the tier-A nvcc pair |

## Code changes

- `voices/export_v2.py`: added `TEXTS` entries for cs/da/el/fi/hu/no/ro/sk/uk/vi (new call-centre-sentence
  translations); appended 13 new `SPEC` entries; added an `EXPECTED_ESPEAK` override table and used it in the
  mismatch-detection/auto-correction check (previously a bare `locale.split("_")[0]` comparison, which would have
  false-positived on `no_NO`'s legitimate `"nb"` espeak tag and incorrectly forced it to the nonexistent `"no"`
  espeak voice).
- `voices/catalog.json`: 13 new entries appended by `export_v2.py`'s own catalog-merge logic (same schema as every
  existing entry: id/language/accent/gender/speaker_id/tier/quality/license/attribution/sample/median_f0_hz/cpu_rtf/
  sample_rate/espeak_voice/onnx_mb/volume_path/notes).
- `voices/LICENSES.md`: new section documenting this batch's licence findings and rejections.
- `voices/samples/<id>/`: sample_1..5.wav (tier A) or sample_1,4.wav (tier B) per voice, same convention as existing
  voices.

## Confirmations

- Nothing was pushed to any remote.
- `voices/publish_voice.py` was never invoked.
- Model weights (`.onnx`) are not committed; they live in the `house-voices` Modal Volume only, matching the existing
  convention for every prior voice in this repo.
