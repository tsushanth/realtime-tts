# Piper voice licence and lineage audit (v2, 2026-09-20)

Engineering audit, not legal advice. Sources: HuggingFace `rhasspy/piper-checkpoints` (dataset repo: MODEL_CARD, config.json,
train.sh, LICENSE per voice) and `rhasspy/piper-voices` (model repo: onnx + cards), plus each linked dataset page, fetched
2026-09-20 (piper-voices head c10ece1a). Evidence and URL are in each row. "Trained from scratch" statements are author
assertions in the cards; weights were not inspected.

## Tiers

- **A** clean data licence (public domain, CC0, CC BY) AND scratch / clean-base lineage. Publishable. CC BY needs attribution
  (the exact string is in `voices/catalog.json` and each voice's `owner.json`).
- **B** clean data licence BUT weights are finetuned from Piper's Lessac checkpoint (Blizzard 2013 licence: "Research
  Purposes only", excludes any commercial purpose, covers things "based upon the Materials", no sub-licensing:
  https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html) or from a base of unknown lineage.
  Whether finetuned weights are "based upon" the data is untested; the cards do not address it. **Owner risk decision.**
  Exported and catalogued, never published by the bulk script without `--allow-tier-b`.
- **C** non-commercial, share-alike, restrictive, unknown/unverifiable data licence, or weights descended from NC data
  (all `*-low` voices descend from Ryan-low, trained on CC BY-NC-SA RyanSpeech). Not exported.

## Corrections vs the first audit

1. **VCTK (en_GB, 109 speakers) is not from scratch**: both MODEL_CARDs say "Finetuned from U.S. English lessac voice
   (medium quality)". Data is CC BY 4.0 (Edinburgh DataShare 10283/3443). Tier B. Every accent voice (Indian, Australian,
   Canadian, NZ, ...) therefore inherits the Lessac question.
2. **libritts_r has two models**: `piper-checkpoints/.../libritts_r/medium/best.ckpt` (replaced 2026-09-17) is scratch-trained
   (MRD+SDP, Piper >= 1.5, byte-identical to `_base_model/base_model.ckpt`) = tier A; the onnx in `piper-voices` is the 2023
   Lessac-finetune = tier B. We export only from the ckpt.
3. **Every `*-low` voice** is finetuned from Ryan-low (NC) = tier C, including the es/fr/nl/pl MLS single-speaker lows and
   pt edresson (clean CC BY data, C by lineage).
4. Missing before, now covered: en_GB cori (scratch, LibriVox PD, tier A), en_US norman (A), libritts high (A), nl_BE rdh
   (A), nl_NL alex (A, chain per card only), es_ES carlfm x_low (A, weak), mimic3-voices repo is CC BY-SA 4.0 (amy/danny/alan = C).
5. `en_GB/aru`: the LICENSE file in the checkpoint dir is a CMU/Festvox notice, not the CC BY 4.0 the card claims, and the
   Liverpool source URL 404s: data licence unverifiable, treated as C. `northern_english_male` is CC BY-SA 4.0 (OpenSLR 83):
   C (a hosted service arguably does not "distribute" the weights, but audio outputs and any distributed model would carry SA
   uncertainty; not worth it while alternatives exist).
6. id_ID news_tts is CC BY-NC-SA (source README): C. Piper has no Australian, Canadian, Indian, Singapore, NZ English, no
   fr_CA, es_CO, pt_PT voice at all.

## What the pinned speakers of multi-speaker voices are

`voices/catalog.json` lists each exported voice; MLS (de/fr/nl) voices are one model with a pinned `speaker_id` (chosen by a
listening-proxy survey, `voices/speaker_survey.py`: median F0 for gender, UTMOS22 as a rough naturalness proxy, speaking rate;
UTMOS is English-trained so treat the picks as shortlists to audition by ear). VCTK accent labels come from the corpus's own
`speaker-info.txt` (`voices/data/`); note the model phonemizes everything with en-gb-x-rp, so only timbre and prosody carry the accent.

## English (en_US, en_GB)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| en_GB-alan-low | en_GB | 1 | low / 16000 | en-gb-x-rp | same | same | Finetuned from Ryan low | **C** |  |
| en_GB-alan-medium | en_GB | 1 | medium / 22050 | en-gb-x-rp | MycroftAI mimic3-voices en_UK/apope_low | 'See URL'; repo CC BY-SA 4.0 | Finetuned from Lessac medium | **C** |  |
| en_GB-alba-medium | en_GB | 1 | medium / 22050 | en-gb-x-rp | CSTR Alba speech corpus (Edinburgh DataShare 10283/3270) | CC BY 4.0 (DataShare dc.rights + license_text.txt fetched) + moral-rights clause: 'Any use derogatory to the voice talent is prohibited' | Finetuned from Lessac medium | **B** | Attribution + moral-rights restriction (Scottish female voice) |
| en_GB-aru-medium | en_GB | 12 | medium / 22050 | en-gb-x-rp | Liverpool ARU speech corpus | Card: CC BY 4.0 (creativecommons.org/licenses/by/4.0/). Source URL (liverpool.ac.uk/.../speech-corpus/) returns 404. The 'LICENSE' file in the checkpoint dir is a CMU/Festvox notice, not CC BY. -> UNVERIFIED | Finetuned from Lessac medium | **B (data licence unverified -> treat as C until confirmed)** | 12 speakers (ids 01..12) |
| en_GB-cori-high | en_GB | 1 | high / 22050 | en | LibriVox (~24h) | LibriVox recordings, public domain (card); Bryce Beattie page https://brycebeattie.com/files/tts/ 'License: public domain' (fetched) | Scratch, 500 epochs (card) | **A** | espeak 'en' |
| en_GB-cori-medium | en_GB | 1 | medium / 22050 | en | LibriVox (~24h, Bryce-curated) | LibriVox recordings, public domain (card); Bryce Beattie page https://brycebeattie.com/files/tts/ 'License: public domain' (fetched) | Scratch, 640 epochs (piper-voices card) | **A** | Config espeak voice is 'en' (not en-gb-x-rp). Ckpt name cori-med-640.ckpt |
| en_GB-jenny_dioco-medium | en_GB | 1 | medium / 22050 | en-gb-x-rp | Jenny (Dioco) 30h (github.com/dioco-group/jenny-tts-dataset) | Custom: 'Attribution is required in software/websites/... that generate audio ... voice must be referred to as "Jenny" / "Jenny (Dioco)" ... Commercial use is permitted ... No further restrictions' (LICENSE in ckpt dir + repo README, fetched). Not CC. | Finetuned from Lessac medium per rhasspy card. CONFLICT: Bryce's page says 'Trained from scratch for 287 epochs'; ckpt name epoch=2748 = 2461+287, consistent with a resume from a ~2461-epoch base, supporting the card | **B** | Mandatory in-product attribution as 'Jenny (Dioco)'. Speculative epoch arithmetic - flag |
| en_GB-northern_english_male-medium | en_GB | 1 | medium / 22050 | en-gb-x-rp | OpenSLR 83 (UK/Ireland dialects) | CC BY-SA 4.0 (openslr.org/83 'Attribution-ShareAlike 4.0 International', fetched) | Finetuned from Lessac medium | **C** | ShareAlike |
| en_GB-semaine-medium | en_GB | 4 | medium / 22050 | en-gb-x-rp | DFKI SEMAINE (marytts/dfki-semaine-data) | CC BY-NC-SA 4.0 (card) | Finetuned from Lessac medium | **C** | NC; speakers prudence, spike, obadiah, poppy |
| en_GB-southern_english_female-low | en_GB | 1 | low / 16000 | en-gb-x-rp | OpenSLR 83 | CC BY-SA 4.0 | Finetuned from Ryan low | **C** | SA + NC-derived |
| en_GB-vctk-medium | en_GB | 109 | medium / 22050 | en-gb-x-rp | CSTR VCTK 0.92 (Edinburgh DataShare 10283/3443) | CC BY 4.0 (DataShare dc.rights 'Creative Commons Attribution 4.0 International Public License' + license_text.txt fetched). Abstract: newspaper texts from Herald Glasgow 'with permission from Herald & Times Group' | Finetuned from Lessac medium (piper-checkpoints + piper-voices MODEL_CARD: 'Finetuned from U.S. English lessac voice (medium quality)'). train.sh has no scratch/resume flags; not from scratch. | **B** | 109 speakers (speaker_id_map p2xx etc.); config espeak en-gb-x-rp. Prior belief 'scratch' is WRONG |
| en_US-amy-low | en_US | 1 | low / 16000 | en-us | MycroftAI/mimic3-voices | same | Finetuned from Ryan low | **C** | NC-derived lineage |
| en_US-amy-medium | en_US | 1 | medium / 22050 | en-us | MycroftAI/mimic3-voices | 'See URL'; repo LICENSE = CC BY-SA 4.0 (GitHub API license key cc-by-sa-4.0, LICENSE text 'Attribution-ShareAlike 4.0 International', fetched); README says nothing per voice | Finetuned from Lessac medium | **C** | SA + unclear recording rights |
| en_US-arctic-medium | en_US | 18 | medium / 22050 | en-us | CMU ARCTIC | Card says 'See LICENSE file' but the arctic dir has NO LICENSE file. festvox page only says texts are out-of-copyright. (The LICENSE in en_GB/aru is a CMU 2003 permissive notice - probably the intended one, misfiled.) UNVERIFIED | Finetuned from Lessac medium | **C (probably B if permissive CMU licence confirmed)** |  |
| en_US-bryce-medium | en_US | 1 | medium / 22050 | en | Author's own recordings (~750 utts) | public domain (card + Bryce page) | Finetuned 1000 epochs from an 'unreleased voice' with 2500 epochs (card) -> base unknown | **B** | Base unknowable; cannot confirm scratch |
| en_US-danny-low | en_US | 1 | low / 16000 | en-us | MycroftAI/mimic3-voices | same | Finetuned from Ryan low | **C** |  |
| en_US-hfc_female-medium | en_US | 1 | medium / 22050 | en-us | Hi-Fi Captain (NICT) | CC BY-NC-SA 4.0 | Finetuned from Lessac medium | **C** | NC |
| en_US-hfc_male-medium | en_US | 1 | medium / 22050 | en-us | Hi-Fi Captain (NICT) | CC BY-NC-SA 4.0 | Finetuned from Lessac medium | **C** | NC |
| en_US-joe-medium | en_US | 1 | medium / 22050 | en-us | OHF-Voice/voice-datasets en_US joe | CC0 (README: 'licensed under CC0 (public domain)', fetched) | Finetuned from Lessac medium (card) | **B** |  |
| en_US-john-medium | en_US | 1 | medium / 22050 | en | LibriVox (~12.5h) | LibriVox recordings, public domain (card); Bryce Beattie page https://brycebeattie.com/files/tts/ 'License: public domain' (fetched) | Finetuned from Kristin (card), Kristin from scratch | **A** | Chain clean: john -> kristin -> scratch |
| en_US-kathleen-low | en_US | 1 | low / 16000 | en-us | rhasspy/dataset-voice-kathleen (CMU Arctic prompts) | CC0 per card; repo README fetched has NO licence line (GitHub API licence: none) -> unverified. OHF README lists kathleen as CC0 | Finetuned from Ryan low | **C** | Lineage NC-derived |
| en_US-kristin-medium | en_US | 1 | medium / 22050 | en | LibriVox (Bryce-curated, ~11.5h) | LibriVox recordings, public domain (card); Bryce Beattie page https://brycebeattie.com/files/tts/ 'License: public domain' (fetched) | Scratch, 2000 epochs (card) | **A** | Individual reader list not published; PD claim rests on LibriVox catalogue policy + Bryce's assertion |
| en_US-kusal-medium | en_US | 1 | medium / 22050 | en-us | MycroftAI/mimic2 | 'See URL'; repo is Apache-2.0 code, recordings unaddressed | Finetuned from Lessac medium | **C** |  |
| en_US-l2arctic-medium | en_US | 24 | medium / 22050 | en-us | L2-ARCTIC | CC BY-NC 4.0 | Finetuned from Lessac medium | **C** | NC |
| en_US-lessac-high | en_US | 1 | high / 22050 | en-us | Blizzard 2013 Lessac | same | Scratch | **C** |  |
| en_US-lessac-low | en_US | 1 | low / 16000 | en-us | Blizzard 2013 Lessac | Research-only licence https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html | Scratch | **C** | Base of most others. 16kHz |
| en_US-lessac-medium | en_US | 1 | medium / 22050 | en-us | Blizzard 2013 Lessac | same | Scratch | **C** |  |
| en_US-libritts-high | en_US | 904 | high / 22050 | en-us | LibriTTS train-clean-360 | CC BY 4.0 (https://www.openslr.org/60/ fetched) | Scratch (piper-voices card 'Trained from scratch on train-clean-360') | **A** | CC BY 4.0 attribution required. onnx only (no ckpt). Not LibriTTS-R. Uses old-style phonemes (works with all Piper versions) |
| en_US-libritts_r-medium | en_US | 904 | medium / 22050 | en-us | LibriTTS-R train-clean-360 | CC BY 4.0 (https://www.openslr.org/141/ fetched) | CHECKPOINT (piper-checkpoints, 2026-09-17): scratch, MRD+SDP, vowel clusters, Piper>=1.5 (card; best.ckpt sha256 e0f93367... identical to _base_model/base_model.ckpt). ONNX in piper-voices is the OLD 2023-09-23 model whose card says 'Fine-tuned from English lessac medium' -> B | **A (ckpt) / B (piper-voices onnx)** | DO NOT use piper-voices onnx if you need clean lineage; export from best.ckpt. Prior audit (scratch) was true only for the new ckpt. Piper>=1.5 required for new model |
| en_US-ljspeech-high | en_US | 1 | high / 22050 | en | LJ Speech | public domain (keithito.com/LJ-Speech-Dataset: 'License Public Domain', fetched) | Scratch (card) | **A** | Card text says 'medium quality settings' in body but dir/config are high; trust config quality=high |
| en_US-ljspeech-medium | en_US | 1 | medium / 22050 | en | LJ Speech | public domain (keithito.com/LJ-Speech-Dataset: 'License Public Domain', fetched) | Scratch (card: 'Trained from scratch for 1000 epochs') | **A** |  |
| en_US-mike-medium | en_US | 1 | medium / 22050 | en-us | OHF-Voice/voice-datasets (mike) | CC0 per card. NOTE: 'mike' is NOT in the OHF README table I fetched (only joe, kathleen listed for en_US) -> dataset licence unverified at source | Finetuned from Lessac medium (card) | **B** | Added 2026-07-16. Checkpoint name has val_mos=4.2686 |
| en_US-norman-medium | en_US | 1 | medium / 22050 | en | LibriVox (~15.5h) | LibriVox recordings, public domain (card); Bryce Beattie page https://brycebeattie.com/files/tts/ 'License: public domain' (fetched) | Scratch, 1200 epochs (piper-voices card + Bryce page) | **A** | No checkpoint published ('forgot to save the ckpt'); onnx only in piper-voices |
| en_US-reza_ibrahim-medium | en_US | 1 | medium / 22050 | en-us | HF mah92/Hedayatfar-Persian-Quran-Audio-Dataset + Ibrahim-Walk-English-Quran-Audio-Dataset | CC0 per card (HF dataset pages not fetched) | Finetuned from Lessac medium (card) | **B** | Quran recitation content; Persian+English bilingual, not a general en_US voice. Dataset licences unverified |
| en_US-ryan-high | en_US | 1 | high / 22050 | en-us | RyanSpeech | CC BY-NC-SA 4.0 | Scratch | **C** | NC |
| en_US-ryan-low | en_US | 1 | low / 16000 | en-us | RyanSpeech | CC BY-NC-SA 4.0 | Scratch | **C** | NC; base of all *-low voices |
| en_US-ryan-medium | en_US | 1 | medium / 22050 | en-us | RyanSpeech | CC BY-NC-SA 4.0 | Finetuned from Lessac medium | **C** | NC |
| en_US-sam-medium | en_US | 1 | medium / 22050 | en-us | Sam Accenture non-binary voice files | Apache-2.0 per card (URL to apache.org text). Source repo github.com/Sam-Accenture-Non-Binary-Voice/non-binary-voice-files returned 404 -> UNVERIFIED | Finetuned from Lessac medium (card) | **B** | Data licence unverifiable at source; treat B-unverified |

## Spanish (es_ES, es_MX, es_AR)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| es_AR-daniela-high | es_AR | 1 | high / 22050 | es-419 | OpenSLR 61 (Argentinian Spanish female) | CC BY-SA 4.0 (openslr.org/61 'Attribution-ShareAlike 4.0 International', fetched) | Finetuned from Lessac high (card, links ckpt epoch=2218) | **C** | SA; only Argentinian voice. espeak es-419 |
| es_ES-carlfm-x_low | es_ES | 1 | x_low / 16000 | es | carlfm01/my-speech-datasets (Spanish single speaker) | 'Public domain' (repo README '## License Public domain', fetched). Underlying audio source not stated | Scratch (card) | **A (weak)** | x_low, 16 kHz. PD is uploader-asserted; original source of audio unstated -> low confidence |
| es_ES-davefx-medium | es_ES | 1 | medium / 22050 | es | OHF-Voice/voice-datasets es_ES 'dave' | CC0 (OHF README, fetched) | Finetuned from Lessac medium | **B** |  |
| es_ES-mls_10246-low | es_ES | 1 | low / 16000 | es | MLS Spanish speaker 10246 | CC BY 4.0 | Finetuned from Ryan low | **C** | NC-derived lineage |
| es_ES-mls_9972-low | es_ES | 1 | low / 16000 | es | MLS Spanish speaker 9972 | CC BY 4.0 | Finetuned from Ryan low | **C** | NC-derived lineage |
| es_ES-sharvard-medium | es_ES | 2 | medium / 22050 | es | Sharvard (Edinburgh DataShare 10283/574), speakers M and F | CC BY 3.0 Unported (DataShare dc.rights: 'License: Creative Commons Attribution 3.0 Unported-CC-BY', fetched) | Finetuned from Lessac medium | **B** | Multi-speaker (2): speaker_id_map M/F; needs speaker_id support |
| es_MX-ald-medium | es_MX | 1 | medium / 22050 | es-419 | Ald Mexican Spanish (HF rmcpantoja/Ald_Mexican_Spanish_speech_dataset, 535 files) | Unlicense (HF dataset card 'license: unlicense', fetched) | Finetuned from Spanish davefx medium -> Lessac | **B** | espeak es-419 |
| es_MX-ald-x_low | es_MX | 1 | x_low / 22050 | es-419 | Synthetic: ~8h TTS output of es_MX-ald-medium on Tatoeba sentences | Synthetic; Tatoeba text is CC BY 2.0 FR / CC0 mix (not verified); no explicit licence in card | Scratch on data generated by ald-medium (Lessac-descended) -> transitive B | **B** | Added 2026-06-15; distillation of a B voice; card gives no licence. Tatoeba attribution unverified |
| es_MX-claude-high | es_MX | 1 | high / 22050 | es-419 | HirCoir/Piper-TTS-Spanish HF space | Apache-2.0 per card; 'See URL' only. Space page content not verifiable | Unknown ('See URL above') | **C** | Data source and base undisclosed |

## French (fr_FR)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| fr_FR-gilles-low | fr_FR | 1 | low / 16000 | fr | Kaggle bryanpark french-single-speaker-speech-dataset | CC0 per card (Kaggle page not verified) | Finetuned from Ryan low | **C** | NC-derived |
| fr_FR-mls-medium | fr_FR | 125 | medium / 22050 | fr | Multilingual LibriSpeech French (OpenSLR 94) | CC BY 4.0 (openslr.org/94 'License: CC BY 4.0', fetched) | Scratch (card 'Trained from scratch') | **A** | 125 speakers in config (card says 125; prior audit assumed fewer). Attribution required |
| fr_FR-mls_1840-low | fr_FR | 1 | low / 16000 | fr | MLS French speaker 1840 | CC BY 4.0 | Finetuned from Ryan low | **C** | NC-derived |
| fr_FR-siwis-low | fr_FR | 1 | low / 16000 | fr | SIWIS | CC BY 4.0 | Finetuned from Ryan low | **C** | NC-derived |
| fr_FR-siwis-medium | fr_FR | 1 | medium / 22050 | fr | SIWIS (Edinburgh DataShare 10283/2353) | CC BY 4.0 (dc.rights 'Creative Commons Attribution 4.0 International Public License'; README.txt 'usable for any purpose under CC BY 4.0', fetched) | Finetuned from Lessac medium | **B** |  |
| fr_FR-tom-medium | fr_FR | 1 | medium / 44100 | fr | git.bksp.space/Tjiho newspaper-sentences + baudelaire-sentences (44.1 kHz) | Repo LICENSE = AGPL-3.0 (fetched). Speaker-audio provenance not disclosed (README lists only sentence sources) | 'See URL' - undisclosed | **C** | AGPL applied to models; recording rights unknown |
| fr_FR-upmc-medium | fr_FR | 2 | medium / 22050 | fr | UPMC Pierre (+Jessica) marytts/upmc-pierre-data | CC BY-SA 4.0 (card; GitHub licence cc-by-sa-4.0) | Finetuned from Lessac medium | **C** | SA; card says 2 speakers |

## Portuguese (pt_BR)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| pt_BR-cadu-medium | pt_BR | 1 | medium / 22050 | pt-br | OHF-Voice pt_BR cadu | CC0 (OHF README) | Finetuned from Lessac medium | **B** | Added 2025-04 |
| pt_BR-edresson-low | pt_BR | 1 | low / 16000 | pt-br | Edresson/TTS-Portuguese-Corpus | CC BY 4.0 (README: 'available under CC BY 4.0', fetched) | Finetuned from Ryan low | **C** | NC-derived lineage; data itself is clean |
| pt_BR-faber-medium | pt_BR | 1 | medium / 22050 | pt-br | OHF-Voice/voice-datasets pt_BR faber | CC0 (OHF README) | Finetuned from Lessac medium | **B** | espeak pt-br |
| pt_BR-jeff-medium | pt_BR | 1 | medium / 22050 | pt-br | OHF-Voice pt_BR jeff | CC0 (OHF README) | Finetuned from Lessac medium | **B** | Added 2025-04 |

## Polish (pl_PL)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| pl_PL-bass-high | pl_PL | 1 | high / 22050 | pl | '~10,000 segments of Polish speech' - source unnamed | Card/HF model tagged Apache-2.0 but that is mislabelled as inherited from lessac; NO dataset licence stated | Finetuned from en_US-lessac-high (card; HF page lists 3301 epochs) | **C** | Added 2026-03-09; data unknown |
| pl_PL-darkman-medium | pl_PL | 1 | medium / 22050 | pl | OHF-Voice pl_PL darkman | CC0 (OHF README) | Finetuned from Lessac medium | **B** |  |
| pl_PL-gosia-medium | pl_PL | 1 | medium / 22050 | pl | OHF-Voice pl_PL gosia | CC0 (OHF README) | Finetuned from Lessac medium | **B** |  |
| pl_PL-mc_speech-medium | pl_PL | 1 | medium / 22050 | pl | Kaggle 'The MC Speech Dataset' (czyzi0) | CC0 per card; Kaggle page not fetchable (JS) -> unverified | Finetuned from Lessac medium | **B** | Kaggle page is JS-rendered and could not be read; CC0 claim and recording provenance unverified |
| pl_PL-mls_6892-low | pl_PL | 1 | low / 16000 | pl | MLS Polish speaker 6892 | CC BY 4.0 | Finetuned from Ryan low | **C** | NC-derived |

## Dutch (nl_NL, nl_BE)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| nl_BE-nathalie-medium | nl_BE | 1 | medium / 22050 | nl | rhasspy/dataset-voice-nathalie | CC0 | Finetuned from Lessac medium | **B** |  |
| nl_BE-nathalie-x_low | nl_BE | 1 | x_low / 16000 | nl | rhasspy/dataset-voice-nathalie (1131 utts) | CC0 (README 'licensed under CC0 1.0', fetched) | Scratch (card) | **A** | x_low; only ~1.1k utterances |
| nl_BE-rdh-medium | nl_BE | 1 | medium / 22050 | nl | r-dh/dutch-vl-tts (15k clips, Flemish male, sentences from Mozilla Common Voice nl) | CC0 (repo LICENSE = CC0 1.0 Universal, fetched; GitHub API licence cc0-1.0) | Scratch (card) | **A** | No ckpt published; onnx only. Sentences from Common Voice (CC0) |
| nl_BE-rdh-x_low | nl_BE | 1 | x_low / 16000 | nl | same | CC0 | Scratch (card) | **A** | x_low 16kHz |
| nl_NL-alex-medium | nl_NL | 1 | medium / 22050 | nl | OHF-Voice nl_NL alex (added 2026-02-20) | CC0 (OHF README) | Finetuned from Dutch rdh medium (piper-voices card) -> rdh is scratch/CC0 | **A** | onnx only, no ckpt. Chain clean per card; unverified beyond card |
| nl_NL-mls-medium | nl_NL | 52 | medium / 22050 | nl | MLS Dutch (OpenSLR 94) | CC BY 4.0 (openslr.org/94) | Scratch (card) | **A** | 52 speakers |
| nl_NL-mls_5809-low | nl_NL | 1 | low / 16000 | nl | MLS Dutch 5809 | CC BY 4.0 | Finetuned from Ryan low | **C** | NC-derived |
| nl_NL-mls_7432-low | nl_NL | 1 | low / 16000 | nl | MLS Dutch 7432 | CC BY 4.0 | Finetuned from Ryan low | **C** | NC-derived |
| nl_NL-pim-medium | nl_NL | 1 | medium / 22050 | nl | OHF-Voice nl_NL pim | CC0 | Finetuned from Lessac medium | **B** | Added 2025-04 |
| nl_NL-ronnie-medium | nl_NL | 1 | medium / 22050 | nl | OHF-Voice nl_NL ronnie | CC0 | Finetuned from Lessac medium | **B** | Added 2025-04 |

## Italian (it_IT)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| it_IT-paola-medium | it_IT | 1 | medium / 22050 | it | paolapersico1/Voice-Dataset-Italian (= OHF it_IT paola) | CC0 (HF dataset tags license:cc0-1.0 fetched; OHF README) | Finetuned from Lessac medium | **B** | piper card says 'See URL'; URL confirms CC0 |
| it_IT-riccardo-x_low | it_IT | 1 | x_low / 16000 | it | M-AILABS Italian (Riccardo Fasol) | 'See URL' - caito.de page unreachable -> UNVERIFIED | Scratch (card) | **C (unverified)** | x_low; M-AILABS licence text could not be fetched |
| it_IT-serena-high | it_IT | 1 | high / 22050 | it | same | same | Scratch (card); epoch 18/105k steps | **A (caveats)** | same caveats |
| it_IT-serena-medium | it_IT | 1 | medium / 22050 | it | committa/serena-synthetic-it-27h (SYNTHETIC: Qwen3-TTS-1.7B-Base voice-clone, undisclosed reference clip) | CC BY 4.0 (HF dataset card front matter 'license: cc-by-4.0' + README, fetched). Sentences ~90% Tatoeba (CC BY 2.0 FR, adapted) + ~10% LLM-written | Scratch (card). Only 15 epochs/83k steps for a from-scratch VITS - unusually short; unverifiable | **A (caveats)** | Caveats: (1) reference-voice clip provenance unknown, (2) Qwen3-TTS model output terms not checked, (3) Tatoeba CC BY 2.0 FR attribution. Updated 2026-09-17 in both repos (word-collapse fix) - re-export |

## Indonesian (id_ID)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| id_ID-news_tts-medium | id_ID | 1 | medium / 22050 | id | s-sakti/data_indsp_news_tts (Indonesian news TTS) | CC BY-NC-SA 4.0 (source README: 'You can use the data free for non-commercial purposes', fetched; checkpoints card agrees). NOTE piper-voices card points to an unrelated Kaggle Malayalam page with 'See URL' (stale) | Finetuned from Lessac medium | **C** | Only Indonesian voice; NC |

## Arabic (ar_JO)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| ar_JO-kareem-low | ar_JO | 1 | low / 16000 | ar | same | same | Finetuned from Lessac low (16 kHz) | **C** |  |
| ar_JO-kareem-medium | ar_JO | 1 | medium / 22050 | ar | AliMokhammad/arabicttstrain | 'See URL' - repo has no licence file (GitHub API: none / unreachable) | Finetuned from Lessac medium | **C** |  |

## Hindi (hi_IN)

| voice id | locale | spk | quality / SR | espeak | dataset | dataset licence (+evidence) | base lineage (+evidence) | tier | notes / unverified |
|---|---|---|---|---|---|---|---|---|---|
| hi_IN-pratham-medium | hi_IN | 1 | medium / 22050 | hi | Card cites AI4Bharat indicnlp_corpus (a TEXT corpus) as 'dataset' | CC BY-NC-SA 4.0 (card) | Not stated | **C** | Audio source unknown; NC |
| hi_IN-priyamvada-medium | hi_IN | 1 | medium / 22050 | hi | same | CC BY-NC-SA 4.0 (card) | Not stated | **C** | Audio source unknown; NC |
| hi_IN-rohan-medium | hi_IN | 1 | medium / 22050 | hi | IndicTTS Hindi Mono Male (IIT Madras) | licence PDF iitm.ac.in/donlab/indictts/downloads/license.pdf unreachable (timeout) -> UNVERIFIED (IndicTTS is historically research-only) | Finetuned from Lessac medium | **C** |  |

## Export paths (tier A and B only)

`pv` = huggingface.co/rhasspy/piper-voices (model repo, has onnx + onnx.json); `cp` = huggingface.co/datasets/rhasspy/piper-checkpoints (has .ckpt + config.json). Raw file URL = `https://huggingface.co/rhasspy/piper-voices/resolve/main/<dir>/<file>` and `https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/<dir>/<file>`.

| tier | dir | pv onnx (+ .onnx.json) | cp checkpoint | cp config | note |
|---|---|---|---|---|---|
| A | `en/en_GB/cori/high` | `en_GB-cori-high.onnx` (pv, 2024-03-12, 114.2 MB) | `cori-high-500.ckpt` | config.json |  |
| A | `en/en_GB/cori/medium` | `en_GB-cori-medium.onnx` (pv, 2024-03-22, 63.5 MB) | `cori-med-640.ckpt` | -(none in cp; use pv onnx.json) | cp dir has only cori-med-640.ckpt (config/card in pv) |
| A | `en/en_US/john/medium` | `en_US-john-medium.onnx` (pv, 2024-05-30, 63.5 MB) | `john-2599.ckpt` | config.json |  |
| A | `en/en_US/kristin/medium` | `en_US-kristin-medium.onnx` (pv, 2024-03-12, 63.5 MB) | `kristin-2000.ckpt` | config.json |  |
| A | `en/en_US/libritts/high` | `en_US-libritts-high.onnx` (pv, 2023-06-26, 136.7 MB) | `NONE (onnx only in pv)` | - |  |
| A (ckpt) / B (piper-voices onnx) | `en/en_US/libritts_r/medium` | `en_US-libritts_r-medium.onnx` (pv, 2023-09-23, 78.6 MB) | `best.ckpt` | config.json | cp best.ckpt is the NEW scratch model (needs Piper>=1.5); pv onnx is OLD lessac-finetune (2023) |
| A | `en/en_US/ljspeech/high` | `en_US-ljspeech-high.onnx` (pv, 2024-03-12, 114.2 MB) | `ljspeech-2000.ckpt` | config.json |  |
| A | `en/en_US/ljspeech/medium` | `en_US-ljspeech-medium.onnx` (pv, 2024-03-12, 63.5 MB) | `lj-med_1000.ckpt` | config.json |  |
| A | `en/en_US/norman/medium` | `en_US-norman-medium.onnx` (pv, 2024-05-30, 63.5 MB) | `NONE (onnx only in pv)` | - |  |
| A (weak) | `es/es_ES/carlfm/x_low` | `es_ES-carlfm-x_low.onnx` (pv, 2023-06-27, 28.1 MB) | `NONE (onnx only in pv)` | - |  |
| A | `fr/fr_FR/mls/medium` | `fr_FR-mls-medium.onnx` (pv, 2024-02-03, 76.7 MB) | `epoch=317-step=3124032.ckpt` | config.json |  |
| A (caveats) | `it/it_IT/serena/high` | `it_IT-serena-high.onnx` (pv, 2026-09-17, 114.2 MB) | `epoch=18-step=105450.ckpt` | config.json |  |
| A (caveats) | `it/it_IT/serena/medium` | `it_IT-serena-medium.onnx` (pv, 2026-09-17, 63.5 MB) | `epoch=14-step=83250.ckpt` | config.json |  |
| A | `nl/nl_BE/nathalie/x_low` | `nl_BE-nathalie-x_low.onnx` (pv, 2023-06-26, 20.6 MB) | `NONE (onnx only in pv)` | - |  |
| A | `nl/nl_BE/rdh/medium` | `nl_BE-rdh-medium.onnx` (pv, 2023-06-26, 63.1 MB) | `NONE (onnx only in pv)` | - |  |
| A | `nl/nl_BE/rdh/x_low` | `nl_BE-rdh-x_low.onnx` (pv, 2023-06-26, 20.6 MB) | `NONE (onnx only in pv)` | - |  |
| A | `nl/nl_NL/alex/medium` | `nl_NL-alex-medium.onnx` (pv, 2026-02-20, 63.5 MB) | `NONE (onnx only in pv)` | - |  |
| A | `nl/nl_NL/mls/medium` | `nl_NL-mls-medium.onnx` (pv, 2024-02-03, 76.6 MB) | `epoch=242-step=9245178.ckpt` | config.json |  |
| B | `en/en_GB/alba/medium` | `en_GB-alba-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=4179-step=2101090.ckpt` | config.json |  |
| B (data licence unverified -> treat as C until confirmed) | `en/en_GB/aru/medium` | `en_GB-aru-medium.onnx` (pv, 2023-06-26, 76.8 MB) | `epoch=3479-step=939600.ckpt` | config.json |  |
| B | `en/en_GB/jenny_dioco/medium` | `en_GB-jenny_dioco-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=2748-step=1729300.ckpt` | config.json |  |
| B | `en/en_GB/vctk/medium` | `en_GB-vctk-medium.onnx` (pv, 2023-06-26, 77.0 MB) | `epoch=545-step=1511328.ckpt` | config.json |  |
| B | `en/en_US/bryce/medium` | `en_US-bryce-medium.onnx` (pv, 2024-05-30, 63.5 MB) | `bryce-3499.ckpt` | config.json |  |
| B | `en/en_US/joe/medium` | `en_US-joe-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=7889-step=1221224.ckpt` | config.json |  |
| B | `en/en_US/mike/medium` | `en_US-mike-medium.onnx` (pv, 2026-07-16, 63.2 MB) | `epoch=5460-val_mos=4.2686.ckpt` | config.json |  |
| B | `en/en_US/reza_ibrahim/medium` | `en_US-reza_ibrahim-medium.onnx` (pv, 2025-04-28, 63.5 MB) | `NONE (onnx only in pv)` | - |  |
| B | `en/en_US/sam/medium` | `en_US-sam-medium.onnx` (pv, 2025-05-02, 63.0 MB) | `epoch=4688-step=106008.ckpt` | config.json |  |
| B | `es/es_ES/davefx/medium` | `es_ES-davefx-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=5629-step=1605020.ckpt` | config.json |  |
| B | `es/es_ES/sharvard/medium` | `es_ES-sharvard-medium.onnx` (pv, 2023-06-26, 76.7 MB) | `epoch=4899-step=215600.ckpt` | config.json |  |
| B | `es/es_MX/ald/medium` | `es_MX-ald-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=9999-step=1753600.ckpt` | config.json |  |
| B | `es/es_MX/ald/x_low` | `es_MX-ald-x_low.onnx` (pv, 2026-06-15, 21.0 MB) | `epoch=1953-step=244064.ckpt` | es_MX-ald-x_low.onnx.json |  |
| B | `fr/fr_FR/siwis/medium` | `fr_FR-siwis-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=3304-step=2050940.ckpt` | config.json |  |
| B | `it/it_IT/paola/medium` | `it_IT-paola-medium.onnx` (pv, 2024-05-30, 63.5 MB) | `NONE (onnx only in pv)` | - |  |
| B | `nl/nl_BE/nathalie/medium` | `nl_BE-nathalie-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=6119-step=1806410.ckpt` | config.json |  |
| B | `nl/nl_NL/pim/medium` | `nl_NL-pim-medium.onnx` (pv, 2025-04-26, 63.5 MB) | `epoch=5120-step=100504.ckpt` | config.json |  |
| B | `nl/nl_NL/ronnie/medium` | `nl_NL-ronnie-medium.onnx` (pv, 2025-04-26, 63.0 MB) | `epoch=5105-step=105876.ckpt` | config.json |  |
| B | `pl/pl_PL/darkman/medium` | `pl_PL-darkman-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=4909-step=1454360.ckpt` | config.json |  |
| B | `pl/pl_PL/gosia/medium` | `pl_PL-gosia-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=5001-step=1457672.ckpt` | config.json |  |
| B | `pl/pl_PL/mc_speech/medium` | `pl_PL-mc_speech-medium.onnx` (pv, 2023-10-02, 63.2 MB) | `epoch=2531-step=1906774.ckpt` | config.json |  |
| B | `pt/pt_BR/cadu/medium` | `pt_BR-cadu-medium.onnx` (pv, 2025-04-28, 63.0 MB) | `epoch=5195-step=109116.ckpt` | config.json |  |
| B | `pt/pt_BR/faber/medium` | `pt_BR-faber-medium.onnx` (pv, 2023-06-26, 63.2 MB) | `epoch=6159-step=1230728.ckpt` | config.json |  |
| B | `pt/pt_BR/jeff/medium` | `pt_BR-jeff-medium.onnx` (pv, 2025-04-28, 63.0 MB) | `epoch=5462-step=118728.ckpt` | config.json |  |

## Cannot-verify list (carry-over)

Recording rights behind "public domain" LibriVox voices (kristin, john, norman, cori) rest on LibriVox policy plus Bryce Beattie's
assertion; all "scratch" claims are unverified; unreachable sources: IITM IndicTTS licence PDF, M-AILABS page, Liverpool ARU page,
Sam-Accenture repo, Kaggle pages, AliMokhammad/arabicttstrain; OHF README does not list `mike` (CC0 by card only); ShareAlike
propagation to weights is unsettled (kept in C); CC BY voices need attribution, jenny_dioco requires "Jenny (Dioco)" credit,
alba has a moral-rights no-derogatory-use clause.

## Data-side coverage research (for languages with no tier-A voice)

See `voices/COVERAGE.md`.

## Mining batch 2026-09-21: 10 new languages

Full detail, per-voice verification level and the espeak-tag check outcome: `voices/MINING_BATCH_2026-09-21.md`.
Summary of new licences introduced (none of these categories were previously in this file):

| voice id | locale | dataset | licence (+evidence) | base lineage | tier |
|---|---|---|---|---|---|
| el-gr-rapunzelina | el_GR | bryanpark/greek-single-speaker-speech-dataset (Kaggle) | CC0 (card) | Scratch (card) | **A** |
| uk-ua-ukrainian_tts-lada / -mykyta | uk_UA | OHF-Voice/voice-datasets (ukrainian_tts, 3 speakers) | CC0 (card) | Scratch (card) | **A** |
| no-no-nvcc-f / -m | no_NO | NB Sprakbanken NVCC (nb.no/sprakbanken, oai-nb-no-sbr-75, 10 speakers) | CC0 (card) | Fine-tuned from the scratch-trained, CC BY 4.0 LibriTTS-R base model (card): clean-base lineage, same rule as es-pilot | **A** |
| cs-cz-jirka | cs_CZ | OHF-Voice/voice-datasets (jirka) | CC0 (card) | Finetuned from Lessac medium (card) | **B** |
| da-dk-talesyntese | da_DK | NB Sprakbanken Talesyntese (nb.no/sprakbanken, oai-nb-no-sbr-21) | CC0 (card) | Finetuned from Lessac medium (card) | **B** |
| fi-fi-harri | fi_FI | bryanpark/finnish-single-speaker-speech-dataset (Kaggle) | CC0 (card) | Finetuned from Lessac medium (card) | **B** |
| hu-hu-anna, hu-hu-imre | hu_HU | OHF-Voice/voice-datasets (anna, imre; berta not exported) | CC0 (card) | Finetuned from Lessac medium (card) | **B** |
| ro-ro-mihai | ro_RO | OHF-Voice/voice-datasets (mihai) | CC0 (card) | Finetuned from Lessac medium (card) | **B** |
| sk-sk-lili | sk_SK | OHF-Voice/voice-datasets (lili) | CC0 (card) | Finetuned from Lessac medium (card) | **B** |
| vi-vn-vais1000 | vi_VN | VAIS-1000 Vietnamese Speech Synthesis Corpus (ieee-dataport.org) | CC BY 4.0 (card) | Finetuned from Lessac medium (card) | **B** |

Same "owner risk decision" caveat as every other tier-B voice: weights descend from Piper's Lessac base (Blizzard 2013,
research-only licence). Not published; catalogued only.

**Rejected as tier C in this same batch** (checked, not exported): `ca/ca_ES/upc_ona` (dataset CC BY-SA 3.0 ES, SA +
Lessac lineage); `sr/sr_RS/serbski_institut` and `tr/tr_TR/dfki` (both CC BY-NC-SA 4.0, non-commercial); `ko/ko_KR/kss`
(CC BY-NC-SA 4.0); `ka/ka_GE/natia` (LICENSE file restricts use to individuals/personal use, prohibits organizations —
explicitly non-commercial); `no/no_NO/talesyntese` (redundant second Norwegian voice, Lessac lineage, skipped in favour
of the tier-A nvcc pair).

**espeak/language-tag note (not the known bug class, but adjacent):** `no_NO` checkpoints ship `espeak.voice="nb"`
(Norwegian Bokmål), not `"no"` — this is *correct*, not a mislabeling, because espeak-ng has no bare `"no"` voice.
`export_v2.py`'s mismatch check was extended with an `EXPECTED_ESPEAK` override table (`{"no": "nb"}`) so this
legitimate divergence doesn't get flagged or auto-"corrected" to a nonexistent espeak voice. All other new-language
checkpoints in this batch shipped an espeak.voice that matched their directory language exactly — no instance of the
upstream de/de_DE/mls Dutch-tag bug class was found in this batch.
