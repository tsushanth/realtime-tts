# Clean training-data coverage research (2026-09-20)

Where clean Piper voices do not exist, this lists corpora usable commercially. "V" = licence verified from the source page or
API on 2026-09-20; "R" = recalled/secondary, NOT verified. Rejected outright: Meta MMS-TTS (CC BY-NC), Coqui XTTS (CPML),
any CC BY-NC(-SA/-ND) set (RyanSpeech, Hi-Fi Captain, L2-ARCTIC, DFKI, Data Baker, RUSLAN, news_tts), HF mirrors of M-AILABS
tagged NC-ND, Emilia (in-the-wild YouTube, tag untrustworthy), synthetic-derived sets with unknown reference voices.
CC BY-SA sets (Google/OpenSLR 32/41/44/61-83) are avoided: SA on model weights is legally unsettled.

## Spanish (priority 2): no tier-A Piper voice; pilot corpus = CML-TTS

| Corpus | Hours | SR | Licence | Verdict |
|---|---|---|---|---|
| CML-TTS es (OpenSLR 146, HF ylacombe/cml-tts) | ~477 h by our parquet sums (card says 279); speaker 3946 M ~181 h, 10246 F ~74 h, 8882 M 37 h, 12367 M 33 h, 11797 F 19 h | 24 kHz | CC BY 4.0 (V openslr.org/146) | Best. LibriVox home recordings, quality varies by reader, accent per speaker NOT labelled. Derived from MLS (audio PD/CC BY, text Gutenberg PD) |
| OpenSLR 61-76 (Google AR/CL/CO/PE/PR/VE Spanish) | 5-7 h per country, ~30 speakers | 48 kHz | CC BY-SA 4.0 (V) | Reject: SA, too little per speaker (Chilean total 7 h / 31 spk) |
| M-AILABS es_ES | ~109 h (Tux, Karen MX, Victor AR) | 16 kHz | BSD-like "commercial use permitted" (text seen only on an unofficial mirror; original page unreachable) | Weak: 16 kHz |
| CSS10 Spanish | 23.8 h, one male (LibriVox) | R 22 kHz | repo Apache-2.0 (V), audio PD; hosted on Kaggle (login), Kaggle licence not verified | Fallback |
| MLS es / FLEURS / Common Voice | large | 16 kHz / 16 kHz / 48 kHz mp3 | CC BY 4.0 (V) / CC BY 4.0 / CC0 | ASR-grade; Common Voice left HF Oct 2025 (login) |
| Panama, US Hispanic, Colombia (dedicated TTS) | none found | | | Only Common Voice/FLEURS es-419 |

## Other languages

| Lang | Best clean corpus | Hours / SR | Licence | Notes |
|---|---|---|---|---|
| pt-BR | TTS-Portuguese-Corpus (Edresson) | 10.5 h single speaker, 48 kHz, RNNoise-denoised | CC BY 4.0 (V) | Good pilot corpus. CML-TTS pt spk 2961 F 47 h also usable |
| pl | CML-TTS pl spk 6892 M (27 h), 7014 F (14 h) | 24 kHz | CC BY 4.0 | M-AILABS pl is 16 kHz |
| it | CML-TTS it spk 1595 M (24 h), 4974 F (18 h) | 24 kHz | CC BY 4.0 | serena (synthetic) exists as tier A with caveats |
| nl | CML-TTS nl (very large), r-dh/dutch-vl-tts (used by rdh voice) | 24 kHz | CC BY 4.0 / CC0 | per-speaker CML hours not computed |
| fr | CML-TTS fr spk 1840 M (216 h), 12899 F (19 h); MLS fr | 24 kHz | CC BY 4.0 | tier-A MLS voice exists |
| fr-CA | none found | | | Use fr-FR; only Common Voice fr |
| fr West Africa | OpenSLR 57 African Accented French, ~22 h, 232 speakers | 16 kHz (R) | Apache-2.0 (V) | ASR-grade prompts, not TTS |
| id | none TTS-grade | | FLEURS id CC BY 4.0, Common Voice id CC0 | OpenSLR 41/44 are Javanese/Sundanese CC BY-SA; news_tts is NC |
| ar (MSA) | ClArTTS (MBZUAI) | ~12 h one male, 40-44 kHz | CC BY 4.0 (HF tag) | Classical register; text needs diacritics |
| ar (Levantine) | Arabic Speech Corpus (Halabi) | ~3.7 h (R), 48 kHz | CC BY 4.0 (site text) | Small |
| ar Egyptian/Gulf | none verified | | | |
| hi | AI4Bharat Rasa (2 spk per language, 48 kHz), IndicVoices-R | | CC BY 4.0 (HF tag; gated: free HF login) | Best Hindi/Indian-English path; filter neutral styles |
| hi | IIT Madras IndicTTS | | licence PDF unreadable (timeout) | Do not use until read |
| en AU/CA/IN/SG/NZ | only VCTK subsets (CC BY 4.0, 0.5 h/speaker) | | | See tier B VCTK voices; Rasa for Indian English |
| en (studio, single speaker) | Hi-Fi TTS (OpenSLR 109, 10 spk, >= 17 h each, 44.1 kHz) | | CC BY 4.0 (V) | Best source to train a clean en voice from scratch |

## espeak-ng (Piper's phonemizer) support, checked locally

es (Spain), es-419 (Latin America), pt (Portugal), pt-br, fr-fr, fr-be, fr-ch (no fr-ca), pl, nl, it, id, ar (no diacritic
restoration), hi; English: en-gb, en-us, en-gb-scotland, en-gb-x-rp, en-gb-x-gbclan, en-gb-x-gbcwmd, en-029, en-us-nyc. There is
no en-au, en-nz, en-ca, en-in, en-sg: those accents can only come from audio, not phonemes.

## Could not verify

Accent of CML es speakers; audio quality (never listened); CSS10 rate and Kaggle licence; IndicTTS licence; IndicVoices-R Hindi
hours; Arabic Speech Corpus hours/rate; Dutch per-speaker CML hours; Egyptian/Gulf/Panama/US-Hispanic/fr-CA/Indonesian TTS corpora.
