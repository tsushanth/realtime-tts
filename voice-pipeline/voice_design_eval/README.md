# voice-design-dev hardening eval artifacts (2026-09-21)

Evidence backing the "Hardening pass" section of `../VOICE_DESIGN_MVP.md`. Not a general eval
framework like `eval/` (WER/MOS vs. ElevenLabs) - specific to this session's two measurements.

- `warm_path_probe.py` - calls the deployed `VoiceDesignModel.generate` Modal method three times
  back-to-back (first cold, next two on the same warm container) and prints wall/load/generation
  timings. Run with `python3 warm_path_probe.py` (needs `modal` installed and authenticated).
- `quality_eval.py` - generates 8 varied voice descriptions (see `VOICE_DESIGN_MVP.md` section 2
  for the full table and caveats), downloads each WAV via `modal volume get`, and scores mean/median
  F0 with `librosa.pyin` against an expected physiological pitch band per description. Needs
  `pip install librosa soundfile numpy modal`. Writes `quality_eval_results.json` and
  `quality_audio/*.wav` next to itself (this repo's copies are `results.json` / `audio/`, renamed
  after the run).
- `results.json` - the actual results from this session's run.
- `audio/*.wav` - the actual generated clips from this session's run (small, ~300-400KB each).

Read `VOICE_DESIGN_MVP.md` before trusting any number here - especially the "what this eval can
and cannot tell you" caveat under section 2: F0 checks age/gender pitch plausibility only, not
adherence to tone/accent/character adjectives, and is not a substitute for a human listening pass.
