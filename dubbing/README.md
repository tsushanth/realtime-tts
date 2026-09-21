# Dubbing MVP: text/audio translated narration

Text/audio only. Explicitly **not** video dubbing — no lip-sync, no video processing, no frame
handling anywhere in this module.

Pipeline: STT (existing worker-stt) → LLM translation with an explicit duration budget in the
prompt → Piper TTS in the target language → silence-aware retiming → bounded (±15%) time-stretch.

## Files

- `catalog.py` — enumerates target languages from `voices/catalog.json` (the single source of
  truth; nothing here hardcodes a language list) and picks a default voice per language.
- `translate.py` — LLM translation behind a `Translator` interface. `OpenRouterTranslator` is the
  only real backend wired up, `StubTranslator` fails loudly (not silently) when no key is present.
- `tts.py` — Piper synthesis via this repo's existing gateway (`gateway/server.js` /tts/authorize,
  same call as `eval/synth.mjs`). `NullTTS` is an explicit stub for exercising the rest of the
  pipeline without live TTS credentials.
- `stt.py` — thin client for the existing `worker-stt` service (same mint scheme as
  `worker-stt/test_client.py`). `TextInput` bypasses STT with manually supplied source text, per
  this task's explicit scope-reduction option.
- `retime.py` — silence-aware retiming (ffmpeg `silencedetect`/`atrim`/`concat`, touches only
  inter-sentence pauses) + bounded time-stretch (ffmpeg `atempo`, capped at ±15%).
- `pipeline.py` — orchestrates all of the above; also a CLI (`python3 -m dubbing.pipeline ...`).

## What's real vs. stubbed in *this* environment

Checked directly (`env`, `df -h`, `which`), not assumed:

| Step | Status here | Why |
|---|---|---|
| Translation | **Real, tested live** | `OPENROUTER_API_KEY` is present. Used `openai/gpt-4o-mini` via OpenRouter. |
| STT | **Stubbed (bypassed via `TextInput`)** | No `STT_BASE` / `STT_SECRET_FILE` for the deployed `worker-stt` Modal app in this environment. The real client (`SttClient`) is implemented and matches `worker-stt/test_client.py`'s mint scheme exactly, but untested here for lack of credentials. |
| TTS | **Stubbed (`NullTTS`)** | No `TTS_GATEWAY_API_KEY` (and no `ADMIN_SECRET` to mint one via `gateway/keys.js`'s `/admin/keys`, which would be needed to get one at all). Real client (`PiperGatewayTTS`) mirrors `eval/synth.mjs`'s auth flow but is untested here. Also: local disk had ~140MB free at build time — not enough to install Piper + onnxruntime + a voice model (~60-80MB each) as a local fallback. |
| Retiming + stretch | **Real, tested live** | Pure ffmpeg (already installed), no external creds needed. |

**Time-stretch tool note** (task asked which of librosa/pyrubberband was used, and to say so if
neither): neither. ffmpeg's `atempo` filter was used instead — same category of tool
(pitch-preserving WSOLA-style tempo change), chosen because ffmpeg was already installed and disk
space (~140MB free) was too tight to safely `pip install librosa` (numpy+scipy+numba+librosa) or
`pyrubberband` (needs the separate `rubberband` CLI) without risking filling the disk further.
Swapping in librosa's `time_stretch` behind `retime.time_stretch()`'s existing signature is a
small change if disk space frees up.

**Translation API note**: an LLM *is* configured in this environment (`OPENROUTER_API_KEY`), so
translation was not stubbed — it's a real, tested step. No other LLM provider key
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) was found set.

## What was actually tested end to end

Ran `python3 -m dubbing.pipeline --text "..." --source-lang en --target-lang de_DE --duration 6.5
--out /tmp/out.wav`:

1. Real OpenRouter translation: `"Thanks for calling, how can I help you today? Your order should
   arrive within three to five business days."` → `"Danke für Ihren Anruf! Wie kann ich helfen?
   Ihre Bestellung kommt in drei bis fünf Tagen."` (budgeted to source duration).
2. `NullTTS` stub synth (clearly logged as non-speech placeholder audio) to exercise the audio
   pipeline mechanically.
3. Real silence-aware retiming + bounded stretch: 5.93s stub clip → 6.146s after silence retiming →
   6.496s after a 0.946x tempo pass, landing within 0.15s of the 6.5s target and within the ±15%
   stretch bound.
4. Also verified the stretch bound itself clamps correctly on an extreme case (0.4s clip targeting
   10s: `stretch_ratio` clamps to 0.85, pipeline honestly reports `hit_target: false` rather than
   silently overshooting the bound).
5. Verified `catalog.validate_target_language` rejects a language not in `voices/catalog.json`
   (e.g. `ja_JP`) with a clear error listing what's actually supported.

STT and real Piper TTS were **not** tested end to end here — see the table above for exactly why,
and `stt.py`/`tts.py` for the real (untested) clients that are ready to run once
`STT_BASE`/`STT_SECRET_FILE` and `TTS_GATEWAY_API_KEY` are provisioned.

## Test-key hygiene (when STT/TTS credentials do get provisioned)

Follow `eval/`'s mint → use → revoke pattern, not a standing key:
- TTS: mint via `gateway/keys.js`'s `/admin/keys` (needs `ADMIN_SECRET`, not this repo's problem to
  generate), use for the test run, then `DELETE /admin/keys` to revoke.
- STT: `stt.mint()` tokens are short-TTL (10 min) stateless HMAC session tokens, not stored
  server-side — "revoke" here means minting a short TTL and not reusing it, there's no server-side
  delete for these (unlike the gateway's `/admin/keys` DELETE flow).
- Never a production-named app, never a real end-user-facing key.

## Language scope

Target languages are read live from `voices/catalog.json`, not hardcoded. As of this build:
`de_DE, en_AU, en_CA, en_GB, en_IE, en_IN, en_NZ, en_US, en_ZA, es_ES, es_MX, fr_FR, it_IT, nl_BE,
nl_NL, pl_PL, pt_BR, ru_RU` (run `python3 dubbing/catalog.py` to regenerate). Kokoro (GPU, English
only) is not used as a target-language engine here since it can't cover any non-English target;
Piper is the only engine in this repo that does.
