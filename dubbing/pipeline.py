#!/usr/bin/env python3
"""Dubbing MVP: STT -> LLM translation (duration-budgeted) -> Piper TTS -> silence-aware retiming
-> bounded time-stretch. Text/audio only. Explicitly NOT video dubbing: no lip-sync, no frame
handling, no video container I/O anywhere in this module.

Usage:
    # Audio in, audio out (needs worker-stt + TTS gateway credentials - see stt.py/tts.py):
    python3 -m dubbing.pipeline --audio in.wav --source-lang en --target-lang de --out out.wav

    # Text in (scope-reduced path, works without STT credentials):
    python3 -m dubbing.pipeline --text "Thanks for calling, how can I help you today?" \\
        --source-lang en --target-lang de --duration 3.2 --out out.wav

See dubbing/README.md for what's real vs. stubbed in this environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile

from . import catalog, stt, translate, tts, retime


def run(
    *,
    audio_path: str | None,
    text: str | None,
    source_lang: str,
    target_lang: str,
    out_path: str,
    duration_s: float | None = None,
    scratch_dir: str | None = None,
) -> dict:
    catalog.validate_target_language(target_lang)
    scratch_dir = scratch_dir or tempfile.mkdtemp(prefix="dubbing_")

    # --- STT (or text bypass) ---
    if audio_path:
        stt_client = stt.SttClient()
        stt_result = stt_client.transcribe(audio_path, language=source_lang)
    else:
        stt_result = stt.TextInput(text=text, language=source_lang, duration_s=duration_s).as_result()
    source_duration_s = stt_result.duration_s or duration_s

    # --- Translation (duration-budgeted) ---
    translator = translate.get_default_translator()
    translation = translator.translate(
        stt_result.text, stt_result.language, target_lang, duration_s=source_duration_s
    )

    # --- TTS ---
    tts_backend = tts.get_default_backend()
    raw_tts_path = f"{scratch_dir}/raw_tts.wav"
    voice_id = tts_backend.synth(translation.text, target_lang, raw_tts_path)

    # --- Retime + bounded stretch to match source duration ---
    if source_duration_s:
        fit = retime.fit_duration(raw_tts_path, out_path, source_duration_s, scratch_dir=scratch_dir)
    else:
        # No known source duration (text-only input with no --duration given): skip fitting,
        # just carry the raw TTS output through unchanged.
        import shutil
        shutil.copy(raw_tts_path, out_path)
        fit = {"note": "no source duration provided; retiming/stretch skipped"}

    return {
        "source_text": stt_result.text,
        "source_lang": stt_result.language,
        "target_lang": target_lang,
        "translated_text": translation.text,
        "voice_id": voice_id,
        "tts_backend": type(tts_backend).__name__,
        "out_path": out_path,
        "fit": fit,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", help="source audio WAV (uses STT)")
    ap.add_argument("--text", help="source text (bypasses STT)")
    ap.add_argument("--source-lang", required=True)
    ap.add_argument("--target-lang", required=True, help=f"one of: {catalog.supported_languages()}")
    ap.add_argument("--duration", type=float, default=None, help="source duration in seconds (for --text mode)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not args.audio and not args.text:
        ap.error("supply --audio or --text")

    result = run(
        audio_path=args.audio,
        text=args.text,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        out_path=args.out,
        duration_s=args.duration,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
