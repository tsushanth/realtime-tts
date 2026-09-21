"""Eval framework, step 2: quality scoring. Reads eval/results/latency.json (written by synth.mjs
+ repair.mjs), transcribes each saved WAV with faster-whisper and scores word error rate against the
known ground-truth text (intelligibility), and scores naturalness with torchaudio's SQUIM_SUBJECTIVE
model (an automatic MOS predictor - NOT a human listener; see eval/README.md for the caveat).
Writes eval/results/quality.json. Re-run after any engine change, same as synth.mjs.

Usage: python3 eval/score.py
"""
import json
import re
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "latency.json"
OUT = HERE / "results" / "quality.json"

results = json.loads(RESULTS.read_text())

print("loading faster-whisper (base)...")
whisper = WhisperModel("base", device="cpu", compute_type="int8")

print("loading SQUIM_SUBJECTIVE (naturalness proxy)...")
squim_bundle = torchaudio.pipelines.SQUIM_SUBJECTIVE
squim_model = squim_bundle.get_model()
# SQUIM_SUBJECTIVE is a reference-based MOS predictor (needs a clean reference utterance alongside the
# signal being scored) - use one fixed, unrelated clean reference clip shipped with torchaudio's test
# fixtures isn't available offline, so instead use each engine's OWN best (shortest silence, clearest)
# clip as a self-reference is circular. Simpler and defensible for a *relative* comparison: synthesize
# one shared "reference" utterance per language with ElevenLabs Multilingual v2 (our best-quality
# available reference) ONCE, and score every engine's output against that SAME reference per language -
# so scores are comparable to each other (same reference), even if not an absolute MOS truth.
REFERENCE_ENGINE = "elevenlabs-multilingual"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_mono16k(wav_path: Path) -> torch.Tensor:
    # torchaudio.load() in this version requires torchcodec, which isn't installed - soundfile
    # (already a dependency throughout this repo) reads the same WAVs fine.
    x, sr = sf.read(str(wav_path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    wav = torch.from_numpy(x).unsqueeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


# Pick one reference clip per language (first sentence available from REFERENCE_ENGINE).
references: dict[str, Path] = {}
for r in results:
    if r["engine"] == REFERENCE_ENGINE and "wav" in r and r["lang"] not in references:
        references[r["lang"]] = HERE / r["wav"]
print("reference clips:", {k: v.name for k, v in references.items()})

quality = []
for i, r in enumerate(results):
    if "wav" not in r:
        quality.append({**r, "wer": None, "naturalness_mos": None, "note": "no audio (synthesis failed)"})
        continue
    wav_path = HERE / r["wav"]
    entry = {"engine": r["engine"], "lang": r["lang"], "id": r["id"], "text": r["text"]}

    # --- Intelligibility: WER via Whisper ---
    try:
        segments, info = whisper.transcribe(str(wav_path), language=r["lang"] if r["lang"] != "en" else "en")
        transcript = " ".join(s.text for s in segments).strip()
        ref_n, hyp_n = normalize(r["text"]), normalize(transcript)
        wer = jiwer.wer(ref_n, hyp_n) if ref_n else None
        entry["transcript"] = transcript
        entry["wer"] = round(wer, 4) if wer is not None else None
    except Exception as e:  # noqa: BLE001
        entry["wer"] = None
        entry["wer_error"] = str(e)

    # --- Naturalness: SQUIM_SUBJECTIVE MOS proxy, relative to the per-language reference clip ---
    try:
        ref_path = references.get(r["lang"])
        if ref_path and ref_path != wav_path:
            sig = load_mono16k(wav_path)
            ref = load_mono16k(ref_path)
            n = min(sig.shape[1], ref.shape[1])
            with torch.no_grad():
                mos = squim_model(sig[:, :n], ref[:, :n])
            entry["naturalness_mos"] = round(float(mos[0]), 3)
        else:
            entry["naturalness_mos"] = None  # this IS the reference clip itself, or no reference for this lang
    except Exception as e:  # noqa: BLE001
        entry["naturalness_mos"] = None
        entry["mos_error"] = str(e)

    quality.append(entry)
    print(f"[{i+1}/{len(results)}] {r['engine']:28s} {r['lang']}/{r['id']:16s} wer={entry.get('wer')} mos={entry.get('naturalness_mos')}")

OUT.write_text(json.dumps(quality, indent=2))
print(f"wrote {OUT}")
