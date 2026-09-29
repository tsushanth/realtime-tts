"""
Turn a customer-uploaded zip of audio clips into the same dataset_manifest.jsonl
row shape build_dataset.py produces from the pilot's Polly corpus:
    {"text": "<voice_tag>: <transcript>", "audio": "clips/<filename>"}

Unlike the pilot (which had ground-truth transcripts from the corpus manifest),
uploaded customer audio has no transcript, so this module transcribes each
clip with faster-whisper -- the same reference-generation approach the STT
real-call eval used for draft references.
"""
import io
import json
import logging
import os
import shutil
import wave
import zipfile

logger = logging.getLogger(__name__)


class DatasetTooSmallError(Exception):
    pass


def transcribe_clip(path: str) -> str:
    """Isolated so tests can monkeypatch it without loading a real Whisper
    model. Production implementation loads faster-whisper lazily (import
    inside the function) since this module is also imported by tests that
    never need the real model."""
    from faster_whisper import WhisperModel

    global _WHISPER_MODEL
    try:
        model = _WHISPER_MODEL
    except NameError:
        model = _WHISPER_MODEL = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(path, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def _is_valid_wav(path: str) -> bool:
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() > 0
    except Exception:
        return False


def prepare_dataset(zip_bytes: bytes, dataset_dir: str, voice_tag: str, min_clips: int = 20) -> list[dict]:
    clips_dir = os.path.join(dataset_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    extracted = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith("/") or not name.lower().endswith((".wav", ".flac", ".mp3")):
                    continue
                out_path = os.path.join(clips_dir, os.path.basename(name))
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(out_path)
    except zipfile.BadZipFile:
        shutil.rmtree(clips_dir, ignore_errors=True)
        raise DatasetTooSmallError("uploaded file is not a valid zip archive")

    rows = []
    for path in extracted:
        if path.lower().endswith(".wav") and not _is_valid_wav(path):
            continue  # corrupt/unreadable clip, skip rather than fail the whole upload
        try:
            text = transcribe_clip(path)
        except Exception as e:
            # Catch any exception during transcription (corrupt/unreadable files of any format).
            # Log and skip that clip; continue with the rest.
            logger.warning(f"Failed to transcribe {path}: {e}, skipping")
            continue
        if not text:
            continue  # unusable clip (silence, noise): no transcript, no training row
        rows.append({
            "text": f"{voice_tag}: {text}",
            "audio": f"clips/{os.path.basename(path)}",
        })

    if len(rows) < min_clips:
        shutil.rmtree(clips_dir, ignore_errors=True)
        raise DatasetTooSmallError(
            f"only {len(rows)} usable clips after extraction and transcription, need at least {min_clips}"
        )

    manifest_path = os.path.join(dataset_dir, "dataset_manifest.jsonl")
    with open(manifest_path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")

    return rows
