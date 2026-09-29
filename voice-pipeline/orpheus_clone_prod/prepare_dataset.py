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

_WHISPER_MODEL = None


class DatasetTooSmallError(Exception):
    pass


class TranscriptionUnavailableError(Exception):
    """The transcription model itself could not be loaded (download failure,
    missing CUDA/cuDNN library, ...). A systemic service failure, NOT a
    problem with the customer's data -- kept distinct from
    DatasetTooSmallError so it is never reported as "not enough usable clips"."""


def load_transcription_model():
    """Load the Whisper model once. Called eagerly by prepare_dataset before
    the per-clip loop, OUTSIDE the per-clip exception handling, so a systemic
    load failure surfaces as TranscriptionUnavailableError instead of being
    swallowed as N individual "bad clip" skips. Isolated so tests can
    monkeypatch it without loading a real model."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel

            # CPU, not CUDA: faster-whisper's GPU backend (ctranslate2) needs
            # its own cuBLAS/cuDNN .so files discoverable via LD_LIBRARY_PATH
            # -- torch's wheel-bundled CUDA libs (installed for the LoRA
            # training step) do NOT satisfy this, and the train_image never
            # provided nvidia-cublas-cu12/nvidia-cudnn-cu12. That caused
            # "Library libcublas.so.12 is not found or cannot be loaded" on
            # every real deployment, load_transcription_model() to raise
            # TranscriptionUnavailableError, and (before that error path
            # existed) silent per-clip failures. The training container
            # already reserves a GPU for the LoRA step, so paying CPU cost
            # for the transcription pass (one-time, not on the training hot
            # path) is a legitimate trade for not depending on a second,
            # separately-provisioned CUDA runtime. int8 is the recommended
            # compute_type for CPU inference with faster-whisper.
            _WHISPER_MODEL = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
        except Exception as e:
            raise TranscriptionUnavailableError(f"transcription service unavailable: {e}") from e
    return _WHISPER_MODEL


def transcribe_clip(path: str) -> str:
    """Isolated so tests can monkeypatch it without loading a real Whisper
    model."""
    model = load_transcription_model()
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

    if extracted:
        # Eager, outside the per-clip try/except below: if the model can't
        # load, that's our failure, not the customer's data. Let it propagate.
        try:
            load_transcription_model()
        except Exception:
            shutil.rmtree(clips_dir, ignore_errors=True)
            raise

    rows = []
    for path in extracted:
        if path.lower().endswith(".wav") and not _is_valid_wav(path):
            continue  # corrupt/unreadable clip, skip rather than fail the whole upload
        try:
            text = transcribe_clip(path)
        except Exception as e:
            # Per-clip decode/transcribe failure (corrupt/unreadable file of
            # any format). The model itself is already loaded at this point,
            # so this is genuinely about this clip. Log and skip.
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
