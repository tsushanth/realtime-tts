"""Streaming synthesis core, shared by the local WS server and the RunPod handler."""
import re
import time
from pathlib import Path

from kokoro_onnx import Kokoro

MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000

_kokoro = None


def get_engine():
    global _kokoro
    if _kokoro is None:
        _kokoro = Kokoro(
            str(MODEL_DIR / "kokoro-v1.0.onnx"),
            str(MODEL_DIR / "voices-v1.0.bin"),
        )
    return _kokoro


_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")


def chunk_text(text, max_chars=90):
    """Split into clause-ish chunks so synthesis can start before the whole utterance arrives."""
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    if not parts:
        parts = [text]
    chunks = []
    buf = ""
    for p in parts:
        if buf and len(buf) + len(p) + 1 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def active_providers():
    """The providers onnxruntime is actually running the session on right now — not just
    what's available, since a provider can be 'available' per get_available_providers()
    but still silently fail to initialize (missing .so at runtime) and fall back."""
    return get_engine().sess.get_providers()


def synthesize_stream(text, voice=DEFAULT_VOICE, speed=1.0):
    """Yields (pcm_f32_bytes_as_wav_chunk, chunk_meta) incrementally, one per text chunk."""
    engine = get_engine()
    for chunk in chunk_text(text):
        t0 = time.perf_counter()
        samples, sr = engine.create(chunk, voice=voice, speed=speed, lang="en-us")
        gen_ms = (time.perf_counter() - t0) * 1000
        audio_s = len(samples) / sr
        yield {
            "text": chunk,
            "samples": samples,
            "sample_rate": sr,
            "gen_ms": gen_ms,
            "audio_s": audio_s,
            "providers": active_providers(),
        }
