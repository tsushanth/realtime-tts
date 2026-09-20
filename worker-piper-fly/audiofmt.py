"""Output formats for the Piper worker: resampling straight from the model's native rate
and numpy G.711 (mu-law / A-law) encoders (audioop was removed in Python 3.13).

Formats (the `format` field of a synthesize message / HTTP body):
  pcm_24000  16-bit LE PCM mono 24 kHz   (default; the historical output, unchanged)
  pcm_8000   16-bit LE PCM mono 8 kHz
  mulaw_8000 G.711 mu-law, 8 kHz, 1 byte/sample (Twilio / telephony)
  alaw_8000  G.711 A-law, 8 kHz, 1 byte/sample
Opus is deliberately not offered: it needs libopus/libogg native libraries (or ffmpeg) in the
image, which is not worth the weight for this worker.
"""
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.signal import firwin, resample_poly

# name -> (sample_rate, kind, http content type)
FORMATS = {
    "pcm_24000": (24000, "pcm", "audio/pcm"),
    "pcm_8000": (8000, "pcm", "audio/pcm"),
    "mulaw_8000": (8000, "mulaw", "audio/basic"),
    "alaw_8000": (8000, "alaw", "audio/x-alaw-basic"),
}
DEFAULT_FORMAT = "pcm_24000"


def parse_format(value) -> str:
    if value is None:
        return DEFAULT_FORMAT
    if not isinstance(value, str) or value not in FORMATS:
        raise ValueError(f"unknown format {value!r} (supported: {', '.join(FORMATS)})")
    return value


_ULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
_ALAW_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def lin2ulaw(pcm: np.ndarray) -> bytes:
    """int16 -> G.711 mu-law; mirrors CPython 3.11 audioop.lin2ulaw(x, 2)."""
    p = pcm.astype(np.int32) >> 2  # 14-bit
    mask = np.where(p < 0, 0x7F, 0xFF)
    p = np.minimum(np.abs(p), 8159) + (0x84 >> 2)
    seg = np.searchsorted(_ULAW_SEG_END, p, side="left")
    uval = (seg << 4) | ((p >> (seg + 1)) & 0xF)
    out = np.where(seg >= 8, 0x7F, uval) ^ mask
    return out.astype(np.uint8).tobytes()


def lin2alaw(pcm: np.ndarray) -> bytes:
    """int16 -> G.711 A-law; mirrors CPython 3.11 audioop.lin2alaw(x, 2)."""
    p = pcm.astype(np.int32) >> 3  # 13-bit
    mask = np.where(p >= 0, 0xD5, 0x55)
    p = np.where(p >= 0, p, -p - 1)
    seg = np.searchsorted(_ALAW_SEG_END, p, side="left")
    sh = np.where(seg < 2, 1, seg)
    aval = (seg << 4) | ((p >> sh) & 0xF)
    out = np.where(seg >= 8, 0x7F, aval) ^ mask
    return out.astype(np.uint8).tobytes()


def ulaw2lin(data: bytes) -> np.ndarray:
    """Reference decoder (tests / client examples)."""
    u = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
    return (np.where(u & 0x80, 0x84 - t, t - 0x84)).astype(np.int16)


def alaw2lin(data: bytes) -> np.ndarray:
    a = np.frombuffer(data, dtype=np.uint8).astype(np.int32) ^ 0x55
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, ((a & 0x0F) << 4) + 8, (((a & 0x0F) << 4) + 0x108) << np.maximum(seg - 1, 0))
    return (np.where(a & 0x80, t, -t)).astype(np.int16)


@lru_cache(maxsize=8)
def _lowpass(native_sr: int, target_sr: int):
    """(up, down, taps): polyphase resample native -> target with a steep anti-alias low-pass
    (pass 92.5% of the target Nyquist, stop at 107.5%, ~80 dB) designed at the upsampled
    rate, so there is exactly one resampling stage from the model's rate."""
    r = Fraction(target_sr, native_sr)
    up, down = r.numerator, r.denominator
    fs_up = native_sr * up
    nyq = target_sr / 2
    fp, fst = 0.925 * nyq, 1.075 * nyq
    dw = 2 * np.pi * (fst - fp) / fs_up
    numtaps = int((80 - 7.95) / (2.285 * dw)) | 1
    taps = firwin(numtaps, (fp + fst) / 2, fs=fs_up, window=("kaiser", 7.86))
    return up, down, taps


def resample(audio: np.ndarray, native_sr: int, target_sr: int) -> np.ndarray:
    if target_sr == native_sr:
        return audio
    up, down, taps = _lowpass(native_sr, target_sr)
    return resample_poly(audio, up, down, window=taps)


def encode(audio: np.ndarray, native_sr: int, fmt: str) -> tuple[bytes, int]:
    """audio: float, peak-normalised, at native_sr. Returns (bytes, sample_rate)."""
    sr, kind, _ = FORMATS[fmt]
    if fmt == "pcm_24000":
        # Historical path, kept literally identical (default scipy window) so default output
        # is byte-for-byte what it always was.
        ratio = Fraction(24000, native_sr).limit_denominator(1000)
        if ratio.numerator != ratio.denominator:
            audio = resample_poly(audio, ratio.numerator, ratio.denominator)
    else:
        audio = resample(audio, native_sr, sr)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    if kind == "mulaw":
        return lin2ulaw(pcm), sr
    if kind == "alaw":
        return lin2alaw(pcm), sr
    return pcm.tobytes(), sr
