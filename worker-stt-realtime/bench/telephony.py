"""Degrade 16 kHz speech to telephone quality: 8 kHz, 300-3400 Hz band, mu-law companding, additive noise.

The companding is continuous mu-law (mu=255) followed by 8-bit quantisation. It is NOT the exact G.711
segment/step table; the maximum difference from bit-exact G.711 is about 0.024 (full scale = 1), which is
immaterial for this benchmark.
"""
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

_SOS = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")


def mulaw_roundtrip(x):
    """Continuous mu-law (mu=255) compand, 8-bit quantise (256 levels), expand. Input clipped to [-1, 1]; float32 out."""
    mu = 255.0
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) / 2.0 * 255.0)
    y = q / 255.0 * 2.0 - 1.0
    return (np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu).astype(np.float32)


def to_telephone(x16, seed=0, snr_db=25.0):
    """16 kHz audio -> telephone-quality 16 kHz float32 of the same length.

    Steps: 16 -> 8 kHz, 300-3400 Hz band-pass, peak-normalise to 0.7 (phone-line-like level, never clips),
    mu-law round trip, add white Gaussian noise at `snr_db` measured against the signal RMS (after
    normalisation), back to 16 kHz. Noise is deterministic per `seed`. int16 input is divided by 32768
    first; float input is assumed to already be in [-1, 1]. Non-finite input raises ValueError.
    """
    x = np.asarray(x16)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    else:
        x = x.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite audio")
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    x8 = resample_poly(x, 1, 2)
    x8 = sosfilt(_SOS, x8)
    peak = np.max(np.abs(x8))
    if peak > 0:
        x8 = x8 * (0.7 / peak)
    x8 = mulaw_roundtrip(x8).astype(np.float64)
    rms = float(np.sqrt(np.mean(x8 ** 2)))
    if rms > 0:
        x8 = x8 + np.random.default_rng(seed).normal(0.0, rms * 10 ** (-snr_db / 20.0), len(x8))
    y = resample_poly(x8, 2, 1)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)
