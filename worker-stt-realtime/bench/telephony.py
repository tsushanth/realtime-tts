"""Degrade 16 kHz speech to telephone quality: 8 kHz, 300-3400 Hz band, G.711 mu-law companding, additive noise."""
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

_SOS = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")


def mulaw_roundtrip(x):
    mu = 255.0
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) / 2.0 * 255.0)
    y = q / 255.0 * 2.0 - 1.0
    return (np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu).astype(np.float32)


def to_telephone(x16, seed=0, snr_db=25.0):
    x = np.asarray(x16, dtype=np.float64)
    n = len(x)
    x8 = resample_poly(x, 1, 2)
    x8 = sosfilt(_SOS, x8)
    peak = np.max(np.abs(x8)) if len(x8) else 0.0
    if peak > 0:
        x8 = x8 * (0.7 / peak)                       # level like a phone line, never clipping
    x8 = mulaw_roundtrip(x8).astype(np.float64)
    rms = float(np.sqrt(np.mean(x8 ** 2))) if len(x8) else 0.0
    if rms > 0:
        x8 = x8 + np.random.default_rng(seed).normal(0.0, rms * 10 ** (-snr_db / 20.0), len(x8))
    y = resample_poly(x8, 2, 1)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)
