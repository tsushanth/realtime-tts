"""Shared objective-QA + automated pass/fail heuristics for the batch voice-conversion MVP
(voice-pipeline/convert_job.py). Pure numpy/soundfile, no `modal` import, so it can run both
standalone (check_conversion_quality.py, a human running it from a laptop) and inside the Modal GPU
container right after seed-vc produces a result (convert_job.py's run_conversion).

The metrics themselves (duration ratio, median-F0 via autocorrelation, RMS energy envelope
correlation) are exactly what check_conversion_quality.py already reported by hand on the original
2 pairs; this module just gives them a name and turns them into an automated pass/fail so a job
result can carry a quality flag instead of requiring someone to eyeball printed numbers.
"""
from __future__ import annotations

import numpy as np


def median_f0(x, sr):
    n = int(0.04 * sr)
    lo, hi = int(sr / 400), int(sr / 60)
    out = []
    for st in range(0, len(x) - n, n):
        fr = x[st:st + n] - np.mean(x[st:st + n])
        if np.sqrt(np.mean(fr ** 2)) < 0.01:
            continue
        ac = np.correlate(fr, fr, "full")[n - 1:]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] / (ac[0] + 1e-9) > 0.5:
            out.append(sr / k)
    return float(np.median(out)) if out else 0.0


def rms_envelope(x, sr, hop_ms=20):
    n = int(hop_ms / 1000 * sr)
    m = len(x) // n
    if m < 2:
        return np.array([])
    fr = x[: m * n].reshape(m, n)
    return np.sqrt(np.mean(fr ** 2, axis=1))


def load_mono(path):
    import soundfile as sf
    x, sr = sf.read(path, always_2d=True)
    return x.mean(axis=1), sr


def envelope_correlation(src, sr_s, out, sr_o):
    env_src = rms_envelope(src, sr_s)
    env_out = rms_envelope(out, sr_o)
    if len(env_src) <= 4 or len(env_out) <= 4:
        return None
    n = min(len(env_src), len(env_out))
    idx_s = np.linspace(0, len(env_src) - 1, n).astype(int)
    idx_o = np.linspace(0, len(env_out) - 1, n).astype(int)
    a, b = env_src[idx_s], env_out[idx_o]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ---- automated quality gate --------------------------------------------------------------------
# Thresholds are deliberately loose heuristics meant to catch OBVIOUS failures (near-silence, wild
# duration mismatch, pitch not moving at all), not to certify subtle quality - a human/listening
# pass is still the real bar per the MVP's original evidence writeup. False negatives (a genuinely
# bad conversion that still passes these checks) are expected; the goal is cheap automatic triage.
MIN_OUTPUT_RMS = 0.003          # near-silence: seed-vc silently producing ~nothing
MIN_DURATION_RATIO = 0.5        # output wildly shorter than source
MAX_DURATION_RATIO = 2.0        # output wildly longer than source
MIN_F0_MOVEMENT_HZ = 3.0        # if |src-tgt| pitch gap is bigger than this, expect output to move
ENVELOPE_CORR_WARN = 0.3        # below this, prosody/timing likely did not survive (warning, not fail)


def validate_conversion_quality(source_path: str, target_path: str, output_path: str) -> dict:
    """Runs the heuristic checks and returns {"passed": bool, "reasons": [...], "warnings": [...],
    "metrics": {...}}. `reasons` non-empty => passed is False. `warnings` don't fail the gate but are
    worth surfacing (e.g. weak envelope correlation)."""
    src, sr_s = load_mono(source_path)
    tgt, sr_t = load_mono(target_path)
    out, sr_o = load_mono(output_path)

    dur_src, dur_out = len(src) / sr_s, len(out) / sr_o
    duration_ratio = dur_out / dur_src if dur_src else 0.0
    out_rms = float(np.sqrt(np.mean(out ** 2))) if len(out) else 0.0

    f0_src = median_f0(src, sr_s)
    f0_tgt = median_f0(tgt, sr_t)
    f0_out = median_f0(out, sr_o)
    corr = envelope_correlation(src, sr_s, out, sr_o)

    reasons, warnings = [], []

    if out_rms < MIN_OUTPUT_RMS:
        reasons.append(f"output is near-silent (RMS={out_rms:.5f} < {MIN_OUTPUT_RMS})")

    if not (MIN_DURATION_RATIO <= duration_ratio <= MAX_DURATION_RATIO):
        reasons.append(
            f"duration ratio out of range (output/source={duration_ratio:.2f}, "
            f"expected {MIN_DURATION_RATIO}-{MAX_DURATION_RATIO})"
        )

    if f0_src and f0_tgt and f0_out:
        gap = abs(f0_tgt - f0_src)
        d_to_src = abs(f0_out - f0_src)
        d_to_tgt = abs(f0_out - f0_tgt)
        if gap > MIN_F0_MOVEMENT_HZ and d_to_tgt >= d_to_src:
            reasons.append(
                f"output pitch did not move toward target (median F0 Hz: source={f0_src:.0f} "
                f"target={f0_tgt:.0f} output={f0_out:.0f} - output stayed closer to source)"
            )
    else:
        warnings.append("could not estimate F0 for source/target/output reliably (low voiced-frame count)")

    if corr is None:
        warnings.append("could not compute envelope correlation (clip too short)")
    elif corr < ENVELOPE_CORR_WARN:
        warnings.append(f"weak energy-envelope correlation ({corr:.2f} < {ENVELOPE_CORR_WARN}); prosody/timing may not be preserved")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": {
            "duration_source_s": round(dur_src, 3),
            "duration_output_s": round(dur_out, 3),
            "duration_ratio": round(duration_ratio, 3),
            "output_rms": round(out_rms, 5),
            "median_f0_source_hz": round(f0_src, 1),
            "median_f0_target_hz": round(f0_tgt, 1),
            "median_f0_output_hz": round(f0_out, 1),
            "envelope_correlation": round(corr, 3) if corr is not None else None,
        },
    }
