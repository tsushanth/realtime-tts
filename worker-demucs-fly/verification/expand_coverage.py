"""Expanded verification coverage beyond the single synthetic clip in make_and_verify.py.

README.md's "What's left" flagged: "only one synthetic clip/noise profile was tested... real
customer audio (phone calls, real background noise/music, multiple speakers) has not been
tried." This script adds:

  1. white_noise_0db   - pure broadband white noise (no drone/music component) at 0dB SNR
  2. white_noise_10db  - same white noise, easier +10dB SNR (background hiss, not drowning)
  3. echo_reverb       - speech convolved with a synthetic room impulse response (decaying
                         echoes), simulating a reverberant room rather than additive noise
  4. second_voice      - a second TTS voice (different speaker) mixed in as "noise" at 0dB SNR,
                         i.e. a crosstalk/competing-speaker case rather than non-speech noise
  5. real_speech_drone - a REAL (non-TTS) public-domain speech recording (Gettysburg Address
                         reading, sourced from archive.org, trimmed to 12s and resampled) mixed
                         with the original drone+pink-noise profile at 0dB SNR - the first
                         non-synthetic-speech test of this pipeline (README's "no real-world
                         audio tested" gap). No "oracle" SNR is computable for this case since we
                         don't have an isolated-speech-only reference for the real recording (it's
                         speech to begin with, no separate clean/noise split) - see "note" in its
                         metrics for what IS and isn't measured for that case.

Each case reuses make_and_verify.py's SNR/band-energy machinery. Writes one metrics JSON per case
to verification/coverage_results.json and prints a summary table. Run after make_and_verify.py
has already produced speech.wav (that file is reused as the base TTS speech signal here).

    .venv/bin/python verification/expand_coverage.py
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from make_and_verify import band_energy_fraction, mix_at_snr, estimate_output_snr  # noqa: E402
from server import Separator  # noqa: E402


def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(n).astype(np.float32)
    return w / (np.std(w) + 1e-9)


def synthetic_rir(sr: int, rt60_s: float, rng: np.random.Generator) -> np.ndarray:
    """A crude synthetic room impulse response: exponentially decaying noise (Schroeder-style),
    good enough to simulate "speech recorded in a reverberant room" for a sanity check - not a
    physically measured RIR."""
    n = int(rt60_s * sr)
    t = np.arange(n) / sr
    decay = np.exp(-6.9 * t / rt60_s)  # -60dB at t=rt60_s
    noise = rng.standard_normal(n)
    rir = (decay * noise).astype(np.float32)
    rir[0] = 1.0  # direct path dominates
    rir /= np.max(np.abs(rir)) + 1e-9
    return rir


def load_or_die(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - run make_and_verify.py first (needs speech.wav)")
    return sf.read(path, dtype="float32", always_2d=False)


def tile_to_length(x: np.ndarray, n: int) -> np.ndarray:
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def run_case(name: str, mix: np.ndarray, sr: int, sep: Separator, clean_ref: np.ndarray | None,
             note: str) -> dict:
    sf.write(os.path.join(HERE, f"case_{name}_mix.wav"), mix, sr, subtype="PCM_16")
    isolated, out_sr = sep.separate(mix.reshape(-1, 1), sr, "vocals")
    sf.write(os.path.join(HERE, f"case_{name}_isolated.wav"), isolated, out_sr, subtype="PCM_16")

    bands = [(0, 80), (6000, 11000)]
    before_frac = band_energy_fraction(mix, sr, bands)
    after_frac = band_energy_fraction(isolated, out_sr, bands)

    result = {
        "case": name,
        "note": note,
        "non_speech_band_energy_fraction_before": round(float(before_frac), 4),
        "non_speech_band_energy_fraction_after": round(float(after_frac), 4),
        "non_speech_band_energy_reduction_pct":
            round(float((before_frac - after_frac) / before_frac * 100), 1) if before_frac > 0 else None,
    }
    if clean_ref is not None:
        from scipy.signal import resample_poly
        ref_at_out_sr = resample_poly(clean_ref, out_sr, sr).astype(np.float32)
        result["estimated_output_snr_db"] = round(float(estimate_output_snr(isolated, ref_at_out_sr, out_sr)), 2)
    return result


def main():
    speech, sr = load_or_die(os.path.join(HERE, "speech.wav"))
    rng = np.random.default_rng(123)
    sep = Separator(os.environ.get("MODEL_NAME", "htdemucs"))
    results = []

    # 1 & 2: white noise at two SNR levels
    for snr_db, label in ((0.0, "white_noise_0db"), (10.0, "white_noise_10db")):
        noise = white_noise(len(speech), rng)
        mix, scaled_noise = mix_at_snr(speech, noise, snr_db)
        input_snr = 10 * np.log10(
            (np.sum(speech.astype(np.float64) ** 2) + 1e-12) /
            (np.sum(scaled_noise.astype(np.float64) ** 2) + 1e-12))
        r = run_case(label, mix, sr, sep, speech,
                     f"pure white noise (no music/drone component), target {snr_db}dB SNR")
        r["measured_input_snr_db"] = round(float(input_snr), 2)
        r["snr_improvement_db"] = round(r["estimated_output_snr_db"] - r["measured_input_snr_db"], 2)
        results.append(r)

    # 3: echo/reverb (convolutional, not additive) at a moderate RT60
    rir = synthetic_rir(sr, rt60_s=0.4, rng=rng)
    reverbed = fftconvolve(speech.astype(np.float64), rir.astype(np.float64), mode="full")[: len(speech)]
    peak = np.max(np.abs(reverbed)) + 1e-9
    reverbed = (reverbed / peak * 0.9).astype(np.float32)
    r = run_case("echo_reverb", reverbed, sr, sep, speech,
                 "speech convolved with a synthetic RT60=0.4s room impulse response - tests "
                 "convolutional distortion, not additive noise. SNR metric here compares "
                 "isolated output to the DRY (non-reverberant) reference, so it also reflects "
                 "how much reverb tail Demucs removes vs. treats as signal.")
    results.append(r)

    # 4: second voice as "noise" (crosstalk / competing speaker)
    second, sr2 = load_or_die(os.path.join(HERE, "second_voice.wav"))
    assert sr2 == sr, f"sample rate mismatch: {sr2} vs {sr}"
    second_tiled = tile_to_length(second, len(speech))
    mix, scaled_noise = mix_at_snr(speech, second_tiled, 0.0)
    input_snr = 10 * np.log10(
        (np.sum(speech.astype(np.float64) ** 2) + 1e-12) /
        (np.sum(scaled_noise.astype(np.float64) ** 2) + 1e-12))
    r = run_case("second_voice", mix, sr, sep, speech,
                 "a second TTS speaker (different voice) mixed in at 0dB as competing speech, "
                 "not non-speech noise - tests whether htdemucs' 'vocals' stem isolates the "
                 "TARGET voice or just 'any speech vs non-speech', which it is not designed to "
                 "distinguish (htdemucs has no speaker-conditioning input). Expect this case to "
                 "perform worse than noise-based cases.")
    r["measured_input_snr_db"] = round(float(input_snr), 2)
    r["snr_improvement_db"] = round(r["estimated_output_snr_db"] - r["measured_input_snr_db"], 2)
    results.append(r)

    # 5: REAL (non-TTS) speech + the original drone/pink-noise profile
    real_speech, sr3 = load_or_die(os.path.join(HERE, "real_speech_gettysburg.wav"))
    assert sr3 == sr, f"sample rate mismatch: {sr3} vs {sr}"
    from make_and_verify import make_noise
    drone_noise = make_noise(len(real_speech), sr, np.random.default_rng(7))
    mix, scaled_noise = mix_at_snr(real_speech, drone_noise, 0.0)
    input_snr = 10 * np.log10(
        (np.sum(real_speech.astype(np.float64) ** 2) + 1e-12) /
        (np.sum(scaled_noise.astype(np.float64) ** 2) + 1e-12))
    r = run_case("real_speech_drone", mix, sr, sep, real_speech,
                 "REAL (non-TTS) public-domain speech recording (archive.org 'Gettysburg "
                 "Address' reading, trimmed to 12s, resampled to match) mixed with the same "
                 "synthetic drone+pink-noise profile as the original verified clip, at 0dB SNR. "
                 "The SNR estimate here uses the real recording itself as the 'clean' reference "
                 "(it was clean before we added synthetic noise to it), so it IS a valid oracle "
                 "SNR measurement, unlike a case with no clean reference at all. This is the "
                 "first test of this pipeline on audio that isn't macOS `say`-synthesized.")
    r["measured_input_snr_db"] = round(float(input_snr), 2)
    r["snr_improvement_db"] = round(r["estimated_output_snr_db"] - r["measured_input_snr_db"], 2)
    results.append(r)

    with open(os.path.join(HERE, "coverage_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'case':<20} {'in_snr':>8} {'out_snr':>8} {'improve':>8} {'band_energy_reduction':>10}")
    for r in results:
        print(f"{r['case']:<20} {r.get('measured_input_snr_db', 'n/a'):>8} "
              f"{r.get('estimated_output_snr_db', 'n/a'):>8} {r.get('snr_improvement_db', 'n/a'):>8} "
              f"{r.get('non_speech_band_energy_reduction_pct', 'n/a'):>9}%")
    print("\nFull results written to verification/coverage_results.json")


if __name__ == "__main__":
    main()
