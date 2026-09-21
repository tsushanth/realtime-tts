"""
Before/after verification for the audio-isolation worker.

How the test clip was made (see README.md "Verification" for the narrative version):
  1. speech.wav: macOS `say -v Samantha` synthesized a ~10s sentence, converted to a 22.05kHz
     mono 16-bit WAV via `afconvert` (both commands run once, by hand; their output is checked
     into this directory so the test is reproducible without macOS).
  2. This script loads speech.wav, synthesizes "background noise/music" as a mix of:
       - broadband pink-ish noise (filtered white noise) at a fixed RMS relative to the speech,
       - a low-frequency musical drone (a few sine tones with slow vibrato, like a synth pad)
     entirely with numpy/scipy - no external samples - and adds it to the speech to make
     noisy_mix.wav at a target input SNR (~0 dB, i.e. noise as loud as the speech - a hard case).
  3. It POSTs noisy_mix.wav to a locally running worker's /v1/isolate (or calls the Separator
     class in-process if SEPARATE_IN_PROCESS=1, avoiding the need for a live server), saving the
     result as isolated.wav.
  4. It computes an SNR-style before/after metric two ways:
       a) "oracle" SNR: since we know the exact speech and noise signals we mixed, before-SNR is
          computed directly from speech vs noise; after-SNR approximates the isolated output as
          (speech proxy) vs (isolated - speech proxy) after time-aligning and scaling, i.e. how much
          of the noise energy survived isolation relative to speech energy retained.
       b) energy-in-non-speech-band: the fraction of total energy in the 0-80Hz and 6-11kHz bands
          (where the synthetic noise/drone concentrates energy that clean speech mostly doesn't),
          before vs after.
  Numbers are printed and written to metrics.json.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, correlate

HERE = os.path.dirname(os.path.abspath(__file__))


def band_energy_fraction(x: np.ndarray, sr: int, low_bands):
    """Fraction of total signal energy falling in the given (lo,hi) Hz bands."""
    total = np.sum(x.astype(np.float64) ** 2) + 1e-12
    frac = 0.0
    for lo, hi in low_bands:
        hi = min(hi, sr / 2 - 1)
        if lo >= hi:
            continue
        if lo <= 0:
            b, a = butter(4, hi / (sr / 2), btype="low")
        else:
            b, a = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
        y = lfilter(b, a, x)
        frac += np.sum(y.astype(np.float64) ** 2) / total
    return frac


def make_noise(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float64)
    b, a = butter(2, 2000 / (sr / 2), btype="low")  # pink-ish: emphasize low/mid freqs
    pink = lfilter(b, a, white)
    t = np.arange(n) / sr
    drone = np.zeros(n)
    for f0, amp in ((70.0, 1.0), (140.0, 0.5), (210.0, 0.3)):
        vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 0.3 * t)
        drone += amp * np.sin(2 * np.pi * f0 * vibrato * t)
    noise = pink / (np.std(pink) + 1e-9) + 0.8 * drone / (np.std(drone) + 1e-9)
    return (noise / (np.std(noise) + 1e-9)).astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    speech_rms = np.sqrt(np.mean(speech.astype(np.float64) ** 2)) + 1e-12
    noise_rms = np.sqrt(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    target_noise_rms = speech_rms / (10 ** (target_snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mix = speech.astype(np.float64) + scaled_noise
    peak = np.max(np.abs(mix))
    if peak > 0.98:
        mix = mix * (0.98 / peak)
        scaled_noise = scaled_noise * (0.98 / peak)
    return mix.astype(np.float32), scaled_noise.astype(np.float32)


def estimate_output_snr(isolated: np.ndarray, speech_ref: np.ndarray, sr: int) -> float:
    """Aligns isolated (which may differ in length/offset from Demucs' framing) to speech_ref via
    cross-correlation, scales to best-fit amplitude, then treats (isolated - scaled speech_ref) as
    residual noise. This is the standard "SNR of an estimate against a known reference" recipe."""
    n = min(len(isolated), len(speech_ref))
    a, b = isolated[:n].astype(np.float64), speech_ref[:n].astype(np.float64)
    corr = correlate(a, b, mode="full")
    lag = np.argmax(np.abs(corr)) - (len(b) - 1)
    if lag >= 0:
        a2, b2 = a[lag:], b[: len(a) - lag]
    else:
        a2, b2 = a[: len(a) + lag], b[-lag:]
    m = min(len(a2), len(b2))
    a2, b2 = a2[:m], b2[:m]
    if m < sr * 0.5:  # alignment degenerate, fall back to unaligned
        a2, b2 = a[: len(b)], b[: len(a)]
    scale = np.dot(a2, b2) / (np.dot(b2, b2) + 1e-12)
    residual = a2 - scale * b2
    signal_energy = np.sum((scale * b2) ** 2) + 1e-12
    noise_energy = np.sum(residual ** 2) + 1e-12
    return 10 * np.log10(signal_energy / noise_energy)


def main():
    speech, sr = sf.read(os.path.join(HERE, "speech.wav"), dtype="float32", always_2d=False)
    rng = np.random.default_rng(42)
    noise = make_noise(len(speech), sr, rng)
    mix, scaled_noise = mix_at_snr(speech, noise, target_snr_db=0.0)
    sf.write(os.path.join(HERE, "noisy_mix.wav"), mix, sr, subtype="PCM_16")

    input_snr_db = 10 * np.log10(
        (np.sum(speech.astype(np.float64) ** 2) + 1e-12) / (np.sum(scaled_noise.astype(np.float64) ** 2) + 1e-12)
    )

    sys.path.insert(0, os.path.dirname(HERE))
    from server import Separator  # noqa: E402

    sep = Separator(os.environ.get("MODEL_NAME", "htdemucs"))
    isolated, out_sr = sep.separate(mix.reshape(-1, 1), sr, "vocals")
    sf.write(os.path.join(HERE, "isolated.wav"), isolated, out_sr, subtype="PCM_16")

    # Resample the float speech reference to the model's output rate for a fair comparison
    # (Demucs runs at 44.1kHz; our source clip is 22.05kHz).
    from scipy.signal import resample_poly
    speech_at_out_sr = resample_poly(speech, out_sr, sr).astype(np.float32)

    output_snr_db = estimate_output_snr(isolated, speech_at_out_sr, out_sr)

    bands = [(0, 80), (6000, 11000)]
    before_frac = band_energy_fraction(mix, sr, bands)
    after_frac = band_energy_fraction(isolated, out_sr, bands)

    metrics = {
        "input_clip": "noisy_mix.wav",
        "output_clip": "isolated.wav",
        "target_mix_snr_db": 0.0,
        "measured_input_snr_db": round(float(input_snr_db), 2),
        "estimated_output_snr_db": round(float(output_snr_db), 2),
        "snr_improvement_db": round(float(output_snr_db - input_snr_db), 2),
        "non_speech_band_energy_fraction_before": round(float(before_frac), 4),
        "non_speech_band_energy_fraction_after": round(float(after_frac), 4),
        "non_speech_band_energy_reduction_pct": round(float((before_frac - after_frac) / before_frac * 100), 1) if before_frac > 0 else None,
        "note": "SNR is an ESTIMATE (cross-correlation-aligned, best-fit-scaled residual against "
                "the known clean speech reference used to build the mix), not a ground-truth "
                "measurement of the model's actual denoising quality on arbitrary audio.",
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
