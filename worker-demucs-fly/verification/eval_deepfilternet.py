"""Runs the exact same test mixes used by make_and_verify.py / expand_coverage.py through
DeepFilterNet instead of htdemucs, and computes the identical metrics (SNR improvement,
non-speech-band energy reduction), for an apples-to-apples comparison.

This does NOT modify make_and_verify.py or expand_coverage.py, and does NOT regenerate the test
audio - it reuses the already-committed mix files those scripts wrote (verification/noisy_mix.wav
and verification/case_*_mix.wav) and the same helper functions (band_energy_fraction, mix_at_snr,
estimate_output_snr) imported directly from make_and_verify.py.

Must be run with a Python environment that has `deepfilternet` + a torch/torchaudio pair it's
compatible with installed (NOT this project's main .venv - see README's DeepFilterNet section for
why: DeepFilterNet 0.5.6 needs an older torchaudio with `torchaudio.backend.common`, incompatible
with this project's torch==2.14.0/torchaudio==2.11.0 pin used for htdemucs). A separate venv was
used to produce the numbers in README.md:

    python3.11 -m venv /tmp/dfn_test_venv
    /tmp/dfn_test_venv/bin/pip install deepfilternet soundfile scipy
    /tmp/dfn_test_venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.1.2" "torchaudio==2.1.2"
    /tmp/dfn_test_venv/bin/python verification/eval_deepfilternet.py
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_and_verify import band_energy_fraction, estimate_output_snr  # noqa: E402

DF_SR = 48000


def dfn_enhance(mix: np.ndarray, sr: int, model, df_state):
    from df.enhance import enhance
    import torch

    mix_48k = resample_poly(mix.astype(np.float64), DF_SR, sr).astype(np.float32)
    t = torch.from_numpy(mix_48k).unsqueeze(0)  # [1, samples], DeepFilterNet wants [channels, T]
    with torch.no_grad():
        enhanced = enhance(model, df_state, t)
    out = enhanced.squeeze(0).cpu().numpy().astype(np.float32)
    return out, DF_SR


def run_case(name, mix_path, sr_expected, clean_ref, clean_ref_sr, model, df_state, note):
    mix, sr = sf.read(mix_path, dtype="float32", always_2d=False)
    assert sr == sr_expected, f"{name}: expected sr {sr_expected}, got {sr}"
    enhanced, out_sr = dfn_enhance(mix, sr, model, df_state)

    sf.write(os.path.join(HERE, f"dfn_case_{name}_enhanced.wav"), enhanced, out_sr, subtype="PCM_16")

    bands = [(0, 80), (6000, 11000)]
    before_frac = band_energy_fraction(mix, sr, bands)
    after_frac = band_energy_fraction(enhanced, out_sr, bands)

    result = {
        "case": name,
        "engine": "deepfilternet3",
        "note": note,
        "non_speech_band_energy_fraction_before": round(float(before_frac), 4),
        "non_speech_band_energy_fraction_after": round(float(after_frac), 4),
        "non_speech_band_energy_reduction_pct":
            round(float((before_frac - after_frac) / before_frac * 100), 1) if before_frac > 0 else None,
    }
    if clean_ref is not None:
        ref_at_out_sr = resample_poly(clean_ref.astype(np.float64), out_sr, clean_ref_sr).astype(np.float32)
        result["estimated_output_snr_db"] = round(float(estimate_output_snr(enhanced, ref_at_out_sr, out_sr)), 2)
    return result


def main():
    from df.enhance import init_df

    model, df_state, _ = init_df()

    speech, speech_sr = sf.read(os.path.join(HERE, "speech.wav"), dtype="float32", always_2d=False)
    real_speech, real_sr = sf.read(os.path.join(HERE, "real_speech_gettysburg.wav"), dtype="float32",
                                    always_2d=False)

    # Known input SNRs (measured_input_snr_db), copied from the already-committed htdemucs runs
    # (metrics.json / coverage_results.json) - these describe the fixed mix files, not the model
    # under test, so reusing them here is correct and not "reusing htdemucs's numbers".
    cases = [
        dict(name="original_drone_0db", mix_path=os.path.join(HERE, "noisy_mix.wav"),
             sr_expected=speech_sr, clean_ref=speech, clean_ref_sr=speech_sr,
             input_snr_db=1.6,
             note="Same clip as make_and_verify.py's original verified case (speech + "
                  "pink-noise/drone at ~0dB), run through DeepFilterNet instead of htdemucs."),
        dict(name="white_noise_0db", mix_path=os.path.join(HERE, "case_white_noise_0db_mix.wav"),
             sr_expected=speech_sr, clean_ref=speech, clean_ref_sr=speech_sr,
             input_snr_db=1.49,
             note="Same mix file expand_coverage.py generated for white_noise_0db, run through "
                  "DeepFilterNet instead of htdemucs."),
        dict(name="real_speech_drone", mix_path=os.path.join(HERE, "case_real_speech_drone_mix.wav"),
             sr_expected=real_sr, clean_ref=real_speech, clean_ref_sr=real_sr,
             input_snr_db=1.63,
             note="Same real (non-TTS) speech + drone/pink-noise mix expand_coverage.py generated "
                  "for real_speech_drone, run through DeepFilterNet instead of htdemucs. This is "
                  "the key comparison this script exists for."),
    ]

    results = []
    for c in cases:
        r = run_case(c["name"], c["mix_path"], c["sr_expected"], c["clean_ref"], c["clean_ref_sr"],
                     model, df_state, c["note"])
        r["measured_input_snr_db"] = c["input_snr_db"]
        r["snr_improvement_db"] = round(r["estimated_output_snr_db"] - c["input_snr_db"], 2)
        results.append(r)

    with open(os.path.join(HERE, "coverage_results_deepfilternet.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'case':<22} {'in_snr':>8} {'out_snr':>8} {'improve':>8} {'band_energy_reduction':>10}")
    for r in results:
        print(f"{r['case']:<22} {r['measured_input_snr_db']:>8} {r['estimated_output_snr_db']:>8} "
              f"{r['snr_improvement_db']:>8} {r['non_speech_band_energy_reduction_pct']:>9}%")
    print("\nFull results written to verification/coverage_results_deepfilternet.json")


if __name__ == "__main__":
    main()
