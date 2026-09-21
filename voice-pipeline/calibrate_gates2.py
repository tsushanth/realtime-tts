"""Calibration data for the quality gates in gates.py. On LibriTTS-R train-clean-100 (CC BY 4.0), per
clip and speaker, measures: SNR estimate, band ratio, median F0, transcript chars/sec on the CLEAN audio,
plus degraded copies made from the same clips: white noise and 'hum+babble-like' coloured noise at known
SNRs (30/20/12/6 dB), telephone versions (300-3400 Hz band-pass, native 8 kHz and re-upsampled to 16 kHz).
Writes calibration_gates.json next to this file. CPU only (~$0.05-0.10).
   modal run calibrate_gates2.py
"""
import modal

app = modal.App("voice-calibrate-gates", image=modal.Image.debian_slim(python_version="3.11").apt_install("wget")
                .pip_install("soundfile", "numpy<2", "scipy").add_local_python_source("gates"))
URL = "https://www.openslr.org/resources/141/train_clean_100.tar.gz"


@app.function(timeout=3 * 3600, cpu=4, memory=8192)
def run(per_gender: int = 10, clips: int = 60, degrade_clips: int = 25):
    import glob, os, random, subprocess
    from collections import defaultdict
    import numpy as np, soundfile as sf
    from scipy.signal import butter, sosfilt, resample_poly
    import gates

    root = "/tmp/libri"; os.makedirs(root, exist_ok=True)
    subprocess.run(f"wget -q -O - {URL} | tar xz -C {root}", shell=True, check=True)
    gender = {}
    for tsv in glob.glob(f"{root}/**/speakers.tsv", recursive=True):
        for line in open(tsv):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2: gender[p[0]] = p[1]
    dur, files = defaultdict(float), defaultdict(list)
    for wav in glob.glob(f"{root}/**/*.wav", recursive=True):
        spk = os.path.basename(wav).split("_")[0]
        info = sf.info(wav); d = info.frames / info.samplerate
        if 1.0 <= d <= 15.0 and os.path.exists(wav[:-4] + ".normalized.txt"):
            dur[spk] += d; files[spk].append(wav)
    rng = np.random.default_rng(0)
    out = {"clean": [], "degraded": []}
    for g in ("M", "F"):
        spks = sorted([s for s in dur if gender.get(s) == g], key=dur.get, reverse=True)[:per_gender]
        if g == "M" and "6209" not in spks and "6209" in dur: spks.append("6209")
        if g == "M" and "7505" in dur and "7505" not in spks: spks.append("7505")
        for spk in spks:
            fl = files[spk]; random.Random(1).shuffle(fl)
            for i, w in enumerate(fl[:clips]):
                x, sr = sf.read(w)
                txt = open(w[:-4] + ".normalized.txt").read().strip()
                out["clean"].append({"spk": spk, "g": g, "sr": sr, "dur": len(x) / sr, "chars": len(txt),
                                     "snr": gates.snr_estimate_db(x, sr), "band": gates.band_ratio_db(x, sr),
                                     "f0": gates.clip_f0_median(x, sr)})
                if i >= degrade_clips: continue
                sp = np.sqrt(np.mean(x ** 2)) + 1e-9
                for snr in (30, 20, 12, 6):
                    n = rng.standard_normal(len(x))
                    kinds = {"white": n,
                             "hum_lowfreq": sosfilt(butter(2, 400, "low", fs=sr, output="sos"), n) * 4 + 0.5 * np.sin(2 * np.pi * 60 * np.arange(len(x)) / sr),
                             "babble_like": sosfilt(butter(2, [300, 3400], "band", fs=sr, output="sos"), n)}
                    for kind, nz in kinds.items():
                        nz = nz / (np.sqrt(np.mean(nz ** 2)) + 1e-12) * sp / (10 ** (snr / 20))
                        out["degraded"].append({"spk": spk, "kind": f"noise_{kind}", "level": snr, "snr": gates.snr_estimate_db(x + nz, sr)})
                sos = butter(6, [300, 3400], "band", fs=sr, output="sos")
                xb = sosfilt(sos, x)
                out["degraded"].append({"spk": spk, "kind": "phone_bandlimited_same_sr", "band": gates.band_ratio_db(xb, sr)})
                x8 = resample_poly(xb, 8000, sr) if sr != 8000 else xb
                out["degraded"].append({"spk": spk, "kind": "phone_native_8k", "band": gates.band_ratio_db(x8, 8000)})
                x16 = resample_poly(x8, 2, 1)
                out["degraded"].append({"spk": spk, "kind": "phone_up_16k", "band": gates.band_ratio_db(x16, 16000)})
                # merely 16 kHz wideband (should NOT be flagged)
                x16w = resample_poly(x, 16000, sr) if sr != 16000 else x
                out["degraded"].append({"spk": spk, "kind": "wideband_16k", "band": gates.band_ratio_db(x16w, 16000)})
    return out


@app.local_entrypoint()
def main():
    import json
    r = run.remote()
    json.dump(r, open("calibration_gates.json", "w"))
    print(len(r["clean"]), len(r["degraded"]))
