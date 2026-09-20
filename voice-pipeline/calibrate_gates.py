"""Calibrates two intake gates on real speakers (LibriTTS-R train-clean-100, CC BY 4.0):
  1. male/female cutoff on median F0 (train_job.py uses 165 Hz), checked against the corpus's
     own gender labels;
  2. a "wildly varied delivery" score: std of voiced-frame F0 in semitones. Known by ear:
     speaker 6209 (male, wide intonation) trained shaky even from a male base; 7505 (male) and the
     rank-1 male trained fine.
Prints a per-speaker table for the top speakers of each gender by audio duration.
modal run --detach calibrate_gates.py   ->  results in the app logs and calibration.json (volume voice-models)
"""
import modal

app = modal.App("voice-calibrate", image=modal.Image.debian_slim(python_version="3.11").apt_install("wget").pip_install("soundfile", "numpy"))
models = modal.Volume.from_name("voice-models", create_if_missing=True)
URL = "https://www.openslr.org/resources/141/train_clean_100.tar.gz"


def f0_track(x, sr):
    import numpy as np
    n = int(0.04 * sr); out = []
    lo, hi = int(sr / 400), int(sr / 60)
    for st in range(0, len(x) - n, n // 2):
        fr = x[st:st + n] - np.mean(x[st:st + n])
        if np.sqrt(np.mean(fr ** 2)) < 0.02:
            continue
        ac = np.correlate(fr, fr, "full")[n - 1:]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] / (ac[0] + 1e-9) > 0.5:
            out.append(sr / k)
    return out


@app.function(timeout=6 * 3600, volumes={"/models": models})
def run(top: int = 14, clips_per_speaker: int = 80):
    import glob, json, os, random, subprocess
    from collections import defaultdict
    import numpy as np
    import soundfile as sf

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
        if 1.0 <= d <= 12.0 and os.path.exists(wav[:-4] + ".normalized.txt"):
            dur[spk] += d; files[spk].append(wav)
    rows = []
    for g in ("M", "F"):
        spks = sorted([s for s in dur if gender.get(s) == g], key=dur.get, reverse=True)
        for rank, spk in enumerate(spks[:top]):
            fl = files[spk]; random.Random(0).shuffle(fl)
            allf, utt_means = [], []
            for w in fl[:clips_per_speaker]:
                x, sr = sf.read(w)
                t = f0_track(x, sr)
                if len(t) > 3:
                    allf += t; utt_means.append(float(np.mean(t)))
            if len(allf) < 50:
                continue
            st = 12 * np.log2(np.array(allf) / np.median(allf))
            rows.append({"speaker": spk, "gender": g, "rank": rank, "minutes": round(dur[spk] / 60, 1),
                         "median_f0": round(float(np.median(allf))),
                         "f0_std_semitones": round(float(np.std(st)), 2),
                         "utt_mean_f0_std_semitones": round(float(np.std(12 * np.log2(np.array(utt_means) / np.median(utt_means)))), 2)})
    json.dump(rows, open("/models/calibration.json", "w"), indent=1); models.commit()
    for r in rows: print(r, flush=True)
    return rows


@app.local_entrypoint()
def main():
    print("Spawned:", run.spawn().object_id)
