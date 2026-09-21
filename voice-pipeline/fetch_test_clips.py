"""Pulls a couple of short clips from two different LibriTTS-R speakers (CC BY 4.0, same public
dataset used by make_test_dataset.py) as test source/target audio for convert_job.py - so the VC
MVP can be run end to end without any real customer audio.

modal run fetch_test_clips.py
"""
import modal

app = modal.App("voice-convert-dev-testdata", image=modal.Image.debian_slim(python_version="3.11").apt_install("wget").pip_install("soundfile"))
URL = "https://www.openslr.org/resources/141/dev_clean.tar.gz"  # smaller split than train_clean_100


@app.function(timeout=1800)
def fetch():
    import glob, os, shutil, subprocess
    from collections import defaultdict
    import soundfile as sf

    root = "/tmp/libri"
    os.makedirs(root, exist_ok=True)
    subprocess.run(f"wget -q -O - {URL} | tar xz -C {root}", shell=True, check=True)
    by_spk = defaultdict(list)
    for wav in glob.glob(f"{root}/**/*.wav", recursive=True):
        spk = os.path.basename(wav).split("_")[0]
        by_spk[spk].append(wav)
    speakers = sorted(by_spk, key=lambda s: -len(by_spk[s]))[:2]
    out = {}
    for role, spk in zip(("source", "target"), speakers):
        # pick a clip in a comfortable 4-10s range
        cand = None
        for wav in by_spk[spk]:
            d = sf.info(wav).frames / sf.info(wav).samplerate
            if 4.0 <= d <= 10.0:
                cand = wav
                break
        cand = cand or by_spk[spk][0]
        data = open(cand, "rb").read()
        out[role] = (spk, os.path.basename(cand), data)
    return out


@app.local_entrypoint()
def main(outdir: str = "./test_clips"):
    import os

    os.makedirs(outdir, exist_ok=True)
    res = fetch.remote()
    for role, (spk, fn, data) in res.items():
        p = f"{outdir}/{role}_{spk}_{fn}"
        open(p, "wb").write(data)
        print(role, "speaker", spk, "->", p, f"({len(data)} bytes)")
