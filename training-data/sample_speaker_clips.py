"""Pull N random raw clips (+ transcripts) of the speaker piper_lj_smalldata.py picks for a
given gender/rank, so the TRAINING DATA can be listened to directly (is the shakiness in the
recordings themselves?). Output: speaker_clips/<gender><rank>/ (wav + clips.txt)."""
import modal

app = modal.App("speaker-clip-sampler", image=modal.Image.debian_slim(python_version="3.11").apt_install("wget").pip_install("soundfile", "numpy"))
URL = "https://www.openslr.org/resources/141/train_clean_100.tar.gz"


@app.function(timeout=3600)
def sample(gender: str = "M", rank: int = 0, n: int = 10, seed: int = 1):
    import glob, os, random, subprocess
    from collections import defaultdict
    import soundfile as sf

    root = "/tmp/libri"; os.makedirs(root, exist_ok=True)
    subprocess.run(f"wget -q -O - {URL} | tar xz -C {root}", shell=True, check=True)
    spk_gender = {}
    for tsv in glob.glob(f"{root}/**/speakers.tsv", recursive=True):
        for line in open(tsv):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2: spk_gender[p[0]] = p[1]
    dur, utts = defaultdict(float), defaultdict(list)
    for wav in glob.glob(f"{root}/**/*.wav", recursive=True):
        spk = os.path.basename(wav).split("_")[0]
        info = sf.info(wav); d = info.frames / info.samplerate
        txt = wav[:-4] + ".normalized.txt"
        if os.path.exists(txt) and 1.0 <= d <= 12.0 and spk_gender.get(spk) == gender:
            dur[spk] += d; utts[spk].append((wav, txt, d))
    spk = sorted(dur, key=dur.get, reverse=True)[rank]
    random.Random(seed).shuffle(utts[spk])
    out = []
    for wav, txt, d in utts[spk][:n]:
        out.append((os.path.basename(wav), open(wav, "rb").read(), open(txt).read().strip(), d))
    return spk, dur[spk] / 60, out


@app.local_entrypoint()
def main(gender: str = "M", rank: int = 0, n: int = 10):
    import os
    spk, mins, clips = sample.remote(gender, rank, n)
    d = f"speaker_clips/{gender}{rank}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/clips.txt", "w") as f:
        f.write(f"speaker {spk}, {mins:.1f} min usable\n")
        for name, data, text, dur in clips:
            open(f"{d}/{name}", "wb").write(data); f.write(f"{name} ({dur:.1f}s): {text}\n")
    print(open(f"{d}/clips.txt").read())
