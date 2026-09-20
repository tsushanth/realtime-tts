"""Builds a customer-style upload zip (FLAC clips + metadata.csv) from one LibriTTS-R speaker (CC BY 4.0),
for end-to-end tests of the intake API.   modal run make_test_zip.py --speaker 7505 --minutes 22 --out /path/x.zip
Also: --mismatch shuffles transcripts (should be rejected), --noise-db 6 adds white noise at that SNR."""
import modal

app = modal.App("voice-test-zip", image=modal.Image.debian_slim(python_version="3.11").apt_install("wget").pip_install("soundfile", "numpy<2"))
URL = "https://www.openslr.org/resources/141/train_clean_100.tar.gz"


@app.function(timeout=3600, cpu=2, memory=8192)
def build(speaker: str, minutes: float, mismatch: bool, noise_db: float) -> bytes:
    import glob, io, os, random, subprocess, zipfile
    import numpy as np, soundfile as sf
    root = "/tmp/libri"; os.makedirs(root, exist_ok=True)
    subprocess.run(f"wget -q -O - {URL} | tar xz -C {root}", shell=True, check=True)
    clips = sorted(w for w in glob.glob(f"{root}/**/{speaker}_*.wav", recursive=True) if os.path.exists(w[:-4] + ".normalized.txt"))
    rows, total = [], 0.0
    buf = io.BytesIO(); z = zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED)
    for w in clips:
        if total >= minutes * 60: break
        x, sr = sf.read(w); d = len(x) / sr
        if not 1.0 <= d <= 15.0: continue
        if noise_db:
            x = x + np.random.default_rng(0).standard_normal(len(x)) * np.sqrt(np.mean(x ** 2)) / 10 ** (noise_db / 20)
        name = os.path.basename(w)[:-4] + ".flac"
        b = io.BytesIO(); sf.write(b, np.clip(x, -1, 1), sr, format="FLAC"); z.writestr(name, b.getvalue())
        rows.append([name, open(w[:-4] + ".normalized.txt").read().strip()]); total += d
    if mismatch:
        t = [r[1] for r in rows]; random.Random(0).shuffle(t)
        for r, tt in zip(rows, t): r[1] = tt
    z.writestr("metadata.csv", "\n".join(f"{n}|{t}" for n, t in rows) + "\n"); z.close()
    print(f"{len(rows)} clips {total/60:.1f} min")
    return buf.getvalue()


@app.local_entrypoint()
def main(speaker: str = "7505", minutes: float = 22.0, out: str = "test.zip", mismatch: bool = False, noise_db: float = 0.0):
    open(out, "wb").write(build.remote(speaker, minutes, mismatch, noise_db))
    print("wrote", out)
