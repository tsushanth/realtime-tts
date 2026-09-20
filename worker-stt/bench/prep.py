"""One-off: download LibriTTS-R dev_clean inside Modal, pick 100 utterances (2-12 s), write 16k clean + 8k telephone (mu-law) versions."""
import modal
from common import image, vol
app = modal.App("stt-bench-prep", image=image)

@app.function(volumes={"/data": vol}, timeout=1800)
def prep(n=100):
    import os, tarfile, random, json, subprocess, numpy as np, soundfile as sf
    from scipy.signal import resample_poly
    os.makedirs("/tmp/x", exist_ok=True)
    subprocess.run("curl -sL https://www.openslr.org/resources/141/dev_clean.tar.gz -o /tmp/dc.tgz", shell=True, check=True)
    print("size", os.path.getsize("/tmp/dc.tgz"))
    subprocess.run("tar xzf /tmp/dc.tgz -C /tmp/x", shell=True, check=True)
    wavs = []
    for r, _, fs in os.walk("/tmp/x"):
        for f in fs:
            if f.endswith(".wav"): wavs.append(os.path.join(r, f))
    print("total wavs", len(wavs))
    random.seed(0); random.shuffle(wavs)
    def mulaw_rt(x):
        mu = 255.0
        y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        q = np.round((y + 1) / 2 * 255) / 255 * 2 - 1
        return np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    out = []
    os.makedirs("/data/clean", exist_ok=True); os.makedirs("/data/tel", exist_ok=True)
    for w in wavs:
        if len(out) >= n: break
        x, sr = sf.read(w, dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        d = len(x) / sr
        txt_path = w[:-4] + ".normalized.txt"
        if not (2 <= d <= 12) or not os.path.exists(txt_path): continue
        ref = open(txt_path).read().strip()
        x16 = resample_poly(x, 2, 3).astype("float32")           # 24k -> 16k
        x8 = resample_poly(x, 1, 3)                              # 24k -> 8k
        x8 = mulaw_rt(np.clip(x8, -1, 1))
        t16 = resample_poly(x8, 2, 1).astype("float32")          # 8k -> 16k (what a call pipeline feeds STT)
        name = os.path.basename(w)[:-4]
        sf.write(f"/data/clean/{name}.wav", x16, 16000, subtype="PCM_16")
        sf.write(f"/data/tel/{name}.wav", t16, 16000, subtype="PCM_16")
        out.append({"id": name, "dur": d, "ref": ref})
    json.dump(out, open("/data/manifest.json", "w"))
    vol.commit()
    print("wrote", len(out), "clips, total s", sum(o["dur"] for o in out))

@app.local_entrypoint()
def main():
    prep.remote()
