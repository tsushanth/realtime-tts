"""Minimum-audio-length experiment. For one LibriTTS-R speaker (CC BY 4.0) trains the SAME recipe
(train_job.steps_for, batch 8, gender-matched public-domain base) at 5, 10, 15 and 25 minutes, synthesizes
the standard 5 sentences, computes simple objective proxies relative to the 25-min voice, and writes ONE
combined listening WAV per speaker (order 5,10,15,25 min for sentence 0, then again for sentence 2).
Objective proxies cannot judge naturalness: the recommendation needs a human listener.

  modal run minlength_experiment.py            # both speakers (F rank 0, M 7505); ~ $1-2 of T4 time
Outputs (local, next to this file): minlength_listen_F.wav, minlength_listen_M.wav, minlength_results.json
Uses its own app name (voice-minlength); does not touch production apps.
"""
import modal

# Recipe constants are duplicated from train_job.py (kept in sync by test below) so the container does not
# need to import train_job; only the local side imports it, for its image (piper stack + both bases).
TEST_SENTENCES = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
]
BASES = {"F": "/ckpt/lj_medium.ckpt", "M": "/ckpt/john_medium.ckpt"}


def steps_for(minutes: float) -> int:
    return int(min(6000, max(2000, 2000 + 80 * minutes)))  # same as train_job.steps_for


if modal.is_local():
    import train_job
    assert train_job.TEST_SENTENCES == TEST_SENTENCES and train_job.BASES == BASES and train_job.steps_for(10) == steps_for(10)
    _image = train_job.image
else:
    _image = None
app = modal.App("voice-minlength", image=_image)
URL = "https://www.openslr.org/resources/141/train_clean_100.tar.gz"
MINUTES = [5, 10, 15, 25]


def proxies(wav_path):
    """Voiced-frame F0 jitter (std of frame-to-frame pitch change, semitones; lower = steadier),
    mean spectral flatness on voiced frames (higher = noisier/breathier), pause fraction and mean pause length."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(wav_path)
    n = int(0.025 * sr)
    m = len(x) // n
    fr = x[: m * n].reshape(m, n)
    rms = np.sqrt((fr ** 2).mean(1))
    thr = max(0.02 * rms.max(), 1e-4)
    voiced_f0, flat, silent = [], [], rms < thr
    lo, hi = int(sr / 400), int(sr / 60)
    for i in range(m):
        if silent[i]:
            continue
        f = fr[i] - fr[i].mean()
        spec = np.fft.rfft(f, 2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] / (ac[0] + 1e-9) > 0.5:
            voiced_f0.append(sr / k)
            p = np.abs(np.fft.rfft(f * np.hanning(n))) ** 2 + 1e-12
            flat.append(float(np.exp(np.log(p).mean()) / p.mean()))
        else:
            voiced_f0.append(np.nan)
    f0 = np.array(voiced_f0, dtype=float)
    ok = ~np.isnan(f0)
    st = 12 * np.log2(f0 / np.nanmedian(f0))
    d = np.diff(st)
    d = d[~np.isnan(d)]
    runs, cur = [], 0
    for s_ in silent:
        if s_:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    return {"seconds": round(len(x) / sr, 2), "f0_jitter_st": round(float(np.std(d)), 3) if len(d) else None,
            "voiced_frames_frac": round(float(ok.sum() / max(1, m)), 3), "spectral_flatness": round(float(np.mean(flat)), 4) if flat else None,
            "pause_frac": round(float(silent.mean()), 3), "mean_pause_ms": round(25 * float(np.mean(runs)), 0) if runs else 0}


@app.function(gpu="T4", timeout=5 * 3600, cpu=4, memory=16384)
def run_speaker(gender: str, rank: int, spk_override: str = ""):
    import csv, glob, io, os, runpy, shutil, subprocess, sys
    from collections import defaultdict
    from math import gcd
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    root = "/tmp/libri"; os.makedirs(root, exist_ok=True)
    subprocess.run(f"wget -q -O - {URL} | tar xz -C {root}", shell=True, check=True)
    gmap = {}
    for tsv in glob.glob(f"{root}/**/speakers.tsv", recursive=True):
        for line in open(tsv):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2: gmap[p[0]] = p[1]
    dur, utts = defaultdict(float), defaultdict(list)
    for wav in glob.glob(f"{root}/**/*.wav", recursive=True):
        spk = os.path.basename(wav).split("_")[0]
        txt = wav[:-4] + ".normalized.txt"
        info = sf.info(wav); d = info.frames / info.samplerate
        if os.path.exists(txt) and 1.0 <= d <= 12.0 and gmap.get(spk) == gender:
            dur[spk] += d; utts[spk].append((wav, txt, d))
    spk = spk_override or sorted(dur, key=dur.get, reverse=True)[rank]
    clips = sorted(utts[spk])
    print(f"speaker {spk} ({gender}): {dur[spk]/60:.1f} min available", flush=True)
    base_ckpt = BASES[gender]

    import torch
    _o = torch.load
    torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})
    import piper.espeakbridge  # noqa: F401
    import piper.train.__main__ as piper_main
    from piper import PiperVoice

    results, wavs_out = {}, {}
    for minutes in MINUTES:
        name = f"m{minutes}"
        wd = f"/tmp/{name}_wavs"; os.makedirs(wd, exist_ok=True)
        rows, total = [], 0.0
        for i, (wav, txt, d) in enumerate(clips):  # same clips in the same order: shorter sets are prefixes
            if total >= minutes * 60: break
            a, sr = sf.read(wav)
            g = gcd(22050, sr)
            a = resample_poly(a, 22050 // g, sr // g)
            a = a / (np.max(np.abs(a)) or 1.0) * 0.9
            fn = f"{i:05d}.wav"; sf.write(f"{wd}/{fn}", a, 22050, "PCM_16")
            rows.append((fn, open(txt).read().strip())); total += d
        with open(f"/tmp/{name}.csv", "w", newline="") as f:
            csv.writer(f, delimiter="|").writerows(rows)
        steps = steps_for(total / 60)  # the production rule
        out = f"/tmp/run_{name}"
        print(f"=== {gender} {name}: {len(rows)} clips {total/60:.1f} min steps={steps}", flush=True)
        os.chdir("/opt/piper-src")
        sys.argv = ["piper.train", "fit", "--data.voice_name", name, "--data.csv_path", f"/tmp/{name}.csv",
                    "--data.audio_dir", wd, "--data.espeak_voice", "en-us", "--data.cache_dir", f"/tmp/cache_{name}",
                    "--data.config_path", f"{out}/config.json", "--data.batch_size", "8",
                    "--model.sample_rate", "22050", "--model.warmstart_ckpt", base_ckpt,
                    "--trainer.max_steps", str(steps), "--trainer.default_root_dir", out]
        piper_main._DEFAULT_CALLBACKS = piper_main._DEFAULT_CALLBACKS[:1]
        try:
            piper_main.main()
        except SystemExit as e:
            print("exit", e.code, flush=True)
        ck = sorted(glob.glob(f"{out}/lightning_logs/version_*/checkpoints/last.ckpt"), key=os.path.getmtime)[-1]
        onnx = f"/tmp/{name}.onnx"
        sys.argv = ["piper.train.export_onnx", "--checkpoint", ck, "--output-file", onnx]
        runpy.run_module("piper.train.export_onnx", run_name="__main__")
        shutil.copy(f"{out}/config.json", onnx + ".json")
        v = PiperVoice.load(onnx)
        results[minutes] = {"clips": len(rows), "actual_minutes": round(total / 60, 1), "steps": steps, "sentences": []}
        for i, t in enumerate(TEST_SENTENCES):
            pcm = np.concatenate([c.audio_int16_array for c in v.synthesize(t)])
            p = f"/tmp/{name}_s{i}.wav"; sf.write(p, pcm, v.config.sample_rate, "PCM_16")
            results[minutes]["sentences"].append(proxies(p))
            wavs_out[(minutes, i)] = (pcm, v.config.sample_rate)
        shutil.rmtree(out, ignore_errors=True)

    # aggregate proxies per length; relative difference to the 25-min voice
    agg = {}
    for m_, r in results.items():
        keys = ["f0_jitter_st", "spectral_flatness", "pause_frac", "mean_pause_ms", "seconds"]
        agg[m_] = {k: round(float(np.mean([s[k] for s in r["sentences"] if s[k] is not None])), 4) for k in keys}
    ref = agg[25]
    for m_ in agg:
        agg[m_]["rel_to_25min"] = {k: round(agg[m_][k] / ref[k], 2) if ref[k] else None for k in ("f0_jitter_st", "spectral_flatness", "pause_frac", "seconds")}
    sr_ = wavs_out[(5, 0)][1]
    gap = np.zeros(int(0.9 * sr_), dtype=np.int16)
    seq = []
    for sent in (0, 2):
        for m_ in MINUTES:
            seq += [wavs_out[(m_, sent)][0], gap]
        seq.append(np.zeros(int(1.8 * sr_), dtype=np.int16))
    buf = io.BytesIO(); sf.write(buf, np.concatenate(seq), sr_, format="WAV", subtype="PCM_16")
    return {"speaker": spk, "gender": gender, "per_length": results, "aggregate": agg, "wav": buf.getvalue()}


@app.local_entrypoint()
def main():
    import json
    calls = [run_speaker.spawn("F", 0), run_speaker.spawn("M", 0, "7505")]
    out = {}
    for c in calls:
        r = c.get()
        open(f"minlength_listen_{r['gender']}.wav", "wb").write(r.pop("wav"))
        out[r["gender"]] = r
        print(r["gender"], r["speaker"], json.dumps(r["aggregate"]))
    json.dump(out, open("minlength_results.json", "w"), indent=1)
