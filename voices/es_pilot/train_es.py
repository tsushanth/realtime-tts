"""Spanish pilot: fine-tune a 2-speaker (F 10246 + M 3946) Piper *medium* voice from the CLEAN English LibriTTS-R base.

Data: CML-TTS Spanish subsets in the `es-pilot` Volume (voices/es_pilot/prep_data.py; CC BY 4.0).
Base: rhasspy/piper-checkpoints en/en_US/libritts_r/medium/best.ckpt (scratch-trained on LibriTTS-R, CC BY 4.0, 904 speakers).
Why it warm-starts across languages: piper1-gpl uses ONE fixed 256-slot phoneme-id map (a universal espeak IPA inventory,
`--data.espeak_voice es` only changes the phonemizer), so embeddings/encoder/flow/decoder/discriminators all line up; only
`emb_g` (904 -> 2 speakers) is re-initialised (non-strict, shape-matched warmstart in VitsModel._warmstart_from_ckpt).

    modal run voices/es_pilot/train_es.py --steps 300 --hours 2 --name smoke       # throughput / OOM smoke test
    modal run --detach voices/es_pilot/train_es.py --steps 60000 --hours 20 --name es2spk
Resumes from last.ckpt in the Volume when re-run with the same --name. Everything lands in es-pilot:/runs/<name>/.
"""
import os

import modal

_PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training-data", "patch_piper_progress.py")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "build-essential", "cmake", "ninja-build", "git", "wget", "ffmpeg")
    .pip_install("torch==2.1.2", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy<2")
    .run_commands("git clone --depth 1 https://github.com/OHF-Voice/piper1-gpl.git /opt/piper-src")
    .apt_install("python3-dev")
    .pip_install("scikit-build", "setuptools", "wheel", "cmake", "ninja")
    .workdir("/opt/piper-src")
    .run_commands("pip install --no-build-isolation -e '.[train]'")
    .run_commands("python3 setup.py build_ext --inplace")
    .run_commands("bash build_monotonic_align.sh")
    .pip_install("numpy<2")
    .add_local_file(_PATCH, remote_path="/tmp/patch_piper_progress.py", copy=True)
    .run_commands("python3 /tmp/patch_piper_progress.py")
    .run_commands("mkdir -p /ckpt && wget -q -O /ckpt/base.ckpt "
                  "'https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/libritts_r/medium/best.ckpt'")
    .pip_install("scipy", "soundfile", "librosa")
)
app = modal.App("es-pilot-train", image=image)
vol = modal.Volume.from_name("es-pilot", create_if_missing=True)
SPEAKERS = {"10246": "f10246", "3946": "m3946"}

TEXTS = [
    "Gracias por llamar, puedo ayudarle con eso. Permítame consultar los datos de su cuenta ahora mismo.",
    "Su pedido debería llegar en un plazo de tres a cinco días hábiles, y le enviaré un correo de confirmación en breve.",
    "Entiendo su frustración, déjeme ver qué puedo hacer para resolverlo.",
    "¿Hay algo más en lo que pueda ayudarle hoy?",
    "Su extensión es el seis seis tres cinco.",
]


@app.function(gpu="A10G", cpu=8, memory=32768, timeout=24 * 3600, volumes={"/data": vol})
def train(steps: int = 60000, hours: float = 20.0, name: str = "es2spk", batch: int = 32, espeak: str = "es"):
    import csv, glob, json, os, random, runpy, shutil, sys, threading, time
    import numpy as np, soundfile as sf
    from scipy.signal import resample_poly

    out = f"/data/runs/{name}"
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    # ---- background volume commits so a crash never loses checkpoints
    stop = threading.Event()
    def committer():
        while not stop.wait(600):
            try: vol.commit()
            except Exception as e: print("commit failed", e, flush=True)
    threading.Thread(target=committer, daemon=True).start()

    # ---- dataset: choose `hours` per speaker, resample 24k -> 22.05k
    rows = [json.loads(l) for p in sorted(glob.glob("/data/manifest/*.jsonl")) for l in open(p, encoding="utf-8")]
    rng = random.Random(0)
    wavs = "/tmp/wavs"; os.makedirs(wavs, exist_ok=True)
    lines, got = [], {}
    for spk, tag in SPEAKERS.items():
        rs = [r for r in rows if str(r["speaker"]) == spk and r["text"].strip()]
        rng.shuffle(rs)
        tot = 0.0
        for r in rs:
            if tot >= hours * 3600: break
            a, sr = sf.read(f"/data/{r['file']}")
            if a.ndim > 1: a = a.mean(1)
            if sr != 22050:
                from math import gcd
                g = gcd(22050, sr); a = resample_poly(a, 22050 // g, sr // g)
            fn = f"{tag}_{len(lines):06d}.wav"
            sf.write(f"{wavs}/{fn}", a, 22050, "PCM_16")
            lines.append((fn, tag, r["text"].strip()))
            tot += r["duration"]
        got[tag] = tot / 3600
    print("dataset hours:", got, "clips:", len(lines), f"prep {time.time()-t_start:.0f}s", flush=True)
    with open("/tmp/train.csv", "w", newline="") as f:
        csv.writer(f, delimiter="|").writerows(lines)

    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    import piper.espeakbridge  # noqa: F401
    import piper.train.__main__ as piper_main

    os.chdir("/opt/piper-src")
    argv = [
        "piper.train", "fit", "--data.voice_name", name, "--data.csv_path", "/tmp/train.csv", "--data.audio_dir", wavs,
        "--data.espeak_voice", espeak, "--data.cache_dir", "/tmp/cache", "--data.config_path", f"{out}/config.json",
        "--data.batch_size", str(batch), "--data.validation_split", "0.005", "--data.num_test_examples", "0",
        "--data.num_workers", "6", "--model.sample_rate", "22050", "--model.num_speakers", "2",
        "--trainer.max_steps", str(steps), "--trainer.default_root_dir", out, "--trainer.precision", "32",
    ]
    last = f"{out}/lightning_logs"
    cks = sorted(glob.glob(f"{last}/version_*/checkpoints/last.ckpt"), key=os.path.getmtime)
    if cks:
        argv += ["--ckpt_path", cks[-1]]; print("resuming from", cks[-1], flush=True)
    else:
        argv += ["--model.warmstart_ckpt", "/ckpt/base.ckpt"]
    sys.argv = argv
    piper_main._DEFAULT_CALLBACKS = piper_main._DEFAULT_CALLBACKS[:1]
    t0 = time.time()
    try:
        piper_main.main()
    except SystemExit as e:
        print("piper.train exit", e.code, flush=True)
    train_s = time.time() - t0
    vol.commit()

    # ---- export + samples
    ck = sorted(glob.glob(f"{out}/lightning_logs/version_*/checkpoints/last.ckpt"), key=os.path.getmtime)[-1]
    onnx = f"{out}/model.onnx"
    sys.argv = ["piper.train.export_onnx", "--checkpoint", ck, "--output-file", onnx]
    runpy.run_module("piper.train.export_onnx", run_name="__main__")
    shutil.copy(f"{out}/config.json", onnx + ".json")
    from piper import PiperVoice, SynthesisConfig
    v = PiperVoice.load(onnx)
    cfgj = json.load(open(onnx + ".json", encoding="utf-8"))
    smap = cfgj["speaker_id_map"]
    os.makedirs(f"{out}/samples", exist_ok=True)
    for tag, sid in smap.items():
        for i, text in enumerate(TEXTS):
            pcm = np.concatenate([c.audio_int16_array for c in v.synthesize(text, SynthesisConfig(speaker_id=sid))])
            sf.write(f"{out}/samples/{tag}_{i+1}.wav", pcm, v.config.sample_rate, "PCM_16")
    stop.set(); vol.commit()
    res = {"name": name, "steps": steps, "hours_per_speaker": got, "train_s": train_s, "total_s": time.time() - t_start,
           "speaker_id_map": smap, "checkpoint": ck}
    json.dump(res, open(f"{out}/result.json", "w"), indent=1); vol.commit()
    print(res, flush=True)
    return res


@app.function(cpu=4, memory=16384, timeout=1800, volumes={"/data": vol})
def sample(name: str = "es2spk", tag: str = ""):
    """Export the newest last.ckpt of run <name> on CPU and write 5 Spanish call-centre samples per speaker to
    /data/runs/<name>/samples_<tag or step>/ (for listening checks while training continues)."""
    import glob, json, os, runpy, shutil, sys
    import numpy as np, soundfile as sf, torch
    vol.reload()
    out = f"/data/runs/{name}"
    ck = sorted(glob.glob(f"{out}/lightning_logs/version_*/checkpoints/last.ckpt"), key=os.path.getmtime)[-1]
    step = torch.load(ck, map_location="cpu", weights_only=False)["global_step"]
    work = f"/tmp/snap"; os.makedirs(work, exist_ok=True)
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    sys.argv = ["piper.train.export_onnx", "--checkpoint", ck, "--output-file", f"{work}/model.onnx"]
    runpy.run_module("piper.train.export_onnx", run_name="__main__")
    shutil.copy(f"{out}/config.json", f"{work}/model.onnx.json")
    from piper import PiperVoice, SynthesisConfig
    v = PiperVoice.load(f"{work}/model.onnx")
    smap = json.load(open(f"{work}/model.onnx.json", encoding="utf-8"))["speaker_id_map"]
    d = f"{out}/samples_{tag or step}"; os.makedirs(d, exist_ok=True)
    for t, sid in smap.items():
        for i, text in enumerate(TEXTS):
            pcm = np.concatenate([c.audio_int16_array for c in v.synthesize(text, SynthesisConfig(speaker_id=sid))])
            sf.write(f"{d}/{t}_{i+1}.wav", pcm, v.config.sample_rate, "PCM_16")
    shutil.copy(f"{work}/model.onnx", f"{out}/model_{step}.onnx"); shutil.copy(f"{work}/model.onnx.json", f"{out}/model_{step}.onnx.json")
    vol.commit()
    return {"step": int(step), "dir": d}


@app.local_entrypoint()
def main(steps: int = 60000, hours: float = 20.0, name: str = "es2spk", batch: int = 32, espeak: str = "es", wait: bool = False, mode: str = "train"):
    if mode == "sample":
        print(sample.remote(name)); return
    if wait:  # smoke tests: block and print the result
        print(train.remote(steps, hours, name, batch, espeak))
    else:  # long runs: use `modal run --detach`
        print("spawned", train.spawn(steps, hours, name, batch, espeak).object_id)
