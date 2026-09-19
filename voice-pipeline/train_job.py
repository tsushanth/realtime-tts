"""Voice pipeline, step 1: dataset in -> trained Piper voice out.

Reads a customer dataset from the `voice-datasets` Volume:
    /<voice_id>/consent.json     REQUIRED - {"consent": true, "speaker_name": ..., "attested_by": ..., "date": ...}
    /<voice_id>/metadata.csv     file|transcript   (one line per clip)
    /<voice_id>/wavs/<file>      any sample rate / channels the audio library can read
and writes to the `voice-models` Volume under /<voice_id>/:
    model.onnx, model.onnx.json, samples/sample_N.wav, manifest.json  (or error.json on rejection)

Fine-tunes from Piper's public-domain LJ Speech checkpoint (see training-data/synthesize_ljspeech.py),
which keeps the base licence-clean. NOT YET VALIDATED BY EAR: training-data/piper_lj_smalldata.py
produces 10-min and ~25-min test voices; whether they sound good is still to be judged.

The consent file is a hard gate: no consent.json with consent=true means no training. It is a
record of attestation, not identity verification - that belongs to the intake service.

Usage:  modal run train_job.py --voice-id acme-agent-1 --dataset-dir ./my_dataset
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "build-essential", "cmake", "ninja-build", "git", "wget")
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
    # dataset.py's prepare_data() logs "Processing utterances..." exactly
    # once at the start and "Processed N utterance(s)" exactly once at the
    # end - nothing in between, no matter how many files. At full-corpus
    # scale that step is silent for a long time by design, not because it's
    # stuck - patch in progress logging every 500 utterances (see
    # patch_piper_progress.py) so this is actually observable instead of
    # indistinguishable from a real hang.
    .add_local_file("patch_piper_progress.py", remote_path="/tmp/patch_piper_progress.py", copy=True)
    .run_commands("python3 /tmp/patch_piper_progress.py")
    .run_commands(
        "mkdir -p /ckpt && wget -q -O /ckpt/lj_medium.ckpt "
        "'https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
        "en/en_US/ljspeech/medium/lj-med_1000.ckpt'"
    )
    .pip_install("scipy", "soundfile")
)

app = modal.App("voice-train", image=image)
datasets = modal.Volume.from_name("voice-datasets", create_if_missing=True)
models = modal.Volume.from_name("voice-models", create_if_missing=True)

TEST_SENTENCES = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
]
MIN_MINUTES, MAX_MINUTES = 20.0, 90.0  # 10 min from the LJ base sounded robotic/shaky (male speaker, by ear); raise/lower with the 25-min result


def steps_for(minutes: float) -> int:
    # 10 min -> 3000 and 25 min -> 4000 are the points being tried (unjudged); extend gently, cap for cost.
    return int(min(6000, max(2000, 2000 + 80 * minutes)))


def reject(voice_id: str, reason: str, **extra):
    import json, os
    os.makedirs(f"/models/{voice_id}", exist_ok=True)
    json.dump({"status": "rejected", "reason": reason, **extra}, open(f"/models/{voice_id}/error.json", "w"))
    models.commit()
    return {"status": "rejected", "reason": reason, **extra}


@app.function(gpu="T4", timeout=4 * 3600, volumes={"/datasets": datasets, "/models": models})
def train_voice(voice_id: str):
    import csv, glob, json, os, runpy, shutil, sys, time

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    d = f"/datasets/{voice_id}"
    try:
        consent = json.load(open(f"{d}/consent.json"))
    except Exception:
        return reject(voice_id, "missing consent.json")
    if consent.get("consent") is not True or not consent.get("speaker_name"):
        return reject(voice_id, "consent.json must have consent=true and a speaker_name")

    rows, total, seen = [], 0.0, set()
    os.makedirs("/tmp/wavs", exist_ok=True)
    skipped = {"unreadable": 0, "too_short_or_long": 0, "clipped": 0, "duplicate": 0, "no_text": 0}
    for i, line in enumerate(csv.reader(open(f"{d}/metadata.csv"), delimiter="|")):
        if len(line) < 2 or not line[1].strip():
            skipped["no_text"] += 1
            continue
        fn, text = line[0], line[1].strip()
        if text.lower() in seen:
            skipped["duplicate"] += 1
            continue
        try:
            audio, sr = sf.read(f"{d}/wavs/{fn}", always_2d=True)
        except Exception:
            skipped["unreadable"] += 1
            continue
        audio = audio.mean(axis=1)  # mono
        dur = len(audio) / sr
        if not 1.0 <= dur <= 15.0:
            skipped["too_short_or_long"] += 1
            continue
        if np.mean(np.abs(audio) > 0.99) > 0.005:  # >0.5% of samples at full scale
            skipped["clipped"] += 1
            continue
        g = gcd(22050, sr)
        audio = resample_poly(audio, 22050 // g, sr // g) if sr != 22050 else audio
        peak = np.max(np.abs(audio)) or 1.0
        audio = audio / peak * 0.9  # peak-normalise so loudness is consistent across clips
        out_fn = f"{len(rows):05d}.wav"
        sf.write(f"/tmp/wavs/{out_fn}", audio, 22050, "PCM_16")
        rows.append((out_fn, text))
        seen.add(text.lower())
        total += dur
    minutes = total / 60
    print(f"{len(rows)} usable clips, {minutes:.1f} min, skipped={skipped}", flush=True)
    if minutes < MIN_MINUTES:
        return reject(voice_id, f"only {minutes:.1f} min of usable audio, need at least {MIN_MINUTES:.0f}", skipped=skipped)
    if minutes > MAX_MINUTES:  # keep cost bounded; longest-first would bias, so take in file order
        cut, acc = 0, 0.0
        for j, (fn, _) in enumerate(rows):
            acc += sf.info(f"/tmp/wavs/{fn}").duration
            if acc > MAX_MINUTES * 60:
                cut = j
                break
        rows = rows[:cut]
    with open("/tmp/train.csv", "w", newline="") as f:
        csv.writer(f, delimiter="|").writerows(rows)

    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    import piper.espeakbridge  # noqa: F401
    import piper.train.__main__ as piper_main
    from piper import PiperVoice

    out = f"/tmp/run"
    steps = steps_for(minutes)
    os.chdir("/opt/piper-src")
    sys.argv = [
        "piper.train", "fit", "--data.voice_name", voice_id,
        "--data.csv_path", "/tmp/train.csv", "--data.audio_dir", "/tmp/wavs",
        "--data.espeak_voice", "en-us", "--data.cache_dir", "/tmp/cache",
        "--data.config_path", f"{out}/config.json", "--data.batch_size", "8",
        "--model.sample_rate", "22050", "--model.warmstart_ckpt", "/ckpt/lj_medium.ckpt",
        "--trainer.max_steps", str(steps), "--trainer.default_root_dir", out,
    ]
    piper_main._DEFAULT_CALLBACKS = piper_main._DEFAULT_CALLBACKS[:1]
    t0 = time.time()
    try:
        piper_main.main()
    except SystemExit as e:
        print(f"piper.train exit {e.code}", flush=True)
    ckpts = sorted(glob.glob(f"{out}/lightning_logs/version_*/checkpoints/last.ckpt"), key=os.path.getmtime)
    if not ckpts:
        return reject(voice_id, "training produced no checkpoint")

    dest = f"/models/{voice_id}"
    os.makedirs(f"{dest}/samples", exist_ok=True)
    onnx = f"{dest}/model.onnx"
    sys.argv = ["piper.train.export_onnx", "--checkpoint", ckpts[-1], "--output-file", onnx]
    runpy.run_module("piper.train.export_onnx", run_name="__main__")
    shutil.copy(f"{out}/config.json", onnx + ".json")
    voice = PiperVoice.load(onnx)
    for i, text in enumerate(TEST_SENTENCES):
        pcm = np.concatenate([c.audio_int16_array for c in voice.synthesize(text)])
        sf.write(f"{dest}/samples/sample_{i}.wav", pcm, voice.config.sample_rate, "PCM_16")
    manifest = {
        "status": "ready", "voice_id": voice_id, "speaker_name": consent["speaker_name"],
        "base": "piper ljspeech medium (public domain)", "clips": len(rows),
        "minutes": round(minutes, 1), "steps": steps, "skipped": skipped,
        "train_seconds": round(time.time() - t0),
    }
    json.dump(manifest, open(f"{dest}/manifest.json", "w"), indent=2)
    models.commit()
    return manifest


@app.local_entrypoint()
def main(voice_id: str, dataset_dir: str = ""):
    """With --dataset-dir, uploads that folder (consent.json, metadata.csv, wavs/) first."""
    if dataset_dir:
        with datasets.batch_upload(force=True) as up:
            up.put_directory(dataset_dir, f"/{voice_id}")
    print("Spawned:", train_voice.spawn(voice_id).object_id)
