"""Synthesize test sentences with the Piper pilot fine-tune (see
piper_pilot_finetune.py), same purpose as synthesize_full_ft.py for
Matcha-TTS: get real audio to actually listen to, not just trust a
decreasing val_mel number.

Piper's own inference path (the `piper` CLI, `onnxruntime`) expects an ONNX
export, not the raw Lightning `.ckpt` - `piper.train.export_onnx` handles
that (`VitsModel.load_from_checkpoint` + `torch.onnx.export`), reusing the
same image/environment as the pilot (same gotchas already solved there).
"""
import modal

# Same image as piper_pilot_finetune.py, duplicated rather than imported
# cross-file: Modal re-imports this entrypoint module INSIDE the remote
# container to locate the decorated function, so a top-level
# `from piper_pilot_finetune import image` fails there with
# ModuleNotFoundError - that file was never added to the image/mounts, only
# this one was. (Same reason synthesize_full_ft.py duplicated Matcha-TTS's
# image instead of importing it from pilot_finetune.py.) See that file for
# the gotchas behind each of these steps.
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
    .run_commands(
        "mkdir -p /ckpt && wget -q -O /ckpt/lessac_medium.ckpt "
        "'https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
        "en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt'"
    )
    .add_local_dir("pilot", remote_path="/pilot_src")
)

app = modal.App("tts-piper-pilot-synth", image=image)

TEST_SENTENCES = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
]


@app.function(gpu="T4", timeout=900)
def train_and_synthesize():
    """Trains the same 300-step pilot fresh (the earlier run's checkpoint
    was never persisted anywhere outside that ephemeral container - same
    class of lesson as Matcha-TTS's pilot needing a Volume for a real run,
    except this pilot is cheap/short enough that just rerunning it is
    simpler than plumbing a Volume through for a one-off check) and
    immediately exports + synthesizes from the result in the same container.
    """
    import csv
    import json
    import os
    import shutil
    import sys

    import piper.espeakbridge  # noqa: F401  sanity check, see pilot script

    os.makedirs("/pilot/audio", exist_ok=True)
    os.makedirs("/pilot/cache", exist_ok=True)

    def convert(src_filelist, dst_csv):
        rows = []
        with open(src_filelist) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, text = line.split("|", 1)
                fname = os.path.basename(path)
                src_wav = os.path.join("/pilot_src/wavs", fname)
                dst_wav = os.path.join("/pilot/audio", fname)
                if not os.path.exists(dst_wav):
                    shutil.copy(src_wav, dst_wav)
                rows.append((fname, text))
        with open(dst_csv, "w", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(rows)
        return len(rows)

    convert("/pilot_src/train.txt", "/pilot/train.csv")
    convert("/pilot_src/val.txt", "/pilot/val.csv")

    import torch
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    os.chdir("/opt/piper-src")
    sys.argv = [
        "piper.train", "fit",
        "--data.voice_name", "pilot_ft",
        "--data.csv_path", "/pilot/train.csv",
        "--data.audio_dir", "/pilot/audio",
        "--data.espeak_voice", "en-us",
        "--data.cache_dir", "/pilot/cache",
        "--data.config_path", "/pilot/config.json",
        "--data.batch_size", "8",
        "--model.sample_rate", "22050",
        "--model.warmstart_ckpt", "/ckpt/lessac_medium.ckpt",
        "--trainer.max_steps", "300",
        "--trainer.default_root_dir", "/output",
    ]
    import piper.train.__main__ as piper_main
    piper_main._DEFAULT_CALLBACKS = [piper_main._DEFAULT_CALLBACKS[0]]
    try:
        piper_main.main()
    except SystemExit as e:
        print(f"piper.train exited with code {e.code}")

    # Find the last checkpoint written
    ckpt_dir = None
    for root, dirs, files in os.walk("/output"):
        if "last.ckpt" in files:
            ckpt_dir = root
    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    print(f"Using checkpoint: {ckpt_path}")

    onnx_path = "/output/pilot_ft.onnx"
    sys.argv = [
        "piper.train.export_onnx",
        "--checkpoint", ckpt_path,
        "--output-file", onnx_path,
    ]
    import runpy
    runpy.run_module("piper.train.export_onnx", run_name="__main__")

    # piper's CLI/onnxruntime inference expects a <model>.onnx.json sidecar -
    # reuse the config written during training (same voice/phoneme config).
    shutil.copy("/pilot/config.json", onnx_path + ".json")

    from piper import PiperVoice
    voice = PiperVoice.load(onnx_path)
    results = []
    for i, text in enumerate(TEST_SENTENCES):
        audio_chunks = list(voice.synthesize(text))
        import numpy as np
        import soundfile as sf
        pcm = np.concatenate([c.audio_int16_array for c in audio_chunks])
        wav_path = f"/tmp/piper_pilot_sample_{i}.wav"
        sf.write(wav_path, pcm, voice.config.sample_rate, "PCM_16")
        import base64
        with open(wav_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        results.append({"text": text, "audio_b64": b64})
        print(f"Synthesized [{i}]: {text}")

    return results


@app.local_entrypoint()
def main():
    import base64
    import os

    results = train_and_synthesize.remote()
    out_dir = os.path.join(os.path.dirname(__file__), "piper_pilot_output")
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(results):
        path = os.path.join(out_dir, f"sample_{i}.wav")
        with open(path, "wb") as f:
            f.write(base64.b64decode(r["audio_b64"]))
        print(f"Saved: {path}  -  \"{r['text']}\"")
