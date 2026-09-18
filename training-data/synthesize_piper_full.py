"""Synthesize + CPU-benchmark the full-corpus Piper fine-tune (see
piper_full_finetune.py). Loads the trained checkpoint from the
tts-checkpoints Volume (no retraining, unlike the pilot's synthesis script
which had to retrain because its checkpoint was never persisted), exports
to ONNX via piper.train.export_onnx, and runs the ONNX inference on a
CPU-only container with real per-sentence timing - same 5 test sentences
used for every other model in this investigation, for a like-for-like
listen and RTF comparison.

Image is duplicated from piper_full_finetune.py rather than imported
(Modal re-imports the entrypoint module inside the remote container, so a
cross-file import fails there - same lesson as synthesize_piper_pilot.py).
No GPU: export_onnx loads the checkpoint with map_location="cpu" and
onnxruntime (base package only) is CPU-only anyway.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "build-essential", "cmake", "ninja-build", "git", "wget")
    .pip_install("torch==2.1.2")
    .pip_install("numpy<2")
    .run_commands("git clone --depth 1 https://github.com/OHF-Voice/piper1-gpl.git /opt/piper-src")
    .apt_install("python3-dev")
    .pip_install("scikit-build", "setuptools", "wheel", "cmake", "ninja")
    .workdir("/opt/piper-src")
    .run_commands("pip install --no-build-isolation -e '.[train]'")
    .run_commands("python3 setup.py build_ext --inplace")
    .run_commands("bash build_monotonic_align.sh")
    .pip_install("numpy<2")
)

app = modal.App("tts-piper-full-synth", image=image)
checkpoint_volume = modal.Volume.from_name("tts-checkpoints")

TEST_SENTENCES = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
]

CKPT = "/checkpoints/piper_full_ft/lightning_logs/version_1/checkpoints/last.ckpt"
CONFIG = "/checkpoints/piper_full_ft/config.json"


@app.function(cpu=4, timeout=1200, volumes={"/checkpoints": checkpoint_volume})
def export_and_synthesize():
    import base64
    import runpy
    import shutil
    import sys
    import time

    import numpy as np
    import soundfile as sf
    import torch

    import piper.espeakbridge  # noqa: F401  sanity check

    # Same weights_only handling as the pilot - patched in-process (runpy
    # below runs in this process, so it inherits the patch).
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    onnx_path = "/tmp/full_ft.onnx"
    t0 = time.time()
    sys.argv = ["piper.train.export_onnx", "--checkpoint", CKPT, "--output-file", onnx_path]
    runpy.run_module("piper.train.export_onnx", run_name="__main__")
    print(f"ONNX export time: {time.time() - t0:.1f}s")
    shutil.copy(CONFIG, onnx_path + ".json")

    from piper import PiperVoice
    t0 = time.time()
    voice = PiperVoice.load(onnx_path)
    print(f"Model load time: {time.time() - t0:.2f}s")

    results = []
    for i, text in enumerate(TEST_SENTENCES):
        t0 = time.time()
        chunks = list(voice.synthesize(text))
        pcm = np.concatenate([c.audio_int16_array for c in chunks])
        synth_time = time.time() - t0
        audio_sec = len(pcm) / voice.config.sample_rate
        rtf = synth_time / audio_sec
        print(f"[{i}] chars={len(text):3d}  synth={synth_time*1000:7.1f}ms  audio={audio_sec:.2f}s  RTF={rtf:.3f}")

        wav_path = f"/tmp/piper_full_sample_{i}.wav"
        sf.write(wav_path, pcm, voice.config.sample_rate, "PCM_16")
        with open(wav_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        results.append({"text": text, "audio_b64": b64, "synth_ms": synth_time * 1000,
                        "audio_sec": audio_sec, "rtf": rtf})
    return results


@app.local_entrypoint()
def main():
    import base64
    import os

    results = export_and_synthesize.remote()
    out_dir = os.path.join(os.path.dirname(__file__), "piper_full_output")
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(results):
        path = os.path.join(out_dir, f"sample_{i}.wav")
        with open(path, "wb") as f:
            f.write(base64.b64decode(r["audio_b64"]))
        print(f"Saved: {path}  -  RTF={r['rtf']:.3f}  ({r['synth_ms']:.0f}ms for {r['audio_sec']:.2f}s audio)  \"{r['text']}\"")
