"""One-off: export the full-corpus Piper checkpoint to ONNX and persist it (plus
the voice config) on the tts-checkpoints Volume, so the serving app can load it
without torch/piper-train in its image. Run: `modal run export_onnx.py`.

Reuses the training image's build steps (export needs torch + piper.train) - see
training-data/piper_pilot_finetune.py for why each build step exists.
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
app = modal.App("piper-export-onnx", image=image)
vol = modal.Volume.from_name("tts-checkpoints")

CKPT = "/checkpoints/piper_full_ft/lightning_logs/version_1/checkpoints/last.ckpt"
CONFIG = "/checkpoints/piper_full_ft/config.json"
OUT_DIR = "/checkpoints/piper_full_ft/serving"


@app.function(cpu=4, timeout=900, volumes={"/checkpoints": vol})
def export():
    import os, runpy, shutil, sys
    import torch

    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

    os.makedirs(OUT_DIR, exist_ok=True)
    onnx = f"{OUT_DIR}/full_ft.onnx"
    sys.argv = ["piper.train.export_onnx", "--checkpoint", CKPT, "--output-file", onnx]
    runpy.run_module("piper.train.export_onnx", run_name="__main__")
    shutil.copy(CONFIG, onnx + ".json")
    vol.commit()
    print({f: os.path.getsize(f"{OUT_DIR}/{f}") for f in os.listdir(OUT_DIR)})


@app.local_entrypoint()
def main():
    export.remote()
