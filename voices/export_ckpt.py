"""Export a piper-checkpoints .ckpt to ONNX into the house-voices Volume under _work/<name>/ (heavy image with piper1-gpl).

    modal run voices/export_ckpt.py --name libritts_r --dir en/en_US/libritts_r/medium --ckpt best.ckpt
Used for the libritts_r checkpoint (scratch-trained 2026-09, Piper >= 1.5; the onnx in rhasspy/piper-voices is the old
Lessac-finetuned model and must NOT be used). Same image/patches as export_voices.py.
"""
import modal
image = (  # same image as export_voices.py
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

app = modal.App("house-voices-export-ckpt", image=image)
volume = modal.Volume.from_name("house-voices", create_if_missing=True)
HF = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"


@app.function(cpu=4, memory=8192, timeout=1800, volumes={"/vol": volume})
def export(name: str, d: str, ckpt_name: str):
    import os, runpy, shutil, sys, urllib.parse, urllib.request
    import torch
    work = f"/vol/_work/{name}"
    os.makedirs(work, exist_ok=True)
    ckpt = f"/tmp/{name}.ckpt"
    urllib.request.urlretrieve(HF + d + "/" + urllib.parse.quote(ckpt_name), ckpt)
    urllib.request.urlretrieve(HF + d + "/config.json", f"{work}/config.json")
    orig = torch.load
    torch.load = lambda *a, **kw: orig(*a, **{**kw, "weights_only": False})
    sys.argv = ["piper.train.export_onnx", "--checkpoint", ckpt, "--output-file", f"{work}/model.onnx"]
    runpy.run_module("piper.train.export_onnx", run_name="__main__")
    shutil.copy(f"{work}/config.json", f"{work}/model.onnx.json")
    volume.commit()
    return {"onnx_mb": os.path.getsize(f"{work}/model.onnx") / 1e6}


@app.local_entrypoint()
def main(name: str, dir: str, ckpt: str):
    print(export.remote(name, dir, ckpt))
