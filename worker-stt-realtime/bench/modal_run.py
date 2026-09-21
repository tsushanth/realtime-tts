"""Run the real-time replay benchmark on Modal CPU (4 dedicated cores ~ proxy for Fly shared-cpu-4x / performance-4x)
and one T4. One container per job, all in parallel, results written to volume `stt-rt-results`.
  modal run --detach modal_run.py::launch          # spawns everything, returns immediately
  modal volume get stt-rt-results / results/       # fetch later
Cost: CPU $0.0000131/core-s + $0.00000222/GiB-s (modal.com/pricing, read 2026-09-20); T4 $0.000164/s."""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__))
GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

def _bake():
    import subprocess
    from moonshine_voice import ModelArch
    from moonshine_voice.download import find_model_info, download_model_from_info
    for a in ("TINY_STREAMING", "SMALL_STREAMING", "MEDIUM_STREAMING"):
        download_model_from_info(find_model_info("en", getattr(ModelArch, a)))
    from faster_whisper.utils import download_model
    for m in ("tiny.en", "base.en", "small.en"):
        download_model(m)

def mk(base):
    img = base.pip_install("sherpa-onnx", "moonshine-voice", "faster-whisper", "jiwer", "soundfile", "scipy", "numpy", "psutil")
    cmds = ["mkdir -p /models && cd /models"]
    for m in ("sherpa-onnx-streaming-zipformer-en-2023-06-26", "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8",
              "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8"):
        cmds.append(f"curl -sL {GH}/{m}.tar.bz2 | tar xj -C /models")
    cmds.append(f"curl -sL -o /models/silero_vad.onnx {GH}/silero_vad.onnx")
    img = img.apt_install("curl", "bzip2").run_commands(*cmds).run_function(_bake)
    return img.env({"STT_MODELS": "/models", "STT_DATA": "/data"}) \
        .add_local_dir(HERE, "/bench", ignore=["__pycache__", "results"]) \
        .add_local_dir(os.path.join(HERE, "..", "data"), "/data", ignore=["dc.tgz"])

cpu_img = mk(modal.Image.debian_slim(python_version="3.11"))
gpu_img = mk(modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"))
vol = modal.Volume.from_name("stt-rt-results", create_if_missing=True)
app = modal.App("stt-rt-bench")

def _run(jobs, tag):
    import subprocess, json, os, time
    os.makedirs("/results", exist_ok=True)
    out = []
    for (engine, sets, extra) in jobs:
        for s, n in sets:
            fn = f"/results/{tag}__{engine}{'_n'+str(extra) if extra else ''}__{s}.json"
            cmd = ["python", "/bench/rtbench.py", "--engine", engine, "--set", s, "--n", str(n), "--out", fn, "--label", tag]
            if extra: cmd += ["--native-ms", str(extra)]
            t = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
            print(engine, s, extra, "rc", r.returncode, f"{time.time()-t:.0f}s"); print(r.stdout[-300:], r.stderr[-800:])
            vol.commit()
    return "ok"

@app.function(image=cpu_img, cpu=4, memory=4096, timeout=6 * 3600, volumes={"/results": vol})
def cpu_job(jobs, tag="modalcpu4"): return _run(jobs, tag)

@app.function(image=gpu_img, gpu="T4", cpu=4, memory=8192, timeout=3 * 3600, volumes={"/results": vol})
def gpu_job(jobs, tag="modalT4"): return _run(jobs, tag)

SETS = [("clean", 100), ("tel", 100), ("call", 48), ("calltel", 48)]

@app.local_entrypoint()
def launch(only: str = ""):
    jobs = [([(e, SETS, 0)], "cpu") for e in ("zip-en-int8", "nemo-480", "nemo-80", "moonshine-small_streaming", "moonshine-tiny_streaming",
                                                 "moonshine-medium_streaming", "fw-tiny.en", "fw-base.en", "fw-small.en")]
    for e in ("zip-en-int8", "nemo-480"):
        for ms in (300, 500, 700):
            jobs.append(([(e, [("tel", 50), ("calltel", 48)], ms)], "cpu"))
    jobs.append(([("fw-small.en-cuda", [("tel", 100), ("calltel", 48)], 0), ("fw-base.en-cuda", [("tel", 100)], 0)], "gpu"))
    for j, kind in jobs:
        if only and only not in j[0][0]: continue
        h = (cpu_job if kind == "cpu" else gpu_job).spawn(j)
        print("spawned", j[0][0], j[0][2], h.object_id)
