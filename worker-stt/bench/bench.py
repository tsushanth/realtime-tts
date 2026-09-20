"""WER / RTF / latency benchmark of faster-whisper models on Modal CPU and GPU.
Run: modal run --detach bench.py   (spawns one function per model x hardware; results -> results.json)"""
import modal
from common import image, vol, MODELS

app = modal.App("stt-bench-run", image=image)

def _norm(s):
    import re
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _bench(model_name, device, compute_type, threads, batched_long=False):
    import json, time, statistics, soundfile as sf, jiwer, numpy as np
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    m = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=threads, download_root="/models")
    man = json.load(open("/data/manifest.json"))
    warm, _ = sf.read(f"/data/clean/{man[0]['id']}.wav", dtype="float32")
    for _ in range(2): list(m.transcribe(warm, language="en", beam_size=1)[0])
    res = {"model": model_name, "device": device, "compute": compute_type}
    for cond in ("clean", "tel"):
        refs, hyps, lat, durs = [], [], [], []
        for c in man:
            a, _ = sf.read(f"/data/{cond}/{c['id']}.wav", dtype="float32")
            t = time.perf_counter()
            segs, _ = m.transcribe(a, language="en", beam_size=1, condition_on_previous_text=False)
            txt = " ".join(s.text for s in segs)
            lat.append(time.perf_counter() - t); durs.append(len(a) / 16000)
            refs.append(_norm(c["ref"])); hyps.append(_norm(txt))
        five = [l for l, d in zip(lat, durs) if 4 <= d <= 6]
        res[cond] = {"wer": jiwer.wer(refs, hyps), "rtf": sum(lat) / sum(durs), "audio_s": sum(durs),
                     "median_lat_5s": statistics.median(five), "n5": len(five),
                     "median_lat_all": statistics.median(lat)}
    if batched_long:
        audio = np.concatenate([sf.read(f"/data/clean/{c['id']}.wav", dtype="float32")[0] for c in man])
        bp = BatchedInferencePipeline(m)
        t = time.perf_counter()
        segs, _ = bp.transcribe(audio, language="en", batch_size=16, beam_size=1)
        n = len(" ".join(s.text for s in segs).split())
        res["batched_long"] = {"rtf": (time.perf_counter() - t) / (len(audio) / 16000), "audio_s": len(audio) / 16000, "words": n}
    return res

@app.function(cpu=4, memory=8192, volumes={"/data": vol}, timeout=3000)
def run_cpu(model_name):
    return _bench(model_name, "cpu", "int8", 8)   # Modal cpu=4 => 4 physical cores = 8 vCPU

@app.function(gpu="T4", volumes={"/data": vol}, timeout=3000)
def run_t4(model_name, batched=False):
    return _bench(model_name, "cuda", "float16", 4, batched)

@app.function(gpu="L4", volumes={"/data": vol}, timeout=3000)
def run_l4(model_name, batched=False):
    return _bench(model_name, "cuda", "float16", 4, batched)

@app.local_entrypoint()
def main():
    import json
    calls = []
    for m in MODELS:
        calls.append(("cpu4", m, run_cpu.spawn(m)))
        calls.append(("t4", m, run_t4.spawn(m, m in ("distil-large-v3", "large-v3-turbo"))))
    for m in ("large-v3-turbo", "distil-large-v3"):
        calls.append(("l4", m, run_l4.spawn(m, True)))
    out = []
    for hw, m, c in calls:
        try:
            r = c.get(); r["hw"] = hw; out.append(r); print(json.dumps(r))
        except Exception as e:
            print("FAIL", hw, m, e)
    json.dump(out, open("results.json", "w"), indent=1)
