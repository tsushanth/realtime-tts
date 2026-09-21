"""Pull two single-speaker Spanish subsets out of CML-TTS (CC BY 4.0, HF ylacombe/cml-tts) into the `es-pilot` Volume.

    modal run --detach voices/es_pilot/prep_data.py
Scans the 203 train parquet files (~70 GB, CPU only, a few dollars of nothing), keeps rows of speakers 10246 (F) and 3946 (M)
with levenshtein >= 0.93 (wav2vec transcript agrees with text) and 9 <= duration <= 17 s, stores them as 24 kHz FLAC +
manifest shards. Speaker accents are NOT labelled anywhere; audition before deciding.
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("pyarrow", "fsspec", "aiohttp", "requests", "soundfile", "numpy")
app = modal.App("es-pilot-prep", image=image)
vol = modal.Volume.from_name("es-pilot", create_if_missing=True)
SPK = {10246, 3946}
BASE = "https://huggingface.co/datasets/ylacombe/cml-tts/resolve/main/"


@app.function(cpu=1, memory=6144, timeout=3000, volumes={"/data": vol}, max_containers=24, retries=3)
def scan(path: str):
    import io, json, os, time
    import fsspec, pyarrow.parquet as pq, soundfile as sf
    f = fsspec.filesystem("https").open(BASE + path, block_size=16 << 20)
    pf = pq.ParquetFile(f)
    tag = os.path.basename(path).split("-")[1]
    rows, kept = [], 0
    for g in range(pf.metadata.num_row_groups):
        sid = pf.read_row_group(g, columns=["speaker_id"])["speaker_id"].to_pylist()
        idx = [i for i, s in enumerate(sid) if s in SPK]
        if not idx:
            continue
        t = pf.read_row_group(g).take(idx).to_pylist()
        for r in t:
            if r["levenshtein"] < 0.93 or not (9.0 <= r["duration"] <= 17.0):
                continue
            d = f"/data/{r['speaker_id']}"
            os.makedirs(d, exist_ok=True)
            name = f"{tag}_{g}_{kept}.flac"
            audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
            sf.write(f"{d}/{name}", audio, sr, format="FLAC")
            rows.append({"speaker": r["speaker_id"], "file": f"{r['speaker_id']}/{name}", "text": r["text"], "duration": r["duration"],
                         "lev": r["levenshtein"], "sr": sr})
            kept += 1
    os.makedirs("/data/manifest", exist_ok=True)
    with open(f"/data/manifest/{tag}.jsonl", "w") as o:
        for r in rows:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    return {"file": path, "kept": kept, "hours": sum(r["duration"] for r in rows) / 3600}


@app.local_entrypoint()
def main():
    import json, urllib.request
    t = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/ylacombe/cml-tts/tree/main/spanish"))
    files = sorted(x["path"] for x in t if x["path"].startswith("spanish/train"))
    print(len(files), "train files")
    tot = 0.0
    for r in scan.map(files, order_outputs=False):
        tot += r["hours"]
        print(r["file"].split("/")[-1][:24], r["kept"], f"total {tot:.1f} h", flush=True)
