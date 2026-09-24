"""Builders for public human-speech benchmark sets in the rtbench manifest format (DATA/<set>/manifest.json + <id>.wav).

  pub_fleurs      FLEURS en-US test, clean 16 kHz
  pub_fleurs_tel  same utterances through telephony.to_telephone
  pub_earnings    Earnings-22 (chunked) real call-like audio
"""
import io
import json
import os
import random
import tarfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from telephony import to_telephone

SR = 16000


def _write_set(out_root, name, ids, refs, audios):
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    for i, x in zip(ids, audios):
        sf.write(os.path.join(d, i + ".wav"), x, SR, subtype="PCM_16")
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump([{"id": i, "ref": r} for i, r in zip(ids, refs)], f, indent=1)
    return {"set": name, "n": len(ids), "hours": sum(len(x) for x in audios) / SR / 3600}


def _to_mono16(x, sr):
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    return x.astype("float32")


def _default_fleurs_files():
    from huggingface_hub import hf_hub_download
    kw = dict(repo_id="google/fleurs", repo_type="dataset")
    return (hf_hub_download(filename="data/en_us/audio/test.tar.gz", **kw),
            hf_hub_download(filename="data/en_us/test.tsv", **kw))


def build_fleurs(out_root, n=50, seed=0, tar_path=None, tsv_path=None):
    if tar_path is None or tsv_path is None:
        tar_path, tsv_path = _default_fleurs_files()
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) >= 6 and 3 * SR <= int(c[5]) <= 15 * SR and c[3].strip():
                rows.append((c[1], c[3].strip()))
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    want = {"test/" + fn: k for k, (fn, _) in enumerate(picked)}
    audio = {}
    with tarfile.open(tar_path, "r:gz") as tf:   # streamed; only wanted members are read
        for m in tf:
            if m.name in want:
                x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                audio[want[m.name]] = _to_mono16(x, sr)
                if len(audio) == len(want):
                    break
    ks = sorted(audio)
    ids = [f"fleurs_{k:03d}" for k in ks]
    refs = [picked[k][1] for k in ks]
    clean = [audio[k] for k in ks]
    tel = [to_telephone(audio[k], seed=k) for k in ks]
    return {"pub_fleurs": _write_set(out_root, "pub_fleurs", ids, refs, clean),
            "pub_fleurs_tel": _write_set(out_root, "pub_fleurs_tel", ids, refs, tel)}


def _default_earnings_source():
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    for p in sorted(fs.glob("datasets/distil-whisper/earnings22/chunked/test-*.parquet")):
        yield fs.open(p, "rb")


def build_earnings(out_root, n=50, seed=0, min_s=3.0, max_s=15.0, parquet_source=None):
    """Takes the first n valid utterances in shard/row order (deterministic; seed is accepted for API symmetry)."""
    import pyarrow.parquet as pq
    src = (parquet_source or _default_earnings_source)()
    ids, refs, audios = [], [], []
    for f in src:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):   # one row group at a time; never the whole shard
            t = pf.read_row_group(g, columns=["audio", "transcription"]).to_pylist()
            for r in t:
                ref = (r["transcription"] or "").strip()
                if not ref:
                    continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                x = _to_mono16(x, sr)
                if not (min_s <= len(x) / SR <= max_s):
                    continue
                ids.append(f"earn_{len(ids):03d}")
                refs.append(ref)
                audios.append(x)
                if len(ids) >= n:
                    return _write_set(out_root, "pub_earnings", ids, refs, audios)
    return _write_set(out_root, "pub_earnings", ids, refs, audios)
