"""Builders for public human-speech benchmark sets in the rtbench manifest format (DATA/<set>/manifest.json + <id>.wav).

  pub_fleurs      FLEURS en-US test, clean 16 kHz
  pub_fleurs_tel  same utterances through telephony.to_telephone
  pub_earnings    Earnings-22 (chunked) call audio, sampled across shards / row groups / calls

Every manifest entry carries source="public" (second key of the engines_cloud pub_* gate).
Sets are staged in a temp directory next to the target and renamed into place only when complete.

Caveat: Earnings-22 references are raw text with disfluencies ("Uh", "um") and digit/word mismatches, so WER on it
is not comparable to FLEURS (whose references are normalised lowercase without punctuation).
"""
import io
import json
import math
import os
import random
import shutil
import tarfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from telephony import to_telephone

SR = 16000


def _stage_set(out_root, name, ids, refs, audios, extra=None):
    """Write the set into a temp dir under out_root; returns (tmp_dir, info). Caller publishes or discards."""
    os.makedirs(out_root, exist_ok=True)
    tmp = os.path.join(out_root, f".{name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        for i, x in zip(ids, audios):
            sf.write(os.path.join(tmp, i + ".wav"), x, SR, subtype="PCM_16")
        man = [dict({"id": i, "ref": r, "source": "public"}, **(extra[k] if extra else {}))
               for k, (i, r) in enumerate(zip(ids, refs))]
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    peaks = [float(np.max(np.abs(x))) if len(x) else 0.0 for x in audios]
    return tmp, {"set": name, "n": len(ids), "hours": sum(len(x) for x in audios) / SR / 3600,
                 "peak_median": float(np.median(peaks)) if peaks else 0.0}


def _publish(out_root, name, tmp):
    dst = os.path.join(out_root, name)
    old = os.path.join(out_root, f".{name}.old-{os.getpid()}")
    if os.path.exists(dst):
        os.rename(dst, old)
    os.rename(tmp, dst)
    shutil.rmtree(old, ignore_errors=True)


def _to_mono16(x, sr):
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    return x.astype("float32")


def _too_short(what, n, got):
    return ValueError(f"{what}: requested {n} utterances but only {got} available (pass allow_short=True to accept)")


def _default_fleurs_files():
    from huggingface_hub import hf_hub_download
    kw = dict(repo_id="google/fleurs", repo_type="dataset")
    return (hf_hub_download(filename="data/en_us/audio/test.tar.gz", **kw),
            hf_hub_download(filename="data/en_us/test.tsv", **kw))


def build_fleurs(out_root, n=50, seed=0, tar_path=None, tsv_path=None, allow_short=False):
    """Builds pub_fleurs (RAW, un-normalised, very quiet source level: median peak ~0.007) and pub_fleurs_tel
    (telephone-degraded). Use derive_normalized_set for a level-normalised clean variant. Each returned info
    dict has peak_median so a level surprise is visible in the build output."""
    if tar_path is None or tsv_path is None:
        tar_path, tsv_path = _default_fleurs_files()
    by_text = {}   # dedupe by transcription: same sentence read by several speakers must not fill the sample
    with open(tsv_path, encoding="utf-8") as f:
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) >= 6 and 3 * SR <= int(c[5]) <= 15 * SR and c[3].strip():
                by_text.setdefault(c[3].strip(), c[1])
    rows = [(fn, t) for t, fn in by_text.items()]
    if len(rows) < n and not allow_short:
        raise _too_short("fleurs", n, len(rows))
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
    if len(audio) < n and not allow_short:
        raise _too_short("fleurs (tar members found)", n, len(audio))
    ks = sorted(audio)
    ids = [f"fleurs_{k:03d}" for k in ks]
    refs = [picked[k][1] for k in ks]
    staged = []
    try:
        clean_tmp, ci = _stage_set(out_root, "pub_fleurs", ids, refs, [audio[k] for k in ks])
        staged.append(clean_tmp)
        tel_tmp, ti = _stage_set(out_root, "pub_fleurs_tel", ids, refs, [to_telephone(audio[k], seed=k) for k in ks])
        staged.append(tel_tmp)
    except BaseException:
        for t in staged:
            shutil.rmtree(t, ignore_errors=True)
        raise
    _publish(out_root, "pub_fleurs", clean_tmp)
    _publish(out_root, "pub_fleurs_tel", tel_tmp)
    return {"pub_fleurs": ci, "pub_fleurs_tel": ti}


def _default_earnings_source():
    """Shard handles (zero-arg callables that open lazily); nothing is downloaded here."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    return [(lambda p=p: fs.open(p, "rb"))
            for p in sorted(fs.glob("datasets/distil-whisper/earnings22/chunked/test-*.parquet"))]


def build_earnings(out_root, n=50, seed=0, min_s=3.0, max_s=15.0, parquet_source=None, allow_short=False):
    """Sample across calls: `seed` picks up to 25 shards, ONE random row group per shard (a row group is one
    call), and up to ceil(n / n_shards) random valid rows from it. parquet_source() returns shard paths, file
    objects, or zero-arg callables returning either; each shard is opened lazily and only one row group is read."""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    shards = list((parquet_source or _default_earnings_source)())
    n_shards = min(len(shards), 25)
    picked = rng.sample(shards, n_shards)
    per_group = math.ceil(n / n_shards) if n_shards else 0
    ids, refs, audios, extra = [], [], [], []
    for sh in picked:
        if len(ids) >= n:
            break
        pf = pq.ParquetFile(sh() if callable(sh) else sh)
        names = pf.schema_arrow.names
        cols = [c for c in ("audio", "transcription", "file_id") if c in names]
        tbl = pf.read_row_group(rng.randrange(pf.num_row_groups), columns=cols)
        audio_col, text_col = tbl.column("audio"), tbl.column("transcription")
        fid_col = tbl.column("file_id") if "file_id" in cols else None
        got = 0
        for r in rng.sample(range(tbl.num_rows), tbl.num_rows):   # random order, decoded lazily
            if got >= per_group or len(ids) >= n:
                break
            ref = (text_col[r].as_py() or "").strip()
            if not ref:
                continue
            x, sr = sf.read(io.BytesIO(audio_col[r].as_py()["bytes"]), dtype="float32")
            x = _to_mono16(x, sr)
            if not (min_s <= len(x) / SR <= max_s):
                continue
            ids.append(f"earn_{len(ids):03d}")
            refs.append(ref)
            audios.append(x)
            extra.append({"file_id": fid_col[r].as_py()} if fid_col is not None else {})
            got += 1
    if len(ids) < n and not allow_short:
        raise _too_short("earnings", n, len(ids))
    tmp, info = _stage_set(out_root, "pub_earnings", ids, refs, audios, extra)
    _publish(out_root, "pub_earnings", tmp)
    info["distinct_file_ids"] = len({e["file_id"] for e in extra if "file_id" in e})
    print(f"pub_earnings: {info['n']} utterances from {info['distinct_file_ids']} distinct file_ids")
    return info


def normalize_peak(x, peak=0.7):
    x = np.asarray(x, dtype="float32")
    if not np.all(np.isfinite(x)):
        raise ValueError("normalize_peak: non-finite samples")
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if m == 0.0 else (x * np.float32(peak / m)).astype("float32")


def derive_normalized_set(out_root, src_set="pub_fleurs", dst_set="pub_fleurs_norm", peak=0.7):
    """Peak-normalise an already-built set into dst_set (same ids/refs/source). No network."""
    from engines_cloud import is_public_set
    if not (dst_set.startswith("pub_") and is_public_set(dst_set)):
        raise ValueError(f"dst_set {dst_set!r} must be a public set name (pub_<name>, not pub_real*)")
    sd = os.path.join(out_root, src_set)
    mp = os.path.join(sd, "manifest.json")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"source set {src_set!r} not built: {mp} missing")
    with open(mp) as f:
        man = json.load(f)
    audios = []
    for m in man:
        x, sr = sf.read(os.path.join(sd, m["id"] + ".wav"), dtype="float32")
        assert sr == SR
        audios.append(normalize_peak(x, peak))
    extra = [{k: v for k, v in m.items() if k not in ("id", "ref", "source")} for m in man]
    tmp, info = _stage_set(out_root, dst_set, [m["id"] for m in man], [m["ref"] for m in man], audios, extra)
    _publish(out_root, dst_set, tmp)
    return info
