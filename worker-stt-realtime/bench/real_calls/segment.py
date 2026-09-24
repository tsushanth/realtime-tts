"""Cut call recordings into utterances and draft references.
  python -m real_calls.segment --raw ../data/real_raw --out ../data/real [--model large-v3-turbo] [--max-calls N]"""
import argparse
import json
import os
import tempfile

import soundfile as sf

FRAME_S = 512 / 16000
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("STT_DATA", os.path.abspath(os.path.join(HERE, "..", "..", "data")))


def _runs(flags, frame_s):
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            runs.append((start * frame_s, i * frame_s))
            start = None
    if start is not None:
        runs.append((start * frame_s, len(flags) * frame_s))
    return runs


def split_speech_ranges(flags, frame_s, min_silence_s=0.6, min_seg_s=1.5, max_seg_s=15.0, pad_s=0.25):
    total = len(flags) * frame_s
    cap = max_seg_s - 2 * pad_s  # so padded output never exceeds max_seg_s
    groups = []
    for r in _runs(flags, frame_s):
        if groups and r[0] - groups[-1][-1][1] < min_silence_s - 1e-9:
            groups[-1].append(r)
        else:
            groups.append([r])
    segs = []
    for g in groups:
        cur = None
        for s, e in g:
            while e - s > cap:
                if cur:
                    segs.append(cur)
                    cur = None
                segs.append((s, s + cap))
                s += cap
            if cur is None:
                cur = (s, e)
            elif e - cur[0] > cap:
                segs.append(cur)
                cur = (s, e)
            else:
                cur = (cur[0], e)
        if cur:
            segs.append(cur)
    return [(max(0.0, s - pad_s), min(total, e + pad_s)) for s, e in segs if e - s >= min_seg_s]


def speech_flags(audio):
    import sherpa_onnx as so
    models = os.environ.get("STT_MODELS", os.path.abspath(os.path.join(HERE, "..", "..", "models")))
    c = so.VadModelConfig()
    c.silero_vad.model = os.path.join(models, "silero_vad.onnx")
    c.silero_vad.min_silence_duration = 0.05
    c.silero_vad.min_speech_duration = 0.1
    c.silero_vad.threshold = 0.5
    c.sample_rate = 16000
    v = so.VoiceActivityDetector(c, buffer_size_in_seconds=60)
    flags = []
    for i in range(0, len(audio) - 511, 512):
        v.accept_waveform(audio[i:i + 512])
        while not v.empty():
            v.pop()
        flags.append(bool(v.is_speech_detected()))
    return flags


def draft_refs(paths, model_size="large-v3-turbo"):
    from faster_whisper import WhisperModel
    m = WhisperModel(model_size, device="cpu", compute_type="int8")
    for p in paths:
        segs, _ = m.transcribe(p, language="en", beam_size=1)
        yield " ".join(s.text.strip() for s in segs).strip()


def build_manifest(raw_dir, out_dir, model_size="large-v3-turbo", max_calls=None, force=False):
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path) and not force:
        with open(manifest_path) as f:
            n_ver = sum(1 for e in json.load(f) if e.get("verified"))
        if n_ver:
            raise RuntimeError(f"{manifest_path} has {n_ver} verified entries that a rebuild would overwrite; pass force=True (--force) to proceed")
    os.makedirs(out_dir, exist_ok=True)
    wavs = sorted(f for f in os.listdir(raw_dir) if f.endswith(".wav"))[:max_calls]
    entries = []
    for w in wavs:
        call_id = w[:-4]
        x, sr = sf.read(os.path.join(raw_dir, w), dtype="float32")
        assert sr == 16000, f"{w}: expected 16 kHz, got {sr}"
        for k, (s, e) in enumerate(split_speech_ranges(speech_flags(x), FRAME_S)):
            uid = f"{call_id}_{k:03d}"
            sf.write(os.path.join(out_dir, uid + ".wav"), x[int(s * sr):int(e * sr)], sr, subtype="PCM_16")
            entries.append({"id": uid, "call_id": call_id, "start_s": round(s, 2), "end_s": round(e, 2),
                            "ref": "", "verified": False, "keyterms": []})
    drafts = draft_refs([os.path.join(out_dir, e["id"] + ".wav") for e in entries], model_size)
    kept = []
    for entry, draft in zip(entries, drafts):
        if draft:
            entry["ref"] = draft
            kept.append(entry)
        else:
            os.remove(os.path.join(out_dir, entry["id"] + ".wav"))
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(kept, f, indent=1)
    os.replace(tmp, manifest_path)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(DATA, "real_raw"))
    ap.add_argument("--out", default=os.path.join(DATA, "real"))
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="overwrite a manifest containing verified entries")
    a = ap.parse_args()
    kept = build_manifest(a.raw, a.out, a.model, a.max_calls, a.force)
    print(f"{len(kept)} utterances in {len({e['call_id'] for e in kept})} calls -> {a.out}")


if __name__ == "__main__":
    main()
