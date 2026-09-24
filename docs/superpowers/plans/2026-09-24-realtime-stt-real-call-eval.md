# Realtime STT Real-Call Evaluation Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build tooling that turns real phone-call recordings into a scored benchmark, run our CPU streaming engines against Deepgram Flux and ElevenLabs Scribe v2 Realtime on the same audio, and produce a data-backed engine decision.

**Architecture:** New `bench/real_calls/` package fetches and screens recordings (local only), cuts them into utterances, and exports a review sheet for hand-verified references. `bench/rtbench.py` gains a `real` set, keyterm scoring and a third-party gate. `bench/engines_cloud.py` adds Deepgram and ElevenLabs adapters behind the existing stream interface (`push/text/final/native_eos`). `bench/report_real.py` renders the comparison and the decision gate.

**Tech Stack:** Python 3.12, pytest, sherpa-onnx 1.13.8, faster-whisper, websockets (sync API), requests, jiwer, soundfile, ffmpeg.

**Spec:** `docs/superpowers/specs/2026-09-24-realtime-stt-real-call-eval-design.md`

**Scope note:** This is Phase 1 only. Phase 2 (deploy on Fly with the autoscaler, load test, speculative end-of-turn, gateway wiring, pricing) depends on the decision this plan produces and gets its own plan after the gate. Do not start it.

## Global Constraints

- All work runs from `worker-stt-realtime/bench/` unless a path says otherwise; tests: `python3 -m pytest tests -q`.
- Real audio, `calls.json`, transcripts and per-run result JSON contain call content: they live only under gitignored `worker-stt-realtime/data/` and `worker-stt-realtime/bench/results/real__*.json`. Never commit them, never print transcripts to shared logs.
- Secrets (`SUPABASE_SERVICE_ROLE_KEY`, `TWILIO_AUTH_TOKEN`, `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`) are read from env files or env vars at run time and never printed, logged, or written to any file.
- Third-party engines (`dg-*`, `el-*`) must refuse to run on any clip whose `call_id` is not in `data/confirmed_calls.txt` (clips with no `call_id` are public/synthetic and allowed).
- Own-call rule is direction-aware: inbound -> `caller_phone`, outbound -> `to_number`, matched against `data/own_numbers.txt` after normalisation.
- Audio format everywhere: mono, 16 kHz, PCM_16 wav. Utterances 1.5-15 s.
- Do not modify `worker-stt-realtime/server.py`, the gateway, or any deploy config in this plan.
- Free disk is tight on the dev machine (about 4 GB when this plan was written): check `df -h /System/Volumes/Data` before downloading models; use `--model small.en` for drafts if under 8 GB free.

## File Structure

- Create `worker-stt-realtime/bench/real_calls/__init__.py` (empty), `numbers.py` (env parsing, number helpers, call selection), `fetch.py` (screen/fetch CLI), `segment.py` (VAD utterance splitting, draft references, manifest build), `review.py` (TSV export/import).
- Create `worker-stt-realtime/bench/textnorm.py` (`norm`, moved out of `rtbench.py`), `metrics.py` (`keyterm_recall`), `engines_cloud.py` (Deepgram/ElevenLabs adapters, gate), `report_real.py`.
- Modify `worker-stt-realtime/bench/rtbench.py` (import `norm`, `real` set, `--only-verified`, keyterms, gate, native strategy for cloud engines) and `bench/engines.py` (`build()` routes `dg-flux`, `el-scribe`).
- Create `worker-stt-realtime/bench/tests/` with `conftest.py` and one test file per module; `worker-stt-realtime/bench/requirements-real.txt`.
- Modify root `.gitignore`.

---

### Task 1: Number helpers, call selection, fetch CLI

**Files:**
- Create: `worker-stt-realtime/bench/requirements-real.txt`, `bench/tests/conftest.py`, `bench/real_calls/__init__.py`, `bench/real_calls/numbers.py`, `bench/real_calls/fetch.py`
- Test: `worker-stt-realtime/bench/tests/test_numbers.py`
- Modify: `.gitignore` (repo root)

**Interfaces:**
- Produces: `parse_env(text: str) -> dict[str,str]`; `normalize_number(n: str|None) -> str|None` (E.164-ish, 10 digits gets `+1`); `mask_number(n) -> str` (`"+1********67"`, `"unknown"` if empty); `counterparty(row: dict) -> str|None`; `select_own_calls(rows, own_numbers, engine="poc", limit=None) -> list[dict]` (newest first); `recording_wav_url(url: str) -> str`. CLI `python -m real_calls.fetch {screen,fetch}` writing `data/real_raw/<call_id>.wav` and `data/real_raw/calls.json`.

- [ ] **Step 1: Environment, requirements, ignores**

`worker-stt-realtime/bench/requirements-real.txt`:
```
sherpa-onnx==1.13.8
numpy
soundfile
jiwer
websockets>=12
requests
faster-whisper
pytest
```
`worker-stt-realtime/bench/tests/conftest.py`:
```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
Append to repo-root `.gitignore`:
```
# Real call audio, transcripts and per-run results: never commit
worker-stt-realtime/data/
worker-stt-realtime/models/
worker-stt-realtime/bench/results/real__*.json
```
Run:
```bash
cd worker-stt-realtime && python3 -m venv .venv && . .venv/bin/activate && pip install -q -r bench/requirements-real.txt
git check-ignore -v data/x models/x bench/results/real__a__real.json
```
Expected: all three paths reported as ignored.

- [ ] **Step 2: Write the failing tests**

`bench/tests/test_numbers.py`:
```python
from real_calls.numbers import (
    counterparty, mask_number, normalize_number, parse_env, recording_wav_url, select_own_calls,
)


def test_parse_env_handles_export_quotes_and_comments():
    text = 'A=1\n# B=2\nexport C="x y"\nD=\'z\'\n  E = 5 \n'
    assert parse_env(text) == {"A": "1", "C": "x y", "D": "z", "E": "5"}


def test_normalize_number():
    assert normalize_number("(555) 123-4567") == "+15551234567"
    assert normalize_number("+44 20 7946 0958") == "+442079460958"
    assert normalize_number("") is None
    assert normalize_number(None) is None


def test_mask_number():
    assert mask_number("+1 (555) 123-4567") == "+1********67"
    assert mask_number(None) == "unknown"


def test_counterparty_is_direction_aware():
    inbound = {"direction": "inbound", "caller_phone": "(555) 123-4567", "to_number": "+18005550100"}
    outbound = {"direction": "outbound", "caller_phone": "+18005550100", "to_number": "(555) 123-4567"}
    assert counterparty(inbound) == "+15551234567"
    assert counterparty(outbound) == "+15551234567"


def _row(i, direction, caller, to, engine="poc", url="https://x/RE1", created="2026-09-20T10:00:00Z"):
    return {"id": i, "direction": direction, "caller_phone": caller, "to_number": to,
            "voice_engine": engine, "recording_url": url, "created_at": created}


def test_select_own_calls_filters_orders_and_limits():
    own = ["+15551234567"]
    rows = [
        _row("old", "inbound", "+15551234567", "+18005550100", created="2026-09-20T10:00:00Z"),
        _row("new", "outbound", "+18005550100", "+15551234567", created="2026-09-21T10:00:00Z"),
        _row("stranger", "inbound", "+15559999999", "+18005550100"),
        _row("retell", "inbound", "+15551234567", "+18005550100", engine="retell"),
        _row("norec", "inbound", "+15551234567", "+18005550100", url=None),
    ]
    assert [r["id"] for r in select_own_calls(rows, own)] == ["new", "old"]
    assert [r["id"] for r in select_own_calls(rows, own, limit=1)] == ["new"]


def test_recording_wav_url():
    base = "https://api.twilio.com/2010-04-01/Accounts/AC1/Recordings/RE1"
    assert recording_wav_url(base) == base + ".wav"
    assert recording_wav_url(base + ".mp3") == base + ".wav"
    assert recording_wav_url(base + ".json?x=1") == base + ".wav"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd worker-stt-realtime/bench && python3 -m pytest tests/test_numbers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'real_calls'`.

- [ ] **Step 4: Implement `numbers.py`**

`bench/real_calls/__init__.py`: empty file. `bench/real_calls/numbers.py`:
```python
import re


def parse_env(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", line)
        if not m:
            continue
        v = m.group(2)
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[m.group(1)] = v
    return out


def normalize_number(n):
    if not n:
        return None
    digits = re.sub(r"\D", "", n)
    if not digits:
        return None
    if len(digits) == 10:
        digits = "1" + digits
    return "+" + digits


def mask_number(n):
    n = normalize_number(n)
    if not n:
        return "unknown"
    return n[:2] + "*" * (len(n) - 4) + n[-2:]


def counterparty(row):
    raw = row.get("caller_phone") if row.get("direction") == "inbound" else row.get("to_number")
    return normalize_number(raw)


def select_own_calls(rows, own_numbers, engine="poc", limit=None):
    own = {x for x in map(normalize_number, own_numbers) if x}
    picked = [r for r in rows
              if r.get("voice_engine") == engine and r.get("recording_url") and counterparty(r) in own]
    picked.sort(key=lambda r: r["created_at"], reverse=True)
    return picked[:limit] if limit else picked


def recording_wav_url(url):
    return re.sub(r"\.(mp3|wav|json)$", "", url.split("?")[0]) + ".wav"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_numbers.py -q`
Expected: 6 passed.

- [ ] **Step 6: Implement `fetch.py`**

`bench/real_calls/fetch.py`:
```python
"""screen: show how many recorded calls exist and which masked numbers they involve.
fetch:  download calls whose counterparty is in data/own_numbers.txt (one number per line) to data/real_raw/.
Secrets are read from env files at run time and never printed."""
import argparse
import json
import os
import subprocess
from collections import Counter

import requests

from .numbers import counterparty, mask_number, parse_env, recording_wav_url, select_own_calls

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("STT_DATA", os.path.abspath(os.path.join(HERE, "..", "..", "data")))
COLS = "id,voice_engine,direction,caller_phone,to_number,recording_url,created_at,duration_seconds"


def load_env(path):
    with open(os.path.expanduser(path)) as f:
        return parse_env(f.read())


def fetch_rows(env, limit):
    key = env["SUPABASE_SERVICE_ROLE_KEY"]
    r = requests.get(
        f"{env['NEXT_PUBLIC_SUPABASE_URL']}/rest/v1/calldesk_call_logs",
        params={"select": COLS, "recording_url": "not.is.null", "voice_engine": "eq.poc",
                "order": "created_at.desc", "limit": str(limit)},
        headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return r.json()


def download_call(row, sid, token, raw_dir):
    dest = os.path.join(raw_dir, row["id"] + ".wav")
    if os.path.exists(dest):
        return dest
    tmp = dest + ".src.wav"
    with requests.get(recording_wav_url(row["recording_url"]), auth=(sid, token), stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-ar", "16000", dest], check=True)
    os.remove(tmp)
    return dest


def cmd_screen(a):
    rows = fetch_rows(load_env(a.calldesk_env), a.limit)
    print(f"{len(rows)} recorded poc-engine calls")
    counts = Counter((r["direction"], mask_number(counterparty(r))) for r in rows)
    for (direction, masked), n in counts.most_common(15):
        print(f"  {direction:8s} {masked:16s} {n}")
    own_path = os.path.join(DATA, "own_numbers.txt")
    if os.path.exists(own_path):
        own = open(own_path).read().split()
        print(f"{len(select_own_calls(rows, own))} of {len(rows)} match data/own_numbers.txt")
    else:
        print("data/own_numbers.txt not found: write your own phone numbers there, one per line")


def cmd_fetch(a):
    own_path = os.path.join(DATA, "own_numbers.txt")
    if not os.path.exists(own_path):
        raise SystemExit("data/own_numbers.txt is required (your own phone numbers, one per line)")
    rows = fetch_rows(load_env(a.calldesk_env), a.limit)
    picked = select_own_calls(rows, open(own_path).read().split(), limit=a.max_calls)
    print(f"downloading {len(picked)} own calls ({len(rows) - len(picked)} excluded)")
    tw = load_env(a.twilio_env)
    raw_dir = os.path.join(DATA, "real_raw")
    os.makedirs(raw_dir, exist_ok=True)
    for r in picked:
        download_call(r, tw["TWILIO_ACCOUNT_SID"], tw["TWILIO_AUTH_TOKEN"], raw_dir)
    with open(os.path.join(raw_dir, "calls.json"), "w") as f:
        json.dump(picked, f, indent=1)
    print(f"done: {raw_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "fetch"])
    ap.add_argument("--calldesk-env", default="~/Documents/GitHub/calldesktech/.env")
    ap.add_argument("--twilio-env", default="~/Documents/GitHub/realtime-tts/call-loop-poc/.env")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--max-calls", type=int, default=None)
    a = ap.parse_args()
    {"screen": cmd_screen, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
```
Run: `python3 -m real_calls.fetch --help`
Expected: usage text, exit 0.

- [ ] **Step 7: Commit**

```bash
git add .gitignore worker-stt-realtime/bench/requirements-real.txt worker-stt-realtime/bench/tests worker-stt-realtime/bench/real_calls
git commit -m "stt-eval: number helpers, own-call selection and fetch CLI for real call recordings"
```

- [ ] **Step 8: CONTROLLER/USER STEP (not for the implementer): fetch the real audio**

The user writes their own phone numbers, one per line, to `worker-stt-realtime/data/own_numbers.txt`. Then:
```bash
cd worker-stt-realtime/bench && python3 -m real_calls.fetch screen
python3 -m real_calls.fetch fetch --max-calls 40
```
Expected: "downloading N own calls (M excluded)", then wavs in `data/real_raw/`. The controller runs this, not a subagent, because it handles real customer-adjacent data and credentials.

---

### Task 2: Utterance segmentation, draft references, review sheet

**Files:**
- Create: `worker-stt-realtime/bench/real_calls/segment.py`, `bench/real_calls/review.py`
- Test: `worker-stt-realtime/bench/tests/test_segment.py`, `bench/tests/test_review.py`

**Interfaces:**
- Consumes: 16 kHz mono wavs in `data/real_raw/` (Task 1).
- Produces: `split_speech_ranges(flags, frame_s, min_silence_s=0.6, min_seg_s=1.5, max_seg_s=15.0, pad_s=0.25) -> list[tuple[float,float]]`; `speech_flags(audio) -> list[bool]` (32 ms frames); `build_manifest(raw_dir, out_dir, model_size="large-v3-turbo", max_calls=None) -> list[dict]` writing `data/real/<call_id>_<k:03d>.wav` and `data/real/manifest.json` entries `{id, call_id, start_s, end_s, ref, verified, keyterms}`; `export_review(manifest_path, tsv_path, call_ids=None)`; `apply_review(manifest_path, tsv_path) -> int`.

- [ ] **Step 1: Write the failing tests**

`bench/tests/test_segment.py`:
```python
import pytest
from real_calls.segment import split_speech_ranges

F = 0.1


def test_two_blocks_separated_by_long_silence_stay_separate():
    flags = [False] * 5 + [True] * 20 + [False] * 10 + [True] * 20 + [False] * 5
    got = split_speech_ranges(flags, F)
    assert got == [pytest.approx((0.25, 2.75)), pytest.approx((3.25, 5.75))]


def test_short_gap_is_merged_and_padding_is_clipped():
    flags = [True] * 20 + [False] * 3 + [True] * 20
    assert split_speech_ranges(flags, F) == [pytest.approx((0.0, 4.3))]


def test_blip_shorter_than_min_segment_is_dropped():
    assert split_speech_ranges([False] * 10 + [True] * 5 + [False] * 10, F) == []


def test_long_continuous_speech_is_hard_split():
    got = split_speech_ranges([True] * 400, F)
    assert len(got) == 3
    assert all(e - s <= 15.5 + 1e-9 for s, e in got)


def test_merged_group_over_max_is_split_at_run_boundary():
    flags = [True] * 100 + [False] * 3 + [True] * 100
    assert len(split_speech_ranges(flags, F)) == 2
```
`bench/tests/test_review.py`:
```python
import csv
import json

from real_calls.review import apply_review, export_review


def _manifest(tmp_path):
    entries = [
        {"id": "c1_000", "call_id": "c1", "start_s": 0.0, "end_s": 3.0, "ref": "draft one", "verified": False, "keyterms": []},
        {"id": "c2_000", "call_id": "c2", "start_s": 1.0, "end_s": 4.0, "ref": "draft two", "verified": False, "keyterms": []},
    ]
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(entries))
    return p


def test_export_filters_by_call_id(tmp_path):
    m, t = _manifest(tmp_path), tmp_path / "review.tsv"
    export_review(str(m), str(t), call_ids={"c1"})
    rows = list(csv.DictReader(open(t), delimiter="\t"))
    assert [r["id"] for r in rows] == ["c1_000"]
    assert rows[0]["draft"] == "draft one" and rows[0]["verified_text"] == ""


def test_apply_review_marks_verified_and_parses_keyterms(tmp_path):
    m, t = _manifest(tmp_path), tmp_path / "review.tsv"
    export_review(str(m), str(t))
    rows = list(csv.DictReader(open(t), delimiter="\t"))
    rows[0]["verified_text"] = "hello this is Sushant"
    rows[0]["keyterms"] = "Sushant; 555 1234"
    with open(t, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    assert apply_review(str(m), str(t)) == 1
    entries = {e["id"]: e for e in json.load(open(m))}
    assert entries["c1_000"]["verified"] is True
    assert entries["c1_000"]["ref"] == "hello this is Sushant"
    assert entries["c1_000"]["keyterms"] == ["Sushant", "555 1234"]
    assert entries["c2_000"]["verified"] is False and entries["c2_000"]["ref"] == "draft two"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_segment.py tests/test_review.py -q`
Expected: FAIL with `ModuleNotFoundError` for `real_calls.segment` / `real_calls.review`.

- [ ] **Step 3: Implement `review.py`**

```python
import csv
import json


def export_review(manifest_path, tsv_path, call_ids=None):
    entries = json.load(open(manifest_path))
    with open(tsv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "call_id", "dur_s", "draft", "verified_text", "keyterms"])
        for e in entries:
            if e.get("verified") or (call_ids is not None and e["call_id"] not in call_ids):
                continue
            w.writerow([e["id"], e["call_id"], round(e["end_s"] - e["start_s"], 1), e["ref"], "",
                        ";".join(e.get("keyterms", []))])


def apply_review(manifest_path, tsv_path):
    entries = json.load(open(manifest_path))
    by_id = {e["id"]: e for e in entries}
    n = 0
    with open(tsv_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = (row.get("verified_text") or "").strip()
            if not text:
                continue
            e = by_id[row["id"]]
            e["ref"] = text
            e["verified"] = True
            e["keyterms"] = [t.strip() for t in (row.get("keyterms") or "").split(";") if t.strip()]
            n += 1
    json.dump(entries, open(manifest_path, "w"), indent=1)
    return n
```

- [ ] **Step 4: Implement `segment.py`**

```python
"""Cut call recordings into utterances and draft references.
  python -m real_calls.segment --raw ../data/real_raw --out ../data/real [--model large-v3-turbo] [--max-calls N]"""
import argparse
import json
import os

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
    groups = []
    for r in _runs(flags, frame_s):
        if groups and r[0] - groups[-1][-1][1] < min_silence_s:
            groups[-1].append(r)
        else:
            groups.append([r])
    segs = []
    for g in groups:
        cur = None
        for s, e in g:
            while e - s > max_seg_s:
                if cur:
                    segs.append(cur)
                    cur = None
                segs.append((s, s + max_seg_s))
                s += max_seg_s
            if cur is None:
                cur = (s, e)
            elif e - cur[0] > max_seg_s:
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


def build_manifest(raw_dir, out_dir, model_size="large-v3-turbo", max_calls=None):
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
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(kept, f, indent=1)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(DATA, "real_raw"))
    ap.add_argument("--out", default=os.path.join(DATA, "real"))
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--max-calls", type=int, default=None)
    a = ap.parse_args()
    kept = build_manifest(a.raw, a.out, a.model, a.max_calls)
    print(f"{len(kept)} utterances in {len({e['call_id'] for e in kept})} calls -> {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests -q`
Expected: all tests from Tasks 1 and 2 pass.

- [ ] **Step 6: Commit**

```bash
git add worker-stt-realtime/bench/real_calls worker-stt-realtime/bench/tests
git commit -m "stt-eval: utterance segmentation, Whisper draft references, review sheet export/import"
```

- [ ] **Step 7: CONTROLLER/USER STEP (not for the implementer): build the manifest and hand-verify**

```bash
cd worker-stt-realtime && . .venv/bin/activate && df -h /System/Volumes/Data | tail -1   # models below need ~1 GB free
mkdir -p models && (cd models && B=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models \
  && for m in sherpa-onnx-streaming-zipformer-en-2023-06-26 sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8; do curl -fsSL $B/$m.tar.bz2 | tar xj; done \
  && curl -fsSL -o silero_vad.onnx $B/silero_vad.onnx)   # NOT fetch_models.sh: it fills docker_models/ and has no NeMo
cd bench && export STT_MODELS=$PWD/../models STT_DATA=$PWD/../data
python3 -m real_calls.segment --max-calls 40
python3 -c "from real_calls.review import export_review; export_review('../data/real/manifest.json','../data/real/review.tsv', call_ids=None)"
```
The user picks about 15 calls (~30 min of audio) with names and numbers, fills `verified_text` and `keyterms` (semicolon-separated: names, digit strings) in `data/real/review.tsv`, then:
```bash
python3 -c "from real_calls.review import apply_review; print(apply_review('../data/real/manifest.json','../data/real/review.tsv'))"
```
Expected: prints the number of verified utterances. The user also writes the ids of calls they confirm are their own tests, one per line, to `data/confirmed_calls.txt`.

---

### Task 3: Shared normaliser, keyterm metric, rtbench `real` set

**Files:**
- Create: `worker-stt-realtime/bench/textnorm.py`, `bench/metrics.py`
- Modify: `worker-stt-realtime/bench/rtbench.py`
- Test: `worker-stt-realtime/bench/tests/test_metrics.py`, `bench/tests/test_rtbench_load.py`

**Interfaces:**
- Produces: `textnorm.norm(text) -> str` (unchanged behaviour, moved); `metrics.keyterm_recall(terms, hyp) -> (hits:int, total:int)`; `rtbench.load(set_name, n, seed=0, verified_only=False)` where each clip dict now also carries `call_id` and `keyterms`; each result row gains `keyterm_hits`, `keyterm_total`; summary gains `keyterm_recall` (float or None).

- [ ] **Step 1: Write the failing tests**

`bench/tests/test_metrics.py`:
```python
from metrics import keyterm_recall
from textnorm import norm


def test_norm_spells_digits_and_strips_punctuation():
    assert norm("Call 555-1234!") == "call five five five one two three four"


def test_keyterm_recall_counts_names_and_numbers():
    hyp = "this is sushant call five five five one two three four"
    assert keyterm_recall(["Sushant", "555 1234"], hyp) == (2, 2)


def test_keyterm_recall_partial_and_word_boundaries():
    assert keyterm_recall(["Sushant", "Ashant"], "hi ashant") == (1, 2)
    assert keyterm_recall(["ash"], "ashant") == (0, 1)
    assert keyterm_recall([], "anything") == (0, 0)
```
`bench/tests/test_rtbench_load.py`:
```python
import json

import numpy as np
import soundfile as sf

import rtbench


def _make_set(tmp_path):
    d = tmp_path / "real"
    d.mkdir()
    man = [
        {"id": "a_000", "call_id": "a", "ref": "hello", "verified": True, "keyterms": ["hello"]},
        {"id": "b_000", "call_id": "b", "ref": "world", "verified": False, "keyterms": []},
    ]
    for m in man:
        sf.write(str(d / (m["id"] + ".wav")), np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
    (d / "manifest.json").write_text(json.dumps(man))


def test_load_real_set_carries_metadata_and_filters_verified(tmp_path, monkeypatch):
    _make_set(tmp_path)
    monkeypatch.setattr(rtbench, "DATA", str(tmp_path))
    clips = rtbench.load("real", 10)
    assert [c["call_id"] for c in clips] == ["a", "b"]
    assert clips[0]["keyterms"] == ["hello"]
    only = rtbench.load("real", 10, verified_only=True)
    assert [c["id"] for c in only] == ["a_000"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_metrics.py tests/test_rtbench_load.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'metrics'`, and `load()` got an unexpected keyword `verified_only`).

- [ ] **Step 3: Create `textnorm.py` and `metrics.py`**

`bench/textnorm.py` (moved verbatim from `rtbench.py`):
```python
import re

_D = "zero one two three four five six seven eight nine".split()


def norm(t):
    t = re.sub(r"\d", lambda m: " " + _D[int(m.group())] + " ", t)   # digits -> spoken (refs are spelled out)
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()
```
`bench/metrics.py`:
```python
from textnorm import norm


def keyterm_recall(terms, hyp):
    h = " " + norm(hyp) + " "
    usable = [norm(t) for t in terms if norm(t)]
    return sum(1 for t in usable if " " + t + " " in h), len(usable)
```

- [ ] **Step 4: Modify `rtbench.py`**

(a) Replace the `_D = ...` line and the whole `def norm(t): ...` function with an import placed right after the `sys.path.insert(0, HERE)` line:
```python
from textnorm import norm            # noqa: E402  (ws_client.py imports norm from here; keep it exported)
from metrics import keyterm_recall   # noqa: E402
```
(b) In `load`, change the signature and manifest line, and add metadata to each clip:
```python
def load(set_name, n, seed=0, verified_only=False):
    mp = os.path.join(DATA, set_name, "manifest.json")
    man = json.load(open(mp if os.path.exists(mp) else os.path.join(DATA, "manifest.json")))
    if verified_only:
        man = [m for m in man if m.get("verified")]
    man = man[:n]
```
change `sigma = 0.0 if set_name in ("clean", "call") else ...` to `set_name in ("clean", "call", "real")`, and change the append to:
```python
        out.append({"id": m["id"], "ref": m["ref"], "call_id": m.get("call_id"), "keyterms": m.get("keyterms", []),
                    "dur": (len(x) + len(lead)) / 16000, "audio": np.concatenate([lead, x, tail]).astype("float32")})
```
(c) In `main()`: add `ap.add_argument("--only-verified", action="store_true")`; replace `clips = load(a.set, a.n)` with `clips = load(a.set, a.n, verified_only=a.only_verified)`; after `row = {...}` is built add:
```python
        kh, kt = keyterm_recall(c.get("keyterms", []), final_text)
        row["keyterm_hits"], row["keyterm_total"] = kh, kt
```
and in the `summ = {...}` dict add:
```python
            "keyterm_total": sum(r["keyterm_total"] for r in rows),
            "keyterm_recall": (sum(r["keyterm_hits"] for r in rows) / sum(r["keyterm_total"] for r in rows))
                              if sum(r["keyterm_total"] for r in rows) else None,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests -q`
Expected: all tests pass. Then `python3 rtbench.py --help` exits 0 and lists `--only-verified`.

- [ ] **Step 6: Commit**

```bash
git add worker-stt-realtime/bench
git commit -m "stt-eval: shared normaliser, keyterm recall metric, rtbench real set and verified filter"
```

---

### Task 4: Deepgram and ElevenLabs adapters with the third-party gate

**Files:**
- Create: `worker-stt-realtime/bench/engines_cloud.py`
- Modify: `worker-stt-realtime/bench/engines.py` (`build()`), `worker-stt-realtime/bench/rtbench.py` (gate and native strategy)
- Test: `worker-stt-realtime/bench/tests/test_engines_cloud.py`

**Interfaces:**
- Consumes: `engines.Engine` base class; stream interface `push(x float32 16 kHz) / text() / final() / native_eos()`; clip dicts with `call_id` (Task 3).
- Produces: `assert_allowed(clips, confirmed_path)` (raises `PermissionError`); `DeepgramFluxEngine(url=None, key=None, eot_threshold=0.7, eot_timeout_ms=5000)` named `dg-flux`; `ElevenLabsRealtimeEngine(url=None, key=None)` named `el-scribe`; `engines.build("dg-flux"|"el-scribe")`. Keys come from `DEEPGRAM_API_KEY` / `ELEVENLABS_API_KEY`. `native_eos()` is true after Deepgram `EndOfTurn` / ElevenLabs `committed_transcript`.

- [ ] **Step 1: Write the failing tests**

`bench/tests/test_engines_cloud.py`:
```python
import json
import threading
import time

import numpy as np
import pytest
from websockets.sync.server import serve

import engines_cloud as ec


def _serve(handler):
    srv = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.socket.getsockname()[1]


def _until(cond, secs=2.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_gate_refuses_unconfirmed_calls_but_allows_public_clips(tmp_path):
    confirmed = tmp_path / "confirmed.txt"
    confirmed.write_text("a\n")
    ec.assert_allowed([{"call_id": None}, {"call_id": "a"}], str(confirmed))
    with pytest.raises(PermissionError, match="b"):
        ec.assert_allowed([{"call_id": "a"}, {"call_id": "b"}], str(confirmed))
    with pytest.raises(PermissionError):
        ec.assert_allowed([{"call_id": "a"}], str(tmp_path / "missing.txt"))


def test_deepgram_flux_turn_flow():
    def handler(ws):
        first = True
        for msg in ws:
            if isinstance(msg, bytes):
                if first:
                    first = False
                    ws.send(json.dumps({"type": "TurnInfo", "event": "StartOfTurn", "transcript": ""}))
                    ws.send(json.dumps({"type": "TurnInfo", "event": "Update", "transcript": "hello"}))
            elif json.loads(msg).get("type") == "CloseStream":
                ws.send(json.dumps({"type": "TurnInfo", "event": "EndOfTurn", "transcript": "hello world"}))
                ws.close()

    srv, port = _serve(handler)
    st = ec.DeepgramFluxEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
    st.push(np.zeros(640, dtype="float32"))
    assert _until(lambda: st.text() == "hello")
    assert not st.native_eos()
    assert st.final() == "hello world"
    assert st.native_eos()
    srv.shutdown()


def test_elevenlabs_partial_then_manual_commit_fallback(monkeypatch):
    monkeypatch.setattr(ec._ELStream, "AUTO_COMMIT_WAIT_S", 0.05)

    def handler(ws):
        ws.send(json.dumps({"message_type": "session_started", "session_id": "s"}))
        for msg in ws:
            m = json.loads(msg)
            if m["message_type"] != "input_audio_chunk":
                continue
            if m["commit"]:
                ws.send(json.dumps({"message_type": "committed_transcript", "text": "testing one two"}))
            else:
                ws.send(json.dumps({"message_type": "partial_transcript", "text": "testing"}))

    srv, port = _serve(handler)
    st = ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
    st.push(np.zeros(640, dtype="float32"))
    assert _until(lambda: st.text() == "testing")
    assert st.final() == "testing one two"
    assert st.native_eos()
    srv.shutdown()


def test_elevenlabs_auth_error_raises_on_new_stream():
    def handler(ws):
        ws.send(json.dumps({"message_type": "auth_error", "error": "bad key"}))
        ws.close()

    srv, port = _serve(handler)
    with pytest.raises(RuntimeError, match="bad key"):
        ec.ElevenLabsRealtimeEngine(url=f"ws://127.0.0.1:{port}", key="k").new_stream()
    srv.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_engines_cloud.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'engines_cloud'`.

- [ ] **Step 3: Implement `engines_cloud.py`**

```python
"""Deepgram Flux and ElevenLabs Scribe v2 Realtime behind the bench stream interface.
Third-party engines only ever see calls listed in a confirmed-calls file (assert_allowed)."""
import base64
import json
import os
import threading
import time

import numpy as np

from engines import Engine


def assert_allowed(clips, confirmed_path):
    ok = set(open(confirmed_path).read().split()) if os.path.exists(confirmed_path) else set()
    bad = sorted({c["call_id"] for c in clips if c.get("call_id") is not None and c["call_id"] not in ok})
    if bad:
        raise PermissionError(f"refusing third-party STT: {len(bad)} call id(s) not in {confirmed_path}: {bad[:5]}")


def _pcm16(x):
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


class _Stream:
    def __init__(self, url, headers):
        from websockets.sync.client import connect
        self.committed, self.partial, self.eos, self.error = [], "", False, None
        self.closed, self.ready = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.ws = connect(url, additional_headers=headers, open_timeout=10, max_size=None)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for raw in self.ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                with self.lock:
                    self.handle(msg)
        except Exception:
            pass
        finally:
            self.closed.set()
            self.ready.set()

    def text(self):
        with self.lock:
            return " ".join(self.committed + ([self.partial] if self.partial else [])).strip()

    def native_eos(self):
        with self.lock:
            return self.eos

    def _wait(self, cond, timeout):
        end = time.time() + timeout
        while time.time() < end and not self.closed.is_set():
            with self.lock:
                if cond():
                    return True
            time.sleep(0.01)
        with self.lock:
            return cond()

    def _turn_done(self):
        return self.eos and not self.partial


class _DGStream(_Stream):
    def handle(self, m):
        if m.get("type") != "TurnInfo":
            return
        ev, tr = m.get("event"), (m.get("transcript") or "").strip()
        if ev == "StartOfTurn":
            self.eos = False
        elif ev in ("Update", "EagerEndOfTurn", "TurnResumed"):
            self.partial, self.eos = tr, False
        elif ev == "EndOfTurn":
            if tr:
                self.committed.append(tr)
            self.partial, self.eos = "", True

    def push(self, x):
        self.ws.send(_pcm16(x))

    def final(self):
        try:
            self.ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:
            pass
        self._wait(self._turn_done, 3.0)
        text = self.text()
        try:
            self.ws.close()
        except Exception:
            pass
        return text


class _ELStream(_Stream):
    AUTO_COMMIT_WAIT_S = 2.0
    _ERRORS = {"quota_exceeded", "rate_limited", "queue_overflow", "resource_exhausted", "session_time_limit_exceeded"}

    def handle(self, m):
        t = m.get("message_type") or ""
        if t == "session_started":
            self.ready.set()
        elif t == "partial_transcript":
            self.partial, self.eos = (m.get("text") or "").strip(), False
        elif t == "committed_transcript":
            tx = (m.get("text") or "").strip()
            if tx:
                self.committed.append(tx)
            self.partial, self.eos = "", True
        elif "error" in t or t in self._ERRORS:
            self.error = m.get("error") or t
            self.ready.set()

    def _chunk(self, x, commit):
        self.ws.send(json.dumps({"message_type": "input_audio_chunk", "commit": commit, "sample_rate": 16000,
                                 "audio_base_64": base64.b64encode(_pcm16(x)).decode()}))

    def push(self, x):
        self._chunk(x, False)

    def final(self):
        if not self._wait(self._turn_done, self.AUTO_COMMIT_WAIT_S):
            self._chunk(np.zeros(1600, dtype="float32"), True)
            self._wait(self._turn_done, 3.0)
        text = self.text()
        try:
            self.ws.close()
        except Exception:
            pass
        return text


class DeepgramFluxEngine(Engine):
    name = "dg-flux"

    def __init__(self, url=None, key=None, eot_threshold=0.7, eot_timeout_ms=5000):
        self.key = key or os.environ["DEEPGRAM_API_KEY"]
        self.url = url or ("wss://api.deepgram.com/v2/listen?model=flux-general-en"
                           f"&eot_threshold={eot_threshold}&eot_timeout_ms={eot_timeout_ms}"
                           "&encoding=linear16&sample_rate=16000")

    def new_stream(self):
        return _DGStream(self.url, {"Authorization": f"Token {self.key}"})


class ElevenLabsRealtimeEngine(Engine):
    name = "el-scribe"

    def __init__(self, url=None, key=None):
        self.key = key or os.environ["ELEVENLABS_API_KEY"]
        self.url = url or ("wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime"
                           "&audio_format=pcm_16000&commit_strategy=vad&language_code=en")

    def new_stream(self):
        s = _ELStream(self.url, {"xi-api-key": self.key})
        if not s.ready.wait(5) or s.error:
            raise RuntimeError(f"elevenlabs session failed: {s.error or 'no session_started within 5 s'}")
        return s
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_engines_cloud.py -q`
Expected: 4 passed.

- [ ] **Step 5: Wire into `engines.build` and `rtbench`**

In `engines.py`, add at the top of `build()` (before the final `raise ValueError(name)` is fine; place before the `if name == "zip-en-int8"` line):
```python
    if name in ("dg-flux", "el-scribe"):
        import engines_cloud
        return engines_cloud.DeepgramFluxEngine() if name == "dg-flux" else engines_cloud.ElevenLabsRealtimeEngine()
```
In `rtbench.py` `main()`, immediately after `clips = load(...)` add:
```python
    if a.engine.startswith(("dg-", "el-")):
        import engines_cloud
        engines_cloud.assert_allowed(clips, os.path.join(DATA, "confirmed_calls.txt"))
```
and change the strategies line so cloud engines also get the `native` strategy:
```python
    strategies = ["commit", "vad300", "vad500", "vad700", "vad500+hint"] + (["native"] if a.native_ms or a.engine.startswith(("moonshine", "dg-", "el-")) else [])
```
Run: `python3 -m pytest tests -q` (all pass) and `python3 -c "import engines; engines.build('nope')"` (raises `ValueError: nope`).

- [ ] **Step 6: Commit**

```bash
git add worker-stt-realtime/bench
git commit -m "stt-eval: Deepgram Flux and ElevenLabs Scribe adapters with confirmed-calls gate"
```

- [ ] **Step 7: CONTROLLER STEP (not for the implementer): live smoke test on a synthetic clip**

Public synthetic audio only (no call data). With keys exported from `call-loop-poc/.env` (read, never printed):
```bash
say -o /tmp/smoke.aiff "Hello, my name is Sushant and my number is five five five one two three four." && ffmpeg -y -loglevel error -i /tmp/smoke.aiff -ac 1 -ar 16000 /tmp/smoke.wav
python3 - <<'EOF'
import soundfile as sf, numpy as np, time, engines
x, _ = sf.read("/tmp/smoke.wav", dtype="float32")
x = np.concatenate([x, np.zeros(16000 * 2, dtype="float32")])
for name in ("dg-flux", "el-scribe"):
    st = engines.build(name).new_stream()
    for i in range(0, len(x), 640):
        st.push(x[i:i + 640]); time.sleep(0.04)
    print(name, "->", st.final())
EOF
```
Expected: each prints a recognisable transcript. If a protocol detail differs from the docs (event names, close message), fix the adapter and its fake-server test together before continuing.

---

### Task 5: Report, decision gate, runbook

**Files:**
- Create: `worker-stt-realtime/bench/report_real.py`
- Test: `worker-stt-realtime/bench/tests/test_report_real.py`
- Output (committed): `worker-stt-realtime/REAL_CALL_EVAL.md` (aggregate numbers only, no transcripts)

**Interfaces:**
- Consumes: `bench/results/real__<engine>__real.json` written by `rtbench.py` (`summary` has `engine, n, wer, keyterm_recall, keyterm_total, first_partial_med_s, cpu_per_audio_s, strat{commit,native}{lat_med,lat_p90,cut_utts}`).
- Produces: `load_results(results_dir, tag="real") -> list[dict]`; `price_per_hour(summary) -> float`; `render_table(summaries) -> str`; `gate(summaries) -> dict` with `best_ours`, `best_cloud`, `checks`, `pass`.

- [ ] **Step 1: Write the failing tests**

`bench/tests/test_report_real.py`:
```python
from report_real import gate, price_per_hour, render_table


def _s(engine, wer, kr, commit_lat, cpu=0.1, native=0.6):
    return {"engine": engine, "n": 40, "wer": wer, "keyterm_recall": kr, "keyterm_total": 30,
            "first_partial_med_s": 0.5, "cpu_per_audio_s": cpu,
            "strat": {"commit": {"lat_med": commit_lat, "lat_p90": commit_lat * 2, "cut_utts": 0},
                      "native": {"lat_med": native, "lat_p90": native * 1.3, "cut_utts": 3}}}


def test_price_per_hour_cloud_is_fixed_and_local_uses_cpu():
    assert price_per_hour(_s("dg-flux", 0.1, 0.9, 0.1)) == 0.39
    assert abs(price_per_hour(_s("nemo-480", 0.1, 0.9, 0.05, cpu=0.1)) - 0.1 * 3600 * 0.0000131) < 1e-9


def test_gate_passes_when_local_is_close_enough():
    res = gate([_s("nemo-480", 0.12, 0.85, 0.06), _s("zip-en-int8", 0.20, 0.7, 0.05),
                _s("dg-flux", 0.09, 0.90, 0.2), _s("el-scribe", 0.10, 0.88, 0.2)])
    assert res["best_ours"]["engine"] == "nemo-480" and res["best_cloud"]["engine"] == "dg-flux"
    assert res["pass"] is True and all(res["checks"].values())


def test_gate_fails_on_wer_and_reports_which_check():
    res = gate([_s("nemo-480", 0.30, 0.85, 0.06), _s("dg-flux", 0.09, 0.90, 0.2)])
    assert res["pass"] is False and res["checks"]["wer_within_1.5x"] is False


def test_gate_needs_both_kinds_of_engine():
    assert "error" in gate([_s("nemo-480", 0.1, 0.9, 0.05)])


def test_render_table_lists_every_engine_and_price():
    out = render_table([_s("nemo-480", 0.12, 0.85, 0.06), _s("dg-flux", 0.09, 0.9, 0.2)])
    assert "nemo-480" in out and "dg-flux" in out and "0.39" in out and "12.0%" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_report_real.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'report_real'`.

- [ ] **Step 3: Implement `report_real.py`**

```python
"""Render the real-call comparison and the Phase 1 decision gate from bench/results/real__*.json.
  python report_real.py [--results results] [--out ../REAL_CALL_EVAL.md]"""
import argparse
import glob
import json
import os

CLOUD_PRICE = {"dg-flux": 0.39, "el-scribe": 0.39}     # $/audio-hour, published rates (2026-09)
MODAL_CPU_PER_CORE_S = 0.0000131                      # $/core-second, worker-stt/DESIGN.md (compute only, packed)


def load_results(results_dir, tag="real"):
    return [json.load(open(f))["summary"] for f in sorted(glob.glob(os.path.join(results_dir, f"{tag}__*.json")))]


def price_per_hour(s):
    return CLOUD_PRICE.get(s["engine"], s["cpu_per_audio_s"] * 3600 * MODAL_CPU_PER_CORE_S)


def _ms(x):
    return "-" if x is None else f"{x * 1000:.0f}"


def _pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


def render_table(summaries):
    lines = ["| engine | n utts | WER | keyterm recall | first partial ms | commit final ms med/p90 | native final ms med/p90 | cpu-s per audio-s | $/audio-hr |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        c, n = s["strat"].get("commit", {}), s["strat"].get("native", {})
        lines.append(f"| {s['engine']} | {s['n']} | {_pct(s['wer'])} | {_pct(s.get('keyterm_recall'))} | "
                     f"{_ms(s['first_partial_med_s'])} | {_ms(c.get('lat_med'))}/{_ms(c.get('lat_p90'))} | "
                     f"{_ms(n.get('lat_med'))}/{_ms(n.get('lat_p90'))} | "
                     f"{'-' if s['engine'] in CLOUD_PRICE else round(s['cpu_per_audio_s'], 3)} | {price_per_hour(s):.4f} |")
    return "\n".join(lines)


def gate(summaries):
    ours = [s for s in summaries if s["engine"] not in CLOUD_PRICE]
    cloud = [s for s in summaries if s["engine"] in CLOUD_PRICE]
    if not ours or not cloud:
        return {"error": "need at least one local and one cloud result"}
    bo, bc = min(ours, key=lambda s: s["wer"]), min(cloud, key=lambda s: s["wer"])
    ko, kc = bo.get("keyterm_recall"), bc.get("keyterm_recall")
    lat = bo["strat"].get("commit", {}).get("lat_med")
    checks = {
        "wer_within_1.5x": bo["wer"] <= 1.5 * bc["wer"],
        "keyterm_within_10pts": ko is not None and kc is not None and ko >= kc - 0.10,
        "commit_final_le_150ms": lat is not None and lat <= 0.150,
        "cpu_le_0.15": bo["cpu_per_audio_s"] <= 0.15,
    }
    return {"best_ours": bo, "best_cloud": bc, "checks": checks, "pass": all(checks.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "REAL_CALL_EVAL.md"))
    a = ap.parse_args()
    summ = load_results(a.results)
    g = gate(summ)
    body = ["# Real-call STT evaluation (Phase 1)", "",
            "Verified utterances only; paced replay as in `bench/rtbench.py`; local engines measured on the dev machine's CPU (a proxy, not Fly).",
            "", render_table(summ), "", "## Decision gate", ""]
    if "error" in g:
        body.append(g["error"])
    else:
        body += [f"Best local: **{g['best_ours']['engine']}**, best cloud: **{g['best_cloud']['engine']}**.", ""]
        body += [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in g["checks"].items()]
        body += ["", f"**Result: {'ship-worthy, proceed to Phase 2 with ' + g['best_ours']['engine'] if g['pass'] else 'not yet, Phase 2 becomes an accuracy investigation'}**"]
    text = "\n".join(body) + "\n"
    with open(a.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add worker-stt-realtime/bench
git commit -m "stt-eval: comparison report and Phase 1 decision gate"
```

- [ ] **Step 6: CONTROLLER/USER STEP (not for the implementer): run the matrix and read the gate**

Prerequisites from Tasks 1-2 (verified utterances, `data/confirmed_calls.txt`). Keys are exported into the shell only for the cloud runs, from the env file, never echoed:
```bash
cd worker-stt-realtime/bench && . ../.venv/bin/activate
export STT_DATA=$PWD/../data STT_MODELS=$PWD/../models
for e in zip-en-int8 nemo-480; do python3 rtbench.py --engine $e --set real --n 500 --only-verified --out results/real__${e}__real.json; done
set -a; . <(grep -E '^(DEEPGRAM|ELEVENLABS)_API_KEY=' ../../call-loop-poc/.env); set +a
for e in dg-flux el-scribe; do python3 rtbench.py --engine $e --set real --n 500 --only-verified --out results/real__${e}__real.json; done
python3 report_real.py
```
Expected: `REAL_CALL_EVAL.md` written with a table and the PASS/FAIL gate. Commit `REAL_CALL_EVAL.md` only (aggregates); result JSON stays ignored. The user reads the gate: PASS means write the Phase 2 plan around the winning engine; FAIL means Phase 2 starts with an accuracy investigation.

---

## Self-review

**Spec coverage:** data source and privacy rules (Tasks 1, 4 gate); own-number direction-aware selection (Task 1); utterance cutting, draft references, hand-verified subset, keyterms (Task 2); candidates and metrics incl. WER, keyterm recall, latency, cuts, cpu, price (Tasks 3-5); decision gate thresholds (Task 5 `gate`); non-goals respected (no `server.py`, gateway or deploy changes). The user's tasks 4-6 (Fly deploy, speculative end-of-turn, gateway wiring/pricing) are deliberately deferred to the Phase 2 plan per the spec's scope, because the deployed artifact depends on the gate result.

**Placeholders:** none; every code step has code, every run step has an expected result. Steps labelled CONTROLLER/USER need real data, credentials or human transcription and are not delegable.

**Type consistency:** clip dict keys `call_id`/`keyterms` (Task 3) are what the gate (Task 4) reads; `keyterm_recall` in the summary (Task 3) is what `gate`/`render_table` read (Task 5); adapter names `dg-flux`/`el-scribe` match `CLOUD_PRICE` and the `startswith(("dg-","el-"))` checks; `_ELStream.AUTO_COMMIT_WAIT_S` is the attribute the test monkeypatches.
