# Public Human-Speech Telephone Benchmark Implementation Plan (Phase 2, step 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run our CPU streaming engines and the two cloud engines on public HUMAN speech with known transcripts (FLEURS en-US degraded to telephone quality, plus real earnings-call audio), to learn whether the accuracy gap seen on the synthetic-persona phone calls also holds on human speech.

**Architecture:** `bench/telephony.py` degrades 16 kHz audio to phone quality (300-3400 Hz band-limit, 8 kHz, G.711 µ-law, noise). `bench/public_sets.py` builds `rtbench`-format sets from FLEURS (one 290 MB tar) and Earnings-22 (row-group reads over HTTP, no full download). `rtbench.py`/`engines_cloud.py` learn that `pub_*` sets are public (allowed for third-party engines, no noise injection). `report_real.py` gains `--tag` so the same gate/report renders public results.

**Tech Stack:** Python 3.12, numpy, scipy, soundfile, huggingface_hub, pyarrow, pytest (the existing `worker-stt-realtime/.venv`; add `pyarrow` to `bench/requirements-real.txt`).

**Spec / background:** `worker-stt-realtime/REAL_CALL_EVAL_METHODOLOGY.md` (why the first evaluation was limited: mostly synthetic far-end speech, small sample) and `worker-stt-realtime/REAL_CALL_EVAL.md` (results: nemo-480 8.3% WER vs el-scribe 2.5%).

## Global Constraints

- Public data only (FLEURS, CC-BY-4.0; Earnings-22, CC-BY-SA-4.0: used for evaluation only). NO real-call data in this plan; the per-call privacy gate for `real` is unchanged and must stay fail-closed.
- Third-party engines may run only on: the existing public sets (`clean`, `tel`, `call`) and the new sets whose name starts with `pub_`; every other set name stays fail-closed.
- Audio format: mono, 16 kHz, PCM_16 wav. Disk is tight (about 1.5 GB free): never keep more than ~500 MB of downloaded/derived audio; delete the FLEURS tar after extracting the chosen clips.
- Work from `worker-stt-realtime/bench/`; tests: `../.venv/bin/python -m pytest tests -q` (119 pass at the start). Do not modify `server.py`, the gateway or any deploy config.
- Secrets are read from env at run time and never printed (cloud runs only, controller-run).

---

### Task 1: Telephone degradation

**Files:** Create `worker-stt-realtime/bench/telephony.py`, `worker-stt-realtime/bench/tests/test_telephony.py`.

**Interfaces:** Produces `mulaw_roundtrip(x: np.ndarray) -> np.ndarray` (float32 in [-1,1], same length, G.711 µ-law encode+decode with 8-bit companding) and `to_telephone(x16: np.ndarray, seed: int = 0, snr_db: float = 25.0) -> np.ndarray` (float32 16 kHz in, float32 16 kHz out, same length within ±2 samples).

- [ ] **Step 1: Write the failing tests** (`tests/test_telephony.py`)

```python
import numpy as np

from telephony import mulaw_roundtrip, to_telephone


def _tone(freq, sr=16000, secs=1.0, amp=0.5):
    t = np.arange(int(sr * secs)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _band_energy(x, lo, hi, sr=16000):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return spec[(f >= lo) & (f < hi)].sum()


def test_mulaw_roundtrip_is_close_and_shape_preserving():
    x = _tone(440, sr=8000)
    y = mulaw_roundtrip(x)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.max(np.abs(y - x)) < 0.05          # companding error bound for a 0.5-amplitude tone


def test_mulaw_roundtrip_clips_out_of_range_input():
    y = mulaw_roundtrip(np.array([2.0, -2.0, 0.0], dtype=np.float32))
    assert np.all(np.abs(y) <= 1.0 + 1e-6)


def test_to_telephone_length_and_dtype():
    x = _tone(1000)
    y = to_telephone(x)
    assert y.dtype == np.float32 and abs(len(y) - len(x)) <= 2


def test_to_telephone_removes_high_band_and_keeps_speech_band():
    x = _tone(1000) + _tone(6000)
    y = to_telephone(x, snr_db=60.0)
    high = _band_energy(y, 4200, 8000); mid = _band_energy(y, 500, 3000)
    assert high < 1e-3 * mid                      # 6 kHz component is gone
    assert mid > 0.05 * _band_energy(x, 500, 3000)  # 1 kHz component survives


def test_to_telephone_is_deterministic_per_seed_and_differs_across_seeds():
    x = _tone(800)
    assert np.array_equal(to_telephone(x, seed=1), to_telephone(x, seed=1))
    assert not np.array_equal(to_telephone(x, seed=1), to_telephone(x, seed=2))


def test_to_telephone_silence_stays_silent():
    y = to_telephone(np.zeros(16000, dtype=np.float32))
    assert np.max(np.abs(y)) < 0.02
```

- [ ] **Step 2: Run to verify they fail:** `../.venv/bin/python -m pytest tests/test_telephony.py -q` -> FAIL (`ModuleNotFoundError: telephony`).

- [ ] **Step 3: Implement** (`telephony.py`)

```python
"""Degrade 16 kHz speech to telephone quality: 8 kHz, 300-3400 Hz band, G.711 mu-law companding, additive noise."""
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

_SOS = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")


def mulaw_roundtrip(x):
    mu = 255.0
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1.0) / 2.0 * 255.0)
    y = q / 255.0 * 2.0 - 1.0
    return (np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu).astype(np.float32)


def to_telephone(x16, seed=0, snr_db=25.0):
    x = np.asarray(x16, dtype=np.float64)
    n = len(x)
    x8 = resample_poly(x, 1, 2)
    x8 = sosfilt(_SOS, x8)
    peak = np.max(np.abs(x8)) if len(x8) else 0.0
    if peak > 0:
        x8 = x8 * (0.7 / peak)                       # level like a phone line, never clipping
    x8 = mulaw_roundtrip(x8).astype(np.float64)
    rms = float(np.sqrt(np.mean(x8 ** 2))) if len(x8) else 0.0
    if rms > 0:
        x8 = x8 + np.random.default_rng(seed).normal(0.0, rms * 10 ** (-snr_db / 20.0), len(x8))
    y = resample_poly(x8, 2, 1)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)
```

- [ ] **Step 4: Run to verify they pass:** `../.venv/bin/python -m pytest tests -q` -> all pass (119 + 6).
- [ ] **Step 5: Commit:** `git add worker-stt-realtime/bench && git commit -m "stt-public: telephone degradation (8 kHz, band-limit, mu-law, noise)"`

---

### Task 2: Public set builders and public-set handling

**Files:** Create `worker-stt-realtime/bench/public_sets.py`, `worker-stt-realtime/bench/tests/test_public_sets.py`; modify `worker-stt-realtime/bench/rtbench.py`, `worker-stt-realtime/bench/engines_cloud.py`, `worker-stt-realtime/bench/requirements-real.txt` (add `pyarrow`), tests in `tests/test_engines_cloud.py` / `tests/test_rtbench_load.py` as needed.

**Interfaces:**
- Consumes: `telephony.to_telephone` (Task 1); `rtbench.load(set_name, n, seed=0, verified_only=False)` reads `DATA/<set>/manifest.json` (list of `{id, ref, ...}`) and `DATA/<set>/<id>.wav` (16 kHz mono).
- Produces: `public_sets.build_fleurs(out_root, n=50, seed=0, tar_path=None, tsv_path=None) -> dict` writing sets `pub_fleurs` (clean) and `pub_fleurs_tel` (same utterances degraded with `to_telephone(seed=hash-stable per id)`) under `out_root`; `public_sets.build_earnings(out_root, n=50, seed=0, min_s=3.0, max_s=15.0, parquet_source=None) -> dict` writing set `pub_earnings`; each returns `{"set": name, "n": int, "hours": float}`-style info per set. `engines_cloud.is_public_set(name) -> bool` (`name in PUBLIC_SETS or name.startswith("pub_")`), used by `assert_allowed` instead of `set_name in PUBLIC_SETS`. `rtbench.load` injects no noise for `pub_*` sets.

Details the implementer must follow (the real downloads are a CONTROLLER step; unit tests use tiny generated fixtures, no network):
- FLEURS: repo `google/fleurs` (dataset), files `data/en_us/audio/test.tar.gz` and `data/en_us/test.tsv` via `huggingface_hub.hf_hub_download(repo_type="dataset")`. TSV columns (tab-separated, no header): `id, file_name, raw_transcription, transcription, characters, num_samples, gender`; use column 3 (`transcription`, lowercase, no punctuation) as the reference. Audio members are named `test/<file_name>`. Select `n` rows with `random.Random(seed)` among rows whose `num_samples` is between 3 s and 15 s at 16 kHz; extract ONLY those members from the tar (stream with `tarfile.open(..., "r:gz")`, do not extract everything), write mono 16 kHz PCM_16 wavs; make `pub_fleurs_tel` from the same audio via `to_telephone` (seed derived deterministically from the utterance index).
- Earnings-22: repo `distil-whisper/earnings22`, config `chunked`, files `chunked/test-*.parquet`. Read with `huggingface_hub.HfFileSystem` + `pyarrow.parquet.ParquetFile` row group by row group (never download a whole shard); columns include `audio` (struct with `bytes`) and `transcription`. Decode `audio.bytes` with `soundfile`, resample to 16 kHz with `scipy.signal.resample_poly` if needed, keep utterances with `min_s <= duration <= max_s`, take `n` (deterministic given `seed`, stop reading as soon as enough are collected), skip rows with empty transcription.
- The functions accept injectable sources (`tar_path`/`tsv_path`, `parquet_source`) so tests build a tiny fake tar/parquet in `tmp_path` and never touch the network.

- [ ] **Step 1: Write failing tests** covering: (a) `build_fleurs` with a generated 6-member tar + TSV picks only rows in the length window, writes `pub_fleurs` and `pub_fleurs_tel` with identical ids/refs, mono 16 kHz PCM_16 wavs, and the `tel` audio has less energy above 4 kHz than the clean audio for the same id; deterministic for the same seed; (b) `build_earnings` with a generated parquet (audio bytes = flac/wav-encoded tones, one row too short, one empty transcription) returns only valid rows and resamples a 24 kHz row to 16 kHz; (c) `is_public_set`: `pub_x` and `clean` True, `real`/`weird` False; `assert_allowed` allows clips with no `call_id` for `pub_fleurs_tel` and still refuses them for `real`; (d) `rtbench.load("pub_fleurs_tel", ...)` returns the audio unchanged apart from the lead/tail padding (an all-zero wav stays exactly zero in the middle section); (e) the existing rtbench `--set real` `--out real__*` guard is unaffected.
- [ ] **Step 2: Run to verify they fail**, **Step 3: Implement**, **Step 4: full suite green** (`../.venv/bin/python -m pytest tests -q`), **Step 5: Commit** `stt-public: FLEURS and Earnings-22 set builders, pub_* sets are public`.

---

### Task 3: Report for public sets

**Files:** Modify `worker-stt-realtime/bench/report_real.py`, `worker-stt-realtime/bench/tests/test_report_real.py`.

**Interfaces:** `load_results(results_dir, tag="real")` already takes a tag; add CLI `--tag` (default `real`), `--title` and keep `--results`/`--out`. The "scored on unverified draft references" warning and its INSUFFICIENT downgrade apply ONLY to summaries whose `set` is `real`; public-set summaries (`set` starting `pub_`) are treated as reference-verified (their transcripts come from the corpus). Result files are named `pub__<engine>__<set>.json` and each engine appears once per set: `load_results` must be able to load ONE set at a time via a new optional `set_name` filter (`--set pub_fleurs_tel`); the gate/keyterm logic stays as is (public sets have no keyterms, so the keyterm check is INSUFFICIENT and the overall verdict is INSUFFICIENT/None: state that in the rendered text; the WER table is the product here).

- [ ] Step 1: tests first (set filter, `--tag pub` file loading from a tmp dir with two sets, unverified warning not applied to `pub_*`, title rendered), Step 2: red, Step 3: implement, Step 4: green (whole suite), Step 5: commit `stt-public: report_real supports --tag/--set/--title for public sets`.

---

### Task 4 (CONTROLLER steps, not for subagents)

1. Build the sets (network): `python -c "import public_sets as p; print(p.build_fleurs('../data', n=50)); print(p.build_earnings('../data', n=50))"`; check sizes (`du -sh ../data/pub_*`), then delete the FLEURS tar from the HF cache if it is large (`~/.cache/huggingface/hub/datasets--google--fleurs`).
2. Run the matrix with `rtbench.py --engine <e> --set <set> --n 50 --out results/pub__<e>__<set>.json` for e in `zip-en-int8 nemo-480 dg-flux el-scribe` and set in `pub_fleurs pub_fleurs_tel pub_earnings` (two chains in parallel: local engines / cloud engines; latency numbers from these runs are NOT comparable because of CPU contention: WER is the product).
3. `python report_real.py --tag pub --set <set> --title ... --out ../PUBLIC_EVAL_<set>.md` for each set, then write `worker-stt-realtime/PUBLIC_EVAL.md` summarising: the three WER tables, how it compares with the real-call result, and the recommendation for what to try next (fine-tune on degraded human speech or not). Commit aggregate results only (per-run result JSON contains transcripts of PUBLIC data, so committing it is allowed, but keep it out of git anyway to stay small: add `results/pub__*.json` to `.gitignore` and commit `results/pub_summary.json`).
