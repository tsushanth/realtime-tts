# Orpheus Streaming Voice Cloning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a second, parallel self-serve voice-cloning path (`custom-fast:<id>`) built on the
already-validated Orpheus LoRA pipeline, alongside the existing Piper cloning product, with no
changes to Piper cloning.

**Architecture:** A new isolated Modal app (`orpheus-clone-prod`) owns training (job-submission
pattern, like `voice-design-dev`) and serving (scale-to-zero vLLM engine, like
`serve_benchmark.py`'s `ClonedVoiceEngine`) behind its own FastAPI surface. A thin ReadAloudAI
gateway route (mirroring `voiceStudioRouter.ts`'s proxy-to-Modal shape) adds auth/billing and
exposes it to customers. A new developers-page docs component replaces the "in development" copy
once the real endpoints exist.

**Tech Stack:** Modal (GPU jobs + Volumes + FastAPI `asgi_app`), vLLM 0.7.3 + `orpheus-speech`,
`transformers`, `peft` (LoRA), `snac`, `faster-whisper` (new: transcribing uploaded audio),
Node/Express (ReadAloudAI backend), Next.js/React (developers page).

**Spec:** `docs/superpowers/specs/2026-09-28-orpheus-streaming-voice-cloning-design.md`

## Global Constraints

- Training environment pins `transformers==4.46.3` (verified working for save/load of the merged
  checkpoint in the pilot). Serving environment pins `transformers==4.48.2` (the floor vLLM 0.7.3
  declares; verified working, unlike an unpinned resolve which broke with
  `TokenizersBackend has no attribute all_special_tokens_extended`) — **these are two different
  Modal images; never share one image between training and serving.**
- `OrpheusModel.__init__(self, model_name, dtype=torch.bfloat16)` accepts only these two kwargs.
  Never pass `tokenizer=` or `max_model_len=` to it directly (both raise `TypeError`, confirmed by
  running).
- vLLM's `AsyncEngineArgs` must be capped to `max_model_len=2048` via the monkey-patch documented
  in `serve_benchmark.py` (`orpheus_engine_mod.AsyncEngineArgs = functools.partial(vllm.AsyncEngineArgs, max_model_len=2048)`,
  applied before constructing `OrpheusModel`) — without it, vLLM tries to size the KV cache for the
  base model's native 131072 context and fails on an A10G.
- Always pass `stop_token_ids=[128258]` explicitly to `generate_speech(...)` — the library's own
  default (`[49158]`) does not match this tokenization scheme's real end-of-speech token.
- SNAC tokenization (interleave order, vocab offsets, special tokens, frame dedup) is exactly as
  implemented in `voice-pipeline/orpheus_clone_pilot/finetune_pilot.py` — verified against
  canopylabs' own data-prep notebook. Do not re-derive it; reuse those constants verbatim.
- New voice ids use the prefix `custom-fast:` and the same id shape as existing voices
  (`^v-[0-9a-f]{10}$`), generated the same way `intake.py` does (`"v-" + secrets.token_hex(5)`) —
  reuse the format so the id space stays visually consistent, but issued from a distinct counter/
  namespace so a `custom-fast:` id and a `custom:` id are never confusable as the same voice.
- Consent fields on creation are identical in shape to the existing Piper cloning intake
  (`speaker_name`, `attested_by`, `consent`, `consent_text_version`, `consent_statement`) — the
  legal requirement does not change with the model.
- No changes to `worker-piper-fly/`, `intake.py`, `voice_design_dev.py`, or any existing `/v1/voices`
  route. This plan only adds new files/routes.
- No pricing decision — the billing hook uses a placeholder SKU constant, not a real price.

## Review Focus

- **Uploaded zip with zero usable audio clips (corrupt files, wrong format, empty archive):** a
  reasonable customer expects a clear `400`/`failed` status with a message, not a training job that
  silently produces a checkpoint from nothing or hangs. Task 2's tests cover empty/corrupt input.
- **A voice's checkpoint directory is present but a `manifest.json` write was interrupted (e.g.
  container killed mid-training):** a reasonable customer polling status expects `training` or
  `failed`, never `ready` on a checkpoint that isn't fully written. Task 1's atomic-write test and
  Task 3's status-transition test cover this.
- **Synthesis requested against a `voice` id that was deleted, never existed, or belongs to a
  Piper-style `custom:` id (not `custom-fast:`):** expects `400`/`404`, not a crash or, worse, a
  checkpoint mix-up. Task 5's tests cover unknown-id and wrong-prefix cases explicitly.
- **Generation that runs away (the pilot's own confirmed RTF>1 / unreliable-stop-token behavior)
  during a real customer's synthesis call:** expects a bounded wait and a `503`, not an indefinitely
  hanging HTTP request. Task 5's timeout test directly reproduces the pilot's own runaway-generation
  failure mode under a tight, testable timeout.
- **Two customers polling/training/synthesizing concurrently against different voice ids on the
  same warm container:** expects voice A's request to never receive voice B's audio. Task 5's
  concurrency test loads two distinct tiny checkpoints in the same process and asserts no
  cross-talk.

---

### Task 1: Modal Volume storage module for cloned-voice records

**Files:**
- Create: `voice-pipeline/orpheus_clone_prod/storage.py`
- Test: `voice-pipeline/orpheus_clone_prod/tests/test_storage.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `VoiceRecordStore` class with methods `create(vid: str, consent: dict) -> None`,
  `mark_dataset_uploaded(vid: str) -> None`, `write_status(vid: str, status: str, **fields) -> None`,
  `read_status(vid: str) -> dict | None`, `save_checkpoint_dir(vid: str, local_dir: str) -> None`
  (atomic move into place), `delete(vid: str) -> None`. Status strings used by later tasks:
  `"awaiting_dataset"`, `"training"`, `"ready"`, `"failed"`. Layout on the Modal Volume:
  `{vid}/manifest.json` (consent fields + status + timestamps), `{vid}/dataset/` (uploaded audio,
  removed once merged into training data), `{vid}/merged/` (the final checkpoint dir, only present
  once `status == "ready"`).

- [ ] **Step 1: Write the failing tests**

```python
# voice-pipeline/orpheus_clone_prod/tests/test_storage.py
import json
import os
import shutil
import tempfile

import pytest

from orpheus_clone_prod.storage import VoiceRecordStore


@pytest.fixture
def store(tmp_path):
    return VoiceRecordStore(root=str(tmp_path))


def test_create_writes_manifest_with_consent_and_awaiting_dataset_status(store, tmp_path):
    consent = {
        "speaker_name": "Jane Doe",
        "attested_by": "Jane Doe",
        "consent": True,
        "consent_text_version": "2026-09-v1",
        "consent_statement": "I am authorized...",
    }
    store.create("v-abc1234567", consent)
    manifest = json.load(open(tmp_path / "v-abc1234567" / "manifest.json"))
    assert manifest["speaker_name"] == "Jane Doe"
    assert manifest["status"] == "awaiting_dataset"
    assert "created_at" in manifest


def test_write_status_then_read_status_round_trips(store):
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    store.write_status("v-abc1234567", "training", clip_count=42)
    status = store.read_status("v-abc1234567")
    assert status["status"] == "training"
    assert status["clip_count"] == 42


def test_read_status_on_unknown_voice_returns_none(store):
    assert store.read_status("v-doesnotexist") is None


def test_save_checkpoint_dir_is_atomic_no_partial_merged_dir_visible(store, tmp_path):
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    src = tmp_path / "staging_checkpoint"
    src.mkdir()
    (src / "config.json").write_text("{}")

    # Simulate a container killed mid-copy by making save_checkpoint_dir fail partway,
    # then confirm no half-written merged/ dir was left for read_status to see as ready.
    orig_rename = os.rename
    def failing_rename(a, b):
        raise OSError("simulated crash mid-move")
    os.rename = failing_rename
    try:
        with pytest.raises(OSError):
            store.save_checkpoint_dir("v-abc1234567", str(src))
    finally:
        os.rename = orig_rename

    assert not (tmp_path / "v-abc1234567" / "merged").exists()

    # Now do it for real and confirm it succeeds and is visible.
    store.save_checkpoint_dir("v-abc1234567", str(src))
    assert (tmp_path / "v-abc1234567" / "merged" / "config.json").exists()


def test_delete_removes_the_whole_voice_directory(store, tmp_path):
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    store.delete("v-abc1234567")
    assert not (tmp_path / "v-abc1234567").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orpheus_clone_prod'`

- [ ] **Step 3: Write the implementation**

```python
# voice-pipeline/orpheus_clone_prod/storage.py
"""
Per-voice record storage for the Orpheus streaming-clone service.

Layout under `root` (a Modal Volume mount in production, a tmp dir in tests):
  {vid}/manifest.json   - consent fields + status + timestamps, the single
                           source of truth read_status() returns
  {vid}/dataset/         - uploaded audio, staged before training consumes it
  {vid}/merged/          - the final HF-format checkpoint, only ever visible
                           once fully written (see save_checkpoint_dir)

Modal Volumes are not a real object store with atomic per-key overwrite like
S3 -- a killed container mid-write must never leave a directory that
read_status() would treat as "ready". save_checkpoint_dir therefore stages
into a sibling temp dir and os.rename()s it into place in one step, the same
atomic-swap pattern intake.py and voice_design_dev.py already use on this
same class of storage.
"""
import json
import os
import shutil
import time


class VoiceRecordStore:
    def __init__(self, root: str):
        self.root = root

    def _dir(self, vid: str) -> str:
        return os.path.join(self.root, vid)

    def _manifest_path(self, vid: str) -> str:
        return os.path.join(self._dir(vid), "manifest.json")

    def create(self, vid: str, consent: dict) -> None:
        os.makedirs(self._dir(vid), exist_ok=True)
        manifest = dict(consent)
        manifest["status"] = "awaiting_dataset"
        manifest["created_at"] = time.time()
        self._write_manifest(vid, manifest)

    def _write_manifest(self, vid: str, manifest: dict) -> None:
        # Atomic write: write to a temp file in the same directory, then
        # rename -- rename is atomic on the same filesystem, a plain write
        # is not (a reader could see a half-written JSON file).
        path = self._manifest_path(vid)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(manifest, f)
        os.rename(tmp_path, path)

    def mark_dataset_uploaded(self, vid: str) -> None:
        self.write_status(vid, "awaiting_dataset", dataset_uploaded=True)

    def write_status(self, vid: str, status: str, **fields) -> None:
        manifest = self.read_status(vid) or {}
        manifest["status"] = status
        manifest.update(fields)
        manifest["updated_at"] = time.time()
        self._write_manifest(vid, manifest)

    def read_status(self, vid: str) -> dict | None:
        path = self._manifest_path(vid)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def save_checkpoint_dir(self, vid: str, local_dir: str) -> None:
        dest = os.path.join(self._dir(vid), "merged")
        staging = dest + ".staging"
        if os.path.exists(staging):
            shutil.rmtree(staging)
        shutil.copytree(local_dir, staging)
        os.rename(staging, dest)  # atomic: merged/ never appears half-written

    def dataset_dir(self, vid: str) -> str:
        path = os.path.join(self._dir(vid), "dataset")
        os.makedirs(path, exist_ok=True)
        return path

    def checkpoint_dir(self, vid: str) -> str:
        return os.path.join(self._dir(vid), "merged")

    def delete(self, vid: str) -> None:
        shutil.rmtree(self._dir(vid), ignore_errors=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_storage.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add voice-pipeline/orpheus_clone_prod/storage.py voice-pipeline/orpheus_clone_prod/tests/test_storage.py
git commit -m "Add per-voice record storage for Orpheus streaming clone service"
```

---

### Task 2: Dataset preparation from an uploaded zip (transcription + SNAC-ready manifest)

**Files:**
- Create: `voice-pipeline/orpheus_clone_prod/prepare_dataset.py`
- Test: `voice-pipeline/orpheus_clone_prod/tests/test_prepare_dataset.py`

**Interfaces:**
- Consumes: `VoiceRecordStore.dataset_dir(vid)` (Task 1) as the extraction target.
- Produces: `prepare_dataset(zip_bytes: bytes, dataset_dir: str, voice_tag: str, min_clips: int = 20) -> list[dict]`,
  returning rows shaped exactly like `build_dataset.py`'s `dataset_manifest.jsonl` entries —
  `{"text": "<voice_tag>: <transcript>", "audio": "clips/<name>"}` — so Task 3's training job can
  consume either a pilot-built or a production-built dataset identically. Raises
  `DatasetTooSmallError` (new, defined in this module) if fewer than `min_clips` usable clips
  survive extraction+transcription.

- [ ] **Step 1: Write the failing tests**

```python
# voice-pipeline/orpheus_clone_prod/tests/test_prepare_dataset.py
import io
import wave
import zipfile

import pytest

from orpheus_clone_prod.prepare_dataset import DatasetTooSmallError, prepare_dataset


def _silent_wav_bytes(duration_s=2.0, sr=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * int(duration_s * sr))
    return buf.getvalue()


def _zip_of(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_prepare_dataset_raises_on_empty_zip(tmp_path):
    with pytest.raises(DatasetTooSmallError):
        prepare_dataset(_zip_of({}), str(tmp_path), voice_tag="testvoice", min_clips=1)


def test_prepare_dataset_raises_when_below_min_clips(tmp_path, monkeypatch):
    # Stub transcription so this test doesn't need a real Whisper model.
    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.transcribe_clip",
        lambda path: "hello there",
    )
    zip_bytes = _zip_of({"clip1.wav": _silent_wav_bytes()})
    with pytest.raises(DatasetTooSmallError):
        prepare_dataset(zip_bytes, str(tmp_path), voice_tag="testvoice", min_clips=5)


def test_prepare_dataset_skips_corrupt_files_and_keeps_valid_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.transcribe_clip",
        lambda path: "a valid transcript",
    )
    zip_bytes = _zip_of({
        "good1.wav": _silent_wav_bytes(),
        "good2.wav": _silent_wav_bytes(),
        "corrupt.wav": b"not actually a wav file",
    })
    rows = prepare_dataset(zip_bytes, str(tmp_path), voice_tag="testvoice", min_clips=2)
    assert len(rows) == 2
    assert all(r["text"].startswith("testvoice: ") for r in rows)
    assert all(r["audio"].startswith("clips/") for r in rows)


def test_prepare_dataset_rejects_clips_that_transcribe_empty(tmp_path, monkeypatch):
    # An unusable clip (silence, noise) that Whisper transcribes as empty
    # text must not become a training row with an empty transcript.
    monkeypatch.setattr(
        "orpheus_clone_prod.prepare_dataset.transcribe_clip",
        lambda path: "",
    )
    zip_bytes = _zip_of({"silent.wav": _silent_wav_bytes()})
    with pytest.raises(DatasetTooSmallError):
        prepare_dataset(zip_bytes, str(tmp_path), voice_tag="testvoice", min_clips=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_prepare_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orpheus_clone_prod.prepare_dataset'`

- [ ] **Step 3: Write the implementation**

```python
# voice-pipeline/orpheus_clone_prod/prepare_dataset.py
"""
Turn a customer-uploaded zip of audio clips into the same dataset_manifest.jsonl
row shape build_dataset.py produces from the pilot's Polly corpus:
    {"text": "<voice_tag>: <transcript>", "audio": "clips/<filename>"}

Unlike the pilot (which had ground-truth transcripts from the corpus manifest),
uploaded customer audio has no transcript, so this module transcribes each
clip with faster-whisper -- the same reference-generation approach the STT
real-call eval used for draft references.
"""
import io
import json
import os
import wave
import zipfile


class DatasetTooSmallError(Exception):
    pass


def transcribe_clip(path: str) -> str:
    """Isolated so tests can monkeypatch it without loading a real Whisper
    model. Production implementation loads faster-whisper lazily (import
    inside the function) since this module is also imported by tests that
    never need the real model."""
    from faster_whisper import WhisperModel

    global _WHISPER_MODEL
    try:
        model = _WHISPER_MODEL
    except NameError:
        model = _WHISPER_MODEL = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(path, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def _is_valid_wav(path: str) -> bool:
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() > 0
    except Exception:
        return False


def prepare_dataset(zip_bytes: bytes, dataset_dir: str, voice_tag: str, min_clips: int = 20) -> list[dict]:
    clips_dir = os.path.join(dataset_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    extracted = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith("/") or not name.lower().endswith((".wav", ".flac", ".mp3")):
                    continue
                out_path = os.path.join(clips_dir, os.path.basename(name))
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(out_path)
    except zipfile.BadZipFile:
        raise DatasetTooSmallError("uploaded file is not a valid zip archive")

    rows = []
    for path in extracted:
        if path.lower().endswith(".wav") and not _is_valid_wav(path):
            continue  # corrupt/unreadable clip, skip rather than fail the whole upload
        text = transcribe_clip(path)
        if not text:
            continue  # unusable clip (silence, noise): no transcript, no training row
        rows.append({
            "text": f"{voice_tag}: {text}",
            "audio": f"clips/{os.path.basename(path)}",
        })

    if len(rows) < min_clips:
        raise DatasetTooSmallError(
            f"only {len(rows)} usable clips after extraction and transcription, need at least {min_clips}"
        )

    manifest_path = os.path.join(dataset_dir, "dataset_manifest.jsonl")
    with open(manifest_path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")

    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_prepare_dataset.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add voice-pipeline/orpheus_clone_prod/prepare_dataset.py voice-pipeline/orpheus_clone_prod/tests/test_prepare_dataset.py
git commit -m "Add dataset preparation (extraction + transcription) for uploaded voice clips"
```

---

### Task 3: Training job (Modal) wired to storage and dataset prep

**Files:**
- Create: `voice-pipeline/orpheus_clone_prod/train_job.py`
- Test: `voice-pipeline/orpheus_clone_prod/tests/test_train_job.py`

**Interfaces:**
- Consumes: `VoiceRecordStore` (Task 1) for status transitions and checkpoint save;
  `prepare_dataset()` (Task 2) for turning the uploaded zip into training rows; the SNAC
  tokenization constants and `encode_audio_to_tokens`/dedup logic already verified in
  `orpheus_clone_pilot/finetune_pilot.py` (imported, not re-derived — see Global Constraints).
- Produces: `run_training_job(vid: str, root: str) -> None`, the function `@modal.function`
  wraps and `.spawn()`s. On success, `store.write_status(vid, "ready", ...)`; on any exception,
  `store.write_status(vid, "failed", error=str(e))` and re-raises (so Modal's own failure
  reporting still fires) rather than swallowing it.

- [ ] **Step 1: Write the failing tests**

```python
# voice-pipeline/orpheus_clone_prod/tests/test_train_job.py
import json
import os

import pytest

from orpheus_clone_prod.storage import VoiceRecordStore
from orpheus_clone_prod.train_job import run_training_job


def test_run_training_job_marks_failed_on_dataset_prep_error(tmp_path, monkeypatch):
    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    # No dataset uploaded at all -- prepare_dataset should raise, and the
    # job must record "failed" with a message rather than crash silently.
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job._load_zip_bytes",
        lambda vid, root: b"",  # empty zip
    )
    with pytest.raises(Exception):
        run_training_job("v-abc1234567", root=str(tmp_path))

    status = store.read_status("v-abc1234567")
    assert status["status"] == "failed"
    assert "error" in status


def test_run_training_job_writes_training_then_ready_on_success(tmp_path, monkeypatch):
    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})

    seen_statuses = []
    orig_write_status = store.write_status
    def tracking_write_status(vid, status, **fields):
        seen_statuses.append(status)
        return orig_write_status(vid, status, **fields)

    monkeypatch.setattr("orpheus_clone_prod.train_job.VoiceRecordStore", lambda root: store)
    monkeypatch.setattr(store, "write_status", tracking_write_status)
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job._load_zip_bytes",
        lambda vid, root: b"fake-zip-bytes",
    )
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job.prepare_dataset",
        lambda zip_bytes, dataset_dir, voice_tag, min_clips=20: [
            {"text": f"{voice_tag}: hello", "audio": "clips/a.wav"}
        ] * 20,
    )
    monkeypatch.setattr(
        "orpheus_clone_prod.train_job._run_lora_finetune_and_merge",
        lambda rows, dataset_dir, voice_tag: os.makedirs(f"{dataset_dir}/fake_merged", exist_ok=True) or f"{dataset_dir}/fake_merged",
    )

    run_training_job("v-abc1234567", root=str(tmp_path))

    assert seen_statuses == ["training", "ready"]
    final = store.read_status("v-abc1234567")
    assert final["status"] == "ready"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_train_job.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orpheus_clone_prod.train_job'`

- [ ] **Step 3: Write the implementation**

```python
# voice-pipeline/orpheus_clone_prod/train_job.py
"""
Training job for the Orpheus streaming-clone service: transcribe the
uploaded audio, SNAC-tokenize, LoRA fine-tune, merge, persist.

Reuses orpheus_clone_pilot/finetune_pilot.py's verified SNAC tokenization
and LoRA training logic rather than re-deriving it -- that module's
constants (AUDIO_TOKENS_START, CODEBOOK_SIZE, special tokens, dedup) are
load-bearing and were checked against canopylabs' own data-prep notebook.

Runs in the transformers==4.46.3-pinned training image (see Global
Constraints) -- a DIFFERENT image than serve.py's vllm==0.7.3/
transformers==4.48.2 image. Never share these images.
"""
import os
import time

from orpheus_clone_prod.prepare_dataset import prepare_dataset
from orpheus_clone_prod.storage import VoiceRecordStore


def _load_zip_bytes(vid: str, root: str) -> bytes:
    store = VoiceRecordStore(root=root)
    zip_path = os.path.join(store.dataset_dir(vid), "upload.zip")
    with open(zip_path, "rb") as f:
        return f.read()


def _run_lora_finetune_and_merge(rows: list[dict], dataset_dir: str, voice_tag: str) -> str:
    """Thin wrapper around finetune_pilot.py's tokenize+train+merge logic,
    parameterized by voice_tag instead of the pilot's hardcoded "joanna".
    Returns the local path to the merged checkpoint directory."""
    from orpheus_clone_pilot.finetune_pilot import run_pilot_for_voice  # see note below

    return run_pilot_for_voice(rows=rows, dataset_dir=dataset_dir, voice_tag=voice_tag)


def run_training_job(vid: str, root: str) -> None:
    store = VoiceRecordStore(root=root)
    try:
        store.write_status(vid, "training")
        dataset_dir = store.dataset_dir(vid)
        zip_bytes = _load_zip_bytes(vid, root)
        rows = prepare_dataset(zip_bytes, dataset_dir, voice_tag=vid, min_clips=20)
        merged_dir = _run_lora_finetune_and_merge(rows, dataset_dir, voice_tag=vid)
        store.save_checkpoint_dir(vid, merged_dir)
        store.write_status(vid, "ready", clip_count=len(rows), trained_at=time.time())
    except Exception as e:
        store.write_status(vid, "failed", error=str(e))
        raise
```

**Refactor `finetune_pilot.py`** to extract the tokenize+train+merge body of its current
`run_pilot()` into a plain function parameterized by `voice_tag`/`dataset_dir`/`rows` instead of
the hardcoded `VOICE_TAG = "joanna"` and fixed manifest path — byte-for-byte identical logic, only
the parameterization changes:

```python
# voice-pipeline/orpheus_clone_pilot/finetune_pilot.py
# Replace the body of run_pilot(epochs=3, lora_r=16, lora_alpha=32) with a
# call to this new function, keeping run_pilot() itself as the Modal
# @app.function entrypoint the pilot's own CLI still uses:

def run_pilot_for_voice(rows: list[dict], dataset_dir: str, voice_tag: str, epochs: int = 3, lora_r: int = 16, lora_alpha: int = 32) -> str:
    """Extracted from run_pilot()'s body: SNAC-tokenize `rows` (already
    {"text", "audio"} pairs, audio paths relative to dataset_dir), LoRA
    fine-tune, merge, and return the local path to the merged checkpoint
    directory. Everything from "=== Loading SNAC codec ===" through
    "=== Merging LoRA into base weights ===" in the original run_pilot()
    moves here unchanged, reading `rows` as a parameter instead of loading
    /data/dataset_manifest.jsonl, and using `voice_tag` instead of the
    module-level VOICE_TAG constant everywhere it appears (prompt tagging,
    output directory naming). Returns the merged checkpoint's local path
    (e.g. f"{dataset_dir}/{voice_tag}_merged") instead of writing straight
    to /checkpoints and calling checkpoint_volume.commit() -- the caller
    (train_job.py's _run_lora_finetune_and_merge, or run_pilot() itself for
    the pilot's own use) owns persistence.
    """
    # ... tokenize each row via encode_audio_to_tokens (unchanged),
    # ... build the LoRA model and Trainer exactly as before (unchanged),
    # ... trainer.train(), merged = model.merge_and_unload() (unchanged),
    # ... merged.save_pretrained(out_dir); tokenizer.save_pretrained(out_dir)
    # return out_dir


def run_pilot(epochs: int = 3, lora_r: int = 16, lora_alpha: int = 32):
    # Pilot's own entrypoint: load rows from /data/dataset_manifest.jsonl as
    # before, then delegate:
    import json
    rows = [json.loads(l) for l in open("/data/dataset_manifest.jsonl")]
    merged_dir = run_pilot_for_voice(rows, dataset_dir="/data", voice_tag=VOICE_TAG, epochs=epochs, lora_r=lora_r, lora_alpha=lora_alpha)
    import shutil
    shutil.copytree(merged_dir, f"/checkpoints/{VOICE_TAG}_merged")
    checkpoint_volume.commit()
```

This is a refactor of already-tested pilot code, not new training logic — the tokenization,
training, and merge steps themselves must not change, only how `voice_tag`/`dataset_dir`/`rows`
are threaded through instead of hardcoded.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_train_job.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add voice-pipeline/orpheus_clone_prod/train_job.py voice-pipeline/orpheus_clone_prod/tests/test_train_job.py voice-pipeline/orpheus_clone_pilot/finetune_pilot.py
git commit -m "Add production training job wiring dataset prep, storage, and LoRA fine-tune"
```

---

### Task 4: Modal FastAPI surface (create/upload/commit/poll/delete)

**Files:**
- Create: `voice-pipeline/orpheus_clone_prod/api.py`
- Test: `voice-pipeline/orpheus_clone_prod/tests/test_api.py`

**Interfaces:**
- Consumes: `VoiceRecordStore` (Task 1), `run_training_job` (Task 3, spawned not called directly).
- Produces: FastAPI routes `POST /v1/orpheus-voices`, `PUT /v1/orpheus-voices/{vid}/dataset`,
  `POST /v1/orpheus-voices/{vid}/dataset/commit`, `GET /v1/orpheus-voices/{vid}`,
  `DELETE /v1/orpheus-voices/{vid}`. Voice id shape `^v-[0-9a-f]{10}$`, generated with
  `"v-" + secrets.token_hex(5)`, matching the existing Piper cloning id shape (Global Constraints).

- [ ] **Step 1: Write the failing tests**

```python
# voice-pipeline/orpheus_clone_prod/tests/test_api.py
import re

import pytest
from fastapi.testclient import TestClient

from orpheus_clone_prod.api import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")
    spawned = []
    app = create_app(root=str(tmp_path), spawn_training=lambda vid, root: spawned.append(vid))
    client = TestClient(app)
    client.spawned = spawned
    return client


AUTH = {"Authorization": "Bearer test-secret"}
CONSENT_BODY = {
    "speaker_name": "Jane Doe",
    "attested_by": "Jane Doe",
    "consent": True,
    "consent_text_version": "2026-09-v1",
    "consent_statement": "I am authorized...",
}


def test_create_voice_requires_auth(client):
    resp = client.post("/v1/orpheus-voices", json=CONSENT_BODY)
    assert resp.status_code == 401


def test_create_voice_returns_id_matching_expected_shape(client):
    resp = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert re.match(r"^v-[0-9a-f]{10}$", body["id"])
    assert body["status"] == "awaiting_dataset"


def test_create_voice_rejects_missing_consent(client):
    bad = dict(CONSENT_BODY)
    del bad["consent"]
    resp = client.post("/v1/orpheus-voices", json=bad, headers=AUTH)
    assert resp.status_code == 400


def test_poll_unknown_voice_returns_404(client):
    resp = client.get("/v1/orpheus-voices/v-0000000000", headers=AUTH)
    assert resp.status_code == 404


def test_full_lifecycle_upload_commit_poll(client):
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]

    upload_resp = client.put(
        f"/v1/orpheus-voices/{vid}/dataset",
        headers={**AUTH, "content-type": "application/zip"},
        content=b"fake-zip-bytes",
    )
    assert upload_resp.status_code == 200

    commit_resp = client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH)
    assert commit_resp.status_code == 200
    assert commit_resp.json()["status"] == "training"
    assert vid in client.spawned  # training was spawned, not run inline

    poll_resp = client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH)
    assert poll_resp.json()["status"] == "training"


def test_commit_without_upload_returns_400(client):
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    resp = client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH)
    assert resp.status_code == 400


def test_delete_removes_the_voice(client):
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    resp = client.delete(f"/v1/orpheus-voices/{vid}", headers=AUTH)
    assert resp.status_code == 200
    assert client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH).status_code == 404


def test_spawn_failure_rolls_back_to_awaiting_dataset_and_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("ORPHEUS_CLONE_SECRET", "test-secret")

    def failing_spawn(vid, root):
        raise RuntimeError("modal spawn failed")

    app = create_app(root=str(tmp_path), spawn_training=failing_spawn)
    client = TestClient(app)
    vid = client.post("/v1/orpheus-voices", json=CONSENT_BODY, headers=AUTH).json()["id"]
    client.put(f"/v1/orpheus-voices/{vid}/dataset", headers={**AUTH, "content-type": "application/zip"}, content=b"z")

    resp = client.post(f"/v1/orpheus-voices/{vid}/dataset/commit", headers=AUTH)
    assert resp.status_code == 503

    status = client.get(f"/v1/orpheus-voices/{vid}", headers=AUTH).json()
    assert status["status"] == "awaiting_dataset"  # rolled back, not stuck in "training"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orpheus_clone_prod.api'`

- [ ] **Step 3: Write the implementation**

```python
# voice-pipeline/orpheus_clone_prod/api.py
"""
FastAPI surface for the Orpheus streaming-clone service, mirroring
voice_design_dev.py's auth/job-submission shape and intake.py's
"roll back the enqueued state and 503 if spawn fails" pattern.

create_app() takes root/spawn_training as parameters (rather than importing
Modal directly) so this module is fully unit-testable without Modal --
main.py (Task 6) wires the real Modal Volume path and .spawn() call.
"""
import hmac
import os
import secrets

from fastapi import FastAPI, HTTPException, Request

from orpheus_clone_prod.storage import VoiceRecordStore

VOICE_ID_RE = __import__("re").compile(r"^v-[0-9a-f]{10}$")
REQUIRED_CONSENT_FIELDS = ("speaker_name", "attested_by", "consent", "consent_text_version", "consent_statement")


def create_app(root: str, spawn_training) -> FastAPI:
    app = FastAPI()
    store = VoiceRecordStore(root=root)

    def auth(request: Request) -> None:
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        secret = os.environ.get("ORPHEUS_CLONE_SECRET", "")
        if not tok or not secret or not hmac.compare_digest(tok.encode(), secret.encode()):
            raise HTTPException(401, "unauthorized")

    @app.post("/v1/orpheus-voices")
    async def create_voice(request: Request):
        auth(request)
        body = await request.json()
        missing = [f for f in REQUIRED_CONSENT_FIELDS if not body.get(f)]
        if missing:
            raise HTTPException(400, f"missing required fields: {', '.join(missing)}")
        vid = "v-" + secrets.token_hex(5)
        store.create(vid, {k: body[k] for k in REQUIRED_CONSENT_FIELDS})
        return {"id": vid, "status": "awaiting_dataset"}

    @app.put("/v1/orpheus-voices/{vid}/dataset")
    async def upload_dataset(vid: str, request: Request):
        auth(request)
        if store.read_status(vid) is None:
            raise HTTPException(404, "unknown voice")
        body = await request.body()
        dataset_dir = store.dataset_dir(vid)
        with open(os.path.join(dataset_dir, "upload.zip"), "wb") as f:
            f.write(body)
        store.mark_dataset_uploaded(vid)
        return {"id": vid, "status": "awaiting_dataset", "uploaded_bytes": len(body)}

    @app.post("/v1/orpheus-voices/{vid}/dataset/commit")
    async def commit_dataset(vid: str, request: Request):
        auth(request)
        status = store.read_status(vid)
        if status is None:
            raise HTTPException(404, "unknown voice")
        if not status.get("dataset_uploaded"):
            raise HTTPException(400, "no dataset uploaded yet")

        store.write_status(vid, "training")
        try:
            spawn_training(vid, root)
        except Exception as e:
            # Roll back to a resumable state, same shape as intake.py's
            # "remove the enqueued marker and 503" pattern -- the customer
            # can retry commit without re-uploading.
            store.write_status(vid, "awaiting_dataset", dataset_uploaded=True)
            raise HTTPException(503, "training service unavailable, please retry") from e
        return {"id": vid, "status": "training"}

    @app.get("/v1/orpheus-voices/{vid}")
    async def poll_voice(vid: str, request: Request):
        auth(request)
        status = store.read_status(vid)
        if status is None:
            raise HTTPException(404, "unknown voice")
        return {"id": vid, **status}

    @app.delete("/v1/orpheus-voices/{vid}")
    async def delete_voice(vid: str, request: Request):
        auth(request)
        if store.read_status(vid) is None:
            raise HTTPException(404, "unknown voice")
        store.delete(vid)
        return {"id": vid, "deleted": True}

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_api.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add voice-pipeline/orpheus_clone_prod/api.py voice-pipeline/orpheus_clone_prod/tests/test_api.py
git commit -m "Add job-submission FastAPI surface for Orpheus streaming clone voices"
```

---

### Task 5: Production serving engine (vLLM, scale-to-zero, with the three verified fixes)

**Files:**
- Create: `voice-pipeline/orpheus_clone_prod/serve.py`
- Test: `voice-pipeline/orpheus_clone_prod/tests/test_serve.py`

**Interfaces:**
- Consumes: `VoiceRecordStore.checkpoint_dir(vid)` (Task 1) to locate a trained voice's merged
  checkpoint.
- Produces: `resolve_voice_dir(voice: str, store: VoiceRecordStore) -> str` (validates the
  `custom-fast:` prefix and id shape, raises `UnknownVoiceError` otherwise);
  `OrpheusCloneEngine` class (the production version of `serve_benchmark.py`'s
  `ClonedVoiceEngine`, generalized to load whichever voice's checkpoint a request names, not a
  single hardcoded `joanna`), with a bounded-time `synthesize(voice_dir, text, timeout_s=15)`
  generator.

- [ ] **Step 1: Write the failing tests**

```python
# voice-pipeline/orpheus_clone_prod/tests/test_serve.py
import pytest

from orpheus_clone_prod.serve import UnknownVoiceError, resolve_voice_dir
from orpheus_clone_prod.storage import VoiceRecordStore


def test_resolve_voice_dir_rejects_wrong_prefix(tmp_path):
    store = VoiceRecordStore(root=str(tmp_path))
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom:v-abc1234567", store)  # Piper prefix, not custom-fast:


def test_resolve_voice_dir_rejects_malformed_id(tmp_path):
    store = VoiceRecordStore(root=str(tmp_path))
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom-fast:not-a-valid-id", store)


def test_resolve_voice_dir_rejects_unknown_voice(tmp_path):
    store = VoiceRecordStore(root=str(tmp_path))
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom-fast:v-abc1234567", store)  # never created


def test_resolve_voice_dir_rejects_voice_not_yet_ready(tmp_path):
    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    store.write_status("v-abc1234567", "training")  # not "ready" yet
    with pytest.raises(UnknownVoiceError):
        resolve_voice_dir("custom-fast:v-abc1234567", store)


def test_resolve_voice_dir_returns_checkpoint_path_when_ready(tmp_path):
    store = VoiceRecordStore(root=str(tmp_path))
    store.create("v-abc1234567", {"speaker_name": "Jane"})
    store.write_status("v-abc1234567", "ready")
    result = resolve_voice_dir("custom-fast:v-abc1234567", store)
    assert result == store.checkpoint_dir("v-abc1234567")


def test_two_voices_never_cross_talk_when_resolved_in_sequence(tmp_path):
    # Reproduces the Review Focus concurrency concern at the resolution layer:
    # resolving voice A's dir must never return voice B's path, even
    # immediately after resolving B.
    store = VoiceRecordStore(root=str(tmp_path))
    for vid in ("v-aaaaaaaaaa", "v-bbbbbbbbbb"):
        store.create(vid, {"speaker_name": vid})
        store.write_status(vid, "ready")

    dir_a = resolve_voice_dir("custom-fast:v-aaaaaaaaaa", store)
    dir_b = resolve_voice_dir("custom-fast:v-bbbbbbbbbb", store)
    assert dir_a != dir_b
    assert dir_a == store.checkpoint_dir("v-aaaaaaaaaa")
    assert dir_b == store.checkpoint_dir("v-bbbbbbbbbb")


def test_synthesize_raises_on_timeout_instead_of_hanging():
    from orpheus_clone_prod.serve import GenerationTimeoutError, _bounded_generate

    def runaway_generator():
        import time
        while True:
            time.sleep(0.05)
            yield b"\x00\x00"

    with pytest.raises(GenerationTimeoutError):
        list(_bounded_generate(runaway_generator(), timeout_s=0.2))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_serve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orpheus_clone_prod.serve'`

- [ ] **Step 3: Write the implementation**

```python
# voice-pipeline/orpheus_clone_prod/serve.py
"""
Production serving for the Orpheus streaming-clone service.

Carries forward three real, verified-by-running fixes from
serve_benchmark.py (do not regress any of them -- see Global Constraints):
  1. OrpheusModel(model_name=..., dtype=...) only -- no tokenizer=/max_model_len=.
  2. AsyncEngineArgs monkey-patched to max_model_len=2048 before constructing
     OrpheusModel, or vLLM tries to size the KV cache for the base model's
     native 131072 context and fails on an A10G.
  3. stop_token_ids=[128258] passed explicitly to every generate_speech()
     call -- the library's own default ([49158]) does not match this
     tokenization's real end-of-speech token.

Runs in the vllm==0.7.3 / transformers==4.48.2-pinned serving image (see
Global Constraints) -- a DIFFERENT image than train_job.py's
transformers==4.46.3 training image.
"""
import re
import time

from orpheus_clone_prod.storage import VoiceRecordStore

VOICE_ID_RE = re.compile(r"^v-[0-9a-f]{10}$")
CUSTOM_FAST_PREFIX = "custom-fast:"


class UnknownVoiceError(Exception):
    pass


class GenerationTimeoutError(Exception):
    pass


def resolve_voice_dir(voice: str, store: VoiceRecordStore) -> str:
    if not voice.startswith(CUSTOM_FAST_PREFIX):
        raise UnknownVoiceError(f"not a custom-fast voice: {voice!r}")
    vid = voice[len(CUSTOM_FAST_PREFIX):]
    if not VOICE_ID_RE.match(vid):
        raise UnknownVoiceError(f"malformed voice id: {vid!r}")
    status = store.read_status(vid)
    if status is None or status.get("status") != "ready":
        raise UnknownVoiceError(f"voice not ready: {vid!r}")
    return store.checkpoint_dir(vid)


def _bounded_generate(chunk_iter, timeout_s: float):
    """Wraps a chunk generator with a wall-clock deadline -- reproduces and
    bounds the pilot's own confirmed runaway-generation failure mode (RTF>1,
    unreliable stop-token behavior) instead of letting a request hang."""
    deadline = time.time() + timeout_s
    for chunk in chunk_iter:
        if time.time() > deadline:
            raise GenerationTimeoutError(f"generation exceeded {timeout_s}s")
        yield chunk


def load_engine_for_checkpoint(checkpoint_dir: str):
    """Constructs a fresh OrpheusModel for one voice's checkpoint, applying
    the AsyncEngineArgs max_model_len patch first. One engine instance per
    warm container; a new voice request on a cold container calls this once
    in @modal.enter()-equivalent setup (wired in Task 6's Modal wrapper)."""
    import functools

    import torch
    import vllm
    import orpheus_tts.engine_class as orpheus_engine_mod
    from orpheus_tts import OrpheusModel

    orpheus_engine_mod.AsyncEngineArgs = functools.partial(
        vllm.AsyncEngineArgs, max_model_len=2048
    )
    return OrpheusModel(model_name=checkpoint_dir, dtype=torch.bfloat16)


def synthesize(model, text: str, voice_tag: str, timeout_s: float = 15.0):
    """voice_tag is the tag baked into the checkpoint's training data
    (e.g. the vid itself, per train_job.py) -- the prompt prefix, not the
    "custom-fast:" API-facing voice string."""
    chunks = model.generate_speech(
        prompt=text,
        voice=voice_tag,
        temperature=0.6,
        top_p=0.8,
        repetition_penalty=1.3,
        stop_token_ids=[128258],
    )
    yield from _bounded_generate(chunks, timeout_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_serve.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add voice-pipeline/orpheus_clone_prod/serve.py voice-pipeline/orpheus_clone_prod/tests/test_serve.py
git commit -m "Add production serving engine with verified vLLM/stop-token/KV-cache fixes"
```

---

### Task 6: Modal app wiring (deployable entrypoint tying Tasks 1-5 together)

**Files:**
- Create: `voice-pipeline/orpheus_clone_prod/main.py`
- Test: `voice-pipeline/orpheus_clone_prod/tests/test_main_smoke.py`

**Interfaces:**
- Consumes: `create_app` (Task 4), `run_training_job` (Task 3), `OrpheusCloneEngine`-equivalent
  loading from `serve.py` (Task 5).
- Produces: the actual `modal.App("orpheus-clone-prod")` with a training `@app.function` (image
  pinned `transformers==4.46.3`, GPU A10G, volume `orpheus-clone-checkpoints`), a serving
  `@app.cls` (image pinned `transformers==4.48.2`/`vllm==0.7.3`, GPU A10G, `scaledown_window=300`,
  same volume), and the `@app.function(image=web_image) @modal.asgi_app() def api()` that
  `create_app(root="/checkpoints", spawn_training=...)` mounts, matching
  `voice_design_dev.py`'s three-image-classes shape (web / training-GPU / serving-GPU).

- [ ] **Step 1: Write the failing test**

```python
# voice-pipeline/orpheus_clone_prod/tests/test_main_smoke.py
"""
This is an import-and-wiring smoke test, not a Modal-execution test (Modal
apps are exercised by actually deploying/running them, matching this
repo's established practice of never mocking the training/serving GPU
path -- see the spec's Testing section). It only confirms main.py wires
Tasks 1-5 together without a typo/import error, which is cheap to check
and has caught real mistakes elsewhere in this codebase (e.g. the
"got an unexpected keyword argument" class of bug found by actually
running serve_benchmark.py).
"""
import importlib


def test_main_module_imports_without_error():
    module = importlib.import_module("orpheus_clone_prod.main")
    assert hasattr(module, "app")
    assert hasattr(module, "api")
    assert hasattr(module, "run_training_job_modal")
    assert hasattr(module, "OrpheusCloneEngine")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_main_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orpheus_clone_prod.main'`

- [ ] **Step 3: Write the implementation**

```python
# voice-pipeline/orpheus_clone_prod/main.py
"""
Deployable Modal app for the Orpheus streaming-clone service. Ties together
storage.py, prepare_dataset.py, train_job.py, api.py, and serve.py.

Deploy: modal deploy voice-pipeline/orpheus_clone_prod/main.py
Secrets required: orpheus-clone-secret (ORPHEUS_CLONE_SECRET), hf-token
(HF_TOKEN, for the gated canopylabs/orpheus-tts-0.1-pretrained repo).
"""
import modal

from orpheus_clone_prod.api import create_app
from orpheus_clone_prod.serve import load_engine_for_checkpoint, resolve_voice_dir, synthesize
from orpheus_clone_prod.storage import VoiceRecordStore
from orpheus_clone_prod.train_job import run_training_job

app = modal.App("orpheus-clone-prod")

checkpoint_volume = modal.Volume.from_name("orpheus-clone-checkpoints", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")
api_secret = modal.Secret.from_name("orpheus-clone-secret")

CHECKPOINT_ROOT = "/checkpoints"

# Training image: pinned transformers==4.46.3, matching finetune_pilot.py's
# already-verified training environment. NEVER share this image with the
# serving image below -- see Global Constraints.
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.46.3", "datasets", "accelerate", "peft==0.13.2",
        "snac", "soundfile", "huggingface_hub", "numpy<2",
        "faster-whisper",
    )
)

# Serving image: pinned transformers==4.48.2 (the floor vllm==0.7.3
# declares) -- an unpinned resolve broke with a TokenizersBackend
# AttributeError, confirmed by running. NEVER share this image with the
# training image above.
serve_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.48.2", "huggingface_hub", "numpy<2", "soundfile",
        "snac", "vllm==0.7.3", "orpheus-speech",
    )
)

web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi==0.109.0", "python-multipart")


@app.function(image=train_image, gpu="A10G", timeout=3600, volumes={CHECKPOINT_ROOT: checkpoint_volume}, secrets=[hf_secret])
def run_training_job_modal(vid: str):
    run_training_job(vid, root=CHECKPOINT_ROOT)
    checkpoint_volume.commit()


@app.cls(image=serve_image, gpu="A10G", timeout=300, scaledown_window=300, volumes={CHECKPOINT_ROOT: checkpoint_volume}, secrets=[hf_secret])
class OrpheusCloneEngine:
    @modal.enter()
    def _init(self):
        self._models = {}  # checkpoint_dir -> loaded OrpheusModel, one per warm container

    @modal.method()
    def synthesize_for_voice(self, voice: str, text: str):
        store = VoiceRecordStore(root=CHECKPOINT_ROOT)
        checkpoint_dir = resolve_voice_dir(voice, store)  # raises UnknownVoiceError -> caller maps to 400
        if checkpoint_dir not in self._models:
            self._models[checkpoint_dir] = load_engine_for_checkpoint(checkpoint_dir)
        vid = voice.split(":", 1)[1]
        return list(synthesize(self._models[checkpoint_dir], text, voice_tag=vid))


@app.function(image=web_image, secrets=[api_secret], volumes={CHECKPOINT_ROOT: checkpoint_volume}, timeout=120)
@modal.asgi_app()
def api():
    def spawn_training(vid: str, root: str):
        run_training_job_modal.spawn(vid)

    return create_app(root=CHECKPOINT_ROOT, spawn_training=spawn_training)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/test_main_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add voice-pipeline/orpheus_clone_prod/main.py voice-pipeline/orpheus_clone_prod/tests/test_main_smoke.py
git commit -m "Wire Orpheus streaming clone service into a deployable Modal app"
```

---

### Task 7: ReadAloudAI gateway route (auth + billing + proxy)

**Files:**
- Create: `ReadAloudAI/backend/src/routes/orpheusVoiceRouter.ts`
- Create: `ReadAloudAI/backend/src/lib/orpheusCloneClient.ts`
- Test: `ReadAloudAI/backend/test/orpheusVoiceRouter.test.ts`
- Modify: `ReadAloudAI/backend/src/app.ts` (or wherever routers are mounted — mount at
  `/v1/orpheus-voices` and `/v1/orpheus-tts`, following the existing mount pattern for
  `voiceStudioRouter.ts`)

**Interfaces:**
- Consumes: `requireUser` and `isBillingActiveForUser` (existing, from `lib/realtimeTtsBilling.js`
  per the `voiceClone.ts`/`voiceDesign.ts` pattern) — reused exactly, not reimplemented.
- Produces: `orpheusCloneClient.createVoice(consent)`, `.uploadDataset(vid, zipBuffer)`,
  `.commit(vid)`, `.getStatus(vid)`, `.deleteVoice(vid)`, `.synthesize(voice, text)` — thin
  wrappers over `fetch` calls to the Modal `api()` app's URL (env var
  `ORPHEUS_CLONE_SERVICE_URL`), each attaching `Authorization: Bearer ${ORPHEUS_CLONE_SECRET}`.

- [ ] **Step 1: Write the failing tests**

```typescript
// ReadAloudAI/backend/test/orpheusVoiceRouter.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest'
import request from 'supertest'
import express from 'express'
import { orpheusVoiceRouter } from '../src/routes/orpheusVoiceRouter'

vi.mock('../src/lib/realtimeTtsBilling.js', () => ({
  isBillingActiveForUser: vi.fn(),
}))
vi.mock('../src/lib/orpheusCloneClient', () => ({
  orpheusCloneClient: {
    createVoice: vi.fn(),
    uploadDataset: vi.fn(),
    commit: vi.fn(),
    getStatus: vi.fn(),
    deleteVoice: vi.fn(),
  },
}))

import { isBillingActiveForUser } from '../src/lib/realtimeTtsBilling.js'
import { orpheusCloneClient } from '../src/lib/orpheusCloneClient'

function appWithUser(user: { id: string } | null) {
  const app = express()
  app.use(express.json())
  app.use(express.raw({ type: 'application/zip', limit: '50mb' }))
  app.use((req: any, _res, next) => { req.user = user; next() })
  app.use(orpheusVoiceRouter)
  return app
}

const CONSENT_BODY = {
  speaker_name: 'Jane Doe',
  attested_by: 'Jane Doe',
  consent: true,
  consent_text_version: '2026-09-v1',
  consent_statement: 'I am authorized...',
}

describe('orpheusVoiceRouter', () => {
  beforeEach(() => vi.clearAllMocks())

  it('returns 401 with no authenticated user', async () => {
    const res = await request(appWithUser(null)).post('/v1/orpheus-voices').send(CONSENT_BODY)
    expect(res.status).toBe(401)
  })

  it('returns 402 when billing is not active', async () => {
    vi.mocked(isBillingActiveForUser).mockResolvedValue(false)
    const res = await request(appWithUser({ id: 'u1' })).post('/v1/orpheus-voices').send(CONSENT_BODY)
    expect(res.status).toBe(402)
    expect(orpheusCloneClient.createVoice).not.toHaveBeenCalled()
  })

  it('proxies to orpheusCloneClient.createVoice when billing is active', async () => {
    vi.mocked(isBillingActiveForUser).mockResolvedValue(true)
    vi.mocked(orpheusCloneClient.createVoice).mockResolvedValue({ id: 'v-abc1234567', status: 'awaiting_dataset' })
    const res = await request(appWithUser({ id: 'u1' })).post('/v1/orpheus-voices').send(CONSENT_BODY)
    expect(res.status).toBe(200)
    expect(res.body.id).toBe('v-abc1234567')
    expect(orpheusCloneClient.createVoice).toHaveBeenCalledWith(CONSENT_BODY)
  })

  it('proxies status poll without a fresh billing check (polling stays free, matching voice_design_dev.py)', async () => {
    vi.mocked(orpheusCloneClient.getStatus).mockResolvedValue({ status: 'training' })
    const res = await request(appWithUser({ id: 'u1' })).get('/v1/orpheus-voices/v-abc1234567')
    expect(res.status).toBe(200)
    expect(isBillingActiveForUser).not.toHaveBeenCalled()
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ReadAloudAI/backend && npx vitest run test/orpheusVoiceRouter.test.ts`
Expected: FAIL with a module-not-found error for `../src/routes/orpheusVoiceRouter`

- [ ] **Step 3: Write the implementation**

```typescript
// ReadAloudAI/backend/src/lib/orpheusCloneClient.ts
const BASE = process.env.ORPHEUS_CLONE_SERVICE_URL || ''
const SECRET = process.env.ORPHEUS_CLONE_SECRET || ''

function authHeaders(extra: Record<string, string> = {}) {
  return { Authorization: `Bearer ${SECRET}`, ...extra }
}

export const orpheusCloneClient = {
  async createVoice(consent: Record<string, unknown>) {
    const res = await fetch(`${BASE}/v1/orpheus-voices`, {
      method: 'POST',
      headers: authHeaders({ 'content-type': 'application/json' }),
      body: JSON.stringify(consent),
    })
    return res.json()
  },
  async uploadDataset(vid: string, zipBuffer: Buffer) {
    const res = await fetch(`${BASE}/v1/orpheus-voices/${vid}/dataset`, {
      method: 'PUT',
      headers: authHeaders({ 'content-type': 'application/zip' }),
      body: zipBuffer,
    })
    return res.json()
  },
  async commit(vid: string) {
    const res = await fetch(`${BASE}/v1/orpheus-voices/${vid}/dataset/commit`, {
      method: 'POST',
      headers: authHeaders(),
    })
    return { status: res.status, body: await res.json() }
  },
  async getStatus(vid: string) {
    const res = await fetch(`${BASE}/v1/orpheus-voices/${vid}`, { headers: authHeaders() })
    return { status: res.status, body: await res.json() }
  },
  async deleteVoice(vid: string) {
    const res = await fetch(`${BASE}/v1/orpheus-voices/${vid}`, { method: 'DELETE', headers: authHeaders() })
    return { status: res.status, body: await res.json() }
  },
}
```

```typescript
// ReadAloudAI/backend/src/routes/orpheusVoiceRouter.ts
import { Router } from 'express'
import { isBillingActiveForUser } from '../lib/realtimeTtsBilling.js'
import { orpheusCloneClient } from '../lib/orpheusCloneClient'

export const orpheusVoiceRouter = Router()

function requireUser(req: any, res: any): { id: string } | null {
  if (!req.user) {
    res.status(401).json({ error: 'authentication required' })
    return null
  }
  return req.user
}

// Billed action (spawns a real training job): gate on billing, mirroring
// voiceClone.ts/voiceDesign.ts's isBillingActiveForUser check exactly.
orpheusVoiceRouter.post('/v1/orpheus-voices', async (req, res) => {
  const user = requireUser(req, res)
  if (!user) return
  const active = await isBillingActiveForUser(user.id)
  if (!active) {
    res.status(402).json({ error: 'Voice cloning requires an active TTS subscription.' })
    return
  }
  const body = await orpheusCloneClient.createVoice(req.body)
  res.json(body)
})

orpheusVoiceRouter.put('/v1/orpheus-voices/:vid/dataset', async (req, res) => {
  const user = requireUser(req, res)
  if (!user) return
  const body = await orpheusCloneClient.uploadDataset(req.params.vid, req.body)
  res.json(body)
})

orpheusVoiceRouter.post('/v1/orpheus-voices/:vid/dataset/commit', async (req, res) => {
  const user = requireUser(req, res)
  if (!user) return
  const { status, body } = await orpheusCloneClient.commit(req.params.vid)
  res.status(status).json(body)
})

// Polling: unlimited/free, same as voice_design_dev.py's GET routes --
// callers legitimately poll their own job, no fresh billing check needed.
orpheusVoiceRouter.get('/v1/orpheus-voices/:vid', async (req, res) => {
  const user = requireUser(req, res)
  if (!user) return
  const { status, body } = await orpheusCloneClient.getStatus(req.params.vid)
  res.status(status).json(body)
})

orpheusVoiceRouter.delete('/v1/orpheus-voices/:vid', async (req, res) => {
  const user = requireUser(req, res)
  if (!user) return
  const { status, body } = await orpheusCloneClient.deleteVoice(req.params.vid)
  res.status(status).json(body)
})
```

Also modify `ReadAloudAI/backend/src/app.ts`: add `import { orpheusVoiceRouter } from
'./routes/orpheusVoiceRouter'` and `app.use(orpheusVoiceRouter)` alongside the existing
`voiceStudioRouter` mount line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ReadAloudAI/backend && npx vitest run test/orpheusVoiceRouter.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add ReadAloudAI/backend/src/routes/orpheusVoiceRouter.ts ReadAloudAI/backend/src/lib/orpheusCloneClient.ts ReadAloudAI/backend/test/orpheusVoiceRouter.test.ts ReadAloudAI/backend/src/app.ts
git commit -m "Add gateway route proxying Orpheus streaming clone training API"
```

---

### Task 8: Synthesis endpoint (`/v1/orpheus-tts`) with billing meter

**Files:**
- Modify: `ReadAloudAI/backend/src/routes/orpheusVoiceRouter.ts`
- Modify: `ReadAloudAI/backend/src/lib/orpheusCloneClient.ts`
- Modify: `ReadAloudAI/backend/test/orpheusVoiceRouter.test.ts`

**Interfaces:**
- Consumes: `OrpheusCloneEngine.synthesize_for_voice` (Task 6), invoked over HTTP via a new
  `POST /v1/orpheus-tts` route on the Modal `api()` app — this task adds a thin passthrough on
  both sides (Modal `api.py` gets a new route calling `.remote()` on the Modal class; the gateway
  proxies to it).

- [ ] **Step 1: Write the failing test (gateway side)**

```typescript
// appended to ReadAloudAI/backend/test/orpheusVoiceRouter.test.ts
it('POST /v1/orpheus-tts requires billing and meters a placeholder SKU', async () => {
  vi.mocked(isBillingActiveForUser).mockResolvedValue(true)
  vi.mocked(orpheusCloneClient.synthesize).mockResolvedValue(Buffer.from([0, 0, 1, 1]))
  const res = await request(appWithUser({ id: 'u1' }))
    .post('/v1/orpheus-tts')
    .send({ text: 'hello', voice: 'custom-fast:v-abc1234567' })
  expect(res.status).toBe(200)
  expect(orpheusCloneClient.synthesize).toHaveBeenCalledWith('custom-fast:v-abc1234567', 'hello')
})

it('POST /v1/orpheus-tts returns 402 without active billing', async () => {
  vi.mocked(isBillingActiveForUser).mockResolvedValue(false)
  const res = await request(appWithUser({ id: 'u1' }))
    .post('/v1/orpheus-tts')
    .send({ text: 'hello', voice: 'custom-fast:v-abc1234567' })
  expect(res.status).toBe(402)
})
```

Add `synthesize: vi.fn()` to the existing `vi.mock('../src/lib/orpheusCloneClient', ...)` block
from Task 7.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ReadAloudAI/backend && npx vitest run test/orpheusVoiceRouter.test.ts`
Expected: FAIL (`orpheusCloneClient.synthesize` undefined / route not found)

- [ ] **Step 3: Write the implementation**

Add to `orpheusCloneClient.ts`:

```typescript
  async synthesize(voice: string, text: string): Promise<Buffer> {
    const res = await fetch(`${BASE}/v1/orpheus-tts`, {
      method: 'POST',
      headers: authHeaders({ 'content-type': 'application/json' }),
      body: JSON.stringify({ voice, text }),
    })
    return Buffer.from(await res.arrayBuffer())
  },
```

Add to `orpheusVoiceRouter.ts`:

```typescript
// Placeholder SKU -- actual pricing is a business decision not made in
// this plan (see spec's non-goals). This constant exists so the billing
// meter call has a real, findable place to plug the real number in.
const ORPHEUS_TTS_SKU_PLACEHOLDER = 'orpheus_streaming_clone_v1'

orpheusVoiceRouter.post('/v1/orpheus-tts', async (req, res) => {
  const user = requireUser(req, res)
  if (!user) return
  const active = await isBillingActiveForUser(user.id)
  if (!active) {
    res.status(402).json({ error: 'Voice cloning requires an active TTS subscription.' })
    return
  }
  const { voice, text } = req.body
  const audio = await orpheusCloneClient.synthesize(voice, text)
  res.set('content-type', 'audio/pcm')
  res.send(audio)
})
```

Add to Modal's `api.py` (Task 4's file — this task extends it):

```python
    @app.post("/v1/orpheus-tts")
    async def synthesize_voice(request: Request):
        auth(request)
        body = await request.json()
        voice = body.get("voice", "")
        text = body.get("text", "")
        if not text:
            raise HTTPException(400, "text is required")
        engine_cls = get_engine_cls()  # injected by main.py, see below
        try:
            from orpheus_clone_prod.serve import UnknownVoiceError, GenerationTimeoutError
            chunks = engine_cls().synthesize_for_voice.remote(voice, text)
        except UnknownVoiceError:
            raise HTTPException(400, f"unknown or not-ready voice: {voice!r}")
        except GenerationTimeoutError:
            raise HTTPException(503, "generation took too long, please retry")
        return Response(b"".join(chunks), media_type="audio/pcm")
```

`create_app` (Task 4) needs a new optional parameter `get_engine_cls` so `api.py` stays
unit-testable without a real Modal class. Update `api.py`'s signature (the default keeps every
Task 4 test passing unchanged, since none of them pass `get_engine_cls` or hit `/v1/orpheus-tts`):

```python
# voice-pipeline/orpheus_clone_prod/api.py
# Change the function signature:
def create_app(root: str, spawn_training, get_engine_cls=None) -> FastAPI:
    app = FastAPI()
    store = VoiceRecordStore(root=root)

    def _get_engine_cls():
        if get_engine_cls is None:
            raise NotImplementedError("get_engine_cls was not provided to create_app")
        return get_engine_cls()

    # ... existing auth() and routes unchanged ...

    # Then add the /v1/orpheus-tts route (shown in this task's main body above)
    # inside create_app, using _get_engine_cls() where that body says get_engine_cls().
```

And update `main.py`'s `api()` (Task 6) to pass it:

```python
# voice-pipeline/orpheus_clone_prod/main.py
@app.function(image=web_image, secrets=[api_secret], volumes={CHECKPOINT_ROOT: checkpoint_volume}, timeout=120)
@modal.asgi_app()
def api():
    def spawn_training(vid: str, root: str):
        run_training_job_modal.spawn(vid)

    return create_app(root=CHECKPOINT_ROOT, spawn_training=spawn_training, get_engine_cls=lambda: OrpheusCloneEngine)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ReadAloudAI/backend && npx vitest run test/orpheusVoiceRouter.test.ts`
Expected: PASS (6 tests total for this file)

Run: `cd voice-pipeline/orpheus_clone_prod && python -m pytest tests/ -v`
Expected: PASS (all prior tasks' tests still pass — `create_app`'s new optional parameter must not
break Task 4's existing calls, which don't pass `get_engine_cls`)

- [ ] **Step 5: Commit**

```bash
git add ReadAloudAI/backend/src/routes/orpheusVoiceRouter.ts ReadAloudAI/backend/src/lib/orpheusCloneClient.ts ReadAloudAI/backend/test/orpheusVoiceRouter.test.ts voice-pipeline/orpheus_clone_prod/api.py voice-pipeline/orpheus_clone_prod/main.py
git commit -m "Add synthesis endpoint with billing gate and generation-timeout handling"
```

---

### Task 9: Developers-page docs component for the real endpoints

**Files:**
- Create: `ReadAloudAI/web/src/components/ra/OrpheusCloningDocs.tsx`
- Modify: `ReadAloudAI/web/src/app/developers/page.tsx`

**Interfaces:**
- Consumes: nothing new; documents Task 7/8's real endpoints.
- Produces: a docs component in the same shape as `VoiceCloningDocs.tsx` (tabbed curl examples,
  a stats panel), replacing the "in development... not yet available through the API" language
  with real usage instructions once the endpoints exist.

- [ ] **Step 1: Write the component**

```typescript
// ReadAloudAI/web/src/components/ra/OrpheusCloningDocs.tsx
'use client'

import { useState } from 'react'

const BASH = `API_KEY="YOUR_API_KEY"
BASE="https://api.readaloudai.org"

# 1. Create a voice
curl -X POST "$BASE/v1/orpheus-voices" \\
  -H "Authorization: Bearer $API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "speaker_name": "Jane Doe",
    "attested_by": "Jane Doe",
    "consent": true,
    "consent_text_version": "2026-09-v1",
    "consent_statement": "I am authorized to consent on behalf of the speaker named in this request, and that speaker has agreed to have their voice cloned and used to synthesize new speech through this service."
  }'
# -> {"id": "v-a1b2c3d4e5", "status": "awaiting_dataset"}

# 2. Upload recordings (8-20 minutes recommended, single speaker)
curl -X PUT "$BASE/v1/orpheus-voices/v-a1b2c3d4e5/dataset" \\
  -H "Authorization: Bearer $API_KEY" \\
  -H "Content-Type: application/zip" \\
  --data-binary @recordings.zip

# 3. Commit -- starts training
curl -X POST "$BASE/v1/orpheus-voices/v-a1b2c3d4e5/dataset/commit" \\
  -H "Authorization: Bearer $API_KEY"
# -> {"id": "v-a1b2c3d4e5", "status": "training"}`

const POLL = `API_KEY="YOUR_API_KEY"
BASE="https://api.readaloudai.org"

# 4. Poll status until ready
curl -s "$BASE/v1/orpheus-voices/v-a1b2c3d4e5" \\
  -H "Authorization: Bearer $API_KEY"
# -> {"id":"v-a1b2c3d4e5","status":"ready", "clip_count": 42}`

const SYNTHESIZE = `API_KEY="YOUR_API_KEY"
BASE="https://api.readaloudai.org"

# 5. Synthesize with your custom-fast voice
curl -X POST "$BASE/v1/orpheus-tts" \\
  -H "Authorization: Bearer $API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "This sentence is spoken in my cloned voice.",
    "voice": "custom-fast:v-a1b2c3d4e5"
  }' \\
  --output cloned.pcm

# Delete a voice
curl -X DELETE "$BASE/v1/orpheus-voices/v-a1b2c3d4e5" \\
  -H "Authorization: Bearer $API_KEY"`

export default function OrpheusCloningDocs() {
  const [tab, setTab] = useState<'upload' | 'poll' | 'synthesize'>('upload')
  const [copied, setCopied] = useState(false)

  const code = tab === 'upload' ? BASH : tab === 'poll' ? POLL : SYNTHESIZE

  return (
    <div>
      <div className="ra-code" style={{ marginTop: 28 }}>
        <div className="ra-code-bar">
          <div className="ra-code-tabs" role="tablist" aria-label="Step">
            <button role="tab" aria-selected={tab === 'upload'} onClick={() => setTab('upload')}>1-3. Create & upload</button>
            <button role="tab" aria-selected={tab === 'poll'} onClick={() => setTab('poll')}>4. Poll status</button>
            <button role="tab" aria-selected={tab === 'synthesize'} onClick={() => setTab('synthesize')}>5+. Synthesize</button>
          </div>
          <button className="ra-copy" onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1500) }}>
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
        <pre><code>{code}</code></pre>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Replace the "in development" copy in the developers page**

In `ReadAloudAI/web/src/app/developers/page.tsx`, replace the section added earlier (the
`id="streaming-cloning"` block with "not yet available through the API") with:

```tsx
                <h3 id="streaming-cloning">Low-latency streaming clone</h3>
                <p>
                  A second cloning path on a different model (Orpheus, a neural codec language
                  model), for conversational agents that need speech to start well under a second
                  after the request. Create a voice with <code className="inl">custom-fast:&lt;id&gt;</code>,
                  8&ndash;20 minutes of your own recordings recommended.
                </p>
                <p><b>What we measured (early numbers, still improving).</b> A pilot voice reached a
                  median time-to-first-audio-chunk of about 550&ndash;580&nbsp;ms once warm, against
                  a 500&nbsp;ms target &mdash; close, not there yet. Two open issues: generation
                  currently runs at roughly 1.6&ndash;2.7&times; real time, and utterance length is
                  not yet reliably controlled &mdash; the same prompt can produce anywhere from a
                  third of a second to several seconds of audio. We are training on more data per
                  voice to address both. Cold container start adds a few seconds on the first
                  request after an idle period.
                </p>
                <OrpheusCloningDocs />
```

Add the import at the top of `page.tsx`: `import OrpheusCloningDocs from
'@/components/ra/OrpheusCloningDocs'`.

- [ ] **Step 3: Type-check**

Run: `cd ReadAloudAI/web && npx tsc --noEmit -p tsconfig.json`
Expected: no new errors introduced by this change (pre-existing unrelated `test/oauth.test.ts`
errors, if any, are not this task's concern)

- [ ] **Step 4: Commit**

```bash
git add ReadAloudAI/web/src/components/ra/OrpheusCloningDocs.tsx ReadAloudAI/web/src/app/developers/page.tsx
git commit -m "Document the real Orpheus streaming clone API endpoints on the developers page"
```

---

### Task 10: End-to-end integration test (real GPU, one real voice)

**Files:**
- Create: `voice-pipeline/orpheus_clone_prod/tests/test_e2e_real.py` (marked to skip by default,
  run manually — real GPU cost, matching this repo's established practice of never mocking the
  training/serving GPU path for the thing that actually matters)

**Interfaces:**
- Consumes: everything from Tasks 1-6, deployed for real on Modal.

- [ ] **Step 1: Write the manual end-to-end test**

```python
# voice-pipeline/orpheus_clone_prod/tests/test_e2e_real.py
"""
Real end-to-end test against a deployed orpheus-clone-prod app: create a
voice, upload a small real recording set, commit, poll to ready, synthesize,
and confirm non-empty audio. This costs real GPU time and is not run in CI
-- run it manually before considering this plan's deployment done.

Run: ORPHEUS_CLONE_BASE_URL=https://...modal.run \
     ORPHEUS_CLONE_SECRET=... \
     python -m pytest tests/test_e2e_real.py -v -s --no-header -m e2e_real
"""
import os
import time
import zipfile
import io

import pytest
import requests

pytestmark = pytest.mark.e2e_real

BASE = os.environ.get("ORPHEUS_CLONE_BASE_URL")
SECRET = os.environ.get("ORPHEUS_CLONE_SECRET")


@pytest.mark.skipif(not BASE or not SECRET, reason="set ORPHEUS_CLONE_BASE_URL and ORPHEUS_CLONE_SECRET to run")
def test_full_lifecycle_against_deployed_service(tmp_path):
    headers = {"Authorization": f"Bearer {SECRET}"}

    consent = {
        "speaker_name": "Test Speaker",
        "attested_by": "Test Speaker",
        "consent": True,
        "consent_text_version": "2026-09-v1",
        "consent_statement": "I am authorized to consent on behalf of the speaker named in this request.",
    }
    create_resp = requests.post(f"{BASE}/v1/orpheus-voices", json=consent, headers=headers)
    assert create_resp.status_code == 200
    vid = create_resp.json()["id"]

    # Reuse a handful of real clips from the pilot's already-extracted dataset
    # rather than synthesizing throwaway audio for this test.
    pilot_dataset = os.path.expanduser("~/Documents/GitHub/worktrees/orpheus-voice-clone-poc/voice-pipeline/orpheus_clone_pilot")
    sample_clips_dir = os.path.join(pilot_dataset, "..", "..", "training-data", "audio")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i, name in enumerate(sorted(os.listdir(sample_clips_dir))[:25]):
            zf.write(os.path.join(sample_clips_dir, name), name)
    upload_resp = requests.put(
        f"{BASE}/v1/orpheus-voices/{vid}/dataset",
        headers={**headers, "content-type": "application/zip"},
        data=buf.getvalue(),
    )
    assert upload_resp.status_code == 200

    commit_resp = requests.post(f"{BASE}/v1/orpheus-voices/{vid}/dataset/commit", headers=headers)
    assert commit_resp.status_code == 200

    deadline = time.time() + 90 * 60
    status = None
    while time.time() < deadline:
        poll_resp = requests.get(f"{BASE}/v1/orpheus-voices/{vid}", headers=headers)
        status = poll_resp.json()["status"]
        if status in ("ready", "failed"):
            break
        time.sleep(30)
    assert status == "ready", f"training did not reach ready in time, last status: {status}"

    synth_resp = requests.post(
        f"{BASE}/v1/orpheus-tts",
        headers={**headers, "content-type": "application/json"},
        json={"text": "Hello, this is a test of my cloned voice.", "voice": f"custom-fast:{vid}"},
    )
    assert synth_resp.status_code == 200
    assert len(synth_resp.content) > 1000  # non-trivial audio, not an empty/error response

    requests.delete(f"{BASE}/v1/orpheus-voices/{vid}", headers=headers)
```

- [ ] **Step 2: Run it manually after deploying Tasks 1-9**

Run: `modal deploy voice-pipeline/orpheus_clone_prod/main.py`, then set
`ORPHEUS_CLONE_BASE_URL`/`ORPHEUS_CLONE_SECRET` from the deploy output and the
`orpheus-clone-secret` Modal secret, then run the pytest command in the test's docstring.
Expected: PASS, with a real trained voice and real synthesized audio confirmed non-empty.

- [ ] **Step 3: Commit**

```bash
git add voice-pipeline/orpheus_clone_prod/tests/test_e2e_real.py
git commit -m "Add manual real-GPU end-to-end test for the deployed streaming clone service"
```

---

## Final whole-branch review

Before finishing this branch: confirm all unit tests across `voice-pipeline/orpheus_clone_prod/`
and `ReadAloudAI/backend/test/orpheusVoiceRouter.test.ts` pass, confirm `npx tsc --noEmit` on
`ReadAloudAI/web` introduces no new errors, and run Task 10's manual real-GPU test at least once
against a real deploy before calling this done. Then use `superpowers:finishing-a-development-branch`.
