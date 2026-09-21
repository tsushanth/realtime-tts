"""Async job manager for the dubbing pipeline: submit -> poll -> fetch, the same shape as
voice-pipeline/intake.py's training jobs (POST creates + kicks off background work; GET polls a
status file; a separate fetch returns the artifact once status is "done").

Deliberately stdlib-only (no `modal`, no `fastapi`) so the job semantics are unit-testable in any
environment, including this one, where neither package is installed. `dubbing/job_app.py` wraps
this in a Modal ASGI app that mirrors voice-pipeline/intake.py's structure 1:1 for when/if it's
deployed; this module is what actually runs the pipeline and tracks state, and is exercised
directly by tests.

Each job runs dubbing.pipeline.run() on a background thread (analogous to intake.py's
`modal.Function...spawn()` - fire the work, return immediately, poll for completion). State is
kept both in memory and mirrored to JSON files under a job directory (analogous to intake.py's
status-by-marker-file-on-a-volume pattern: `training.json` / `manifest.json` / `error.json`), so a
restarted process can still answer GET /dubbing/jobs/{id} for jobs already on disk.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import pipeline

JOB_ID_PREFIX = "dub-"
_TERMINAL = ("done", "error")


@dataclasses.dataclass
class Job:
    id: str
    status: str  # queued | running | done | error
    created_at: float
    params: dict
    updated_at: float
    result: dict | None = None
    error: str | None = None

    def to_public_dict(self) -> dict:
        """What a GET /dubbing/jobs/{id} response body carries - no local filesystem paths."""
        out = {
            "job_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.status == "done" and self.result:
            out["result"] = {k: v for k, v in self.result.items() if k != "out_path"}
            out["result_ready"] = True
        if self.status == "error":
            out["error"] = self.error
        return out


class JobStore:
    """In-memory + on-disk job tracker. One process-wide instance is normal (see
    `default_store()`); tests construct their own with an isolated `jobs_dir`.
    """

    def __init__(self, jobs_dir: str | None = None, max_workers: int = 4,
                 pipeline_run: Callable[..., dict] = pipeline.run):
        self.jobs_dir = jobs_dir or os.path.join(tempfile.gettempdir(), "dubbing_jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pipeline_run = pipeline_run

    # ---- submit ----
    def submit(self, params: dict) -> Job:
        job_id = JOB_ID_PREFIX + uuid.uuid4().hex[:12]
        now = time.time()
        job = Job(id=job_id, status="queued", created_at=now, updated_at=now, params=params)
        with self._lock:
            self._jobs[job_id] = job
        self._write_status(job)
        self._executor.submit(self._run, job_id)
        return job

    # ---- poll ----
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        return self._read_status(job_id)

    # ---- fetch result audio ----
    def result_audio_path(self, job_id: str) -> str | None:
        job = self.get(job_id)
        if job is None or job.status != "done" or not job.result:
            return None
        path = job.result.get("out_path")
        return path if path and os.path.exists(path) else None

    # ---- internals ----
    def _job_scratch_dir(self, job_id: str) -> str:
        d = os.path.join(self.jobs_dir, job_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _status_path(self, job_id: str) -> str:
        return os.path.join(self._job_scratch_dir(job_id), "status.json")

    def _write_status(self, job: Job) -> None:
        with open(self._status_path(job.id), "w") as f:
            json.dump(dataclasses.asdict(job), f)

    def _read_status(self, job_id: str) -> Job | None:
        path = self._status_path(job_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return Job(**data)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.updated_at = time.time()
        self._write_status(job)
        scratch = self._job_scratch_dir(job_id)
        out_path = os.path.join(scratch, "out.wav")
        try:
            result = self._pipeline_run(
                audio_path=job.params.get("audio_path"),
                text=job.params.get("text"),
                source_lang=job.params["source_lang"],
                target_lang=job.params["target_lang"],
                out_path=out_path,
                duration_s=job.params.get("duration_s"),
                scratch_dir=scratch,
            )
            with self._lock:
                job.status = "done"
                job.result = result
                job.updated_at = time.time()
        except Exception as e:  # noqa: BLE001 - job errors must surface via status, never crash the worker thread
            with self._lock:
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.updated_at = time.time()
        self._write_status(job)


_default_store: JobStore | None = None
_default_store_lock = threading.Lock()


def default_store() -> JobStore:
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = JobStore()
        return _default_store
