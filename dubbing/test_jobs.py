"""Tests for dubbing/jobs.py's submit -> poll -> fetch lifecycle. Uses a fake pipeline_run (no
network calls, no ffmpeg) so these run anywhere, including without OPENROUTER_API_KEY/ffmpeg.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import unittest

from dubbing import jobs


def fake_pipeline_run(*, audio_path, text, source_lang, target_lang, out_path, duration_s, scratch_dir):
    with open(out_path, "wb") as f:
        f.write(b"RIFF....WAVEfake")
    return {
        "source_text": text,
        "target_lang": target_lang,
        "translated_text": f"[{target_lang}] {text}",
        "out_path": out_path,
    }


def failing_pipeline_run(**kwargs):
    raise RuntimeError("boom: simulated pipeline failure")


def slow_pipeline_run(*, out_path, **kwargs):
    time.sleep(0.2)
    with open(out_path, "wb") as f:
        f.write(b"ok")
    return {"out_path": out_path}


class JobStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dub_jobs_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _store(self, pipeline_run=fake_pipeline_run):
        return jobs.JobStore(jobs_dir=self.tmp, pipeline_run=pipeline_run)

    def test_submit_then_poll_to_done(self):
        store = self._store()
        job = store.submit({"text": "hello", "source_lang": "en", "target_lang": "de_DE"})
        self.assertTrue(job.id.startswith("dub-"))
        self.assertIn(job.status, ("queued", "running", "done"))

        deadline = time.time() + 5
        while time.time() < deadline:
            polled = store.get(job.id)
            if polled.status == "done":
                break
            time.sleep(0.01)
        self.assertEqual(polled.status, "done")
        self.assertEqual(polled.result["translated_text"], "[de_DE] hello")
        d = polled.to_public_dict()
        self.assertNotIn("out_path", d.get("result", {}))  # local path never leaks to the API surface

    def test_result_audio_path_only_after_done(self):
        store = self._store(pipeline_run=slow_pipeline_run)
        job = store.submit({"text": "hi", "source_lang": "en", "target_lang": "de_DE"})
        self.assertIsNone(store.result_audio_path(job.id))  # still queued/running
        deadline = time.time() + 5
        while time.time() < deadline and store.get(job.id).status != "done":
            time.sleep(0.01)
        path = store.result_audio_path(job.id)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

    def test_failed_job_surfaces_error_not_a_crash(self):
        store = self._store(pipeline_run=failing_pipeline_run)
        job = store.submit({"text": "hi", "source_lang": "en", "target_lang": "de_DE"})
        deadline = time.time() + 5
        while time.time() < deadline and store.get(job.id).status not in ("done", "error"):
            time.sleep(0.01)
        polled = store.get(job.id)
        self.assertEqual(polled.status, "error")
        self.assertIn("boom", polled.error)
        d = polled.to_public_dict()
        self.assertEqual(d["error"], polled.error)
        self.assertNotIn("result", d)

    def test_unknown_job_id_returns_none(self):
        store = self._store()
        self.assertIsNone(store.get("dub-doesnotexist"))

    def test_status_survives_a_fresh_store_reading_the_same_dir(self):
        store = self._store()
        job = store.submit({"text": "hi", "source_lang": "en", "target_lang": "de_DE"})
        deadline = time.time() + 5
        while time.time() < deadline and store.get(job.id).status != "done":
            time.sleep(0.01)
        # A brand new JobStore over the same jobs_dir (simulating a process restart) can still
        # answer a poll for a job it didn't submit in-process, from the on-disk status file.
        fresh = jobs.JobStore(jobs_dir=self.tmp, pipeline_run=fake_pipeline_run)
        reread = fresh.get(job.id)
        self.assertIsNotNone(reread)
        self.assertEqual(reread.status, "done")

    def test_concurrent_jobs_do_not_clobber_each_other(self):
        store = self._store()
        submitted = [store.submit({"text": f"msg-{i}", "source_lang": "en", "target_lang": "de_DE"}) for i in range(5)]
        deadline = time.time() + 5
        while time.time() < deadline and any(store.get(j.id).status != "done" for j in submitted):
            time.sleep(0.01)
        for i, j in enumerate(submitted):
            self.assertEqual(store.get(j.id).result["translated_text"], f"[de_DE] msg-{i}")


if __name__ == "__main__":
    unittest.main()
