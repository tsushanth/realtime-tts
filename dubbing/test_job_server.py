"""Tests for dubbing/job_server.py's HTTP surface: POST /dubbing/jobs, GET /dubbing/jobs/{id},
GET /dubbing/jobs/{id}/result. Uses a fake pipeline_run (no network calls, no ffmpeg) and a real
loopback HTTP server on an ephemeral port.
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from dubbing import gateway_auth, job_server, jobs
from dubbing.test_jobs import fake_pipeline_run

SECRET = "test-dubbing-job-secret"


def fake_authorize_ok(api_key):
    """Stands in for gateway_auth.authorize so these tests don't need a live Node gateway -
    the wire-level contract with gateway/server.js's /dubbing/authorize is exercised separately
    (this is the same pattern the store/pipeline_run injection above already uses)."""
    if not api_key:
        raise gateway_auth.AuthorizeError(400, "api_key is required")
    if api_key != "valid-customer-key":
        raise gateway_auth.AuthorizeError(401, "invalid or missing API key")
    return {"authorized": True, "id": "key-abc123"}


class JobServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DUBBING_JOB_SECRET"] = SECRET
        cls.tmp = tempfile.mkdtemp(prefix="dub_job_server_test_")
        cls.store = jobs.JobStore(jobs_dir=cls.tmp, pipeline_run=fake_pipeline_run)
        cls.server = job_server.make_server(port=0, store=cls.store)
        cls.server.RequestHandlerClass.authorize_fn = staticmethod(fake_authorize_ok)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        os.environ.pop("DUBBING_JOB_SECRET", None)

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)

    def _post_job(self, body: dict, auth: str | None = SECRET, api_key: str | None = "valid-customer-key"):
        conn = self._conn()
        headers = {"Content-Type": "application/json"}
        if auth is not None:
            headers["Authorization"] = f"Bearer {auth}"
        full_body = dict(body)
        if api_key is not None:
            full_body["api_key"] = api_key
        conn.request("POST", "/dubbing/jobs", body=json.dumps(full_body), headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
        return resp.status, data

    def _get(self, path: str, auth: str | None = SECRET):
        conn = self._conn()
        headers = {}
        if auth is not None:
            headers["Authorization"] = f"Bearer {auth}"
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, body

    def test_submit_requires_auth(self):
        status, data = self._post_job({"text": "hi", "source_lang": "en", "target_lang": "de_DE"}, auth=None)
        self.assertEqual(status, 401)
        self.assertEqual(data, {"error": "unauthorized"})

    def test_submit_rejects_wrong_key(self):
        status, _ = self._post_job({"text": "hi", "source_lang": "en", "target_lang": "de_DE"}, auth="wrong")
        self.assertEqual(status, 401)

    def test_submit_validates_target_language(self):
        status, data = self._post_job({"text": "hi", "source_lang": "en", "target_lang": "xx_XX"})
        self.assertEqual(status, 400)
        self.assertIn("xx_XX", data["error"])

    def test_submit_requires_text_or_audio(self):
        status, data = self._post_job({"source_lang": "en", "target_lang": "de_DE"})
        self.assertEqual(status, 400)

    def test_full_lifecycle_submit_poll_fetch(self):
        status, data = self._post_job({"text": "Thanks for calling", "source_lang": "en", "target_lang": "de_DE"})
        self.assertEqual(status, 202)
        job_id = data["job_id"]
        self.assertIn(data["status"], ("queued", "running", "done"))

        deadline = time.time() + 5
        polled = None
        while time.time() < deadline:
            status, body = self._get(f"/dubbing/jobs/{job_id}")
            self.assertEqual(status, 200)
            polled = json.loads(body)
            if polled["status"] == "done":
                break
            time.sleep(0.02)
        self.assertEqual(polled["status"], "done")
        self.assertEqual(polled["result"]["translated_text"], "[de_DE] Thanks for calling")
        self.assertNotIn("out_path", polled["result"])

        status, audio = self._get(f"/dubbing/jobs/{job_id}/result")
        self.assertEqual(status, 200)
        self.assertEqual(audio, b"RIFF....WAVEfake")

    def test_poll_unknown_job_is_404(self):
        status, data = self._get("/dubbing/jobs/dub-nope")
        self.assertEqual(status, 404)

    def test_result_before_done_is_409(self):
        status, data = self._post_job({"text": "slow one", "source_lang": "en", "target_lang": "de_DE"})
        job_id = data["job_id"]
        # Immediately try to fetch the result; the job store's fake pipeline is fast but we don't
        # rely on timing - a status other than "done" must 409, whichever way the race falls.
        status, body = self._get(f"/dubbing/jobs/{job_id}/result")
        self.assertIn(status, (200, 409))  # allow either if it finished first; assert the contract below
        if status == 409:
            data = json.loads(body)
            self.assertIn("not done", data["error"])

    def test_poll_requires_auth(self):
        status, data = self._post_job({"text": "hi", "source_lang": "en", "target_lang": "de_DE"})
        job_id = data["job_id"]
        status, _ = self._get(f"/dubbing/jobs/{job_id}", auth=None)
        self.assertEqual(status, 401)

    # --- consent/ownership gating: a valid, billing-enabled gateway API key is required per job,
    # separate from the DUBBING_JOB_SECRET bearer above (which only proves "trusted backend") ---

    def test_submit_requires_api_key(self):
        status, data = self._post_job({"text": "hi", "source_lang": "en", "target_lang": "de_DE"}, api_key=None)
        self.assertEqual(status, 400)
        self.assertIn("api_key", data["error"])

    def test_submit_rejects_invalid_api_key(self):
        status, data = self._post_job({"text": "hi", "source_lang": "en", "target_lang": "de_DE"}, api_key="not-a-real-key")
        self.assertEqual(status, 401)
        self.assertEqual(data, {"error": "invalid or missing API key"})

    def test_submit_with_valid_api_key_succeeds(self):
        status, data = self._post_job({"text": "hi", "source_lang": "en", "target_lang": "de_DE"}, api_key="valid-customer-key")
        self.assertEqual(status, 202)


if __name__ == "__main__":
    unittest.main()
