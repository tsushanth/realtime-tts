"""Callable async job API for the dubbing pipeline: POST /dubbing/jobs -> GET /dubbing/jobs/{id}
(poll) -> GET /dubbing/jobs/{id}/result (fetch), the same submit/poll/fetch shape
voice-pipeline/intake.py + train_job.py use for voice training jobs (POST kicks off a background
run and returns immediately; GET polls a status marker; the artifact is fetched separately once
status is "done").

Why this lives here (Python, stdlib HTTP) rather than in gateway/server.js (Node) or as a newly
deployed Modal app - see dubbing/README.md "Where this API lives" section for the full reasoning;
short version: every step (translate.py's OpenRouter call, tts.py's gateway call, stt.py's
worker-stt call, retime.py's ffmpeg subprocess calls) is already implemented in Python and none of
it needs GPU/CPU Modal compute of its own - it's orchestration of outbound HTTP calls plus local
ffmpeg, exactly like gateway/server.js's /tts/authorize and /stt/authorize already are for their
single-call endpoints, just multi-stage and long-running instead of one round trip. Reimplementing
stt.py/tts.py/translate.py/retime.py in Node would duplicate real, already-written, already-tested
logic for no benefit. A dedicated Modal app (dubbing/job_app.py) mirroring intake.py is sketched
alongside this file for when/if that's wanted, but per this task's "prefer NOT deploying" default
it is NOT deployed - this stdlib server is what's actually implemented, tested and runnable today,
with no new dependency (no `modal`, no `fastapi`, neither of which is installed in this
environment) and no deploy step at all.

Auth: same Bearer-token-over-a-shared-secret pattern as gateway/server.js's `requireAdmin` and
voice-pipeline/intake.py's `auth()` - Bearer DUBBING_JOB_SECRET. Unlike gateway's per-customer API
keys (keys.js), this is a single shared secret for whoever is allowed to submit dubbing jobs
(e.g. the product backend), matching intake.py's INTAKE_SECRET model, since dubbing jobs are
submitted server-to-server, not directly by end users.

Run: DUBBING_JOB_SECRET=... python3 -m dubbing.job_server [--port 8090]
"""
from __future__ import annotations

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import audit, catalog, gateway_auth, jobs

JOB_ID_RE = re.compile(r"^/dubbing/jobs/([A-Za-z0-9_-]+)(/result)?$")


def _secret() -> str | None:
    return os.environ.get("DUBBING_JOB_SECRET")


class DubbingJobHandler(BaseHTTPRequestHandler):
    store: jobs.JobStore = jobs.default_store()  # overridden per-instance by make_server() / tests
    # Overridden per-instance by make_server()/tests to a fake so tests don't need a live gateway.
    authorize_fn = staticmethod(gateway_auth.authorize)

    # --- helpers ---
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = _secret()
        if not expected:
            # Fails closed: an unconfigured secret means nobody is authorized, never "auth
            # disabled" - same posture as gateway/server.js's requireAdmin when ADMIN_SECRET unset.
            return False
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {expected}"

    def _require_auth(self) -> bool:
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return False
        return True

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw)

    # --- routes ---
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming convention
        path = urlparse(self.path).path
        if path != "/dubbing/jobs":
            self._send_json(404, {"error": "not found"})
            return
        if not self._require_auth():
            return
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "body must be JSON"})
            return

        source_lang = body.get("source_lang")
        target_lang = body.get("target_lang")
        text = body.get("text")
        audio_path = body.get("audio_path")  # server-local path; see README for the upload-path caveat
        duration_s = body.get("duration_s")
        api_key = body.get("api_key")

        if not source_lang or not target_lang:
            self._send_json(400, {"error": "source_lang and target_lang are required"})
            return
        if not text and not audio_path:
            self._send_json(400, {"error": "text or audio_path is required"})
            return
        try:
            catalog.validate_target_language(target_lang)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
            return

        # Consent/ownership gating: the DUBBING_JOB_SECRET bearer above only proves "this is the
        # trusted backend calling" (server-to-server, like intake.py's INTAKE_SECRET) - it says
        # nothing about whether there's a real, billing-enabled customer behind this specific job.
        # Match voice-api's /tts/authorize and /audio/authorize: require a valid, billing-enabled
        # gateway API key per job, not an open endpoint once the shared secret is known.
        try:
            result = self.authorize_fn(api_key)
        except gateway_auth.AuthorizeError as e:
            audit.audit_log("job_submit_denied", reason=e.message, status=e.status,
                             target_lang=target_lang, ip=self.client_address[0])
            self._send_json(e.status, {"error": e.message})
            return
        key_id = result.get("id")
        audit.audit_log("job_submit_authorized", id=key_id, target_lang=target_lang,
                         mode="audio_path" if audio_path else "text", ip=self.client_address[0])

        job = self.store.submit({
            "text": text,
            "audio_path": audio_path,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "duration_s": duration_s,
            "key_id": key_id,
        })
        self._send_json(202, job.to_public_dict())

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        m = JOB_ID_RE.match(path)
        if not m:
            self._send_json(404, {"error": "not found"})
            return
        if not self._require_auth():
            return
        job_id, is_result = m.group(1), bool(m.group(2))
        job = self.store.get(job_id)
        if job is None:
            self._send_json(404, {"error": "unknown job"})
            return
        if not is_result:
            self._send_json(200, job.to_public_dict())
            return
        # /result: only serves the artifact once done, and never leaks the local scratch path
        if job.status != "done":
            self._send_json(409, {"error": f"job is not done (status: {job.status})"})
            return
        audio_path = self.store.result_audio_path(job_id)
        if not audio_path:
            self._send_json(410, {"error": "result no longer available"})
            return
        with open(audio_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:  # quieter test/CLI output
        pass


def make_server(host: str = "127.0.0.1", port: int = 8090, store: jobs.JobStore | None = None) -> ThreadingHTTPServer:
    handler = type("BoundDubbingJobHandler", (DubbingJobHandler,), {"store": store or jobs.default_store()})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    if not _secret():
        print("WARNING: DUBBING_JOB_SECRET is not set; every request will be rejected as unauthorized.")
    server = make_server(args.host, args.port)
    print(f"dubbing job server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
