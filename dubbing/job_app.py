"""SKETCH, NOT DEPLOYED: a Modal-app wrapper around dubbing/jobs.py, mirroring
voice-pipeline/intake.py's conventions (app naming, secret-based auth, asgi_app + a spawned
background function) for the case where this pipeline later needs to run inside Modal itself -
e.g. GPU-backed local Piper synthesis instead of calling out to the existing gateway.

This file is NOT deployed and NOT exercised by tests: `modal` and `fastapi` are not installed in
this environment (checked via `python3 -c "import modal"` / `import fastapi`, both
ModuleNotFoundError), and per this task's constraints ("prefer NOT deploying anything and just
building/testing the code locally") nothing here was deployed. The actually-implemented, tested,
running API is dubbing/job_server.py (stdlib `http.server`, zero new dependencies) - see
dubbing/README.md's "Where this API lives" section for the full reasoning on why that's the real
deliverable and this is only a sketch of the alternative.

If this is deployed for real, deploy it under a clearly non-production name, e.g.:
    VOICE_APP_SUFFIX=  ->  app = modal.App("dubbing-mvp-dev")   # never "dubbing-mvp" unprefixed
    modal deploy dubbing/job_app.py
and confirm scale-to-zero the same way intake.py's `api()` does: no GPU function, a short
`timeout`, and Modal's default scale-down on idle for the asgi_app container - nothing keeps
running (or billing) once requests stop.
"""
from __future__ import annotations

import os

try:
    import modal
except ImportError:  # pragma: no cover - modal is not installed in this environment; see module docstring
    modal = None

APP_NAME = os.environ.get("DUBBING_APP_NAME", "dubbing-mvp-dev")  # test/dev name, never a production one

if modal is not None:
    image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi==0.109.0").apt_install("ffmpeg")
    app = modal.App(APP_NAME, image=image)
    job_status_volume = modal.Volume.from_name("dubbing-jobs-dev", create_if_missing=True)
    secret = modal.Secret.from_name("dubbing-job-secret")  # holds DUBBING_JOB_SECRET, OPENROUTER_API_KEY, etc.

    @app.function(volumes={"/jobs": job_status_volume}, timeout=600, cpu=1, memory=1024)
    def run_dub_job(job_id: str, params: dict) -> None:
        """Background job, spawned per submission - analogous to intake.py's train_voice spawn.
        Runs the same dubbing.pipeline.run() as jobs.JobStore._run(), against the Modal volume
        instead of local disk, so status survives across container restarts."""
        from . import jobs  # local import: keeps `modal` optional for the rest of the package

        store = jobs.JobStore(jobs_dir="/jobs")
        store._run(job_id)  # status already written as "queued" by the submitting API call below
        job_status_volume.commit()

    @app.function(secrets=[secret], volumes={"/jobs": job_status_volume}, timeout=60, cpu=1, memory=512)
    @modal.asgi_app()
    def api():
        """Same three routes as job_server.py, expressed as a FastAPI app - see intake.py's `api()`
        for the structural precedent (auth() helper, per-resource ID validation, status_of()-style
        GET). Left unimplemented in detail here since this is a sketch, not the shipped API."""
        import hmac

        from fastapi import FastAPI, HTTPException, Request

        from . import catalog, jobs

        web = FastAPI()
        store = jobs.JobStore(jobs_dir="/jobs")

        def auth(request: Request):
            tok = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not tok or not hmac.compare_digest(tok.encode(), os.environ["DUBBING_JOB_SECRET"].encode()):
                raise HTTPException(401, "unauthorized")

        @web.post("/dubbing/jobs")
        async def submit(request: Request):
            auth(request)
            body = await request.json()
            target_lang = body.get("target_lang")
            try:
                catalog.validate_target_language(target_lang)
            except ValueError as e:
                raise HTTPException(400, str(e))
            job = store.submit(body)
            job_status_volume.commit()
            run_dub_job.spawn(job.id, body)
            return job.to_public_dict()

        @web.get("/dubbing/jobs/{job_id}")
        async def poll(job_id: str, request: Request):
            auth(request)
            job_status_volume.reload()
            job = store.get(job_id)
            if job is None:
                raise HTTPException(404, "unknown job")
            return job.to_public_dict()

        return web
