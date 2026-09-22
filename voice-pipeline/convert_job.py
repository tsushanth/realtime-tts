"""Batch (non-realtime) speech-to-speech voice conversion MVP, built on Plachtaa/seed-vc's
zero-shot `inference.py` (NOT `real-time-gui.py` / the streaming app.py - explicitly the batch path).

Given a source clip (what to say) and a target reference clip (whose voice), produces the source's
content in the target's voice, preserving the source's original timing/prosody: true voice
conversion (mel-spectrogram -> DiT diffusion -> vocoder), not an STT->TTS re-synthesis. No
per-speaker training: the target reference is just conditioning at inference time.

DEV / NON-PRODUCTION: app name is "voice-convert-dev" and does not touch the voice-intake /
voice-train apps, their volumes, or the Piper serving machine. Scale-to-zero GPU: containers only
run for the duration of a job and cost nothing while idle. Do not deploy this under a
production-sounding name.

Async job pattern (mirrors train_job.py): submit writes source/target + a job.json onto a Volume
and spawns the GPU function; status/result are read back from the Volume.

HTTP API surface (hardening pass): this MVP shipped CLI-only (`modal run ... --source --target`).
To make it callable as a service we added a `@modal.asgi_app()` FastAPI wrapper (`api()`, below)
that is a thin HTTP front door onto the *same* submit/status/fetch_result/cleanup Modal functions
the CLI already used - no new job logic, just HTTP routing + auth around it.

Why this pattern over dubbing/jobs.py + job_server.py's stdlib-HTTP-thread pattern: that pattern
was chosen there specifically because dubbing's pipeline is pure orchestration of outbound HTTP
calls + local ffmpeg with no GPU/CPU compute of its own, so a Python thread was enough and adding
`modal`/`fastapi` would have been new, unneeded dependencies (see dubbing/job_server.py's own
docstring). Voice conversion is the opposite case: the actual work is a GPU (A10G) subprocess that
needs Modal's scale-to-zero container scheduling, which a stdlib thread server cannot provide (it
would need its own always-on host and its own GPU). Modal's own `asgi_app()` is already how the
sibling intake.py exposes voice-intake as HTTP, so this follows intake.py's pattern (submit -> poll
status -> fetch result, Bearer-secret auth) rather than reimplementing dubbing's - same shape as
this file's own submit()/status()/fetch_result()/cleanup() functions already use internally, so the
HTTP layer is close to a 1:1 mapping onto functions that already existed and were already tested via
the CLI. This keeps the CLI (`modal run convert_job.py ...`) working unchanged for local/dev use.

Consent/ownership: like intake.py and dubbing/job_server.py, this is a single shared-secret gate
(Bearer CONVERT_SECRET, `modal.Secret.from_name("voice-convert")`), not a per-customer API-key /
consent-attestation model - this MVP is meant to be called server-to-server by a backend that has
already decided whose audio this is and whether they consented to have their voice used as a
target reference, the same division of responsibility intake.py's docstring states explicitly
("this API only checks one shared secret"). If this is ever exposed to end users directly (rather
than through a trusted backend), port intake.py's per-caller consent-attestation fields
(speaker_name/attested_by/consent/consent_text_version) and the gateway's per-key
isValidKey/checkAccess billing gate first - a bare shared secret is not sufficient for that case.

Usage:
    modal run convert_job.py --source ./clip.wav --target ./ref.wav --out ./out.wav   (CLI, unchanged)
    modal serve convert_job.py                                                         (HTTP, dev)
    modal deploy convert_job.py                                                        (HTTP, deployed)
"""
import os as _os
import modal

APP_SUFFIX = _os.environ.get("VOICE_CONVERT_APP_SUFFIX", "")  # for spinning up a second throwaway copy
app = modal.App("voice-convert-dev" + APP_SUFFIX)

jobs = modal.Volume.from_name("voice-convert-dev-jobs", create_if_missing=True)
# Optional: absent in dev by default (mirrors intake.py's INTAKE_SECRET requirement, but does not
# hard-fail import so `modal run` for local CLI testing keeps working without provisioning a secret).
try:
    convert_secret = modal.Secret.from_name("voice-convert", required_keys=["CONVERT_SECRET"])
except Exception:
    convert_secret = None

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.1.2", "torchaudio==2.1.2", extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands("git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /opt/seed-vc")
    .pip_install(
        "einops", "transformers==4.41.2", "huggingface_hub", "librosa", "soundfile", "PyYAML",
        "munch", "scipy", "modelscope", "onnxruntime", "gradio", "sounddevice", "descript-audio-codec",
        "openai-whisper", "matplotlib", "accelerate", "numpy<2",
    )
    .add_local_python_source("quality")  # shared metrics/quality-gate module (voice-pipeline/quality.py)
)
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi==0.109.0")

MAX_SOURCE_SECONDS = 120.0
MAX_TARGET_SECONDS = 60.0
MIN_SECONDS = 0.5
MAX_BYTES = 50 * 1024 * 1024
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


def _validate(path: str, kind: str, max_seconds: float):
    """Format + duration gate. Raises ValueError with a plain-language reason (mirrors train_job's
    reject() style: cheap, readable checks before any GPU time is spent)."""
    import soundfile as sf

    ext = _os.path.splitext(path)[1].lower()
    if ext not in AUDIO_EXT:
        raise ValueError(f"{kind}: unsupported format '{ext}' (allowed: {', '.join(AUDIO_EXT)})")
    size = _os.path.getsize(path)
    if size == 0:
        raise ValueError(f"{kind}: empty file")
    if size > MAX_BYTES:
        raise ValueError(f"{kind}: file too large ({size} bytes, max {MAX_BYTES})")
    try:
        info = sf.info(path)
        duration = info.frames / info.samplerate
    except Exception:
        # soundfile can't read mp3/m4a on some builds; fall back to librosa (slower, decodes fully)
        import librosa

        y, sr = librosa.load(path, sr=None)
        duration = len(y) / sr
    if duration < MIN_SECONDS:
        raise ValueError(f"{kind}: too short ({duration:.2f}s, min {MIN_SECONDS}s)")
    if duration > max_seconds:
        raise ValueError(f"{kind}: too long ({duration:.1f}s, max {max_seconds:.0f}s)")
    return duration


def _job_dir(job_id: str) -> str:
    return f"/jobs/{job_id}"


@app.function(image=image, volumes={"/jobs": jobs}, timeout=120)
def submit(job_id: str, source_bytes: bytes, source_ext: str, target_bytes: bytes, target_ext: str,
           diffusion_steps: int = 30, length_adjust: float = 1.0) -> dict:
    """Validates inputs, stores them on the jobs Volume, writes job.json (status=queued), and spawns
    the GPU conversion. Returns immediately - poll status() / fetch_result()."""
    import json, time

    d = _job_dir(job_id)
    _os.makedirs(d, exist_ok=True)
    src_path = f"{d}/source{source_ext}"
    tgt_path = f"{d}/target{target_ext}"
    open(src_path, "wb").write(source_bytes)
    open(tgt_path, "wb").write(target_bytes)

    try:
        src_dur = _validate(src_path, "source", MAX_SOURCE_SECONDS)
        tgt_dur = _validate(tgt_path, "target reference", MAX_TARGET_SECONDS)
    except ValueError as e:
        json.dump({"status": "rejected", "error": str(e)}, open(f"{d}/job.json", "w"))
        jobs.commit()
        return {"job_id": job_id, "status": "rejected", "error": str(e)}

    json.dump({
        "status": "queued", "submitted_at": int(time.time()),
        "source_seconds": round(src_dur, 2), "target_seconds": round(tgt_dur, 2),
        "diffusion_steps": diffusion_steps, "length_adjust": length_adjust,
    }, open(f"{d}/job.json", "w"))
    jobs.commit()

    run_conversion.spawn(job_id, diffusion_steps, length_adjust)
    return {"job_id": job_id, "status": "queued", "source_seconds": round(src_dur, 2), "target_seconds": round(tgt_dur, 2)}


@app.function(image=image, gpu="A10G", volumes={"/jobs": jobs}, timeout=900)
def run_conversion(job_id: str, diffusion_steps: int = 30, length_adjust: float = 1.0):
    """Runs seed-vc's batch inference.py (zero-shot VC, no training) as a subprocess: this is the
    library's own non-realtime entrypoint, deliberately not real-time-gui.py or the streaming app.
    length-adjust=1.0 (default) keeps the source's original timing/prosody."""
    import glob, json, subprocess, time

    d = _job_dir(job_id)
    job = json.load(open(f"{d}/job.json"))
    job["status"] = "processing"
    job["started_at"] = int(time.time())
    json.dump(job, open(f"{d}/job.json", "w"))
    jobs.commit()

    source = glob.glob(f"{d}/source.*")[0]
    target = glob.glob(f"{d}/target.*")[0]
    out_dir = f"{d}/out"
    _os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    cmd = [
        "python3", "inference.py",
        "--source", source, "--target", target, "--output", out_dir,
        "--diffusion-steps", str(diffusion_steps), "--length-adjust", str(length_adjust),
        "--inference-cfg-rate", "0.7", "--f0-condition", "False", "--fp16", "True",
    ]
    proc = subprocess.run(cmd, cwd="/opt/seed-vc", capture_output=True, text=True)
    elapsed = time.time() - t0

    outputs = glob.glob(f"{out_dir}/*.wav")
    if proc.returncode != 0 or not outputs:
        job.update({
            "status": "failed", "finished_at": int(time.time()), "gpu_seconds": round(elapsed, 1),
            "stderr_tail": proc.stderr[-4000:], "stdout_tail": proc.stdout[-2000:],
        })
        json.dump(job, open(f"{d}/job.json", "w"))
        jobs.commit()
        return

    result_path = f"{d}/result.wav"
    _os.rename(outputs[0], result_path)

    quality = None
    try:
        from quality import validate_conversion_quality
        quality = validate_conversion_quality(source, target, result_path)
    except Exception as e:  # a broken quality check must never fail a conversion that otherwise succeeded
        quality = {"passed": None, "reasons": [], "warnings": [f"quality check itself errored: {e!r}"], "metrics": {}}

    job.update({
        "status": "done", "finished_at": int(time.time()), "gpu_seconds": round(elapsed, 1),
        "quality": quality,
    })
    json.dump(job, open(f"{d}/job.json", "w"))
    jobs.commit()


@app.function(image=modal.Image.debian_slim(), volumes={"/jobs": jobs}, timeout=30)
def status(job_id: str) -> dict:
    import json

    jobs.reload()
    p = f"{_job_dir(job_id)}/job.json"
    if not _os.path.exists(p):
        return {"job_id": job_id, "status": "unknown"}
    return {"job_id": job_id, **json.load(open(p))}


@app.function(image=modal.Image.debian_slim(), volumes={"/jobs": jobs}, timeout=30)
def fetch_result(job_id: str) -> bytes:
    jobs.reload()
    p = f"{_job_dir(job_id)}/result.wav"
    if not _os.path.exists(p):
        raise FileNotFoundError(f"no result for {job_id} (check status first)")
    return open(p, "rb").read()


@app.function(image=modal.Image.debian_slim(), volumes={"/jobs": jobs}, timeout=30)
def cleanup(job_id: str) -> dict:
    """Removes a job's audio from the Volume (source/target/result) - call after fetching the
    result so nothing lingers billing Volume storage."""
    import shutil

    d = _job_dir(job_id)
    existed = _os.path.exists(d)
    shutil.rmtree(d, ignore_errors=True)
    jobs.commit()
    return {"job_id": job_id, "deleted": existed}


@app.function(
    image=web_image,
    secrets=[convert_secret] if convert_secret else [],
    volumes={"/jobs": jobs},
    timeout=60,
)
@modal.asgi_app()
def api():
    """HTTP front door onto submit/status/fetch_result/cleanup - see the module docstring for why
    this asgi_app pattern was chosen over dubbing/job_server.py's stdlib-thread pattern.

      POST   /convert            multipart form: source=<file>, target=<file>,
                                  diffusion_steps?=int (default 30), length_adjust?=float (default 1.0)
                                  -> 202 {job_id, status: queued|rejected, source_seconds, target_seconds}
      GET    /convert/{job_id}   -> job.json contents, e.g. {status: queued|processing|done|failed,
                                  quality: {passed, reasons, warnings, metrics}} once done
      GET    /convert/{job_id}/result   -> audio/wav bytes (409 if not done yet, 404 if unknown)
      DELETE /convert/{job_id}   -> removes the job's audio from the Volume, {deleted: bool}

    Auth: Bearer CONVERT_SECRET on every route (hmac.compare_digest, fail-closed if the secret is
    unset - same posture as intake.py's auth() and dubbing/job_server.py's _authorized()). This is a
    shared secret for a trusted caller (e.g. a product backend), not a per-customer API key/consent
    model - see the module docstring for what porting that would require.
    """
    import hmac
    import json
    import secrets as pysecrets

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response

    web = FastAPI()

    def auth(request: Request):
        secret = _os.environ.get("CONVERT_SECRET")
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not secret or not tok or not hmac.compare_digest(tok.encode(), secret.encode()):
            raise HTTPException(401, "unauthorized")

    @web.post("/convert")
    async def convert(request: Request, diffusion_steps: int = 30, length_adjust: float = 1.0):
        auth(request)
        form = await request.form()
        source = form.get("source")
        target = form.get("target")
        if source is None or target is None:
            raise HTTPException(400, "multipart fields 'source' and 'target' (audio files) are required")
        job_id = "job-" + pysecrets.token_hex(5)
        source_bytes, target_bytes = await source.read(), await target.read()
        src_ext = _os.path.splitext(source.filename or "")[1].lower() or ".wav"
        tgt_ext = _os.path.splitext(target.filename or "")[1].lower() or ".wav"
        result = submit.remote(job_id, source_bytes, src_ext, target_bytes, tgt_ext, diffusion_steps, length_adjust)
        status_code = 400 if result["status"] == "rejected" else 202
        return Response(content=json.dumps(result), media_type="application/json", status_code=status_code)

    @web.get("/convert/{job_id}")
    async def get_status(job_id: str, request: Request):
        auth(request)
        return status.remote(job_id)

    @web.get("/convert/{job_id}/result")
    async def get_result(job_id: str, request: Request):
        auth(request)
        st = status.remote(job_id)
        if st["status"] == "unknown":
            raise HTTPException(404, "unknown job")
        if st["status"] != "done":
            raise HTTPException(409, f"job is not done (status: {st['status']})")
        try:
            audio = fetch_result.remote(job_id)
        except FileNotFoundError:
            raise HTTPException(410, "result no longer available")
        return Response(audio, media_type="audio/wav")

    @web.delete("/convert/{job_id}")
    async def delete(job_id: str, request: Request):
        auth(request)
        return cleanup.remote(job_id)

    return web


@app.local_entrypoint()
def main(source: str, target: str, out: str = "./converted.wav", job_id: str = "", diffusion_steps: int = 30,
         length_adjust: float = 1.0, poll_seconds: int = 5, timeout_seconds: int = 600, keep: bool = False):
    """modal run convert_job.py --source ./clip.wav --target ./ref.wav --out ./converted.wav"""
    import secrets, time

    jid = job_id or "job-" + secrets.token_hex(5)
    src_ext = _os.path.splitext(source)[1].lower()
    tgt_ext = _os.path.splitext(target)[1].lower()
    src_bytes = open(source, "rb").read()
    tgt_bytes = open(target, "rb").read()

    res = submit.remote(jid, src_bytes, src_ext, tgt_bytes, tgt_ext, diffusion_steps, length_adjust)
    print("submit:", res)
    if res["status"] == "rejected":
        raise SystemExit(f"rejected: {res['error']}")

    t0 = time.time()
    while time.time() - t0 < timeout_seconds:
        st = status.remote(jid)
        print("status:", st["status"], f"(+{int(time.time() - t0)}s)")
        if st["status"] in ("done", "failed"):
            break
        time.sleep(poll_seconds)
    else:
        raise SystemExit("timed out waiting for conversion")

    if st["status"] == "failed":
        print("STDERR TAIL:\n", st.get("stderr_tail", ""))
        raise SystemExit("conversion failed")

    audio = fetch_result.remote(jid)
    open(out, "wb").write(audio)
    print(f"wrote {out} ({len(audio)} bytes), gpu_seconds={st.get('gpu_seconds')}")
    if not keep:
        print(cleanup.remote(jid))
