"""Voice-design MVP (v1, "direct-serve"): text description -> previewable audio via Parler-TTS,
async, on Modal scale-to-zero GPU. Follows the auth / submit-poll / error-handling shape of
intake.py + train_job.py, but does NOT do any fine-tuning: Parler-TTS is a single text+description
-conditioned model, so "designing a voice" here just means writing a good natural-language voice
description and getting back generated speech. No training job, no per-user checkpoint, no serving
machine push - this is intentionally the smallest thing that lets someone iterate on a voice
description and hear results.

THIS IS A DEV/PROTOTYPE APP. Non-production name on purpose: "voice-design-dev". Do not point any
product surface at it and do not rename it to reuse voice-intake/voice-train's names.

API (Bearer VOICE_DESIGN_SECRET):
  POST /designs            {description, text} -> {job_id}         submit a generation job
  GET  /designs/{job_id}   -> {status: queued|running|ready|failed, ...}  poll for the result
  GET  /designs/{job_id}/audio  -> wav bytes (only once status == ready)

Deploy:  modal deploy voice_design_dev.py
Secret `voice-design-dev` must hold VOICE_DESIGN_SECRET.
"""
import os as _os
import time
import modal

APP_SUFFIX = _os.environ.get("VOICE_DESIGN_APP_SUFFIX", "")  # "" = voice-design-dev; set to try a second throwaway copy
app = modal.App("voice-design-dev" + APP_SUFFIX)

# Parler-TTS needs a recent transformers/torch; kept in its own image, separate from voice-train's
# Piper image (different model family entirely).
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "espeak-ng")
    .pip_install(
        "torch==2.1.2", "torchaudio==2.1.2", extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "git+https://github.com/huggingface/parler-tts.git",
        "soundfile", "numpy<2",
        # parler-tts pulls torchaudio unpinned as a transitive dep, which grabs a version built
        # against a different CUDA runtime (libcudart.so.13 missing) than torch==2.1.2+cu121 above.
        # Re-pin after, so the matched pair from the line above wins.
        "torchaudio==2.1.2",
    )
)
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi==0.109.0")

jobs = modal.Volume.from_name("voice-design-dev-jobs", create_if_missing=True)
secret = modal.Secret.from_name("voice-design-dev")

MAX_DESCRIPTION_CHARS = 800
MIN_DESCRIPTION_CHARS = 10
MAX_TEXT_CHARS = 500
MIN_TEXT_CHARS = 1
JOB_ID_RE = r"^d-[0-9a-f]{10}$"

MODEL_ID = "parler-tts/parler-tts-mini-v1"

# Rate limiting: fixed-window, per-token, applied only to POST /designs (the GPU-spawning,
# billable action) - GET polling is free and unlimited since callers legitimately poll their own
# job repeatedly. Default is deliberately low: this is a dev/preview GPU path, not a production
# throughput target. Override per-deployment with VOICE_DESIGN_RATE_LIMIT_PER_HOUR.
RATE_LIMIT_PER_HOUR = int(_os.environ.get("VOICE_DESIGN_RATE_LIMIT_PER_HOUR", "10"))


@app.cls(
    image=gpu_image, gpu="A10G", timeout=600, volumes={"/jobs": jobs},
    scaledown_window=60,  # scale to zero quickly after the last request - this is a dev/preview path, not a hot service
)
class VoiceDesignModel:
    """Class-based (not @app.function) specifically so @modal.enter() can load Parler-TTS ONCE per
    container and keep it resident in GPU memory for the container's life. The original v1
    function-based `generate` called `from_pretrained()` inside the function body on every single
    invocation - so even a warm container (same process, same scaledown_window) still paid a
    ~10-12s reload every call (confirmed by measurement: a back-to-back warm call showed
    model_load_seconds=11.49s, vs 26.15s cold - faster because the weights were already on local
    disk, but still not truly warm because the model object itself was rebuilt and re-transferred
    to the GPU each time). Moving load into @modal.enter() is the actual fix, not just a
    measurement exercise: it makes warm requests skip loading entirely."""

    @modal.enter()
    def load(self):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.container_load_seconds = round(time.time() - t0, 2)

    @modal.method()
    def generate(self, job_id: str, description: str, text: str):
        """Writes status.json + (on success) audio.wav into the job's volume dir. model_load_seconds
        in the result is now ~0 for every call after the first on a given container (the container's
        own one-time load cost is reported separately as container_load_seconds, not billed to any
        one job) - this is the number that answers "what does a warm request actually cost"."""
        import json
        import os
        import soundfile as sf

        d = f"/jobs/{job_id}"
        os.makedirs(d, exist_ok=True)  # defensive: api() also creates this before spawning, but a
        # direct caller (e.g. a latency probe invoking this function without going through
        # /designs) should not crash on a missing directory.

        def write_status(**kw):
            json.dump({"job_id": job_id, **kw}, open(f"{d}/status.json", "w"))
            jobs.commit()

        write_status(status="running", started_at=int(time.time()))
        try:
            t1 = time.time()
            desc_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
            prompt_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            generation = self.model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids)
            audio = generation.cpu().numpy().squeeze()
            gen_s = time.time() - t1

            sf.write(f"{d}/audio.wav", audio, self.model.config.sampling_rate)
            result = dict(
                status="ready", finished_at=int(time.time()),
                container_load_seconds=self.container_load_seconds,  # one-time per container, not per job
                generation_seconds=round(gen_s, 2),
                audio_seconds=round(len(audio) / self.model.config.sampling_rate, 2),
                sampling_rate=self.model.config.sampling_rate,
            )
            write_status(**result)
            # Returned in addition to the volume write so a caller invoking this function directly
            # (e.g. a warm-path latency probe via `modal.Cls.from_name(...)().generate.remote()`,
            # not through the /designs HTTP API) can read timings without a second round trip
            # through the volume. api() below still only reads status.json/audio.wav, unaffected.
            return {"job_id": job_id, **result}
        except Exception as e:
            write_status(status="failed", finished_at=int(time.time()), error=str(e)[:500])
            raise


@app.function(image=web_image, secrets=[secret], volumes={"/jobs": jobs}, timeout=60)
@modal.asgi_app()
def api():
    import hashlib
    import hmac
    import json
    import os
    import re
    import secrets as pysecrets
    import time as _time

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response

    web = FastAPI()
    ID_RE = re.compile(JOB_ID_RE)

    def auth(request: Request) -> str:
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not tok or not hmac.compare_digest(tok.encode(), os.environ["VOICE_DESIGN_SECRET"].encode()):
            raise HTTPException(401, "unauthorized")
        return tok

    def jid_ok(jid: str):
        if not ID_RE.match(jid):
            raise HTTPException(400, "bad job id")

    # Fixed-window per-token rate limit, backed by the same volume /jobs already uses for job
    # state (no new infra). Bucketed by wall-clock hour so the window resets cleanly, tracked per
    # sha256(token) so the shared secret today - or any per-caller token added later, see
    # VOICE_DESIGN_MVP.md's rate-limiting section - each gets an independent counter. This is a
    # basic MVP-grade limiter, not a production one: the read-modify-write against the volume is
    # NOT atomic across concurrent requests hitting different containers, so a burst landing across
    # two containers in the same instant could both read the same pre-increment count and both
    # pass ("undercounting" under concurrency), i.e. this bounds sustained abuse, not a hard cap
    # under adversarial concurrent load. A real per-key quota (Cloudflare KV / Redis / Postgres
    # counter with atomic INCR, as sketched in VOICE_DESIGN_MVP.md) is the fix if this ever needs
    # to be abuse-resistant rather than just "not unlimited."
    def check_rate_limit(token: str):
        os.makedirs("/jobs/.ratelimit", exist_ok=True)
        th = hashlib.sha256(token.encode()).hexdigest()[:16]
        path = f"/jobs/.ratelimit/{th}.json"
        hour = int(_time.time() // 3600)
        jobs.reload()
        try:
            rec = json.load(open(path))
        except Exception:
            rec = {}
        if rec.get("hour") != hour:
            rec = {"hour": hour, "count": 0}
        rec["count"] += 1
        if rec["count"] > RATE_LIMIT_PER_HOUR:
            raise HTTPException(
                429,
                f"rate limit exceeded: {RATE_LIMIT_PER_HOUR} design submissions/hour per key; retry next hour",
            )
        json.dump(rec, open(path, "w"))
        jobs.commit()

    @web.post("/designs")
    async def create(request: Request):
        tok = auth(request)
        check_rate_limit(tok)
        body = await request.json()
        description = body.get("description")
        text = body.get("text")
        if not isinstance(description, str) or not (MIN_DESCRIPTION_CHARS <= len(description.strip()) <= MAX_DESCRIPTION_CHARS):
            raise HTTPException(400, f"description must be {MIN_DESCRIPTION_CHARS}-{MAX_DESCRIPTION_CHARS} characters")
        if not isinstance(text, str) or not (MIN_TEXT_CHARS <= len(text.strip()) <= MAX_TEXT_CHARS):
            raise HTTPException(400, f"text must be {MIN_TEXT_CHARS}-{MAX_TEXT_CHARS} characters")
        jid = "d-" + pysecrets.token_hex(5)
        os.makedirs(f"/jobs/{jid}", exist_ok=True)
        json.dump({"job_id": jid, "status": "queued", "submitted_at": int(__import__("time").time())}, open(f"/jobs/{jid}/status.json", "w"))
        jobs.commit()
        try:
            model_cls = modal.Cls.from_name("voice-design-dev" + APP_SUFFIX, "VoiceDesignModel")
            model_cls().generate.spawn(jid, description.strip(), text.strip())
        except Exception as e:
            import shutil
            shutil.rmtree(f"/jobs/{jid}", ignore_errors=True)
            jobs.commit()
            print("spawn failed:", repr(e), flush=True)
            raise HTTPException(503, "generation service unavailable, please retry")
        return {"job_id": jid, "status": "queued"}

    @web.get("/designs/{jid}")
    async def get(jid: str, request: Request):
        auth(request); jid_ok(jid)
        jobs.reload()
        p = f"/jobs/{jid}/status.json"
        if not os.path.exists(p):
            raise HTTPException(404, "unknown job")
        return json.load(open(p))

    @web.get("/designs/{jid}/audio")
    async def audio(jid: str, request: Request):
        auth(request); jid_ok(jid)
        jobs.reload()
        sp = f"/jobs/{jid}/status.json"
        if not os.path.exists(sp):
            raise HTTPException(404, "unknown job")
        st = json.load(open(sp))
        if st.get("status") != "ready":
            raise HTTPException(409, f"job is not ready (status={st.get('status')})")
        ap = f"/jobs/{jid}/audio.wav"
        if not os.path.exists(ap):
            raise HTTPException(404, "audio missing")
        return Response(open(ap, "rb").read(), media_type="audio/wav")

    return web
