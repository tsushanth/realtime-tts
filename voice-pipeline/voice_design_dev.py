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


@app.function(
    image=gpu_image, gpu="A10G", timeout=600, volumes={"/jobs": jobs},
    scaledown_window=60,  # scale to zero quickly after the last request - this is a dev/preview path, not a hot service
)
def generate(job_id: str, description: str, text: str):
    """Runs on GPU. Writes status.json + (on success) audio.wav into the job's volume dir."""
    import json
    import soundfile as sf
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    d = f"/jobs/{job_id}"

    def write_status(**kw):
        json.dump({"job_id": job_id, **kw}, open(f"{d}/status.json", "w"))
        jobs.commit()

    write_status(status="running", started_at=int(time.time()))
    t0 = time.time()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        load_s = time.time() - t0

        t1 = time.time()
        desc_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        generation = model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids)
        audio = generation.cpu().numpy().squeeze()
        gen_s = time.time() - t1

        sf.write(f"{d}/audio.wav", audio, model.config.sampling_rate)
        write_status(
            status="ready", finished_at=int(time.time()),
            model_load_seconds=round(load_s, 2), generation_seconds=round(gen_s, 2),
            audio_seconds=round(len(audio) / model.config.sampling_rate, 2),
            sampling_rate=model.config.sampling_rate,
        )
    except Exception as e:
        write_status(status="failed", finished_at=int(time.time()), error=str(e)[:500])
        raise


@app.function(image=web_image, secrets=[secret], volumes={"/jobs": jobs}, timeout=60)
@modal.asgi_app()
def api():
    import hmac
    import json
    import os
    import re
    import secrets as pysecrets

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response

    web = FastAPI()
    ID_RE = re.compile(JOB_ID_RE)

    def auth(request: Request):
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not tok or not hmac.compare_digest(tok.encode(), os.environ["VOICE_DESIGN_SECRET"].encode()):
            raise HTTPException(401, "unauthorized")

    def jid_ok(jid: str):
        if not ID_RE.match(jid):
            raise HTTPException(400, "bad job id")

    @web.post("/designs")
    async def create(request: Request):
        auth(request)
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
            modal.Function.from_name("voice-design-dev" + APP_SUFFIX, "generate").spawn(jid, description.strip(), text.strip())
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
