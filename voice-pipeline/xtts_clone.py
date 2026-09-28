"""XTTS v2 instant voice cloning: ~6 seconds of reference audio -> synthesizable voice.
No training. Scale-to-zero GPU: model loads once per container, stays warm for repeated syntheses.

API (Bearer XTTS_ADMIN_SECRET):
  POST /voices                    multipart {name?, language?, reference} (audio file)
                                  -> {voice_id, status: "ready", name, language}
  POST /voices/{id}/synthesize    {text, language?} -> audio/wav
  GET  /voices/{id}               -> {voice_id, status, name, language, created_at}
  GET  /voices/{id}/reference     -> audio/* (original reference audio)
  DELETE /voices/{id}             -> {deleted: true}

Deploy: modal deploy voice-pipeline/xtts_clone.py
Secret `xtts-clone` must hold XTTS_ADMIN_SECRET.
"""
import os as _os
import time
import modal

APP_SUFFIX = _os.environ.get("XTTS_APP_SUFFIX", "")
app = modal.App("xtts-clone" + APP_SUFFIX)

# XTTS v2 needs espeak-ng for phonemization, torch for inference, and the TTS package.
# Pin transformers<5 because TTS 0.22.0 is incompatible with transformers 5.x.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "git", "wget", "libsndfile1")
    .pip_install(
        "torch==2.1.2", "torchaudio==2.1.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("transformers==4.40.2", "numpy<2")
    .pip_install("TTS", "soundfile")
    .add_local_file(_os.path.join(_os.path.dirname(__file__), "xtts_download.py"), remote_path="/tmp/xtts_download.py", copy=True)
    .run_commands("python3 /tmp/xtts_download.py")
)

web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi==0.109.0", "python-multipart")

voices = modal.Volume.from_name("xtts-clone-voices", create_if_missing=True)
secret = modal.Secret.from_name("xtts-clone")

# Synthesis config
MIN_REF_SECONDS = 3.0   # below this, quality degrades
MAX_REF_SECONDS = 30.0  # XTTS v2 supports up to ~30s well
MAX_TEXT_CHARS = 5000
DEFAULT_LANG = "en"
SUPPORTED_LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"}

ID_RE = _os


@app.cls(
    image=gpu_image,
    gpu="A10G",
    timeout=600,
    volumes={"/voices": voices},
    scaledown_window=120,  # keep warm for 2 min after last request
)
class XttsSynthesizer:
    """Loads XTTS v2 once per container. Speaker embeddings are cached in memory after first
    use per voice so repeated syntheses of the same voice are fast."""

    @modal.enter()
    def load(self):
        from TTS.api import TTS
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        # XTTS v2 multilingual model; downloaded once on first container startup
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.container_load_seconds = round(time.time() - t0, 2)
        print(f"XTTS v2 loaded on {self.device} in {self.container_load_seconds}s", flush=True)

    @modal.method()
    def clone(self, voice_id: str, reference_bytes: bytes, name: str, language: str) -> dict:
        """Store reference audio, validate duration, return metadata."""
        import json
        import os
        import soundfile as sf

        d = f"/voices/{voice_id}"
        os.makedirs(d, exist_ok=True)

        ref_path = f"{d}/reference.wav"
        with open(ref_path, "wb") as f:
            f.write(reference_bytes)

        # Validate audio
        try:
            audio, sr = sf.read(ref_path)
        except Exception as e:
            raise RuntimeError(f"Invalid audio file: {e}")

        dur = len(audio) / sr
        if dur < MIN_REF_SECONDS:
            raise RuntimeError(f"Reference audio too short ({dur:.1f}s < {MIN_REF_SECONDS}s)")
        if dur > MAX_REF_SECONDS:
            raise RuntimeError(f"Reference audio too long ({dur:.1f}s > {MAX_REF_SECONDS}s)")

        # Warm-up: compute speaker embedding once by synthesizing a short phrase.
        # This caches the embedding in the model's speaker_manager for subsequent calls.
        # We discard the output audio.
        dummy_text = "Hello."
        warm_lang = language if language in SUPPORTED_LANGS else DEFAULT_LANG
        _ = self.tts.tts(text=dummy_text, speaker_wav=ref_path, language=warm_lang)

        meta = {
            "voice_id": voice_id,
            "status": "ready",
            "name": name,
            "language": language,
            "reference_seconds": round(dur, 2),
            "sample_rate": sr,
            "created_at": int(time.time()),
            "container_load_seconds": self.container_load_seconds,
        }
        json.dump(meta, open(f"{d}/meta.json", "w"))
        voices.commit()
        return meta

    @modal.method()
    def synthesize(self, voice_id: str, text: str, language: str) -> bytes:
        """Synthesize text using the cached speaker embedding for this voice."""
        import io
        import os

        import soundfile as sf

        d = f"/voices/{voice_id}"
        ref_path = f"{d}/reference.wav"
        voices.reload()
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Voice {voice_id} not found")

        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"Text too long ({len(text)} > {MAX_TEXT_CHARS})")

        lang = language if language in SUPPORTED_LANGS else DEFAULT_LANG

        t0 = time.time()
        wav = self.tts.tts(text=text, speaker_wav=ref_path, language=lang)
        gen_s = time.time() - t0

        # Convert to standard WAV (mono, 24kHz, PCM16) — same as Piper output
        buf = io.BytesIO()
        sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        print(f"synthesize voice={voice_id} chars={len(text)} lang={lang} gen={gen_s:.2f}s", flush=True)
        return wav_bytes

    @modal.method()
    def get_meta(self, voice_id: str) -> dict:
        import json
        import os

        d = f"/voices/{voice_id}"
        path = f"{d}/meta.json"
        voices.reload()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Voice {voice_id} not found")
        return json.load(open(path))

    @modal.method()
    def delete(self, voice_id: str) -> dict:
        import shutil
        import os

        d = f"/voices/{voice_id}"
        voices.reload()
        existed = os.path.exists(d)
        if existed:
            shutil.rmtree(d)
            voices.commit()
        return {"deleted": existed}


@app.function(image=web_image, secrets=[secret], volumes={"/voices": voices}, timeout=120)
@modal.asgi_app()
def api():
    import hashlib
    import hmac
    import io
    import os
    import re

    from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
    from fastapi.responses import Response

    web = FastAPI()
    ID_RE = re.compile(r"^x-[0-9a-f]{12}$")

    def auth(request: Request) -> str:
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not tok:
            raise HTTPException(401, "unauthorized")
        ok = hmac.compare_digest(tok.encode(), os.environ.get("XTTS_ADMIN_SECRET", "").encode())
        if not ok:
            raise HTTPException(401, "unauthorized")
        return tok

    def vid_ok(vid: str):
        if not ID_RE.match(vid):
            raise HTTPException(400, "bad voice id")

    def _synth_cls():
        return modal.Cls.from_name("xtts-clone" + APP_SUFFIX, "XttsSynthesizer")()

    @web.post("/voices")
    async def create_voice(
        request: Request,
        name: str = Form(""),
        language: str = Form(DEFAULT_LANG),
        reference: UploadFile = File(...),
    ):
        auth(request)
        if language not in SUPPORTED_LANGS:
            raise HTTPException(400, f"unsupported language: {language}. Supported: {', '.join(sorted(SUPPORTED_LANGS))}")

        ref_bytes = await reference.read()
        if not ref_bytes or len(ref_bytes) < 1024:
            raise HTTPException(400, "reference audio is empty or too small")

        # Generate deterministic id from hash of reference bytes so re-uploading the same
        # reference gives the same voice_id (idempotent)
        vid = "x-" + hashlib.sha256(ref_bytes).hexdigest()[:12]

        # Check if already exists (avoid re-computing embedding)
        voices.reload()
        if os.path.exists(f"/voices/{vid}/meta.json"):
            import json
            meta = json.load(open(f"/voices/{vid}/meta.json"))
            return meta

        try:
            meta = _synth_cls().clone.remote(vid, ref_bytes, name or vid, language)
            return meta
        except Exception as e:
            print("clone failed:", repr(e), flush=True)
            raise HTTPException(503, "clone service unavailable, please retry")

    @web.get("/voices/{vid}")
    async def get_voice(vid: str, request: Request):
        auth(request)
        vid_ok(vid)
        try:
            meta = _synth_cls().get_meta.remote(vid)
            return meta
        except FileNotFoundError:
            raise HTTPException(404, "unknown voice")

    @web.get("/voices/{vid}/reference")
    async def get_reference(vid: str, request: Request):
        auth(request)
        vid_ok(vid)
        voices.reload()
        ref_path = f"/voices/{vid}/reference.wav"
        if not os.path.exists(ref_path):
            raise HTTPException(404, "unknown voice")
        with open(ref_path, "rb") as f:
            data = f.read()
        return Response(data, media_type="audio/wav")

    @web.post("/voices/{vid}/synthesize")
    async def synthesize(vid: str, request: Request):
        auth(request)
        vid_ok(vid)
        body = await request.json()
        text = str(body.get("text", ""))
        language = str(body.get("language", DEFAULT_LANG))
        if not text:
            raise HTTPException(400, "text is required")
        if language not in SUPPORTED_LANGS:
            raise HTTPException(400, f"unsupported language: {language}")

        try:
            wav = _synth_cls().synthesize.remote(vid, text, language)
            return Response(wav, media_type="audio/wav")
        except FileNotFoundError:
            raise HTTPException(404, "unknown voice")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @web.delete("/voices/{vid}")
    async def delete_voice(vid: str, request: Request):
        auth(request)
        vid_ok(vid)
        return _synth_cls().delete.remote(vid)

    return web
