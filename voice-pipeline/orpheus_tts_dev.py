"""Orpheus TTS streaming prototype on Modal.
Uses the finetuned prod model (~3B) with vLLM for fast inference.
Streams audio chunks as LLM tokens arrive.

API (Bearer ORPHEUS_SECRET):
  POST /v1/tts
    Body: {text, voice?: "tara" | "leah" | "zoe" | "zac" | "jess" | "leo" | "mia" | "julia"}
    -> audio/pcm (24kHz, 16-bit mono) streamed in chunks
  GET /v1/voices
    -> {"voices": [...]}
  GET /v1/health
    -> {"status": "ok"}

Deploy: modal deploy voice-pipeline/orpheus_tts_dev.py
Secret `orpheus-tts-dev` must hold ORPHEUS_SECRET.
"""
import os
import time
import modal

APP_SUFFIX = os.environ.get("ORPHEUS_APP_SUFFIX", "")
app = modal.App("orpheus-tts-dev" + APP_SUFFIX)


def download_model():
    from huggingface_hub import snapshot_download
    print("Downloading Orpheus TTS model...", flush=True)
    snapshot_download(
        repo_id="canopylabs/orpheus-tts-0.1-finetune-prod",
        local_dir="/models/orpheus-tts",
        local_dir_use_symlinks=False,
    )
    print("Model download complete.", flush=True)


gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers", "huggingface_hub", "numpy", "soundfile",
        "snac", "vllm==0.7.3", "orpheus-speech",
    )
    .run_function(download_model)
)

web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.109.0", "python-multipart"
)

secret = modal.Secret.from_name("orpheus-tts-dev")

SAMPLE_RATE = 24000

# Available voices for the finetuned prod model
VOICES = ["tara", "leah", "zoe", "zac", "jess", "leo", "mia", "julia"]


@app.cls(
    image=gpu_image,
    gpu="A10G",
    timeout=1800,
    scaledown_window=300,
)
class OrpheusEngine:
    """Loads Orpheus TTS 3B model once per container."""

    @modal.enter()
    def load(self):
        import torch
        from orpheus_tts import OrpheusModel

        t0 = time.time()
        self.model = OrpheusModel(
            model_name="/models/orpheus-tts",
            dtype=torch.bfloat16,
            tokenizer="canopylabs/orpheus-3b-0.1-pretrained",
            max_model_len=2048,
        )
        self.container_load_seconds = round(time.time() - t0, 2)
        print(
            f"Orpheus TTS loaded in {self.container_load_seconds}s",
            flush=True,
        )

    @modal.method()
    def generate(self, text: str, voice: str = "tara"):
        """Generate speech synchronously, yielding PCM16 audio chunks.
        Returns a list of bytes for easy serialization across Modal."""
        import io
        import wave

        chunks = []
        for audio_chunk in self.model.generate_speech(
            prompt=text,
            voice=voice,
            temperature=0.6,
            top_p=0.8,
            repetition_penalty=1.3,
        ):
            chunks.append(audio_chunk)

        # Wrap all chunks in a WAV container
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            for chunk in chunks:
                wf.writeframes(chunk)
        return buf.getvalue()

    @modal.method()
    def generate_streaming(self, text: str, voice: str = "tara"):
        """Generate speech and return raw PCM chunks for true streaming."""
        chunks = []
        for audio_chunk in self.model.generate_speech(
            prompt=text,
            voice=voice,
            temperature=0.6,
            top_p=0.8,
            repetition_penalty=1.3,
        ):
            chunks.append(audio_chunk)
        return chunks


@app.function(image=web_image, secrets=[secret], timeout=120)
@modal.asgi_app()
def api():
    import hmac
    import os
    import io
    import wave
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response, StreamingResponse

    web = FastAPI()

    def auth(request: Request) -> str:
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not tok:
            raise HTTPException(401, "unauthorized")
        ok = hmac.compare_digest(tok.encode(), os.environ.get("ORPHEUS_SECRET", "").encode())
        if not ok:
            raise HTTPException(401, "unauthorized")
        return tok

    def _engine_cls():
        return modal.Cls.from_name("orpheus-tts-dev" + APP_SUFFIX, "OrpheusEngine")()

    @web.get("/v1/health")
    async def health():
        return {"status": "ok"}

    @web.get("/v1/voices")
    async def voices(request: Request):
        auth(request)
        return {"voices": VOICES}

    @web.post("/v1/tts")
    async def tts(request: Request):
        auth(request)
        body = await request.json()
        text = str(body.get("text", ""))
        voice = str(body.get("voice", "tara"))
        if not text:
            raise HTTPException(400, "text is required")
        if voice not in VOICES:
            raise HTTPException(400, f"voice must be one of: {', '.join(VOICES)}")

        streaming = bool(body.get("streaming", True))

        try:
            if streaming:
                chunks = _engine_cls().generate_streaming.remote(text, voice)

                def chunk_generator():
                    for pcm in chunks:
                        yield pcm

                return StreamingResponse(
                    chunk_generator(),
                    media_type="audio/pcm",
                    headers={
                        "X-Sample-Rate": str(SAMPLE_RATE),
                        "Content-Disposition": "attachment; filename=audio.pcm",
                    },
                )
            else:
                wav = _engine_cls().generate.remote(text, voice)
                return Response(wav, media_type="audio/wav")
        except Exception as e:
            print("synthesis failed:", repr(e), flush=True)
            raise HTTPException(503, f"synthesis failed: {e}")

    return web
