"""Fish Speech v1.5 streaming TTS with zero-shot voice cloning.
Loads S2-Pro model on A10G GPU.

API (Bearer FISH_SPEECH_SECRET):
  POST /v1/tts
    Body: {text, references?: [{audio: base64, text: string}], streaming?: true}
    -> audio/wav
  GET /v1/health -> {"status": "ok"}

Deploy: modal deploy voice-pipeline/fish_speech_dev.py
"""
import os
import time
import modal
import base64
import sys
import importlib.util

APP_SUFFIX = os.environ.get("FISH_SPEECH_APP_SUFFIX", "")
app = modal.App("fish-speech-dev" + APP_SUFFIX)


def download_checkpoint():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="fishaudio/fish-speech-1.5",
        local_dir="/checkpoints/s2-pro",
        local_dir_use_symlinks=False,
    )


gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ffmpeg", "espeak-ng", "portaudio19-dev", "libportaudio2")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.45.2", "huggingface_hub", "loguru", "omegaconf",
        "hydra-core", "einops", "einx", "vector_quantize_pytorch",
        "loralib", "ormsgpack", "cachetools", "tiktoken",
        "numpy<2", "soundfile", "pydub", "librosa", "scipy",
        "sentencepiece", "natsort", "rich", "gradio>5.0.0",
        "kui>=1.6.0", "uvicorn>=0.30.0", "pydantic==2.9.2",
        "baize", "aiohttp", "typer", "fsspec", "datasets==2.18.0",
        "lightning>=2.1.0", "tensorboard>=2.14.1", "wandb>=0.15.11",
        "grpcio>=1.58.0", "pyrootutils>=1.0.4", "resampy>=0.4.3",
        "zstandard>=0.22.0", "funasr==1.1.5", "modelscope==1.17.1",
        "opencc-python-reimplemented==0.1.7", "silero-vad",
        "faster_whisper", "onnxruntime", "numba", "click", "pooch",
    )
    .run_commands(
        # Remove any stale fish-speech install from earlier builds
        "pip uninstall -y fish-speech 2>/dev/null || true",
        "git clone --depth 1 --branch v1.5.1 https://github.com/fishaudio/fish-speech.git /fish-speech",
        # Touch .project-root so pyrootutils works
        "touch /fish-speech/.project-root",
        # Install fish-speech in editable mode (no deps — already installed above)
        "cd /fish-speech && pip install -e . --no-deps --force-reinstall",
    )
    .run_function(download_checkpoint)
)

web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.109.0", "python-multipart", "numpy", "soundfile"
)

references_vol = modal.Volume.from_name("fish-speech-refs", create_if_missing=True)
secret = modal.Secret.from_name("fish-speech-dev")

SAMPLE_RATE = 44100
AMPLITUDE = 32768


def _load_module_from_path(name, path):
    """Load a module from a specific file path, bypassing __init__.py issues."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@app.cls(
    image=gpu_image,
    gpu="A10G",
    timeout=1800,
    volumes={"/refs": references_vol},
    scaledown_window=120,
)
class FishSpeechEngine:
    """Loads Fish Speech S2-Pro once per container."""

    @modal.enter()
    def load(self):
        import torch
        import sys
        sys.path.insert(0, "/fish-speech")
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.vqgan.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = torch.bfloat16
        ckpt_dir = "/checkpoints/s2-pro"

        t0 = time.time()

        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=ckpt_dir,
            device=self.device,
            precision=precision,
            compile=False,
        )

        self.decoder_model = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path=os.path.join(ckpt_dir, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
            device=self.device,
        )

        self.engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=precision,
            compile=False,
        )

        self.container_load_seconds = round(time.time() - t0, 2)
        print(
            f"Fish Speech S2-Pro loaded on {self.device} in {self.container_load_seconds}s",
            flush=True,
        )

    @modal.method()
    def synthesize(self, text: str, references_b64: list[dict], streaming: bool) -> bytes:
        import io
        import numpy as np
        import soundfile as sf
        import sys
        sys.path.insert(0, "/fish-speech")
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

        refs = []
        for r in references_b64:
            audio_bytes = base64.b64decode(r["audio"])
            refs.append(ServeReferenceAudio(audio=audio_bytes, text=r.get("text", "")))

        req = ServeTTSRequest(
            text=text,
            references=refs,
            streaming=streaming,
            format="wav",
        )

        t0 = time.time()
        chunks = []
        for result in self.engine.inference(req):
            if result.code == "segment" and isinstance(result.audio, tuple):
                pcm = (result.audio[1] * AMPLITUDE).astype(np.int16).tobytes()
                chunks.append(pcm)
            elif result.code == "final" and isinstance(result.audio, tuple):
                pcm = (result.audio[1] * AMPLITUDE).astype(np.int16).tobytes()
                chunks.append(pcm)
                break

        gen_s = time.time() - t0
        print(f"synthesize text={len(text)}chars refs={len(refs)} streaming={streaming} gen={gen_s:.2f}s", flush=True)

        all_pcm = b"".join(chunks)
        if not all_pcm:
            raise RuntimeError("No audio generated")

        buf = io.BytesIO()
        arr = np.frombuffer(all_pcm, dtype=np.int16)
        sf.write(buf, arr, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @modal.method()
    def synthesize_streaming_chunks(self, text: str, references_b64: list[dict]) -> list[bytes]:
        import numpy as np
        import sys
        sys.path.insert(0, "/fish-speech")
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

        refs = []
        for r in references_b64:
            audio_bytes = base64.b64decode(r["audio"])
            refs.append(ServeReferenceAudio(audio=audio_bytes, text=r.get("text", "")))

        req = ServeTTSRequest(
            text=text,
            references=refs,
            streaming=True,
            format="wav",
        )

        chunks = []
        for result in self.engine.inference(req):
            if result.code == "segment" and isinstance(result.audio, tuple):
                pcm = (result.audio[1] * AMPLITUDE).astype(np.int16).tobytes()
                chunks.append(pcm)
            elif result.code == "final" and isinstance(result.audio, tuple):
                pcm = (result.audio[1] * AMPLITUDE).astype(np.int16).tobytes()
                chunks.append(pcm)
                break

        return chunks


@app.function(image=web_image, secrets=[secret], timeout=600)
@modal.asgi_app()
def api():
    import hmac
    import os
    import numpy as np
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response, StreamingResponse

    web = FastAPI()

    def auth(request: Request) -> str:
        tok = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not tok:
            raise HTTPException(401, "unauthorized")
        ok = hmac.compare_digest(tok.encode(), os.environ.get("FISH_SPEECH_SECRET", "").encode())
        if not ok:
            raise HTTPException(401, "unauthorized")
        return tok

    def _engine_cls():
        return modal.Cls.from_name("fish-speech-dev" + APP_SUFFIX, "FishSpeechEngine")()

    @web.get("/v1/health")
    async def health():
        return {"status": "ok"}

    @web.post("/v1/tts")
    async def tts(request: Request):
        auth(request)
        body = await request.json()
        text = str(body.get("text", ""))
        if not text:
            raise HTTPException(400, "text is required")

        refs_raw = body.get("references", [])
        if not isinstance(refs_raw, list):
            raise HTTPException(400, "references must be a list")

        refs = []
        for r in refs_raw:
            if not isinstance(r, dict) or "audio" not in r:
                raise HTTPException(400, "each reference must have 'audio' (base64)")
            refs.append({"audio": str(r["audio"]), "text": str(r.get("text", ""))})

        streaming = bool(body.get("streaming", False))

        try:
            if streaming:
                chunks = _engine_cls().synthesize_streaming_chunks.remote(text, refs)

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
                wav = _engine_cls().synthesize.remote(text, refs, False)
                return Response(wav, media_type="audio/wav")
        except Exception as e:
            print("synthesis failed:", repr(e), flush=True)
            raise HTTPException(503, f"synthesis failed: {e}")

    return web
