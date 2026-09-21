"""
Audio isolation (source separation / denoising) worker - platform-independent build in the
style of ../worker-piper-fly/server.py: plain FastAPI + uvicorn, configured by env vars, no
Modal. Runs Demucs (HTDemucs) to split an uploaded clip into stems and returns the "vocals"
stem as an isolated speech track.

Backend: PyTorch Demucs (the `demucs` package), NOT an ONNX export. An ONNX export of HTDemucs
was evaluated first per the task brief and deliberately not pursued: HTDemucs is a hybrid
time+spectrogram model (STFT/iSTFT layers, complex-valued ops, an LSTM bottleneck, learned
"transformer" cross-domain attention in some variants) - none of which export cleanly through
torch.onnx without custom op surgery, and the published ~1.3x CPU speedup figure is for the
plain conv/LSTM stack, not something to be verified from scratch here. Given the task's own
guidance ("don't burn excessive time on ONNX export... correctness and a working verified
pipeline matter more than which backend"), we went straight to the documented, supported
PyTorch path. See README.md "Backend choice" for the full writeup.

Endpoint: POST /v1/isolate
  multipart/form-data: file=<audio>, stem?=vocals (default; one of demucs' 4 htdemucs stems:
  vocals, drums, bass, other)
  Auth: Authorization: Bearer <AUTH_TOKEN | gateway session token>, same shape as
  worker-piper-fly (see verify_session_claims below and README.md "Auth").
  Response: audio/wav body of the isolated stem at the model's native 44.1kHz, mono-summed.
  Errors: JSON {"error": "..."} - see README.md "Errors" for the exact status codes.

Other env: MODEL_NAME (default htdemucs), MAX_UPLOAD_BYTES (default 25MB), MAX_DURATION_S
(default 120s), MAX_CONNECTIONS (default 2 - Demucs is heavy per-request; this is much lower
than worker-piper-fly's default), TORCH_THREADS (default: os.cpu_count()), PORT (8080),
USAGE_REPORT_URL/USAGE_REPORT_SECRET (usage metering hook, same shape as worker-piper-fly's
report_usage - fires after each successful /v1/isolate with {id, audio_seconds, engine:
"denoise"}; unset USAGE_REPORT_SECRET means usage simply isn't reported).
"""
import base64
import hashlib
import hmac
import io
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response

MODEL_NAME = os.environ.get("MODEL_NAME", "htdemucs")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_DURATION_S = float(os.environ.get("MAX_DURATION_S", "120"))
MIN_DURATION_S = float(os.environ.get("MIN_DURATION_S", "0.1"))
MAX_CONNECTIONS = int(os.environ.get("MAX_CONNECTIONS", "2"))
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", str(os.cpu_count() or 2)))
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
SESSION_SECRET = os.environ.get("SESSION_SECRET")
# Usage/billing hook, same endpoint+auth shape as worker-piper-fly/server.py's report_usage -
# fire-and-forget POST to the gateway after each successful separation, tagged engine "denoise"
# with clip duration (not TTS chars - see gateway/keys.js recordUsageById's audioSeconds param).
# Unset USAGE_REPORT_SECRET => usage is simply not reported (no error), same as piper's worker
# when it's run standalone/locally without the gateway wired up.
USAGE_REPORT_URL = os.environ.get("USAGE_REPORT_URL", "https://api.readaloudai.org/admin/usage/report")
USAGE_REPORT_SECRET = os.environ.get("USAGE_REPORT_SECRET")
VALID_STEMS = ("vocals", "drums", "bass", "other")
# Formats soundfile can read without ffmpeg. Mirrors worker-piper-fly's "reject early, clear
# message" style rather than letting a decode error surface as a 500 deep in the model call.
ALLOWED_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif")

if not (AUTH_TOKEN or SESSION_SECRET):
    raise SystemExit("Set AUTH_TOKEN and/or SESSION_SECRET - refusing to start an unauthenticated worker")

torch.set_num_threads(TORCH_THREADS)


def verify_session_claims(token: str):
    """Identical HMAC session-token check to worker-piper-fly/server.py's verify_session_claims -
    kept as a self-contained copy (not imported across worker dirs) so this worker can be built
    and deployed independently. If the two ever drift, worker-piper-fly's is authoritative."""
    if not SESSION_SECRET or not token:
        return (None, None)
    try:
        payload_b64, sig_b64 = token.split(".")
    except ValueError:
        return (None, None)
    expected = base64.urlsafe_b64encode(
        hmac.new(SESSION_SECRET.encode(), payload_b64.encode(), hashlib.sha256).digest()
    ).decode().rstrip("=")
    if not hmac.compare_digest(sig_b64, expected):
        return (None, None)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    except Exception:
        return (None, None)
    if "id" not in payload or "exp" not in payload or time.time() * 1000 > payload["exp"]:
        return (None, None)
    uid = payload.get("uid")
    return payload["id"], (uid if isinstance(uid, str) and uid else None)


class ValidationError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


def check_auth(request: Request):
    auth = request.headers.get("authorization", "")
    presented = auth[len("Bearer "):] if auth.startswith("Bearer ") else ""
    key_id, uid = verify_session_claims(presented)
    static_ok = bool(AUTH_TOKEN) and bool(presented) and hmac.compare_digest(presented.encode(), AUTH_TOKEN.encode())
    if key_id is None and not static_ok:
        raise ValidationError(401, "unauthorized")
    return key_id, uid


def validate_stem(stem: str) -> str:
    """Raises ValidationError(400) for an unknown stem name; returns it unchanged otherwise."""
    if stem is None:
        return "vocals"
    if stem not in VALID_STEMS:
        raise ValidationError(400, f"stem must be one of {list(VALID_STEMS)}")
    return stem


def validate_filename(filename: str) -> None:
    if not filename or "." not in filename:
        raise ValidationError(415, f"unsupported or missing file extension (allowed: {list(ALLOWED_EXTENSIONS)})")
    ext = filename[filename.rfind("."):].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(415, f"unsupported file extension {ext!r} (allowed: {list(ALLOWED_EXTENSIONS)})")


def validate_size(nbytes: int) -> None:
    if nbytes == 0:
        raise ValidationError(400, "empty file")
    if nbytes > MAX_UPLOAD_BYTES:
        raise ValidationError(413, f"file too large ({nbytes} bytes, max {MAX_UPLOAD_BYTES})")


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Returns (float32 samples[frames, channels], sample_rate). Raises ValidationError(400) for
    anything soundfile can't parse (corrupt file, unsupported codec inside an allowed container)."""
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - libsndfile raises various things for bad input
        raise ValidationError(400, f"could not decode audio: {e}")
    if audio.shape[0] == 0:
        raise ValidationError(400, "decoded audio has zero frames")
    return audio, sr


def validate_duration(nframes: int, sr: int) -> float:
    duration_s = nframes / sr
    if duration_s < MIN_DURATION_S:
        raise ValidationError(400, f"clip too short ({duration_s:.3f}s, min {MIN_DURATION_S}s)")
    if duration_s > MAX_DURATION_S:
        raise ValidationError(413, f"clip too long ({duration_s:.1f}s, max {MAX_DURATION_S}s)")
    return duration_s


class Separator:
    """Lazily loads the Demucs model on first use (downloads pretrained weights on the first
    call if not already cached under ~/.cache/torch), then reuses it. Loading is slow (seconds)
    and the model is not thread-safe for concurrent forward passes on the same instance in a
    way we want to rely on, so calls are serialized through a lock - matches worker-piper-fly's
    _phonemize_lock pattern for espeak's global state."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from demucs.pretrained import get_model
                    m = get_model(self.model_name)
                    m.eval()
                    self._model = m
        return self._model

    def separate(self, audio: np.ndarray, sr: int, stem: str) -> tuple[np.ndarray, int]:
        """audio: [frames, channels] float32 at any sample rate. Returns (mono_float32, model_sr)
        for the requested stem."""
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        model = self.model
        with self._lock:
            wav = torch.from_numpy(audio.T).float()  # [channels, frames]
            wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
            ref = wav.mean(0)
            std = ref.std() + 1e-8
            wav = (wav - ref.mean()) / std
            with torch.no_grad():
                sources = apply_model(model, wav[None], device="cpu", progress=False)[0]
            sources = sources * std + ref.mean()
            idx = model.sources.index(stem)
            out = sources[idx]  # [channels, frames]
            mono = out.mean(0).cpu().numpy().astype(np.float32)
            return mono, model.samplerate


def report_usage(key_id: str, audio_seconds: float):
    """Best-effort, off the request path (runs in the thread pool): a failed report must never
    affect the caller's response. Same endpoint/payload shape the piper worker uses, engine
    "denoise" and audio_seconds instead of chars - see gateway/keys.js applyUsage."""
    if not key_id or not audio_seconds or not USAGE_REPORT_SECRET:
        return
    try:
        req = urllib.request.Request(
            USAGE_REPORT_URL,
            data=json.dumps({"id": key_id, "audio_seconds": audio_seconds, "engine": "denoise"}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {USAGE_REPORT_SECRET}"},
            method="POST")
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:  # noqa: BLE001
        print(f"usage report failed for key {key_id}: {e}")


separator = Separator(MODEL_NAME)
pool = ThreadPoolExecutor(max_workers=max(1, MAX_CONNECTIONS))
app = FastAPI()
_active = 0
_active_lock = threading.Lock()


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME, "device": "cpu", "active": _active, "max": MAX_CONNECTIONS}


def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.post("/v1/isolate")
async def isolate(request: Request, file: UploadFile = File(...), stem: str = Form(default="vocals")):
    global _active
    try:
        key_id, _uid = check_auth(request)
    except ValidationError as e:
        return _err(e.status, e.message)

    try:
        stem = validate_stem(stem)
        validate_filename(file.filename)
    except ValidationError as e:
        return _err(e.status, e.message)

    data = await file.read()
    try:
        validate_size(len(data))
    except ValidationError as e:
        return _err(e.status, e.message)

    try:
        audio, sr = decode_audio(data)
        duration_s = validate_duration(audio.shape[0], sr)
    except ValidationError as e:
        return _err(e.status, e.message)

    with _active_lock:
        if _active >= MAX_CONNECTIONS:
            return _err(503, "at capacity, retry shortly")
        _active += 1
    try:
        loop_result = await _run_separation(audio, sr, stem)
    except ValidationError as e:
        return _err(e.status, e.message)
    except Exception as e:  # noqa: BLE001 - report to client, keep the process alive
        return _err(500, f"separation failed: {e}")
    finally:
        with _active_lock:
            _active -= 1

    mono, out_sr = loop_result
    pool.submit(report_usage, key_id, duration_s)
    buf = io.BytesIO()
    sf.write(buf, mono, out_sr, format="WAV", subtype="PCM_16")
    return Response(
        content=buf.getvalue(), media_type="audio/wav",
        headers={"X-Sample-Rate": str(out_sr), "X-Stem": stem, "Cache-Control": "no-store"},
    )


async def _run_separation(audio: np.ndarray, sr: int, stem: str):
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(pool, separator.separate, audio, sr, stem)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")
