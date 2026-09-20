"""worker-stt-prod: BATCH speech-to-text API (faster-whisper large-v3-turbo) on Modal L4.

Intended production app name: `realtime-stt-worker`. Do NOT deploy under that name from a dev machine;
for testing:  STT_APP_NAME=stt-prod-test STT_SECRET=stt-prod-test-secret modal deploy app.py   (then `modal app stop stt-prod-test`).

  POST /v1/stt?format=&language=&word_timestamps=
      body: raw audio bytes (Content-Type anything but multipart)  OR  multipart/form-data with a `file` part
      (format/language/word_timestamps may also be form fields; query string wins).
      format: auto (default; container sniffed) | wav | flac | mp3 | ogg | m4a | mulaw_8000 | alaw_8000 | pcm_16000
              (raw formats are headerless: G.711 u-law / A-law 8 kHz mono, or signed 16-bit LE PCM 16 kHz mono)
      language: auto (default, Whisper detection) | ISO-639-1 code such as en
      -> {text, language, language_probability, duration, words:[{word,start,end}], segments:[{id,start,end,text}]}
  GET /health

Limits: 200 MB body, 3 h of audio, REQUEST_TIMEOUT_S wall clock per request (queueing + decode + inference).
Auth: `Authorization: Bearer <session token>` minted by gateway/keys.js createSessionToken (HMAC-SHA256, same scheme as
worker-piper-fly/server.py verify_session_token, secret MODAL_SESSION_SECRET). Verified BEFORE the body is read.
Usage: after a successful response, best-effort POST {id, audio_seconds, engine:"stt"} to USAGE_REPORT_URL with
Bearer MODAL_USAGE_REPORT_SECRET (gateway /admin/usage/report). Billed audio = decoded duration, not speech-only.
NOTE: Modal web endpoints answer requests running > ~150 s with a 303 redirect the client must follow (curl -L).
"""
import os
import modal

APP_NAME = os.environ.get("STT_APP_NAME", "realtime-stt-worker")
MODEL = "large-v3-turbo"
SECRET_NAME = os.environ.get("STT_SECRET", "realtime-stt-secrets")

MAX_BYTES = 200 * 1024 * 1024
MAX_AUDIO_SECONDS = 3 * 3600
REQUEST_TIMEOUT_S = 900
BATCH_SIZE = 16


def _bake():
    from faster_whisper.utils import download_model
    download_model(MODEL, cache_dir="/models")


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg")
    .pip_install("faster-whisper==1.1.1", "numpy<2", "fastapi[standard]", "python-multipart", "requests")
    .run_function(_bake)
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu="L4",
    min_containers=0,
    max_containers=int(os.environ.get("STT_MAX_CONTAINERS", "4")),   # hard cap on GPU spend
    scaledown_window=120,
    timeout=REQUEST_TIMEOUT_S + 120,
    memory=16384,                     # 3 h of 16 kHz audio decodes to ~350 MB int16 / ~700 MB float32
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
@modal.concurrent(max_inputs=4)      # upload/ffmpeg overlap; GPU inference is serialized by a lock
class STT:
    @modal.enter()
    def load(self):
        import threading
        import numpy as np
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        self.model = WhisperModel(MODEL, device="cuda", compute_type="float16", download_root="/models")
        self.pipe = BatchedInferencePipeline(model=self.model)
        self.lock = threading.Lock()
        segs, _ = self.pipe.transcribe(np.zeros(16000 * 5, dtype="float32"), language="en", batch_size=BATCH_SIZE)
        list(segs)  # warm CUDA kernels

    @modal.asgi_app()
    def web(self):
        import asyncio, base64, hashlib, hmac, json, subprocess, tempfile, time, threading, urllib.request
        import numpy as np
        from fastapi import BackgroundTasks, FastAPI, Request
        from fastapi.responses import JSONResponse
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        from starlette.formparsers import MultiPartParser
        from starlette.datastructures import UploadFile

        SESSION_SECRET = os.environ.get("MODAL_SESSION_SECRET", "")
        USAGE_URL = os.environ.get("USAGE_REPORT_URL", "")
        USAGE_SECRET = os.environ.get("MODAL_USAGE_REPORT_SECRET", "")
        pipe, lock = self.pipe, self.lock
        api = FastAPI(title="realtime-stt-worker", docs_url=None, redoc_url=None, openapi_url=None)

        class ApiError(Exception):
            def __init__(self, status, message):
                self.status, self.message = status, message

        @api.exception_handler(ApiError)
        async def _err(_req, e):
            return JSONResponse({"error": e.message}, status_code=e.status)

        def verify_session_token(token):
            """Mirrors gateway/keys.js verifySessionToken exactly (same as worker-piper-fly/server.py)."""
            if not SESSION_SECRET or not token:
                return None
            try:
                payload_b64, sig_b64 = token.split(".")
            except ValueError:
                return None
            expected = base64.urlsafe_b64encode(
                hmac.new(SESSION_SECRET.encode(), payload_b64.encode(), hashlib.sha256).digest()).decode().rstrip("=")
            if not hmac.compare_digest(sig_b64, expected):
                return None
            try:
                p = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
            except Exception:
                return None
            if "id" not in p or "exp" not in p or time.time() * 1000 > p["exp"]:
                return None
            return p["id"]

        def report_usage(key_id, seconds, req_id):
            print(f"usage id={key_id} audio_seconds={seconds:.2f} req={req_id}")
            if not (USAGE_URL and USAGE_SECRET and key_id and seconds > 0):
                return
            body = json.dumps({"id": key_id, "audio_seconds": round(seconds, 3), "engine": "stt"}).encode()
            for attempt in range(3):   # lost reports are lost revenue: retry, but never affect the caller
                try:
                    req = urllib.request.Request(USAGE_URL, data=body, method="POST", headers={
                        "Authorization": f"Bearer {USAGE_SECRET}", "Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=5).read()
                    return
                except Exception as e:
                    print(f"usage report attempt {attempt + 1} failed: {type(e).__name__}")
                    time.sleep(1 + attempt * 2)

        # ---- request parsing ---------------------------------------------------------------------
        RAW = {   # format -> ffmpeg input args for headerless audio; value 2nd = bytes per second
            "mulaw_8000": (["-f", "mulaw", "-ar", "8000", "-ac", "1"], 8000),
            "alaw_8000": (["-f", "alaw", "-ar", "8000", "-ac", "1"], 8000),
            "pcm_16000": (["-f", "s16le", "-ar", "16000", "-ac", "1"], 32000),
        }
        CONTAINER_FORMATS = {"auto", "wav", "flac", "mp3", "ogg", "m4a"}
        # ffprobe format_name tokens we accept; blocks playlist/concat/etc demuxers (SSRF / local-file tricks)
        ALLOWED_DEMUXERS = {"wav", "flac", "mp3", "ogg", "mov", "mp4", "m4a", "matroska", "webm", "aac"}

        async def counted(stream):
            n = 0
            async for chunk in stream:
                n += len(chunk)
                if n > MAX_BYTES:
                    raise ApiError(413, f"request body exceeds {MAX_BYTES // (1024 * 1024)} MB limit")
                yield chunk

        async def read_body(request, tmp_path):
            """Streams the (raw or multipart) upload into tmp_path with a hard size cap. Returns extra form fields."""
            cl = request.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > MAX_BYTES + 65536:
                raise ApiError(413, f"request body exceeds {MAX_BYTES // (1024 * 1024)} MB limit")
            ctype = request.headers.get("content-type", "")
            if ctype.lower().startswith("multipart/form-data"):
                try:
                    form = await MultiPartParser(request.headers, counted(request.stream()), max_files=1, max_fields=10,
                                                 max_part_size=1024 * 1024).parse()
                except ApiError:
                    raise
                except Exception:
                    raise ApiError(400, "malformed multipart body")
                f = form.get("file")
                if not isinstance(f, UploadFile):
                    raise ApiError(400, "multipart body needs a `file` part")
                with open(tmp_path, "wb") as out:
                    while True:
                        chunk = await f.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                await f.close()
                return {k: v for k, v in form.items() if isinstance(v, str)}
            with open(tmp_path, "wb") as out:
                async for chunk in counted(request.stream()):
                    out.write(chunk)
            return {}

        def probe_and_decode(path, fmt, deadline):
            """-> float32 mono 16 kHz samples. Raises ApiError."""
            size = os.path.getsize(path)
            if size == 0:
                raise ApiError(400, "empty audio body")
            max_out = MAX_AUDIO_SECONDS * 32000   # bytes of s16le at 16 kHz
            if fmt in RAW:
                args, bps = RAW[fmt]
                if size / bps > MAX_AUDIO_SECONDS:
                    raise ApiError(413, f"audio longer than {MAX_AUDIO_SECONDS // 3600} hours")
                in_args = args
            else:
                try:
                    pr = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_entries",
                                         "format=format_name,duration", "-of", "json", path],
                                        capture_output=True, timeout=30)
                    info = json.loads(pr.stdout or b"{}").get("format") if pr.returncode == 0 else None
                except Exception:
                    info = None
                if not info or not (set(info.get("format_name", "").split(",")) & ALLOWED_DEMUXERS):
                    raise ApiError(400, "unsupported or undecodable audio (accepted: wav, flac, mp3, ogg, m4a/mp4, webm, "
                                        "or raw mulaw_8000 / alaw_8000 / pcm_16000 via ?format=)")
                try:
                    if float(info.get("duration", 0)) > MAX_AUDIO_SECONDS:
                        raise ApiError(413, f"audio longer than {MAX_AUDIO_SECONDS // 3600} hours")
                except (TypeError, ValueError):
                    pass
                in_args = []
            proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file", *in_args, "-i", path,
                                     "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunks, total = [], 0
            try:
                while True:
                    if time.monotonic() > deadline:
                        raise ApiError(504, "request timed out while decoding")
                    b = proc.stdout.read(1 << 20)
                    if not b:
                        break
                    total += len(b)
                    if total > max_out:
                        raise ApiError(413, f"audio longer than {MAX_AUDIO_SECONDS // 3600} hours")
                    chunks.append(b)
            finally:
                if proc.poll() is None:
                    proc.kill()
                err = proc.stderr.read(400)
                proc.wait()
                proc.stdout.close(); proc.stderr.close()
            if total == 0:
                raise ApiError(400, "could not decode audio: " + err.decode(errors="replace")[:200])
            pcm = np.frombuffer(b"".join(chunks)[: total // 2 * 2], dtype=np.int16)
            return pcm.astype(np.float32) / 32768.0

        def transcribe(audio, language, words, deadline):
            if not lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                raise ApiError(504, "request timed out waiting for a GPU slot")
            try:
                segs, info = pipe.transcribe(audio, language=language, beam_size=1, batch_size=BATCH_SIZE,
                                             word_timestamps=words, vad_filter=True)
                out = []
                for s in segs:   # lazily produced per batch: lets us enforce the deadline mid-file
                    if time.monotonic() > deadline:
                        raise ApiError(504, "request timed out during transcription")
                    out.append(s)
            finally:
                lock.release()
            return out, info

        @api.get("/health")
        def health():
            return {"status": "healthy", "model": MODEL}

        @api.post("/v1/stt")
        async def stt(request: Request, background: BackgroundTasks):
            t0 = time.monotonic()
            deadline = t0 + REQUEST_TIMEOUT_S
            h = request.headers.get("authorization", "")
            key_id = verify_session_token(h[7:] if h.lower().startswith("bearer ") else None)
            if not key_id:
                raise ApiError(401, "invalid or expired session token")
            qp = request.query_params
            fd, tmp = tempfile.mkstemp(prefix="stt-")
            os.close(fd)
            try:
                form = await read_body(request, tmp)
                t_read = time.monotonic()
                def param(name, default):
                    return (qp.get(name) or form.get(name) or default)
                fmt = str(param("format", "auto")).lower()
                if fmt not in CONTAINER_FORMATS and fmt not in RAW:
                    raise ApiError(400, "format must be one of auto, wav, flac, mp3, ogg, m4a, mulaw_8000, alaw_8000, pcm_16000")
                lang = str(param("language", "auto")).lower()
                if lang == "auto":
                    lang = None
                elif lang not in _LANGUAGE_CODES:
                    raise ApiError(400, "unsupported language code (use auto or an ISO-639-1 code like en)")
                wt = str(param("word_timestamps", "true")).lower()
                if wt not in ("true", "false", "1", "0"):
                    raise ApiError(400, "word_timestamps must be true or false")
                audio = await asyncio.to_thread(probe_and_decode, tmp, fmt, deadline)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            dur = len(audio) / 16000
            if dur > MAX_AUDIO_SECONDS:
                raise ApiError(413, f"audio longer than {MAX_AUDIO_SECONDS // 3600} hours")
            t1 = time.monotonic()
            segs, info = await asyncio.to_thread(transcribe, audio, lang, wt in ("true", "1"), deadline)
            words = [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
                     for s in segs for w in (s.words or [])]
            segments = [{"id": i, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
                        for i, s in enumerate(segs)]
            text = " ".join(x["text"] for x in segments if x["text"]).strip()
            print(f"req id={key_id} dur={dur:.1f}s read={t_read - t0:.2f}s decode={t1 - t_read:.2f}s infer={time.monotonic() - t1:.2f}s lang={info.language}")
            background.add_task(report_usage, key_id, dur, hex(id(segs))[2:])   # only on success, after the response is sent
            return {"text": text, "language": info.language, "language_probability": round(info.language_probability, 4),
                    "duration": round(dur, 3), "words": words, "segments": segments}

        return api
