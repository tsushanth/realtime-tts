"""worker-stt: Whisper-based speech-to-text API on Modal (counterpart to ElevenLabs Scribe).

  POST /v1/stt            multipart: file=<audio>, format=(auto|wav|flac|mp3|ogg|mulaw), language=(auto|xx), word_timestamps=(true|false)
                          -> {text, language, words:[{word,start,end}], duration}
  WS   /v1/stt/stream     client: optional JSON {"encoding":"pcm16"|"mulaw","sample_rate":16000,"endpoint_ms":500,"partials":true}
                          then binary PCM16 (or mulaw 8k) chunks. server: {"type":"partial"|"final","text",...} + {"type":"usage",...}
  GET  /health

Auth: same short-lived HMAC-SHA256 session tokens as gateway/keys.js (see verify_session_token; mirrors
worker-piper-fly/server.py and worker-modal-readaloud/app.py). Header `Authorization: Bearer <token>` or `?token=`.
Usage: reported per audio second (POST USAGE_REPORT_URL {"id","audio_seconds","engine":"whisper-stt"}); the gateway's
/admin/usage/report currently only understands `chars`, so it needs a small extension before production (see DESIGN.md).

Deploy ONLY as an ephemeral test app:  STT_APP_NAME=stt-proto-test-<x> modal deploy app.py  (then `modal app stop`).
"""
import os
import modal

APP_NAME = os.environ.get("STT_APP_NAME", "stt-proto-test")
MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")
GPU = os.environ.get("STT_GPU", "T4")

def _bake():
    from faster_whisper.utils import download_model
    download_model(MODEL, cache_dir="/models")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg")
    .pip_install("faster-whisper==1.1.1", "numpy<2", "fastapi[standard]", "python-multipart", "websockets", "requests")
    .env({"STT_MODEL": MODEL})
    .run_function(_bake)
)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU,
    scaledown_window=60,          # never min_containers>0; scale to zero
    max_containers=2,
    timeout=900,
    secrets=[modal.Secret.from_name(os.environ.get("STT_SECRET", "stt-proto-test-secret"))],
)
@modal.concurrent(max_inputs=8)
class STT:
    @modal.enter()
    def load(self):
        from faster_whisper import WhisperModel
        import threading
        self.model = WhisperModel(MODEL, device="cuda", compute_type="float16", download_root="/models")
        self.lock = threading.Lock()   # one decode at a time per GPU container
        list(self.model.transcribe(__import__("numpy").zeros(16000, dtype="float32"), language="en")[0])

    @modal.asgi_app()
    def web(self):
        import asyncio, base64, hashlib, hmac, json, subprocess, time, urllib.request
        import numpy as np
        from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        SESSION_SECRET = os.environ.get("MODAL_SESSION_SECRET", "")
        USAGE_URL = os.environ.get("USAGE_REPORT_URL", "")
        USAGE_SECRET = os.environ.get("MODAL_USAGE_REPORT_SECRET", "")
        model, lock = self.model, self.lock
        api = FastAPI(title="worker-stt")

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

        def report_usage(key_id, seconds, mode):
            print(f"usage id={key_id} audio_seconds={seconds:.2f} mode={mode}")   # always logged
            if not (USAGE_URL and USAGE_SECRET and key_id and seconds):
                return
            try:
                req = urllib.request.Request(USAGE_URL, method="POST",
                    data=json.dumps({"id": key_id, "audio_seconds": round(seconds, 3), "engine": "whisper-stt", "mode": mode}).encode(),
                    headers={"Authorization": f"Bearer {USAGE_SECRET}", "Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:  # never affect the caller
                print("usage report failed:", e)

        def bearer(h):
            return h[7:] if h and h.lower().startswith("bearer ") else None

        # ---- decoding -------------------------------------------------------------------------
        MU = None
        def mulaw_to_float(b):
            nonlocal MU
            if MU is None:
                u = (~np.arange(256, dtype=np.uint8)).astype(np.int32)
                sign, exp, man = u & 0x80, (u >> 4) & 7, u & 15
                s = (((man << 3) + 0x84) << exp) - 0x84
                MU = np.where(sign != 0, -s, s).astype(np.float32) / 32768.0
            return MU[np.frombuffer(b, dtype=np.uint8)]

        def upsample8k(x):   # 8k -> 16k linear interpolation
            return np.interp(np.arange(len(x) * 2) / 2.0, np.arange(len(x)), x).astype(np.float32)

        def decode(data, fmt):
            if fmt == "mulaw":
                return upsample8k(mulaw_to_float(data))
            p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
                               input=data, capture_output=True)
            if p.returncode != 0 or not p.stdout:
                raise ValueError("could not decode audio: " + p.stderr.decode()[:200])
            return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0

        def transcribe(audio, language=None, words=False, vad=False):
            with lock:
                segs, info = model.transcribe(audio, language=language, beam_size=1, word_timestamps=words,
                                              vad_filter=vad, condition_on_previous_text=False)
                segs = list(segs)
            text = " ".join(s.text.strip() for s in segs).strip()
            w = [{"word": x.word.strip(), "start": round(x.start, 3), "end": round(x.end, 3)}
                 for s in segs for x in (s.words or [])]
            return text, info.language, w

        @api.get("/health")
        def health():
            return {"status": "healthy", "model": MODEL}

        @api.post("/v1/stt")
        async def stt(file: UploadFile = File(...), format: str = Form("auto"), language: str = Form("auto"),
                      word_timestamps: str = Form("true"), authorization: str = Header(None)):
            key_id = verify_session_token(bearer(authorization))
            if not key_id:
                raise HTTPException(401, "invalid or expired session token")
            if format not in ("auto", "wav", "flac", "mp3", "ogg", "mulaw"):
                raise HTTPException(400, "format must be auto|wav|flac|mp3|ogg|mulaw")
            data = await file.read()
            if len(data) > 100 * 1024 * 1024:
                raise HTTPException(413, "file too large")
            try:
                audio = await asyncio.to_thread(decode, data, format)
            except ValueError as e:
                raise HTTPException(400, str(e))
            dur = len(audio) / 16000
            text, lang, words = await asyncio.to_thread(
                transcribe, audio, None if language == "auto" else language, word_timestamps.lower() == "true", True)
            asyncio.get_event_loop().run_in_executor(None, report_usage, key_id, dur, "batch")
            return {"text": text, "language": lang, "words": words, "duration": round(dur, 3)}

        # ---- streaming: VAD endpointing + re-decode of the growing utterance --------------------
        @api.websocket("/v1/stt/stream")
        async def stream(ws: WebSocket, token: str = ""):
            key_id = verify_session_token(token or bearer(ws.headers.get("authorization")))
            await ws.accept()
            if not key_id:
                await ws.close(code=4401); return
            cfg = {"encoding": "pcm16", "endpoint_ms": 500, "partials": True, "language": "en"}
            buf = np.zeros(0, dtype=np.float32)   # current utterance (16k float)
            billed = 0.0
            vad_opts = VadOptions(threshold=0.5, min_silence_duration_ms=100, speech_pad_ms=0, min_speech_duration_ms=150)
            last_check = last_partial = 0
            partial_txt = ""
            first = True
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("text") is not None:
                        c = json.loads(msg["text"])
                        if c.get("type") == "end":
                            break
                        cfg.update({k: v for k, v in c.items() if k in cfg})
                        continue
                    b = msg.get("bytes")
                    if not b:
                        continue
                    chunk = (upsample8k(mulaw_to_float(b)) if cfg["encoding"] == "mulaw"
                             else np.frombuffer(b[: len(b) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0)
                    billed += len(chunk) / 16000
                    buf = np.concatenate([buf, chunk])
                    if len(buf) - last_check < 3200:   # VAD check every 200 ms of audio
                        continue
                    last_check = len(buf)
                    ts = await asyncio.to_thread(get_speech_timestamps, buf, vad_opts)
                    if not ts:
                        if len(buf) > 16000:            # drop pure silence, keep a 1 s lead-in
                            buf = buf[-16000:]; last_check = len(buf); last_partial = 0
                        continue
                    trailing_ms = (len(buf) - ts[-1]["end"]) / 16 
                    if trailing_ms >= cfg["endpoint_ms"]:
                        t0 = time.perf_counter()
                        text, lang, words = await asyncio.to_thread(transcribe, buf[: ts[-1]["end"] + 1600], cfg["language"], True)
                        await ws.send_json({"type": "final", "text": text, "language": lang, "words": words,
                                            "audio_seconds": round(ts[-1]["end"] / 16000, 2),
                                            "decode_ms": round((time.perf_counter() - t0) * 1000)})
                        buf = np.zeros(0, dtype=np.float32); last_check = last_partial = 0; partial_txt = ""
                    elif cfg["partials"] and len(buf) - last_partial >= 16000 and not lock.locked():
                        last_partial = len(buf)
                        text, _, _ = await asyncio.to_thread(transcribe, buf, cfg["language"], False)
                        if text and text != partial_txt:
                            partial_txt = text
                            await ws.send_json({"type": "partial", "text": text})
                # flush whatever is left at end of stream
                if len(buf) > 8000:
                    ts = get_speech_timestamps(buf, vad_opts)
                    if ts:
                        text, lang, words = await asyncio.to_thread(transcribe, buf, cfg["language"], True)
                        await ws.send_json({"type": "final", "text": text, "language": lang, "words": words})
            except WebSocketDisconnect:
                pass
            finally:
                asyncio.get_event_loop().run_in_executor(None, report_usage, key_id, billed, "stream")
                try:
                    await ws.send_json({"type": "usage", "audio_seconds": round(billed, 2)})
                    await ws.close()
                except Exception:
                    pass

        return api
