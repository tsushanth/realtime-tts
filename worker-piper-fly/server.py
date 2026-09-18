"""
Piper (our fine-tuned voice) over WebSocket - platform-independent build of
worker-modal-piper/app.py: plain FastAPI + uvicorn, configured by env vars, no
Modal. Meant for an always-on CPU box (Fly Machine next to call-loop-poc, Hetzner, a
home server...). Same protocol as the Kokoro worker, so call-loop-poc switches by
changing TTS_GATEWAY_WS_URL only:
  client -> {"type": "synthesize", "text": "...", "voice": "<ignored>", "speed": 1.0}
  client -> {"type": "stop"}
  server -> {"type": "chunk_meta", "text": "", "gen_ms": .., "audio_s": .., "providers": [..]}
  server -> <binary PCM16LE mono 24kHz>      # immediately follows chunk_meta
  server -> {"type": "done"} | {"type": "cancelled"} | {"type": "error", "message": ".."}
See worker-modal-piper/README.md for the design reasons (24kHz resample, working `stop`,
phonemization serialized, inference off the event loop).

Auth (at least one of these two must be set):
  SESSION_SECRET  - HMAC secret shared with gateway/keys.js. Accepts the short-lived session
                    tokens api.readaloudai.org's /tts/authorize issues; completed requests are
                    billed len(text) chars via USAGE_REPORT_URL/USAGE_REPORT_SECRET, exactly
                    like worker-modal-readaloud/app.py (cancelled requests are not billed).
  AUTH_TOKEN      - static bearer for private internal clients (call-loop-poc); not metered.
Other env: MODEL_PATH (default /models/full_ft.onnx; the .json config must sit beside it),
ORT_INTRA_THREADS (default 2), MAX_CONNECTIONS (default 4: beyond it new sockets get a
clean "at capacity" error + close 1013 instead of everyone slowing to a crawl - see the
capacity numbers in worker-modal-piper/README.md), MAX_TEXT_CHARS (default 5000), PORT (default 8080),
HOST (unset => bind both 0.0.0.0 and :: explicitly; set to force a single address).
"""
import asyncio
import base64
import hashlib
import hmac
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import onnxruntime
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from piper import PiperVoice, SynthesisConfig
from scipy.signal import resample_poly

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/full_ft.onnx")
ORT_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "2"))
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
SESSION_SECRET = os.environ.get("SESSION_SECRET")
USAGE_REPORT_URL = os.environ.get("USAGE_REPORT_URL", "https://api.readaloudai.org/admin/usage/report")
USAGE_REPORT_SECRET = os.environ.get("USAGE_REPORT_SECRET")
MAX_CONNECTIONS = int(os.environ.get("MAX_CONNECTIONS", "4"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))
if not (AUTH_TOKEN or SESSION_SECRET):
    raise SystemExit("Set AUTH_TOKEN and/or SESSION_SECRET - refusing to start an unauthenticated TTS server")
OUT_SR = 24000
WARMUP_TEXT = "Thanks for calling, I can help you with that. Let me pull up your account details right now."


class PiperEngine:
    def __init__(self, model_path: str, intra_threads: int):
        self.voice = PiperVoice.load(model_path)
        # PiperVoice.load builds a default SessionOptions, which sizes onnxruntime's
        # thread pool from the HOST core count - oversubscribes a small VM/container.
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = 1
        self.voice.session = onnxruntime.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        ratio = Fraction(OUT_SR, self.voice.config.sample_rate).limit_denominator(1000)
        self.up, self.down = ratio.numerator, ratio.denominator
        self._phonemize_lock = threading.Lock()  # espeak-ng has global state

    def sentences(self, text: str) -> list[list[int]]:
        with self._phonemize_lock:
            return [self.voice.phonemes_to_ids(p) for p in self.voice.phonemize(text) if p]

    def synth(self, phoneme_ids: list[int], speed: float = 1.0) -> tuple[bytes, float]:
        cfg = SynthesisConfig(length_scale=self.voice.config.length_scale / max(speed, 0.1))
        audio = self.voice.phoneme_ids_to_audio(phoneme_ids, cfg)
        peak = float(np.max(np.abs(audio)))
        audio = audio / peak if peak > 1e-8 else np.zeros_like(audio)  # piper's normalize_audio
        if self.up != self.down:
            audio = resample_poly(audio, self.up, self.down)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes(), len(pcm) / OUT_SR


engine = PiperEngine(MODEL_PATH, ORT_INTRA_THREADS)
for _ids in engine.sentences(WARMUP_TEXT):  # pay first-inference costs at startup, not on a caller
    engine.synth(_ids)
pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 2)
app = FastAPI()
_active = 0  # open, authenticated sockets (single event loop, so plain int is safe)


def verify_session_token(token: str):
    """Mirrors gateway/keys.js verifySessionToken exactly (HMAC-SHA256 over the base64url
    payload string, unpadded base64url). Returns the API-key id, or None if invalid/expired."""
    if not SESSION_SECRET or not token:
        return None
    try:
        payload_b64, sig_b64 = token.split(".")
    except ValueError:
        return None
    expected = base64.urlsafe_b64encode(
        hmac.new(SESSION_SECRET.encode(), payload_b64.encode(), hashlib.sha256).digest()
    ).decode().rstrip("=")
    if not hmac.compare_digest(sig_b64, expected):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    except Exception:
        return None
    if "id" not in payload or "exp" not in payload or time.time() * 1000 > payload["exp"]:
        return None
    return payload["id"]


def report_usage(key_id: str, chars: int):
    """Best-effort, off the request path (runs in the thread pool): a failed report must never
    affect the caller's session. Same endpoint/payload the Modal worker uses."""
    if not key_id or not chars or not USAGE_REPORT_SECRET:
        return
    try:
        req = urllib.request.Request(
            USAGE_REPORT_URL, data=json.dumps({"id": key_id, "chars": chars}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {USAGE_REPORT_SECRET}"},
            method="POST")
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:  # noqa: BLE001
        print(f"usage report failed for key {key_id}: {e}")


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "piper-full-ft", "device": "cpu", "active": _active, "max": MAX_CONNECTIONS}


@app.websocket("/tts")
async def tts(ws: WebSocket, token: str = Query(default="")):
    global _active
    auth_header = ws.headers.get("authorization", "")
    bearer = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else ""
    presented = token or bearer
    key_id = verify_session_token(presented)  # metered API-key session, or None
    static_ok = bool(AUTH_TOKEN) and bool(presented) and hmac.compare_digest(presented.encode(), AUTH_TOKEN.encode())
    if key_id is None and not static_ok:
        await ws.close(code=4401)
        return
    await ws.accept()
    if _active >= MAX_CONNECTIONS:
        await ws.send_json({"type": "error", "message": "at capacity, retry shortly"})
        await ws.close(code=1013)
        return
    _active += 1

    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()
    cancel = asyncio.Event()

    async def reader():
        # Keeps receiving during synthesis so `stop` lands mid-utterance; only the main
        # loop below writes to the socket.
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    await inbox.put({"type": "_invalid"})
                    continue
                if msg.get("type") == "stop":
                    cancel.set()
                else:
                    await inbox.put(msg)
        except WebSocketDisconnect:
            cancel.set()
            await inbox.put(None)

    reader_task = asyncio.create_task(reader())
    try:
        while True:
            msg = await inbox.get()
            if msg is None:
                break
            if msg.get("type") == "_invalid":
                await ws.send_json({"type": "error", "message": "invalid JSON"})
                continue
            if msg.get("type") != "synthesize":
                await ws.send_json({"type": "error", "message": "unknown message type"})
                continue

            cancel.clear()
            text = msg.get("text", "")
            speed = float(msg.get("speed", 1.0))
            if len(text) > MAX_TEXT_CHARS:
                await ws.send_json({"type": "error", "message": f"text too long (max {MAX_TEXT_CHARS} chars)"})
                continue
            try:
                sentences = await loop.run_in_executor(pool, engine.sentences, text)
                cancelled = False
                for ids in sentences:
                    if cancel.is_set():
                        cancelled = True
                        break
                    t0 = time.perf_counter()
                    pcm, audio_s = await loop.run_in_executor(pool, engine.synth, ids, speed)
                    await ws.send_json({
                        "type": "chunk_meta", "text": "",
                        "gen_ms": (time.perf_counter() - t0) * 1000,
                        "audio_s": audio_s, "providers": ["cpu-onnx"],
                    })
                    await ws.send_bytes(pcm)
                await ws.send_json({"type": "cancelled" if cancelled else "done"})
                if key_id and not cancelled:  # cancelled requests are not billed (matches the Modal worker)
                    pool.submit(report_usage, key_id, len(text))
            except WebSocketDisconnect:
                break
            except Exception as e:  # noqa: BLE001 - report to client, keep the process alive
                await ws.send_json({"type": "error", "message": str(e)})
    finally:
        reader_task.cancel()
        _active -= 1


if __name__ == "__main__":
    import uvicorn
    import socket
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST")
    if host:
        uvicorn.run(app, host=host, port=port, log_level="info")
    else:
        # Fly's edge proxy dials the machine over IPv4 while 6PN private traffic is IPv6, and
        # a lone "::" socket only served the latter there (proxy: "instance refused connection").
        # So bind one socket per stack explicitly.
        socks = []
        for fam, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
            try:
                sk = socket.socket(fam, socket.SOCK_STREAM)
                sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if fam == socket.AF_INET6:
                    sk.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sk.bind((addr, port))
                socks.append(sk)
            except OSError as e:
                print(f"skip {addr}: {e}")
        uvicorn.Server(uvicorn.Config(app, log_level="info")).run(sockets=socks)
