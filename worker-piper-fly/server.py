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
import re
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import numpy as np
import onnxruntime
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from piper import PiperVoice, SynthesisConfig

import audiofmt

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/full_ft.onnx")
ORT_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "2"))
VOICES_DIR = os.environ.get("VOICES_DIR", "/voices")  # customer voices: <id>/model.onnx(+.json)(+owner.json)
VOICES_ADMIN_TOKEN = os.environ.get("VOICES_ADMIN_TOKEN")  # unset => admin endpoints disabled
MAX_VOICE_BYTES = int(os.environ.get("MAX_VOICE_BYTES", str(200 * 1024 * 1024)))
MAX_VOICES = int(os.environ.get("MAX_VOICES", "6"))  # loaded customer voices kept in memory (LRU)
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
SESSION_SECRET = os.environ.get("SESSION_SECRET")
USAGE_REPORT_URL = os.environ.get("USAGE_REPORT_URL", "https://api.readaloudai.org/admin/usage/report")
USAGE_REPORT_SECRET = os.environ.get("USAGE_REPORT_SECRET")
MAX_CONNECTIONS = int(os.environ.get("MAX_CONNECTIONS", "4"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "5000"))
# Time-to-first-audio: split ONLY the first sentence at a clause boundary (default off, see README).
FIRST_CHUNK_SPLIT = os.environ.get("FIRST_CHUNK_SPLIT", "").lower() in ("1", "true", "yes", "on")
SPLIT_MIN_WORDS = 8   # first sentence must have MORE than this many words to be split
SPLIT_MIN_HALF = 3    # each half must keep at least this many words
CLAUSE_PHONEMES = {",", ";", ":", "-", "\u2014", "\u2013"}
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
        self.native_sr = self.voice.config.sample_rate
        self._phonemize_lock = threading.Lock()  # espeak-ng has global state

    def sentences(self, text: str, split_first: bool = None) -> list[list[int]]:
        """Phoneme-id chunks to synthesize in order. One per sentence; with split_first (default:
        FIRST_CHUNK_SPLIT) the first sentence may be cut at a clause boundary into two chunks."""
        if split_first is None:
            split_first = FIRST_CHUNK_SPLIT
        with self._phonemize_lock:
            sents = [p for p in self.voice.phonemize(text) if p]
            if split_first and sents:
                sents[0:1] = split_clause(sents[0])
            return [self.voice.phonemes_to_ids(p) for p in sents]

    def synth(self, phoneme_ids: list[int], speed: float = 1.0, fmt: str = audiofmt.DEFAULT_FORMAT) -> tuple[bytes, float, int]:
        """Returns (encoded audio, duration in seconds, sample rate of the encoded audio)."""
        cfg = SynthesisConfig(length_scale=self.voice.config.length_scale / max(speed, 0.1))
        audio = self.voice.phoneme_ids_to_audio(phoneme_ids, cfg)
        peak = float(np.max(np.abs(audio)))
        audio = audio / peak if peak > 1e-8 else np.zeros_like(audio)  # piper's normalize_audio
        data, sr = audiofmt.encode(audio, self.native_sr, fmt)
        width = 2 if audiofmt.FORMATS[fmt][1] == "pcm" else 1  # bytes per sample
        return data, len(data) / width / sr, sr


def split_clause(phonemes: list[str]) -> list[list[str]]:
    """Cut one sentence's phoneme list at its earliest clause boundary (, ; : dash) that leaves
    >= SPLIT_MIN_HALF words on both sides; sentences of <= SPLIT_MIN_WORDS words, or with no such
    boundary, are returned whole. The clause punctuation stays on the first half so espeak's
    continuation intonation is kept."""
    words = 1 + sum(1 for p in phonemes if p == " ")
    if words <= SPLIT_MIN_WORDS:
        return [phonemes]
    seen = 1
    for i, p in enumerate(phonemes):
        if p == " ":
            seen += 1
        elif p in CLAUSE_PHONEMES:
            first, rest = phonemes[: i + 1], phonemes[i + 1:]
            while rest and rest[0] == " ":
                rest = rest[1:]
            w1 = len([w for w in "".join(first).split(" ") if w.strip(",;:-\u2014\u2013")])
            w2 = len([w for w in "".join(rest).split(" ") if w.strip(",;:-\u2014\u2013")])
            if w1 >= SPLIT_MIN_HALF and w2 >= SPLIT_MIN_HALF:
                return [first, rest]
    return [phonemes]


engine = PiperEngine(MODEL_PATH, ORT_INTRA_THREADS)
for _ids in engine.sentences(WARMUP_TEXT):  # pay first-inference costs at startup, not on a caller
    engine.synth(_ids)
pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 2)
app = FastAPI()

VOICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_voices: "OrderedDict[str, PiperEngine]" = OrderedDict()  # LRU of loaded customer voices
_voices_lock = threading.Lock()


class VoiceError(Exception):
    pass


def get_engine(voice: str, key_id, uid=None) -> "PiperEngine":
    """Resolves the request's `voice` to an engine. Anything not prefixed `custom:` (existing
    clients send arbitrary names such as a Kokoro voice) gets the default voice, unchanged.
    `custom:<id>` loads <VOICES_DIR>/<id>/model.onnx on first use (blocking - call from the thread
    pool). A voice with owner.json {"key_ids": [...]} is only usable by those API keys; internal
    static-token clients (key_id None) may use any. Missing voice or wrong owner gives the same
    error so voice ids can't be probed. owner.json may also list "user_ids": a session token whose `uid`
    (the key's owning user, embedded by the gateway) is in that list may use the voice, so access follows
    the user across newly created keys; a token without uid (key issued without an owner) never matches.
    Either rule (key_ids or user_ids) suffices. owner.json {"public": true} makes the voice usable by any
    authenticated client (house voices)."""
    if not isinstance(voice, str) or not voice.startswith("custom:"):
        return engine
    vid = voice[len("custom:"):]
    if not VOICE_ID_RE.match(vid):
        raise VoiceError("unknown voice")
    d = os.path.join(VOICES_DIR, vid)
    model = os.path.join(d, "model.onnx")
    if not os.path.isfile(model):
        raise VoiceError("unknown voice")
    if key_id is not None:
        try:
            meta = json.load(open(os.path.join(d, "owner.json")))
        except FileNotFoundError:
            raise VoiceError("unknown voice")  # customer voices must declare an owner
        except (ValueError, OSError):
            raise VoiceError("unknown voice")
        if not isinstance(meta, dict):
            raise VoiceError("unknown voice")
        if meta.get("public") is not True:  # {"public": true} = house voice, any authenticated client
            owners, users = meta.get("key_ids"), meta.get("user_ids")
            by_key = isinstance(owners, list) and key_id in owners
            by_user = uid is not None and isinstance(users, list) and uid in users
            if not (by_key or by_user):
                raise VoiceError("unknown voice")
    with _voices_lock:
        eng = _voices.get(vid)
        if eng is not None:
            _voices.move_to_end(vid)
            return eng
        eng = PiperEngine(model, ORT_INTRA_THREADS)
        for _ids in eng.sentences(WARMUP_TEXT):
            eng.synth(_ids)
        _voices[vid] = eng
        while len(_voices) > MAX_VOICES:
            _voices.popitem(last=False)
        return eng
_active = 0  # open, authenticated sockets (single event loop, so plain int is safe)


# ---- voice admin (server-to-server: the voice pipeline pushes finished voices here) ----
VOICE_FILES = {"model.onnx", "model.onnx.json", "owner.json"}


def _admin(request: Request, vid: str = None):
    tok = request.headers.get("authorization", "").removeprefix("Bearer ")
    if not VOICES_ADMIN_TOKEN or not tok or not hmac.compare_digest(tok.encode(), VOICES_ADMIN_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="unauthorized")
    if vid is not None and not VOICE_ID_RE.match(vid):
        raise HTTPException(status_code=400, detail="bad voice id")


def _evict(vid: str):
    with _voices_lock:
        _voices.pop(vid, None)  # in-flight sessions keep their own reference


@app.put("/admin/voices/{vid}")
async def admin_put_voice(vid: str, request: Request):
    """Body: a tar (optionally gzipped) holding model.onnx, model.onnx.json, owner.json. Replaces
    an existing voice atomically (extract to a temp dir, then rename over)."""
    _admin(request, vid)
    os.makedirs(VOICES_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=VOICES_DIR, prefix=".incoming-")
    try:
        tarpath, size = os.path.join(tmp, "in.tar"), 0
        with open(tarpath, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_VOICE_BYTES:
                    raise HTTPException(status_code=413, detail="voice archive too large")
                f.write(chunk)
        out = os.path.join(tmp, "voice"); os.makedirs(out)
        try:
            with tarfile.open(tarpath, "r:*") as tf:
                names = set()
                for m in tf.getmembers():
                    base = os.path.basename(m.name)
                    if base.startswith("._"):  # macOS AppleDouble metadata
                        continue
                    if not m.isfile() or base != m.name.lstrip("./") or base not in VOICE_FILES:
                        raise HTTPException(status_code=400, detail=f"unexpected archive member {m.name!r}")
                    with tf.extractfile(m) as src, open(os.path.join(out, base), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    names.add(base)
        except tarfile.TarError:
            raise HTTPException(status_code=400, detail="not a valid tar archive")
        if not VOICE_FILES <= names:
            raise HTTPException(status_code=400, detail=f"archive must contain {sorted(VOICE_FILES)}")
        try:
            meta = json.load(open(os.path.join(out, "owner.json")))
            assert isinstance(meta, dict)
            def _ne(k): return isinstance(meta.get(k), list) and len(meta[k]) > 0 and all(isinstance(x, str) and x for x in meta[k])
            assert meta.get("public") is True or _ne("key_ids") or _ne("user_ids")
        except Exception:
            raise HTTPException(status_code=400, detail='owner.json must be {"key_ids": [...]} and/or {"user_ids": [...]} (non-empty), or {"public": true}')
        dest = os.path.join(VOICES_DIR, vid)
        old = dest + ".old"
        shutil.rmtree(old, ignore_errors=True)
        if os.path.isdir(dest):
            os.rename(dest, old)
        os.rename(out, dest)
        shutil.rmtree(old, ignore_errors=True)
        _evict(vid)
        return {"voice": f"custom:{vid}", "bytes": size}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.delete("/admin/voices/{vid}")
async def admin_delete_voice(vid: str, request: Request):
    _admin(request, vid)
    existed = os.path.isdir(os.path.join(VOICES_DIR, vid))
    shutil.rmtree(os.path.join(VOICES_DIR, vid), ignore_errors=True)
    _evict(vid)
    return {"deleted": existed}


@app.get("/admin/voices")
async def admin_list_voices(request: Request):
    _admin(request)
    ids = sorted(d for d in (os.listdir(VOICES_DIR) if os.path.isdir(VOICES_DIR) else []) if VOICE_ID_RE.match(d))
    return {"voices": ids, "loaded": list(_voices.keys())}


def verify_session_token(token: str):
    return verify_session_claims(token)[0]


def verify_session_claims(token: str):
    """Returns (key_id, uid) - (None, None) if invalid. uid is the owning user id or None. Mirrors gateway/keys.js verifySessionToken exactly (HMAC-SHA256 over the base64url
    payload string, unpadded base64url). Returns the API-key id, or None if invalid/expired."""
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


def report_usage(key_id: str, chars: int):
    """Best-effort, off the request path (runs in the thread pool): a failed report must never
    affect the caller's session. Same endpoint/payload the Modal worker uses."""
    if not key_id or not chars or not USAGE_REPORT_SECRET:
        return
    try:
        req = urllib.request.Request(
            USAGE_REPORT_URL, data=json.dumps({"id": key_id, "chars": chars, "engine": "piper"}).encode(),
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
    key_id, uid = verify_session_claims(presented)  # metered API-key session, or (None, None)
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
            try:
                speed = float(msg.get("speed", 1.0))
                fmt = audiofmt.parse_format(msg.get("format"))
            except (ValueError, TypeError) as e:
                await ws.send_json({"type": "error", "message": str(e) if "format" in str(e) else "invalid speed"})
                continue
            if not isinstance(text, str):
                await ws.send_json({"type": "error", "message": "text must be a string"})
                continue
            if len(text) > MAX_TEXT_CHARS:
                await ws.send_json({"type": "error", "message": f"text too long (max {MAX_TEXT_CHARS} chars)"})
                continue
            try:
                try:
                    eng = await loop.run_in_executor(pool, get_engine, msg.get("voice"), key_id, uid)
                except VoiceError as e:
                    await ws.send_json({"type": "error", "message": str(e)})
                    continue
                sentences = await loop.run_in_executor(pool, eng.sentences, text)
                cancelled = False
                for ids in sentences:
                    if cancel.is_set():
                        cancelled = True
                        break
                    t0 = time.perf_counter()
                    pcm, audio_s, sr = await loop.run_in_executor(pool, eng.synth, ids, speed, fmt)
                    await ws.send_json({
                        "type": "chunk_meta", "text": "",
                        "gen_ms": (time.perf_counter() - t0) * 1000,
                        "audio_s": audio_s, "providers": ["cpu-onnx"],
                        "format": fmt, "sample_rate": sr,
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


# ---- HTTP streaming: POST /v1/tts/stream ----
# Body {text, voice?, speed?, format?}; auth = Authorization: Bearer <session token | AUTH_TOKEN>.
# Response: chunked body of raw audio in the requested format, one sentence at a time as it is
# synthesized. Content-Type: audio/pcm (pcm_*), audio/basic (mulaw_8000), audio/x-alaw-basic
# (alaw_8000); X-Sample-Rate and X-Audio-Format describe the stream.
CORS = {
    "Access-Control-Allow-Origin": "*",  # bearer-token auth, no cookies, so * is safe here
    "Access-Control-Expose-Headers": "X-Sample-Rate, X-Audio-Format, Retry-After",
}


def _http_err(status: int, message: str, **headers) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers={**CORS, **headers})


@app.options("/v1/tts/stream")
async def tts_stream_preflight():
    return Response(status_code=204, headers={
        **CORS,
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
        "Access-Control-Max-Age": "86400",
    })


@app.post("/v1/tts/stream")
async def tts_stream(request: Request):
    global _active
    auth = request.headers.get("authorization", "")
    presented = auth[len("Bearer "):] if auth.startswith("Bearer ") else ""
    key_id, uid = verify_session_claims(presented)
    static_ok = bool(AUTH_TOKEN) and bool(presented) and hmac.compare_digest(presented.encode(), AUTH_TOKEN.encode())
    if key_id is None and not static_ok:
        return _http_err(401, "unauthorized", **{"WWW-Authenticate": "Bearer"})
    if _active >= MAX_CONNECTIONS:
        return _http_err(503, "at capacity, retry shortly", **{"Retry-After": "1"})
    _active += 1  # from here on, every exit path must release
    released = False

    def release():
        nonlocal released
        global _active
        if not released:
            released = True
            _active -= 1

    try:
        try:
            body = await request.json()
            assert isinstance(body, dict)
        except Exception:
            release()
            return _http_err(400, "body must be a JSON object")
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            release()
            return _http_err(400, "text must be a non-empty string")
        if len(text) > MAX_TEXT_CHARS:
            release()
            return _http_err(413, f"text too long (max {MAX_TEXT_CHARS} chars)")
        try:
            speed = float(body.get("speed", 1.0))
            fmt = audiofmt.parse_format(body.get("format"))
        except (ValueError, TypeError) as e:
            release()
            return _http_err(400, str(e) if "format" in str(e) else "invalid speed")
        loop = asyncio.get_running_loop()
        try:
            eng = await loop.run_in_executor(pool, get_engine, body.get("voice", "default"), key_id, uid)
        except VoiceError as e:
            release()
            return _http_err(404, str(e))
        chunks = await loop.run_in_executor(pool, eng.sentences, text)
    except BaseException:
        release()
        raise

    sr = audiofmt.FORMATS[fmt][0]

    async def gen():
        try:
            for ids in chunks:
                if await request.is_disconnected():
                    return  # client went away: stop, don't bill
                data, _, _ = await loop.run_in_executor(pool, eng.synth, ids, speed, fmt)
                yield data
            if key_id and not await request.is_disconnected():  # bill only fully delivered requests
                pool.submit(report_usage, key_id, len(text))
        finally:
            release()

    return StreamingResponse(
        gen(), media_type=audiofmt.FORMATS[fmt][2], background=BackgroundTask(release),
        headers={**CORS, "X-Sample-Rate": str(sr), "X-Audio-Format": fmt, "Cache-Control": "no-store"},
    )


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
