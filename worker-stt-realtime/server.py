"""
Realtime (streaming) English STT over WebSocket, CPU-first. FastAPI + sherpa-onnx streaming transducer
(default: Apache-2.0 streaming Zipformer, int8) + silero VAD endpointing. Same shape as worker-piper-fly/server.py so it
runs on a Fly Machine (scale-to-zero) or any container.

Protocol  WS /v1/stt/realtime?token=<session token>[&encoding=pcm16|mulaw][&sample_rate=16000|8000][&endpoint_ms=500]
  client -> (optional first text frame) {"type":"config","encoding":"pcm16"|"mulaw","sample_rate":16000|8000,
                                        "endpoint_ms":500,"semantic_hint":true,"partials":true}
  client -> binary frames: PCM16LE mono 16 kHz (default) or G.711 mu-law 8 kHz, any chunk size (20-100 ms typical)
  client -> {"type":"commit"}  end of speech signalled by the client: finalize immediately (no silence wait)
  client -> {"type":"end"}     flush, send usage, close
  server -> {"type":"loading"}  (model still warming; audio is buffered, up to MAX_BUFFER_S)   then {"type":"ready"}
  server -> {"type":"partial","text":..,"start_s":..,"audio_s":..}      audio_s = audio received so far
  server -> {"type":"speculative_final","text":..,"silence_ms":..}   early end-of-turn guess at SPECULATIVE_MS (default 300) of
                                  silence; start your LLM now. Followed by "final" (same text) at endpoint_ms, or by
                                  {"type":"resume"} if the caller kept talking (discard the speculative work).
  server -> {"type":"final","segment":n,"text":..,"start_s":..,"end_s":..,"reason":"vad|commit|end|max_len","latency_hint_ms":..}
  server -> {"type":"usage","id":<key id|null>,"audio_seconds":..,"engine":"stt-realtime"}   (also POSTed to USAGE_REPORT_URL)
  server -> {"type":"error","message":..}   close 4401 unauthorized, 1013 at capacity
Timestamps are seconds of audio received on this connection (stream clock), not wall clock.

Auth (>=1 required): SESSION_SECRET (HMAC session tokens from gateway/keys.js, identical to worker-piper-fly verify_session_token)
and/or AUTH_TOKEN (static bearer, unmetered).  Env: MODEL_DIR, ENGINE (zipformer|nemo480), MAX_CONNECTIONS (default 6),
NUM_THREADS (1), ENDPOINT_MS (500), USAGE_REPORT_URL/USAGE_REPORT_SECRET, PORT (8080), SILERO_PATH.
"""
import asyncio, base64, hashlib, hmac, json, os, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from scipy.signal import firwin, lfilter

MODEL_ROOT = os.environ.get("MODEL_DIR", "/models")
ENGINE = os.environ.get("ENGINE", "zipformer")
SILERO = os.environ.get("SILERO_PATH", os.path.join(MODEL_ROOT, "silero_vad.onnx"))
MAX_CONNECTIONS = int(os.environ.get("MAX_CONNECTIONS", "6"))
NUM_THREADS = int(os.environ.get("NUM_THREADS", "1"))
DEFAULT_ENDPOINT_MS = int(os.environ.get("ENDPOINT_MS", "500"))
SPECULATIVE_MS = int(os.environ.get("SPECULATIVE_MS", "300"))   # 0 disables
MAX_BUFFER_S = float(os.environ.get("MAX_BUFFER_S", "20"))
MAX_UTT_S = float(os.environ.get("MAX_UTT_S", "30"))
AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
SESSION_SECRET = os.environ.get("SESSION_SECRET")
USAGE_REPORT_URL = os.environ.get("USAGE_REPORT_URL", "https://api.readaloudai.org/admin/usage/report")
USAGE_REPORT_SECRET = os.environ.get("USAGE_REPORT_SECRET")
if not (AUTH_TOKEN or SESSION_SECRET):
    raise SystemExit("Set AUTH_TOKEN and/or SESSION_SECRET - refusing to start an unauthenticated STT server")

FUNC = set("a an the and or but so of to in on at for with from by as that which who if then than because my your his her "
           "their our its is are was were be been am will would can could should i we you they he she it um uh like about into over".split())

_rec = None
_ready = threading.Event()
_load_err = None
_active = 0
_t_boot = time.time()
_load_s = None
pool = ThreadPoolExecutor(max_workers=max(4, MAX_CONNECTIONS))


def _load():
    global _rec, _load_err, _load_s
    t = time.time()
    try:
        import sherpa_onnx as so
        if ENGINE == "nemo480":
            d = os.path.join(MODEL_ROOT, "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8")
            f = dict(encoder="encoder.int8.onnx", decoder="decoder.int8.onnx", joiner="joiner.int8.onnx")
        else:
            d = os.path.join(MODEL_ROOT, "sherpa-onnx-streaming-zipformer-en-2023-06-26")
            s = "chunk-16-left-128.int8.onnx"
            f = dict(encoder=f"encoder-epoch-99-avg-1-{s}", decoder=f"decoder-epoch-99-avg-1-{s}", joiner=f"joiner-epoch-99-avg-1-{s}")
        r = so.OnlineRecognizer.from_transducer(
            tokens=os.path.join(d, "tokens.txt"), num_threads=NUM_THREADS, sample_rate=16000,
            decoding_method="greedy_search", **{k: os.path.join(d, v) for k, v in f.items()})
        # warm the graph (first decode allocates)
        st = r.create_stream(); st.accept_waveform(16000, np.zeros(16000, np.float32))
        while r.is_ready(st): r.decode_stream(st)
        _rec = r
    except Exception as e:  # noqa: BLE001
        _load_err = repr(e)
        print("model load failed:", e, flush=True)
    _load_s = time.time() - t
    _ready.set()


threading.Thread(target=_load, daemon=True).start()
app = FastAPI()


def verify_session_token(token: str):
    """Mirrors gateway/keys.js verifySessionToken (and worker-piper-fly/server.py). Returns key id or None."""
    if not SESSION_SECRET or not token:
        return None
    try:
        payload_b64, sig_b64 = token.split(".")
    except ValueError:
        return None
    expected = base64.urlsafe_b64encode(hmac.new(SESSION_SECRET.encode(), payload_b64.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    if not hmac.compare_digest(sig_b64, expected):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    except Exception:
        return None
    if "id" not in payload or "exp" not in payload or time.time() * 1000 > payload["exp"]:
        return None
    return payload["id"]


def report_usage(key_id, seconds):
    if not key_id or seconds <= 0 or not USAGE_REPORT_SECRET:
        return
    try:
        req = urllib.request.Request(USAGE_REPORT_URL, data=json.dumps({"id": key_id, "audio_seconds": round(seconds, 2), "engine": "stt-realtime"}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {USAGE_REPORT_SECRET}"}, method="POST")
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:  # noqa: BLE001
        print(f"usage report failed for {key_id}: {e}", flush=True)


# ---- audio decoding -------------------------------------------------------------------------------------
_MU = None
def _mulaw_table():
    global _MU
    if _MU is None:
        u = (~np.arange(256, dtype=np.uint8)).astype(np.int32)
        sign = u & 0x80; exp = (u >> 4) & 7; man = u & 15
        mag = ((man << 3) + 0x84) << exp
        _MU = (np.where(sign != 0, 0x84 - mag, mag - 0x84)).astype(np.float32) / 32768.0
    return _MU


class Up2:
    """Stateful 8k -> 16k: zero-stuff + 63-tap lowpass (gain 2), state carried across chunks."""
    def __init__(self):
        self.h = firwin(63, 0.5) * 2.0
        self.zi = np.zeros(len(self.h) - 1)
    def __call__(self, x):
        y = np.zeros(len(x) * 2); y[::2] = x
        out, self.zi = lfilter(self.h, [1.0], y, zi=self.zi)
        return out.astype(np.float32)


class Session:
    def __init__(self, endpoint_ms, hint, partials):
        import sherpa_onnx as so
        self.so = so
        self.endpoint_ms, self.hint, self.partials = endpoint_ms, hint, partials
        self.stream = _rec.create_stream()
        c = so.VadModelConfig()
        c.silero_vad.model = SILERO; c.silero_vad.min_silence_duration = 0.05
        c.silero_vad.min_speech_duration = 0.1; c.silero_vad.threshold = 0.5; c.sample_rate = 16000
        self.vad = so.VoiceActivityDetector(c, buffer_size_in_seconds=30)
        self.t = 0.0                  # audio clock (s)
        self.utt_start = None; self.last_txt = ""; self.sil_start = None; self.seg = 0
        self.speech_seen = False; self.last_speech_t = 0.0; self.last_partial_wall = 0.0
        self.spec_sent = False

    def _text(self):
        r = _rec.get_result(self.stream)
        return (r if isinstance(r, str) else r.text).strip()

    def push(self, x):
        """Blocking (runs in the thread pool). Returns list of events."""
        ev = []
        self.stream.accept_waveform(16000, x)
        while _rec.is_ready(self.stream):
            _rec.decode_stream(self.stream)
        self.vad.accept_waveform(x)
        while not self.vad.empty():
            self.vad.pop()
        self.t += len(x) / 16000
        sp = self.vad.is_speech_detected()
        if sp:
            if not self.speech_seen:
                self.utt_start = max(0.0, self.t - len(x) / 16000)
            self.speech_seen = True; self.sil_start = None; self.last_speech_t = self.t
        elif self.speech_seen and self.sil_start is None:
            self.sil_start = self.t - 0.05
        txt = self._text()
        if txt and txt != self.last_txt:
            self.last_txt = txt
            if self.spec_sent:
                self.spec_sent = False
                ev.append({"type": "resume"})
            now = time.time()
            if self.partials and now - self.last_partial_wall >= 0.08:
                self.last_partial_wall = now
                ev.append({"type": "partial", "text": txt, "start_s": round(self.utt_start or 0.0, 2), "audio_s": round(self.t, 2)})
        if self.speech_seen and self.sil_start is not None:
            thr = self.endpoint_ms
            words = re.sub(r"[^a-z' ]", "", self.last_txt.lower()).split()
            if self.hint and words and words[-1] in FUNC:
                thr = min(thr + 400, 1500)    # sentence obviously unfinished ("... my order number is")
            sil_ms = (self.t - self.sil_start) * 1000
            if SPECULATIVE_MS and not self.spec_sent and self.last_txt and SPECULATIVE_MS <= sil_ms < thr:
                self.spec_sent = True
                ev.append({"type": "speculative_final", "text": self.last_txt, "silence_ms": round(sil_ms)})
            if sil_ms >= thr:
                ev += self.finalize("vad")
        elif self.speech_seen and self.t - (self.utt_start or 0) > MAX_UTT_S:
            ev += self.finalize("max_len")
        return ev

    def finalize(self, reason):
        if not self.speech_seen and not self.last_txt:
            return []
        self.stream.accept_waveform(16000, np.zeros(int(0.32 * 16000), np.float32))   # flush encoder right-context
        self.stream.input_finished()
        while _rec.is_ready(self.stream):
            _rec.decode_stream(self.stream)
        txt = self._text()
        ev = []
        if txt:
            end = self.last_speech_t if self.last_speech_t else self.t
            ev.append({"type": "final", "segment": self.seg, "text": txt, "start_s": round(self.utt_start or 0.0, 2),
                       "end_s": round(end, 2), "reason": reason})
            self.seg += 1
        self.stream = _rec.create_stream()
        self.vad.reset()
        self.speech_seen = False; self.sil_start = None; self.last_txt = ""; self.utt_start = None; self.spec_sent = False
        return ev


@app.get("/health")
async def health():
    return {"status": "healthy" if _ready.is_set() and _rec else ("loading" if not _ready.is_set() else "error"), "engine": ENGINE,
            "active": _active, "max": MAX_CONNECTIONS, "load_s": _load_s, "uptime_s": round(time.time() - _t_boot, 1), "error": _load_err}


@app.get("/warm")
async def warm():
    """Cheap wake-up target for the gateway's authorize call: starting the machine is the point; returns readiness."""
    return {"ready": bool(_ready.is_set() and _rec), "active": _active, "max": MAX_CONNECTIONS}


@app.websocket("/v1/stt/realtime")
async def stt(ws: WebSocket, token: str = Query(default=""), encoding: str = Query(default="pcm16"),
              sample_rate: int = Query(default=16000), endpoint_ms: int = Query(default=DEFAULT_ENDPOINT_MS)):
    global _active
    auth = ws.headers.get("authorization", "")
    presented = token or (auth[7:] if auth.startswith("Bearer ") else "")
    key_id = verify_session_token(presented)
    static_ok = bool(AUTH_TOKEN) and bool(presented) and hmac.compare_digest(presented.encode(), AUTH_TOKEN.encode())
    if key_id is None and not static_ok:
        await ws.close(code=4401); return
    if _active >= MAX_CONNECTIONS:
        await ws.accept(); await ws.send_json({"type": "error", "message": "at capacity, retry shortly"}); await ws.close(code=1013); return
    _active += 1
    await ws.accept()
    loop = asyncio.get_running_loop()
    cfg = dict(encoding=encoding, sample_rate=sample_rate, endpoint_ms=endpoint_ms, semantic_hint=True, partials=True)
    total = 0.0
    sess = None
    up = None
    pending = []          # audio buffered while the model loads
    closed = False
    try:
        if not (_ready.is_set() and _rec):
            await ws.send_json({"type": "loading"})

        async def ensure_session():
            nonlocal sess, up
            if sess is not None:
                return True
            if not _ready.is_set():
                return False
            if _rec is None:
                await ws.send_json({"type": "error", "message": "model failed to load"}); return False
            sess = Session(int(cfg["endpoint_ms"]), bool(cfg["semantic_hint"]), bool(cfg["partials"]))
            if cfg["encoding"] == "mulaw" or int(cfg["sample_rate"]) == 8000:
                up = Up2()
            await ws.send_json({"type": "ready", "engine": ENGINE, "load_s": _load_s})
            return True

        def decode(b):
            if cfg["encoding"] == "mulaw":
                x = _mulaw_table()[np.frombuffer(b, np.uint8)]
            else:
                x = np.frombuffer(b[: len(b) // 2 * 2], "<i2").astype(np.float32) / 32768.0
            return up(x) if up is not None else x

        async def run(fn, *a):
            return await loop.run_in_executor(pool, fn, *a)

        async def send_all(evs):
            for e in evs:
                await ws.send_json(e)

        while True:
            if sess is None and not _ready.is_set():
                # wait for either a client message or model readiness
                rd = asyncio.ensure_future(ws.receive())
                while not rd.done():
                    await asyncio.wait([rd], timeout=0.05)
                    if _ready.is_set() and not rd.done():
                        break
                if not rd.done():
                    await ensure_session()
                    msg = await rd
                else:
                    msg = rd.result()
            else:
                await ensure_session()
                msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                closed = True; break
            if msg.get("text") is not None:
                try:
                    m = json.loads(msg["text"])
                except Exception:
                    continue
                t = m.get("type")
                if t == "config":
                    for k in ("encoding", "sample_rate", "endpoint_ms", "semantic_hint", "partials"):
                        if k in m: cfg[k] = m[k]
                elif t == "commit" and sess:
                    await send_all(await run(sess.finalize, "commit"))
                elif t == "end":
                    break
                continue
            b = msg.get("bytes")
            if not b:
                continue
            if sess is None:            # still loading: buffer (bounded)
                pending.append(b)
                if sum(len(p) for p in pending) / (2 if cfg["encoding"] != "mulaw" else 1) / max(1, int(cfg["sample_rate"])) > MAX_BUFFER_S:
                    pending.pop(0)
                continue
            for b2 in ([*pending, b] if pending else [b]):
                x = decode(b2); total += len(x) / 16000
                await send_all(await run(sess.push, x))
            pending.clear()
        if sess:
            await send_all(await run(sess.finalize, "end"))
    except WebSocketDisconnect:
        closed = True
    except Exception as e:  # noqa: BLE001
        print("stt error:", repr(e), flush=True)
        try: await ws.send_json({"type": "error", "message": "internal error"})
        except Exception: pass
    finally:
        _active -= 1
        if total > 0:
            pool.submit(report_usage, key_id, total)
            if not closed:
                try: await ws.send_json({"type": "usage", "id": key_id, "audio_seconds": round(total, 2), "engine": "stt-realtime"})
                except Exception: pass
        if not closed:
            try: await ws.close()
            except Exception: pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8080")), ws_max_size=1 << 20, access_log=False, log_level="warning")
