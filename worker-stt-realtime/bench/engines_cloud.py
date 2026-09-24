"""Deepgram Flux and ElevenLabs Scribe v2 Realtime behind the bench stream interface.
Third-party engines only ever see calls listed in a confirmed-calls file (assert_allowed)."""
import base64
import json
import os
import re
import threading
import time

import numpy as np

from engines import Engine


PUBLIC_SETS = {"clean", "tel", "call"}   # public/synthetic sets; anything else is treated as real call audio


def _confirmed_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    return {ln for ln in lines if ln and not ln.startswith("#")}


def assert_allowed(clips, confirmed_path, set_name):
    """Fail closed: outside PUBLIC_SETS every clip needs a call_id exactly listed in confirmed_path."""
    ok = _confirmed_ids(confirmed_path)
    public = set_name in PUBLIC_SETS
    missing = sum(1 for c in clips if not c.get("call_id") and not public)
    bad = sorted({c["call_id"] for c in clips if c.get("call_id") and c["call_id"] not in ok})
    if missing:
        raise PermissionError(f"refusing third-party STT: {missing} clip(s) in set {set_name!r} have no call_id "
                              f"(only sets {sorted(PUBLIC_SETS)} may omit it)")
    if bad:
        raise PermissionError(f"refusing third-party STT: {len(bad)} call id(s) not in {confirmed_path}: {bad[:5]}")


def _pcm16(x):
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


class _Stream:
    def __init__(self, url, headers):
        from websockets.sync.client import connect
        self.committed, self.partial, self.eos, self.error = [], "", False, None
        self.closed, self.ready = threading.Event(), threading.Event()
        self._closing = False
        self._final_started = self._final_done = False
        self.lock = threading.Lock()
        self.ws = connect(url, additional_headers=headers, open_timeout=10, max_size=None)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        lost = "closed"
        try:
            for raw in self.ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                with self.lock:
                    self.handle(msg)
        except Exception as e:
            lost = type(e).__name__     # type only: message text could carry secrets
        finally:
            with self.lock:
                if not self._closing and not (self._final_started and self._final_done) and self.error is None:
                    self.error = f"connection lost: {lost}"
            self.closed.set()
            self.ready.set()

    def close(self):
        """Idempotent; marks the close as expected so the reader does not report it as a loss."""
        self._closing = True
        try:
            self.ws.close()
        except Exception:
            pass

    def _begin_final(self):
        """Mark final() started; a turn already complete counts as the flush being done."""
        with self.lock:
            self._final_started = True
            self._final_done = self._turn_done()

    @staticmethod
    def _code(v):
        """Machine code only: free-form server prose is never put into an exception."""
        return v if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", v) else "unknown"

    def _raise_if_error(self):
        with self.lock:
            err = self.error
        if err:
            raise RuntimeError(err)

    def _send(self, data):
        self._raise_if_error()
        try:
            self.ws.send(data)
        except Exception as e:
            self._raise_if_error()
            raise RuntimeError(f"send failed: {type(e).__name__}") from None

    def text(self):
        with self.lock:
            return " ".join(self.committed + ([self.partial] if self.partial else [])).strip()

    def native_eos(self):
        with self.lock:
            return self.eos

    def _wait(self, cond, timeout):
        end = time.time() + timeout
        while time.time() < end and not self.closed.is_set():
            with self.lock:
                if cond():
                    return True
            time.sleep(0.01)
        with self.lock:
            return cond()

    def _turn_done(self):
        return self.eos and not self.partial


class _DGStream(_Stream):
    def handle(self, m):
        if m.get("type") == "Error":
            self.error = f"deepgram error: {self._code(m.get('code'))}"
            return
        if m.get("type") != "TurnInfo":
            return
        ev, tr = m.get("event"), (m.get("transcript") or "").strip()
        if ev == "StartOfTurn":
            self.eos = False
        elif ev in ("Update", "EagerEndOfTurn", "TurnResumed"):
            self.partial, self.eos = tr, False
        elif ev == "EndOfTurn":
            if tr:
                self.committed.append(tr)
            self.partial, self.eos = "", True
            if self._final_started:
                self._final_done = True

    def push(self, x):
        self._send(_pcm16(x))

    def final(self):
        try:
            self._raise_if_error()
            self._begin_final()
            try:
                self.ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            # a drop before the flush EndOfTurn is a reader error; EndOfTurn then close is a normal finish
            self._wait(lambda: self._final_done or self.error, 3.0)
            self._raise_if_error()
            return self.text()
        finally:
            self.close()


class _ELStream(_Stream):
    AUTO_COMMIT_WAIT_S = 2.0
    _ERRORS = {"quota_exceeded", "rate_limited", "queue_overflow", "resource_exhausted", "session_time_limit_exceeded"}

    def handle(self, m):
        t = m.get("message_type") or ""
        if t == "session_started":
            self.ready.set()
        elif t == "partial_transcript":
            self.partial, self.eos = (m.get("text") or "").strip(), False
        elif t == "committed_transcript":
            tx = (m.get("text") or "").strip()
            if tx:
                self.committed.append(tx)
            self.partial, self.eos = "", True
            if self._final_started:
                self._final_done = True
        elif "error" in t or t in self._ERRORS:
            self.error = self._code(t)
            self.ready.set()

    def _chunk(self, x, commit):
        self._send(json.dumps({"message_type": "input_audio_chunk", "commit": commit, "sample_rate": 16000,
                                 "audio_base_64": base64.b64encode(_pcm16(x)).decode()}))

    def push(self, x):
        self._chunk(x, False)

    def final(self):
        try:
            self._raise_if_error()
            self._begin_final()
            done = lambda: self._final_done or self.error
            if not self._wait(done, self.AUTO_COMMIT_WAIT_S):
                self._chunk(np.zeros(1600, dtype="float32"), True)
                self._wait(done, 3.0)
            self._raise_if_error()
            return self.text()
        finally:
            self.close()


class DeepgramFluxEngine(Engine):
    name = "dg-flux"

    def __init__(self, url=None, key=None, eot_threshold=0.7, eot_timeout_ms=5000):
        self.key = key or os.environ["DEEPGRAM_API_KEY"]
        self.url = url or ("wss://api.deepgram.com/v2/listen?model=flux-general-en"
                           f"&eot_threshold={eot_threshold}&eot_timeout_ms={eot_timeout_ms}"
                           "&encoding=linear16&sample_rate=16000")

    def new_stream(self):
        return _DGStream(self.url, {"Authorization": f"Token {self.key}"})


class ElevenLabsRealtimeEngine(Engine):
    name = "el-scribe"

    def __init__(self, url=None, key=None):
        self.key = key or os.environ["ELEVENLABS_API_KEY"]
        self.url = url or ("wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime"
                           "&audio_format=pcm_16000&commit_strategy=vad&language_code=en")

    def new_stream(self):
        s = _ELStream(self.url, {"xi-api-key": self.key})
        if not s.ready.wait(5) or s.error:
            s.close()
            raise RuntimeError(f"elevenlabs session failed: {s.error or 'no session_started within 5 s'}")
        return s
