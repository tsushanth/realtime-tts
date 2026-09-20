from __future__ import annotations

import json
import threading
from typing import AsyncIterator, Iterator, Optional

import requests
from websockets.asyncio.client import connect as aconnect
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from .errors import (ApiError, CapacityError, from_message, from_status)

DEFAULT_BASE = "https://api.readaloudai.org"


def _err_text(resp: requests.Response) -> str:
    try:
        body = resp.json()
        return str(body.get("error") or body.get("message") or body)
    except Exception:
        return resp.text[:200] or f"HTTP {resp.status_code}"


def _closed_error(exc: ConnectionClosed) -> ApiError:
    code = exc.rcvd.code if exc.rcvd else None
    if code == 1013:
        return CapacityError()
    return ApiError(f"connection closed unexpectedly (code {code})")


class ReadAloud:
    """Client for the ReadAloud streaming TTS API."""

    def __init__(self, api_key: str, engine: str = "piper", api_base: str = DEFAULT_BASE,
                 timeout: float = 30.0):
        self.api_key = api_key
        self.engine = engine
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self._ws = None
        self._lock = threading.Lock()

    # -- auth ---------------------------------------------------------------
    def authorize(self) -> dict:
        """Exchange the API key for a short-lived token. Returns {token, url, http_url?}."""
        r = requests.post(f"{self.api_base}/tts/authorize",
                          json={"key": self.api_key, "engine": self.engine}, timeout=self.timeout)
        if r.status_code != 200:
            raise from_status(r.status_code, _err_text(r))
        return r.json()

    @staticmethod
    def _request(text, voice, speed, format) -> dict:
        return {"type": "synthesize", "text": text, "voice": voice, "speed": speed, "format": format}

    # -- sync streaming -----------------------------------------------------
    def stream(self, text: str, voice: str = "default", speed: float = 1.0,
               format: str = "pcm_24000") -> Iterator[bytes]:
        """Yield audio chunks. Closing the generator early cancels synthesis."""
        yield from self._ws_stream(self.authorize(), text, voice, speed, format)

    def _ws_stream(self, auth: dict, text, voice, speed, format) -> Iterator[bytes]:
        cm = connect(f"{auth['url']}?token={auth['token']}", max_size=None,
                     open_timeout=self.timeout)
        ws = cm.__enter__()   # websockets >= 15 deprecates using connect() without `with`
        with self._lock:
            self._ws = ws
        try:
            ws.send(json.dumps(self._request(text, voice, speed, format)))
            while True:
                try:
                    msg = ws.recv()
                except ConnectionClosed as exc:
                    raise _closed_error(exc) from None
                if isinstance(msg, bytes):
                    yield msg
                    continue
                data = json.loads(msg)
                t = data.get("type")
                if t == "error":
                    raise from_message(data.get("message", "unknown error"))
                if t in ("done", "cancelled"):
                    return
        finally:
            try:
                ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            cm.__exit__(None, None, None)
            with self._lock:
                if self._ws is ws:
                    self._ws = None

    def stop(self) -> None:
        """Cancel the in-flight sync stream (callable from another thread)."""
        with self._lock:
            ws = self._ws
        if ws is not None:
            try:
                ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass

    # -- async streaming ----------------------------------------------------
    async def astream(self, text: str, voice: str = "default", speed: float = 1.0,
                      format: str = "pcm_24000") -> AsyncIterator[bytes]:
        """Async variant of stream(). Breaking out of the loop cancels synthesis."""
        auth = await _to_thread(self.authorize)
        acm = aconnect(f"{auth['url']}?token={auth['token']}", max_size=None,
                       open_timeout=self.timeout)
        ws = await acm.__aenter__()
        try:
            await ws.send(json.dumps(self._request(text, voice, speed, format)))
            while True:
                try:
                    msg = await ws.recv()
                except ConnectionClosed as exc:
                    raise _closed_error(exc) from None
                if isinstance(msg, bytes):
                    yield msg
                    continue
                data = json.loads(msg)
                t = data.get("type")
                if t == "error":
                    raise from_message(data.get("message", "unknown error"))
                if t in ("done", "cancelled"):
                    return
        finally:
            try:
                await ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass
            await acm.__aexit__(None, None, None)

    # -- HTTP streaming -----------------------------------------------------
    def _http_stream(self, auth: dict, text, voice, speed, format,
                     chunk_size: int) -> Iterator[bytes]:
        http_url = auth.get("http_url")
        if not http_url:
            raise ApiError("HTTP streaming is not available for this engine; use stream()")
        r = requests.post(http_url, headers={"Authorization": f"Bearer {auth['token']}"},
                          json={"text": text, "voice": voice, "speed": speed, "format": format},
                          timeout=self.timeout, stream=True)
        try:
            if r.status_code != 200:
                ra = r.headers.get("Retry-After")
                try:
                    retry_after = float(ra) if ra else None
                except ValueError:
                    retry_after = None
                raise from_status(r.status_code, _err_text(r), retry_after)
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    yield chunk
        finally:
            r.close()

    def stream_http(self, text: str, voice: str = "default", speed: float = 1.0,
                    format: str = "pcm_24000", chunk_size: int = 4096) -> Iterator[bytes]:
        """Yield audio bytes from the HTTP streaming endpoint (``POST http_url`` with a
        Bearer token, chunked response). Available for Piper; raises ApiError otherwise.
        Chunks are arbitrary byte slices, not sentence-aligned; for 16-bit formats
        they may split a sample. Closing the generator drops the connection."""
        yield from self._http_stream(self.authorize(), text, voice, speed, format, chunk_size)

    # -- convert ------------------------------------------------------------
    def convert(self, text: str, voice: str = "default", speed: float = 1.0,
                format: str = "pcm_24000") -> bytes:
        """Return the complete audio. Uses the HTTP endpoint when the server offers it
        (Piper), otherwise collects the WebSocket stream."""
        auth = self.authorize()
        if auth.get("http_url"):
            return b"".join(self._http_stream(auth, text, voice, speed, format, 65536))
        return b"".join(self._ws_stream(auth, text, voice, speed, format))


async def _to_thread(fn):
    import asyncio
    return await asyncio.to_thread(fn)
