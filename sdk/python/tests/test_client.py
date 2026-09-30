import struct

import pytest

from readaloud import (ApiError, AuthError, CapacityError, QuotaError, ReadAloud, VoiceError, wav)


def test_wav_header():
    w = wav(b"\x00\x01" * 10, 24000)
    assert w[:4] == b"RIFF" and w[8:12] == b"WAVE"
    assert struct.unpack("<I", w[24:28])[0] == 24000
    assert struct.unpack("<I", w[40:44])[0] == 20 and len(w) == 64


def test_stream(mock):
    c = ReadAloud("k", api_base=mock.base)
    chunks = list(c.stream("hi", voice="default"))
    assert chunks == [b"\x00" * 4, b"\x01" * 4, b"\x02" * 4]
    assert mock.last_request == {"type": "synthesize", "text": "hi", "voice": "default",
                                 "speed": 1.0, "format": "pcm_24000"}


def test_convert_falls_back_to_ws(mock):
    assert len(ReadAloud("k", api_base=mock.base).convert("hi")) == 12


def test_convert_http(mock):
    mock.with_http = True
    out = ReadAloud("k", api_base=mock.base).convert("hi", format="mulaw_8000")
    assert out == b"abcdefgh"
    assert mock.last_auth == "Bearer tok" and mock.last_http["format"] == "mulaw_8000"


def test_convert_http_capacity(mock):
    mock.with_http = True
    mock.behavior = "capacity_http"
    with pytest.raises(CapacityError) as e:
        ReadAloud("k", api_base=mock.base).convert("hi")
    assert e.value.retry_after == 2.0


def test_auth_and_quota(mock):
    with pytest.raises(AuthError):
        list(ReadAloud("bad", api_base=mock.base).stream("x"))
    with pytest.raises(QuotaError):
        ReadAloud("poor", api_base=mock.base).convert("x")


def test_capacity_ws(mock):
    mock.behavior = "capacity_ws"
    with pytest.raises(CapacityError):
        list(ReadAloud("k", api_base=mock.base).stream("x"))


def test_voice_error(mock):
    with pytest.raises(VoiceError):
        list(ReadAloud("k", api_base=mock.base).stream("x", voice="custom:nope"))
    assert issubclass(VoiceError, ApiError)


def test_early_close_cancels(mock):
    mock.behavior = "slow"
    g = ReadAloud("k", api_base=mock.base).stream("x")
    assert next(g)
    g.close()


def test_stop_from_client(mock):
    mock.behavior = "slow"
    c = ReadAloud("k", api_base=mock.base)
    n = 0
    for _ in c.stream("x"):
        n += 1
        c.stop()
    assert n < 50 and mock.stops == 1


async def test_astream(mock):
    got = [ch async for ch in ReadAloud("k", api_base=mock.base).astream("hi")]
    assert len(got) == 3


async def test_astream_auth(mock):
    with pytest.raises(AuthError):
        async for _ in ReadAloud("bad", api_base=mock.base).astream("hi"):
            pass


def test_stream_http(mock):
    mock.with_http = True
    got = list(ReadAloud("k", api_base=mock.base).stream_http(
        "hi", voice="custom:abc", speed=1.2, format="alaw_8000"))
    assert b"".join(got) == b"abcdefgh"
    assert mock.last_auth == "Bearer tok"
    assert mock.last_http == {"text": "hi", "voice": "custom:abc", "speed": 1.2, "format": "alaw_8000"}


def test_stream_http_unavailable(mock):
    with pytest.raises(ApiError):
        list(ReadAloud("k", api_base=mock.base, engine="kokoro").stream_http("hi"))


def test_stream_http_errors(mock):
    mock.with_http = True
    mock.behavior = "capacity_http"
    with pytest.raises(CapacityError) as e:
        list(ReadAloud("k", api_base=mock.base).stream_http("hi"))
    assert e.value.retry_after == 2.0 and e.value.status == 503
    with pytest.raises(AuthError):
        list(ReadAloud("bad", api_base=mock.base).stream_http("hi"))


@pytest.mark.parametrize("fmt", ["pcm_24000", "pcm_8000", "mulaw_8000", "alaw_8000", "mp3_24000_64", "mp3_24000_128", "opus_24000"])
def test_formats_ws_and_http(mock, fmt):
    c = ReadAloud("k", api_base=mock.base)
    list(c.stream("hi", format=fmt))
    assert mock.last_request["format"] == fmt
    mock.with_http = True
    c.convert("hi", format=fmt)
    assert mock.last_http["format"] == fmt


def test_custom_voice_passthrough(mock):
    c = ReadAloud("k", api_base=mock.base)
    list(c.stream("hi", voice="custom:abc123"))
    assert mock.last_request["voice"] == "custom:abc123"


def test_from_status_unknown_voice_404():
    from readaloud.errors import from_status
    e = from_status(404, "unknown voice")
    assert isinstance(e, VoiceError) and e.status == 404
    assert not isinstance(from_status(404, "not found"), VoiceError)


# ---- POST /v1/text-to-speech (mp3/opus) ----------------------------------
import asyncio
import json as _json
import threading as _threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@pytest.fixture
def gateway():
    class G:
        status = 200
        error = {"error": "x"}
        headers = ()
        seen = []
    g = G()
    g.seen = []
    MP3 = b"\xff\xfb\x90\x00" + bytes(range(10))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["content-length"])))
            g.seen.append((self.path, self.headers["Authorization"], body))
            if g.status != 200:
                b = _json.dumps(g.error).encode()
                self.send_response(g.status)
                self.send_header("content-type", "application/json")
                for k, v in g.headers:
                    self.send_header(k, v)
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self.send_response(200)
            self.send_header("content-type", "audio/mpeg")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in (MP3[:5], MP3[5:]):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
            self.wfile.write(b"0\r\n\r\n")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    _threading.Thread(target=srv.serve_forever, daemon=True).start()
    g.base = f"http://127.0.0.1:{srv.server_port}"
    g.mp3 = MP3
    yield g
    srv.shutdown()


def test_text_to_speech_mp3(gateway):
    c = ReadAloud("sk", engine="kokoro", api_base=gateway.base)
    out = c.text_to_speech("hello", voice="af_heart", speed=1.2, format="mp3_24000_64")
    assert out == gateway.mp3
    path, auth, body = gateway.seen[0]
    assert path == "/v1/text-to-speech" and auth == "Bearer sk"
    assert body == {"text": "hello", "voice": "af_heart", "speed": 1.2,
                    "format": "mp3_24000_64", "engine": "kokoro"}


def test_text_to_speech_defaults_and_engine_override(gateway):
    c = ReadAloud("sk", api_base=gateway.base)
    c.text_to_speech("hi", engine="kokoro")
    assert gateway.seen[0][2]["format"] == "mp3_24000_128"
    assert gateway.seen[0][2]["engine"] == "kokoro"
    c.text_to_speech("hi")
    assert gateway.seen[1][2]["engine"] == "piper"


def test_stream_text_to_speech(gateway):
    c = ReadAloud("sk", api_base=gateway.base)
    chunks = list(c.stream_text_to_speech("hi", chunk_size=3))
    assert len(chunks) > 1 and b"".join(chunks) == gateway.mp3
    # early close doesn't raise
    it = c.stream_text_to_speech("hi", chunk_size=2)
    assert next(it)
    it.close()


@pytest.mark.parametrize("status,cls,exact", [
    (401, AuthError, False), (402, QuotaError, False), (503, CapacityError, False),
    (400, ApiError, True), (413, ApiError, True), (501, ApiError, True), (502, ApiError, True),
])
def test_text_to_speech_errors(gateway, status, cls, exact):
    gateway.status = status
    gateway.error = {"error": f"boom {status}"}
    c = ReadAloud("sk", api_base=gateway.base)
    for call in (lambda: c.text_to_speech("x"), lambda: list(c.stream_text_to_speech("x"))):
        with pytest.raises(cls) as e:
            call()
        assert e.value.status == status and str(e.value) == f"boom {status}"
        if exact:
            assert type(e.value) is ApiError


def test_text_to_speech_voice_error(gateway):
    gateway.status = 400
    gateway.error = {"error": "unknown voice"}
    with pytest.raises(VoiceError):
        ReadAloud("sk", api_base=gateway.base).text_to_speech("x", voice="custom:nope")


def test_text_to_speech_503_retry_after(gateway):
    gateway.status = 503
    gateway.error = {"error": "encoder at capacity"}
    gateway.headers = (("Retry-After", "2"),)
    with pytest.raises(CapacityError) as e:
        ReadAloud("sk", api_base=gateway.base).text_to_speech("x")
    assert e.value.retry_after == 2.0
    gateway.headers = ()
    with pytest.raises(CapacityError) as e:
        ReadAloud("sk", api_base=gateway.base).text_to_speech("x")
    assert e.value.retry_after is None


def test_async_text_to_speech(gateway):
    c = ReadAloud("sk", api_base=gateway.base)

    async def run():
        chunks = [ch async for ch in c.astream_text_to_speech("hi", chunk_size=3)]
        whole = await c.atext_to_speech("hi", format="opus_24000")
        gateway.status, gateway.error = 402, {"error": "quota"}
        with pytest.raises(QuotaError):
            await c.atext_to_speech("hi")
        with pytest.raises(QuotaError):
            [ch async for ch in c.astream_text_to_speech("hi")]
        return chunks, whole

    chunks, whole = asyncio.run(run())
    assert b"".join(chunks) == gateway.mp3 and whole == gateway.mp3
    assert gateway.seen[1][2]["format"] == "opus_24000"
