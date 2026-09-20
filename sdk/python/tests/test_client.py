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


@pytest.mark.parametrize("fmt", ["pcm_24000", "pcm_8000", "mulaw_8000", "alaw_8000"])
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
