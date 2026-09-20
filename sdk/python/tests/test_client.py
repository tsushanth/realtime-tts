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
