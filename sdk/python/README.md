# readaloud (Python)

Client for the [ReadAloud](https://readaloudai.org) streaming text-to-speech API. Audio is
raw PCM16LE mono (24 kHz by default), or 8 kHz PCM / mu-law / A-law for telephony.

- Docs: https://readaloudai.org/developers
- MCP server (use ReadAloud from Claude, Cursor, etc.): https://readaloudai.org/developers/mcp

```
pip install readaloud
```

Python 3.9+. Depends on `websockets` and `requests`.

## Quickstart

```python
from readaloud import ReadAloud, wav

client = ReadAloud("YOUR_API_KEY", engine="piper")   # or "kokoro"

# Stream chunks as they are synthesized (WebSocket, one chunk per sentence)
for chunk in client.stream("Hello from ReadAloud.", voice="default"):
    play(chunk)

# HTTP chunked streaming (Piper): plain POST, arbitrary byte slices
for chunk in client.stream_http("Hello from ReadAloud.", format="mulaw_8000"):
    send_to_phone(chunk)

# Compressed audio (mp3 / opus) via POST /v1/text-to-speech
open("hello.mp3", "wb").write(client.text_to_speech("Hello.", format="mp3_24000_128"))
for chunk in client.stream_text_to_speech("Hello.", format="opus_24000"):
    send(chunk)

# Whole clip -> WAV file
pcm = client.convert("Hello there.")
open("out.wav", "wb").write(wav(pcm, 24000))
```

Kokoro voices: `voice="af_heart"`. Custom voices: `voice="custom:<id>"`.

## API

- `ReadAloud(api_key, engine="piper", api_base="https://api.readaloudai.org", timeout=30)`
- `stream(text, voice="default", speed=1.0, format="pcm_24000")` -> iterator of `bytes`
  (WebSocket). Closing the generator early cancels synthesis.
- `astream(...)` -> async iterator, same semantics.
- `stream_http(text, voice, speed, format, chunk_size=4096)` -> iterator of `bytes` from the
  HTTP streaming endpoint (`POST http_url`, Bearer token, chunked). Piper only; raises
  `ApiError` when the server offers no `http_url`.
- `convert(...)` -> `bytes`. Uses HTTP streaming when `/tts/authorize` returns `http_url`
  (Piper), otherwise collects the WebSocket stream.
- `text_to_speech(text, voice="default", speed=1.0, format="mp3_24000_128", engine=None)` -> `bytes`
  from the one-shot gateway endpoint `POST /v1/text-to-speech` (Bearer API key, no authorize step).
  The way to get mp3/opus. `engine` defaults to the client's. HTTP errors map to the exceptions below
  (`CapacityError.retry_after` comes from `Retry-After`).
- `stream_text_to_speech(..., chunk_size=4096)` -> iterator of `bytes` from the same endpoint;
  `astream_text_to_speech(...)` / `atext_to_speech(...)` are the async variants (blocking I/O runs in a thread).
- `stop()` cancels the in-flight sync `stream()` from another thread.
- `wav(pcm, sample_rate=24000)` wraps PCM in a WAV header.
- `format`: `pcm_24000` (default), `pcm_8000`, `mulaw_8000`, `alaw_8000`, plus the compressed
  `mp3_24000_64`, `mp3_24000_128` (`audio/mpeg`) and `opus_24000` (`audio/ogg`). Compressed formats are
  only served by `text_to_speech()` / `stream_text_to_speech()` (the WebSocket `stream()` and Piper
  `stream_http()`/`convert()` are PCM/G.711 only). Only PCM formats can be wrapped with `wav()`; pass the matching sample rate.

## Errors

All derive from `ReadAloudError`; `ApiError` has `.status`.

| Exception | When |
|---|---|
| `AuthError` | 401, invalid key/token |
| `QuotaError` | 402, free tier exhausted |
| `CapacityError` | WS close 1013, "at capacity", HTTP 503 (`.retry_after`) |
| `VoiceError` | unknown voice (including an unknown `custom:<id>`) |
| `ApiError` | anything else (from `text_to_speech`: 400 bad request, 413 text over 5000 chars, 501 engine unavailable, 502 worker unavailable) |

```python
import time
from readaloud import CapacityError

try:
    audio = client.convert(text)
except CapacityError as e:
    time.sleep(e.retry_after or 1)
    audio = client.convert(text)
```

## Security

The client takes your **API key**. Run it server-side (backend, worker, CLI) and keep the key
in an environment variable or secret store. Never ship the key in a browser or mobile app:
have your server call `POST /tts/authorize` and hand the short-lived `token`/`url` to the
client instead.

## Development

```
pip install -e ".[dev]" && pytest
```

MIT licensed.
