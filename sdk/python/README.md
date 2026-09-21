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
- `stop()` cancels the in-flight sync `stream()` from another thread.
- `wav(pcm, sample_rate=24000)` wraps PCM in a WAV header.
- `format`: `pcm_24000` (default), `pcm_8000`, `mulaw_8000`, `alaw_8000`. Only PCM formats
  can be wrapped with `wav()`; pass the matching sample rate.

## Errors

All derive from `ReadAloudError`; `ApiError` has `.status`.

| Exception | When |
|---|---|
| `AuthError` | 401, invalid key/token |
| `QuotaError` | 402, free tier exhausted |
| `CapacityError` | WS close 1013, "at capacity", HTTP 503 (`.retry_after`) |
| `VoiceError` | unknown voice (including an unknown `custom:<id>`) |
| `ApiError` | anything else |

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
