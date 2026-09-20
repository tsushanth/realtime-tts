# readaloud (Python)

Small client for the ReadAloud streaming text-to-speech API. Audio is raw PCM16LE mono
(24 kHz by default).

```
pip install ./sdk/python
```

```python
from readaloud import ReadAloud, wav

client = ReadAloud("YOUR_API_KEY", engine="piper")   # or "kokoro"

# Stream chunks as they are synthesized (one per sentence)
for chunk in client.stream("Hello from ReadAloud.", voice="default"):
    play(chunk)

# Whole clip -> WAV file
pcm = client.convert("Hello there.")
open("out.wav", "wb").write(wav(pcm, 24000))
```

Kokoro voices: `client.stream(text, voice="af_heart")`. Custom voices: `voice="custom:<id>"`.

## API

- `ReadAloud(api_key, engine="piper", api_base="https://api.readaloudai.org", timeout=30)`
- `stream(text, voice="default", speed=1.0, format="pcm_24000")` -> iterator of `bytes`
  (WebSocket). Breaking out of the loop / closing the generator cancels synthesis.
- `astream(...)` -> async iterator, same semantics.
- `convert(...)` -> `bytes`. Uses the HTTP streaming endpoint when the server offers one
  (`http_url` from authorize, Piper), otherwise collects the WebSocket stream.
- `stop()` cancels the in-flight sync stream from another thread.
- `wav(pcm, sample_rate=24000)` wraps PCM in a WAV header.
- `format`: `pcm_24000` (default), `pcm_8000`, `mulaw_8000`, `alaw_8000`, ... (server-dependent).
  Only PCM formats can be wrapped with `wav()`; use the matching sample rate.

## Errors

All derive from `ReadAloudError`; `ApiError` has `.status`.

| Exception | When |
|---|---|
| `AuthError` | 401, invalid key/token |
| `QuotaError` | 402, free tier exhausted |
| `CapacityError` | WS close 1013, "at capacity", HTTP 503 (`.retry_after`) |
| `VoiceError` | "unknown voice" |
| `ApiError` | anything else |

```python
try:
    audio = client.convert(text)
except CapacityError as e:
    time.sleep(e.retry_after or 1)
```

Tests: `pip install -e "sdk/python[dev]" && pytest sdk/python`.
