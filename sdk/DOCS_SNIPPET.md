## SDKs

Official client libraries wrap authorization, WebSocket streaming, HTTP streaming, audio formats
(`pcm_24000`, `pcm_8000`, `mulaw_8000`, `alaw_8000`) and `custom:<id>` voices. They take your API
key, so use them server-side; for browsers, mint a short-lived token on your server via
`POST /tts/authorize`.

### Python (3.9+)

```
pip install readaloud
```

```python
from readaloud import ReadAloud, wav

client = ReadAloud("YOUR_API_KEY")                     # engine="piper" by default
for chunk in client.stream("Hello from ReadAloud."):   # WebSocket, one chunk per sentence
    play(chunk)                                        # raw PCM16LE mono, 24 kHz
for chunk in client.stream_http("Hi.", format="mulaw_8000"):  # HTTP chunked (Piper)
    send(chunk)
open("out.wav", "wb").write(wav(client.convert("Hello there."), 24000))
```

### JavaScript / TypeScript (Node 22+, browsers)

```
npm install readaloud
```

```js
import { ReadAloud, pcmToWav } from 'readaloud';

const client = new ReadAloud({ apiKey: process.env.READALOUD_API_KEY });   // engine 'piper' by default
for await (const chunk of client.stream('Hello from ReadAloud.')) {        // WebSocket, Uint8Array PCM
  play(chunk);
}
for await (const chunk of client.streamHttp('Hi.', { format: 'mulaw_8000' })) send(chunk); // HTTP chunked (Piper)
const wav = pcmToWav(await client.convert('Hello there.'), 24000);
```

Errors: `AuthError` (401), `QuotaError` (402), `CapacityError` (503 / at capacity, has retry-after),
`VoiceError` (unknown voice). Source: https://github.com/tsushanth/realtime-tts/tree/main/sdk.
Using an AI assistant instead? See the MCP server: https://readaloudai.org/developers/mcp

### Audio isolation (beta, `worker-demucs-fly`)

Given a noisy/mixed clip, isolates the speech track (Demucs `htdemucs`, +9.4dB SNR improvement
measured on a synthetic 0dB-SNR test clip — see `worker-demucs-fly/README.md`). Not yet wrapped
by the official SDKs above — authorize, then call the worker directly, same two-step shape as
the TTS/STT engines:

```
curl -s -X POST https://api.readaloudai.org/audio/authorize \
  -H "Content-Type: application/json" -d '{"key":"YOUR_API_KEY"}'
# => {"token":"...","url":"https://<demucs-worker>.fly.dev"}

curl -s -X POST https://<demucs-worker>.fly.dev/v1/isolate \
  -H "Authorization: Bearer <token>" -F "file=@noisy.wav" -F "stem=vocals" -o isolated.wav
```

`stem` defaults to `vocals` (the speech track); `drums`/`bass`/`other` are also available (htdemucs'
other separated sources). Clip limits: 25MB / 120s by default. **Undeployed as of this writing** —
`worker-demucs-fly/fly.toml` is configured (app `demucs-isolation-dev`) but not yet pushed to Fly.
