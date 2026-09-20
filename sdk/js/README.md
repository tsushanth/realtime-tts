# readaloud (JavaScript)

Zero-dependency ESM client for the ReadAloud streaming TTS API. Works in Node >= 22 and
browsers (uses global `fetch` and `WebSocket`). No build step. Audio is raw PCM16LE mono
(24 kHz by default).

```js
import { ReadAloud, pcmToWav } from 'readaloud';

const client = new ReadAloud({ apiKey: 'YOUR_API_KEY', engine: 'piper' }); // or 'kokoro'

// Stream chunks (Uint8Array, one per sentence)
for await (const chunk of client.stream('Hello from ReadAloud.', { voice: 'default' })) {
  play(chunk);
}

// Whole clip -> WAV
const pcm = await client.convert('Hello there.');
const wav = pcmToWav(pcm, 24000);
```

Stop early with `break` or an `AbortSignal`:

```js
const ac = new AbortController();
setTimeout(() => ac.abort(), 1000);
try { for await (const c of client.stream(text, { signal: ac.signal })) play(c); }
catch (e) { if (e.name !== 'AbortError') throw e; }
```

## API

- `new ReadAloud({ apiKey, engine = 'piper', apiBase = 'https://api.readaloudai.org' })`
- `stream(text, { voice = 'default', speed = 1, format = 'pcm_24000', signal })` -> `AsyncGenerator<Uint8Array>`
- `convert(text, opts)` -> `Promise<Uint8Array>`; uses the HTTP endpoint when the server offers
  `http_url` (Piper), otherwise collects the WebSocket stream.
- `pcmToWav(pcm, sampleRate = 24000, channels = 1)`
- Voices: Piper `default`; Kokoro e.g. `af_heart`; custom `custom:<id>`.
- Browsers: an API key in client-side code is visible to users; prefer calling `authorize`
  from your server. (This client always authorizes with the key.)

## Errors

`ReadAloudError` > `ApiError` (`.status`) > `AuthError` (401), `QuotaError` (402),
`CapacityError` (close 1013 / "at capacity" / 503, `.retryAfter`), `VoiceError` ("unknown voice").

Tests: `node --test sdk/js/test/`.
