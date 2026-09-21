# readaloud (JavaScript)

Zero-dependency ESM client for the [ReadAloud](https://readaloudai.org) streaming
text-to-speech API. Node >= 22 (global `fetch` and `WebSocket`) and modern browsers. Ships
TypeScript types. Audio is raw PCM16LE mono (24 kHz by default), or 8 kHz PCM / mu-law / A-law.

- Docs: https://readaloudai.org/developers
- MCP server (use ReadAloud from Claude, Cursor, etc.): https://readaloudai.org/developers/mcp

```
npm install readaloud
```

## Quickstart

```js
import { ReadAloud, pcmToWav } from 'readaloud';

const client = new ReadAloud({ apiKey: process.env.READALOUD_API_KEY, engine: 'piper' }); // or 'kokoro'

// Stream chunks over WebSocket (Uint8Array, one per sentence)
for await (const chunk of client.stream('Hello from ReadAloud.', { voice: 'default' })) {
  play(chunk);
}

// HTTP chunked streaming (Piper)
for await (const chunk of client.streamHttp('Hello.', { format: 'mulaw_8000' })) send(chunk);

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
- `stream(text, { voice = 'default', speed = 1, format = 'pcm_24000', signal })` -> `AsyncGenerator<Uint8Array>` (WebSocket)
- `streamHttp(text, opts)` -> `AsyncGenerator<Uint8Array>` from the HTTP streaming endpoint
  (`POST http_url`, Bearer token). Piper only; throws `ApiError` if the server offers no `http_url`.
  Chunks are arbitrary byte slices.
- `convert(text, opts)` -> `Promise<Uint8Array>`; uses HTTP streaming when `/tts/authorize`
  returns `http_url` (Piper), otherwise collects the WebSocket stream.
- `pcmToWav(pcm, sampleRate = 24000, channels = 1)`
- `format`: `pcm_24000`, `pcm_8000`, `mulaw_8000`, `alaw_8000`.
- Voices: Piper `default`; Kokoro e.g. `af_heart`; custom `custom:<id>`.

## Errors

`ReadAloudError` > `ApiError` (`.status`) > `AuthError` (401), `QuotaError` (402),
`CapacityError` (close 1013 / "at capacity" / 503, `.retryAfter`), `VoiceError` (unknown voice,
including an unknown `custom:<id>`).

```js
import { CapacityError } from 'readaloud';
try { await client.convert(text); }
catch (e) { if (e instanceof CapacityError) await new Promise(r => setTimeout(r, (e.retryAfter ?? 1) * 1000)); else throw e; }
```

## Security

The client takes your **API key**, so run it server-side and keep the key in an environment
variable or secret store. Never put the key in browser code: every user could read it. This
client always authorizes with the key itself, so for browser apps run a small server endpoint
that calls `POST /tts/authorize` and returns only the short-lived `token` and `url` to the
page, then open the WebSocket (`${url}?token=${token}`) from the browser yourself.

## Development

```
npm test
```

MIT licensed.
