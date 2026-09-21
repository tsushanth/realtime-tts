# worker-piper-fly: portable always-on Piper server (no Modal)

`server.py` is the platform-independent build of `../worker-modal-piper/app.py`: plain
FastAPI + uvicorn, env-configured, same protocol as the Kokoro worker (24kHz PCM16, working
mid-utterance `stop`). Built for a cheap always-on CPU box: a Fly Machine in sjc next to
`call-loop-poc` (reachable over Fly's private network, no public hop), Hetzner, or any VM.

Why: Modal always-on CPU is ~$34/core-month (cpu=4, 2GiB ~= $149/month). Fly published
prices: shared-cpu-4x/2GB $13.27/mo, shared-cpu-2x/1GB $6.64/mo, performance-2x/4GB
$64.39/mo (Amsterdam baseline, other regions differ). Piper needs ~0.1 core per live
stream, so one small box covers a few concurrent calls. NOT yet verified: how Fly shared
CPUs behave under a sustained burst - benchmark on the real target before choosing a size.

## Run
    # model is not in git (63MB): fetch from the Modal volume
    mkdir -p models && modal volume get tts-checkpoints piper_full_ft/serving/full_ft.onnx models/ \
      && modal volume get tts-checkpoints piper_full_ft/serving/full_ft.onnx.json models/
    docker build -t piper-tts . && docker run -e AUTH_TOKEN=... -e HOST=0.0.0.0 -p 8080:8080 piper-tts
Env: AUTH_TOKEN (required), MODEL_PATH, ORT_INTRA_THREADS (2), PORT (8080),
HOST ("::" for Fly's IPv6 private network; 0.0.0.0 for IPv4-only).

## Verified locally (Docker, Apple-silicon Mac; latency there is meaningless)
Wrong token rejected; every chunk_meta followed by one correctly sized binary frame;
`stop` cancelled an 8-sentence text after 2 chunks; bad JSON recovered. Not yet deployed
anywhere; no Fly app exists.

## Custom (customer) voices

Send `"voice": "custom:<id>"` in the synthesize message. Any other `voice` value (e.g. a Kokoro
name from an existing client) uses the default voice, unchanged. Voices live in `VOICES_DIR`
(default `/voices`) as `<id>/model.onnx`, `model.onnx.json` and `owner.json` (`{"key_ids": [...]}`,
the gateway API-key ids allowed to use it; required for session-token clients, ignored for the
internal static token). owner.json may also hold `"user_ids": [...]`: the gateway embeds the key's owning user (`uid`, set when the key is issued with an `owner`, or backfilled through `POST /admin/keys/owner`) in session tokens, and a token whose uid is listed may use the voice, so access follows the user across new keys; a token without uid never matches user_ids (key_ids still work for such keys). Either rule suffices. `voice-pipeline/train_job.py` produces the model files. Loaded lazily with
an LRU of `MAX_VOICES` (default 6). Local test: cold first request ~1.9 s, warm ~116 ms; wrong
owner, missing voice and path traversal all return the same "unknown voice" error. Not yet
deployed: needs a Fly volume (or baked image) holding the voices and a sync from the Modal
`voice-models` volume.

## Output formats

Add `"format"` to the WebSocket `synthesize` message (or the HTTP body below):

| format | payload | rate |
|---|---|---|
| `pcm_24000` (default) | 16-bit LE PCM mono, byte-identical to before this field existed | 24 kHz |
| `pcm_8000` | 16-bit LE PCM mono | 8 kHz |
| `mulaw_8000` | G.711 mu-law, 1 byte/sample (Twilio media streams) | 8 kHz |
| `alaw_8000` | G.711 A-law, 1 byte/sample | 8 kHz |

`chunk_meta` now also carries `"format"` and `"sample_rate"`. An unknown format returns
`{"type":"error","message":"unknown format ..."}` and the connection stays usable. 8 kHz output is
resampled once, directly from the model's native 22.05 kHz (polyphase, 80 dB Kaiser low-pass, pass
3.7 kHz / stop 4.3 kHz; tones at 4.6-9 kHz are attenuated by 89-103 dB). G.711 is implemented in
numpy (`audiofmt.py`; `audioop` is gone in Python 3.13) and matches Python 3.11 `audioop` bit for bit on
all 65536 input values; round-trip SNR is ~37 dB. **Opus is not offered**: it needs libopus/ffmpeg
native code in the image, not worth the weight here.

## HTTP streaming: `POST /v1/tts/stream`

Same auth, `MAX_TEXT_CHARS`, capacity (`MAX_CONNECTIONS`, counted in the same `_active`), voice
ownership and billing rules as the WebSocket. Body: `{"text", "voice"?: "default", "speed"?: 1.0, "format"?: "pcm_24000"}`.
The response is chunked raw audio, one sentence written as soon as it is synthesized.

    # stream to a file (session token from POST /tts/authorize {engine:"piper"} -> token + http_url)
    curl -N -X POST https://piper-tts-sjc.fly.dev/v1/tts/stream \
      -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
      -d '{"text":"Thanks for calling, how can I help?","format":"pcm_24000"}' -o out.pcm
    ffmpeg -f s16le -ar 24000 -ac 1 -i out.pcm out.wav
    # telephony: mu-law 8 kHz
    curl -N -X POST .../v1/tts/stream -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
      -d '{"text":"Hello","format":"mulaw_8000"}' | ffmpeg -f mulaw -ar 8000 -ac 1 -i - out.wav

Headers: `Content-Type` is `audio/pcm` (both `pcm_*`), `audio/basic` (`mulaw_8000`) or
`audio/x-alaw-basic` (`alaw_8000`); `X-Sample-Rate`, `X-Audio-Format`. Errors are JSON `{"error"}` with
401 (auth), 400 (bad body/format/empty text), 404 (`unknown voice`, same message for missing and
forbidden), 413 (too long), 503 + `Retry-After: 1` (at capacity). Usage is reported only after the
last sentence was delivered; a client that disconnects mid-stream is not billed and its slot is freed.
CORS (`Access-Control-Allow-Origin: *`, allows `Authorization`/`Content-Type`, OPTIONS preflight) applies to this
path only; auth is the bearer token, never cookies. The gateway's `/tts/authorize` returns `http_url`
(derived from `PIPER_WORKER_URL`) for `engine:"piper"`.

## Public (house) voices

`owner.json` may be `{"key_ids": [<non-empty>]}` (as before) or `{"public": true}`: any authenticated client
may then use `custom:<id>`. Admin PUT rejects anything else. Missing and forbidden voices give the same error.

Optional `"speaker_id": <int>` in `owner.json` pins a speaker of a multi-speaker model (e.g. the MLS or VCTK
voices), so one model file can back several distinct voices. It is passed as `SynthesisConfig(speaker_id=...)`.
Omitted = unchanged behaviour. A non-integer, negative or out-of-range value (or non-zero on a single-speaker
model) fails the voice with `voice misconfigured: ...` at first use.

## First-chunk split (`FIRST_CHUNK_SPLIT=1`, default OFF)

Piper synthesizes a whole sentence before sending anything, so time-to-first-audio grows with the first
sentence's length. With the flag on, only the FIRST sentence is cut at its earliest clause boundary
(`,` `;` `:`, em/en dash as espeak reports them) when it has more than 8 words and both halves keep at
least 3; otherwise nothing changes. The punctuation stays on the first half so espeak's continuation
intonation is preserved. A spaced hyphen (" - ") is dropped by espeak and is not a boundary.

Measured with `bench_first_chunk.py` (Apple-silicon Mac, native Python 3.11 + piper 1.8.0 + onnxruntime 1.30, 4 threads,
20 runs, six 18-23 word call-center sentences; absolute numbers are NOT Fly numbers, compare off vs on;
phonemize + first-chunk inference + encode, median):

| sentence | first-chunk words off -> on | off ms | on ms |
|---|---|---|---|
| 1 | 21 -> 6 | 157.9 | 72.2 |
| 2 | 23 -> 9 | 159.7 | 85.4 |
| 3 | 19 -> 5 | 143.9 | 55.9 |
| 4 | 18 -> 18 (only comma leaves 2 words: no split) | 156.1 | 156.1 |
| 5 | 18 -> 3 | 176.3 | 48.8 |
| 6 | 23 -> 5 | 198.9 | 53.8 |
| mean | | 165.5 | 78.7 |

Model time is roughly linear in words, so this is about -55% on long first sentences. Second chunks take
50-200 ms while the first plays for 1.5-3 s, so there is no underrun in a streaming client.

Prosody risk (honest): each chunk is synthesized independently, so the second half loses the model's
context: pitch reset, slightly different pace, and a real gap at the seam (each chunk carries its own near-silent edges; not measured in ms).
Measured loudness step at the seam is small (-0.4 to +2.0 dB, chunks are peak-normalized) but audibility is a
judgement call: listen to `split_samples/off/NN.wav` vs `split_samples/on/NN.wav` (24 kHz mono, chunks
concatenated exactly as a client hears them, seam included) before enabling. Very short first chunks
(sentence 5, "Yes, of course,") are the most likely to sound choppy; raising the 3-word minimum trades away
part of the gain. Default stays off.

## Tests

    # unit (no model): G.711 vs audioop, SNR, anti-aliasing, default-path identity
    PYTHONPATH=. python3.11 -m unittest discover -s tests -p 'test_audiofmt.py' -v
    # needs the model: byte-identical default output, format consistency, split_clause
    SESSION_SECRET=x MODEL_PATH=models/full_ft.onnx PYTHONPATH=. python3.11 -m unittest discover -s tests -p 'test_engine_inprocess.py' -v
    # end to end against a running server on :8099 (WS + HTTP + billing + capacity + voices; needs Node >= 22)
    node tests/integration.mjs
