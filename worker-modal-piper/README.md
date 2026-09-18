# worker-modal-piper: our fine-tuned Piper voice, CPU-only, WebSocket

Drop-in for `worker-modal/` (Kokoro on a T4): same protocol, so `call-loop-poc`
switches by changing `TTS_GATEWAY_WS_URL`. No GPU, so no CUDA init and no ~17.5s
GPU cold start. Deployed as its OWN app (`realtime-tts-worker-piper`), scale-to-zero;
nothing in production points at it yet.

Files: `app.py` (service + `bench` + `phonemize_probe`), `export_onnx.py` (checkpoint ->
ONNX on the `tts-checkpoints` volume), `test_protocol.py` (end-to-end test of the
deployed service, run inside Modal so the auth token never leaves it),
`rtt_probe.py` (network round-trip with zero synthesis).

## What was verified (deployed service, 2026-09-18)
- Auth: wrong token rejected (HTTP 403 at handshake).
- Framing: every `chunk_meta` is followed by exactly one binary frame; PCM byte
  length == `audio_s` x 24000 x 2 for all chunks (24kHz mono PCM16, resampled from
  Piper's native 22,050Hz).
- `stop` works mid-utterance: 8-sentence text stopped after 3 chunks, terminal
  `cancelled`. (The Kokoro worker only reads the socket between requests, so its
  `stop` cannot interrupt synthesis.)
- `speed` works (0.8x -> 3.11s, 1.25x -> 2.29s for the same sentence).
- Bad JSON / unknown type return `error` and leave the connection usable.
- 4 / 8 / 16 concurrent callers x 3 requests: all completed, all framing correct.

## Latency, decomposed (do not quote a single number without this)
Warm request, ~2.6-4.3s sentences:
- Server-side synthesis (`gen_ms`): **160-265ms** with `ORT_INTRA_THREADS=2`.
- Phonemization: 0.5ms. Resample + framing: negligible.
- Network round trip in the test topology: **260ms** (measured with an error reply that
  does no synthesis). Test client and container are in different regions, so this is a
  property of the test, not the service. `*.modal.run` is us-east-1 (see DECISIONS.md).
- Cold (scale-to-zero, one measurement): **~3.0s** connect -> first audio, vs ~17.5s
  for Kokoro/T4.

Baselines from DECISIONS.md: Kokoro/T4 production ~408ms warm per turn, ~970ms first
turn; ElevenLabs ~170-430ms. **Honest read:** warm server-side latency is the same
ballpark as Kokoro's GPU, not a clear win. The wins are cold start (~3s vs ~17.5s),
no GPU floor, and working `stop`. A true end-to-end number (Fly -> Cloudflare -> Modal)
has NOT been measured; it needs a real call routed here.

## Capacity (`modal run app.py::bench_main`, cpu=4, single container)
| ORT threads | 1 caller p50 | 4 callers p50 | audio-s per wall-s @ 4 / 12 callers |
|---|---|---|---|
| 1 | 499ms | 465ms | 36 / 94 |
| 2 | 346ms | 433ms | 40 / 97 |
| 4 | 245ms | 337ms | 50 / 107 |
Caveat: at 12 callers throughput (94-107) exceeds what 4 cores can do (one core ~10x
realtime), so Modal's `cpu=4` reservation was bursting. Treat **~35-40 continuously
speaking streams per container** as the safe figure; more is burst. Real calls are
bursty (TTS only while the agent speaks), so calls-per-container is higher than streams.
`ORT_INTRA_THREADS=2` was chosen conservatively because of that burst caveat; 4 gave
lower latency everywhere in the benchmark (~100ms less on a single call) and is worth
trying if the container proves not to be core-starved.

espeak-ng thread safety: 200 concurrent phonemizations, 0 mismatches, no crash. That
does not prove it is safe, so phonemization stays serialized under a lock (0.5ms cost).

## Open decisions (not made)
1. `MIN_CONTAINERS`: 0 = no standing cost, ~3s worst-case first call; 1 = always warm,
   continuous billing. CPU cost not verified here - check Modal billing after enabling.
2. Pin the container region to match where call-loop-poc runs (Modal `region=`), to cut RTT.
3. Route a real call here (change `TTS_GATEWAY_WS_URL`, or the Cloudflare edge worker
   origin) for the first true end-to-end latency measurement.
4. Sample-rate contract is 24kHz; if call-loop-poc downsamples to 8kHz telephony,
   resampling straight to 8k would skip a step (not investigated).
