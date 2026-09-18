# Benchmark scripts

Node scripts (need the `ws` package) used for the Piper vs ElevenLabs numbers. Run them
from a machine in the same region as the service (we used a Fly.io machine in San Jose,
reaching Piper over private networking) - running from a laptop measures your home internet.

- `latency_probe.js` - interleaved time-to-first-audio for Piper and ElevenLabs. Env:
  `TTS_GATEWAY_API_KEY` (Piper static token), `ELEVENLABS_API_KEY`, optional `ELEVENLABS_VOICE_ID`.
- `load_test.js` / `load_test_continuous.js` - realistic-call and continuous concurrent load.
  Env: `TTS_GATEWAY_API_KEY`. URL is hardcoded to the private Fly address; edit for yours.
- `elevenlabs_samples.js` - renders the five test sentences via ElevenLabs.

Results and caveats: see ../worker-modal-piper/README.md and ../worker-piper-fly/PUBLIC_DRAFT.md.
