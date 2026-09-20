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
internal static token). `voice-pipeline/train_job.py` produces the model files. Loaded lazily with
an LRU of `MAX_VOICES` (default 6). Local test: cold first request ~1.9 s, warm ~116 ms; wrong
owner, missing voice and path traversal all return the same "unknown voice" error. Not yet
deployed: needs a Fly volume (or baked image) holding the voices and a sync from the Modal
`voice-models` volume.
