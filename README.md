# realtime-tts

Standalone realtime (streaming, low-latency, barge-in capable) TTS service.
Gateway on Fly.io, GPU inference worker designed for RunPod. See `DECISIONS.md` for the
reasoning behind each choice.

## Architecture

```
client (WS) --> gateway (Fly, server.js, dumb proxy) --> worker (RunPod GPU or local, Kokoro-82M)
                     |
                 /health

Protocol (client <-> worker, proxied verbatim by the gateway):
  -> {"type":"synthesize","text":"...","voice":"af_heart","speed":1.0}
  -> {"type":"stop"}                          # barge-in / cancel
  <- {"type":"chunk_meta","text":..,"gen_ms":..,"audio_s":..}
  <- <binary PCM16LE mono 24kHz>              # immediately follows chunk_meta
  <- {"type":"done"} | {"type":"cancelled"} | {"type":"error","message":".."}
```

## Current real status (as of this build)

- **Worker**: implemented, tested, and RUNNING locally on CPU (Kokoro-82M via
  `kokoro-onnx`). Not deployed to RunPod — no RunPod account/API key exists yet.
- **Gateway**: implemented (`gateway/server.js`), NOT deployed to Fly in this session
  (a valid `WORKER_WS_URL` needs to point at a real worker first — pointing it at
  `localhost` from Fly's network doesn't work; deploy the worker to RunPod first, then
  the gateway).
- **Harness**: implemented and actually run end-to-end against the local worker. Real
  numbers, from `harness/last_local_run.txt`:
  - concurrency=1: TTFB p50=502ms p95=502ms, RTF=2.77x — **PASS**
  - concurrency=4: TTFB p50=1552ms p95=4106ms, RTF=2.40x — **FAIL** (single CPU process
    serializes synthesis under load; see DECISIONS.md)

## Run the worker locally

```bash
cd worker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models
curl -sL -o models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -sL -o models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
python server.py   # ws://0.0.0.0:8765
```

## Run the harness

```bash
cd harness
../worker/.venv/bin/python run.py --endpoint ws://127.0.0.1:8765 --concurrency 4
```

Results accumulate in `harness/history.sqlite3`; regression detection compares each
run's p50 against the rolling baseline of the last 100 successful runs on that endpoint.

For continuous running (cron / a scheduled Fly Machine / GH Actions cron):

```bash
ENDPOINT=wss://realtime-tts-gateway.fly.dev/tts CONCURRENCY=8 INTERVAL=300 ./loop.sh
# or a single pass for cron:
ONE_SHOT=1 ENDPOINT=... ./loop.sh
```

## Deploy the gateway to Fly

```bash
cd gateway
flyctl launch --copy-config --name realtime-tts-gateway   # or flyctl deploy if app exists
flyctl secrets set WORKER_WS_URL=wss://<your-runpod-endpoint>
```

## Deploy the worker to RunPod (NOT done — needs your API key)

1. Get a RunPod account + API key: https://www.runpod.io/console/user/settings
2. `export RUNPOD_API_KEY=...` locally, install the RunPod CLI or use their web console
3. Build and push `worker/Dockerfile` to a registry RunPod can pull from (Docker Hub, GHCR)
4. Create a RunPod Serverless endpoint pointing at that image, GPU tier of your choice
   (a T4 or L4 is plenty for Kokoro-82M — it's a small model)
5. RunPod gives you an endpoint URL — put it in the gateway's `WORKER_WS_URL` Fly secret
   (note: `handler.py` uses RunPod's request/response streaming API, not raw WebSockets —
   the gateway's `WORKER_WS_URL` proxy assumes a WS-speaking worker like `worker/server.py`;
   if you deploy via RunPod Serverless's HTTP job API instead, the gateway needs a small
   adapter to translate WS <-> RunPod's HTTP polling/streaming — not built here, flagged
   as the next real piece of work)

## Known limitations

- Single-process worker serializes concurrent synthesis (see DECISIONS.md) — needs
  either RunPod's own concurrency handling per-endpoint or multiple worker replicas
  behind the gateway (not built — gateway currently proxies to one fixed WORKER_WS_URL).
- Gateway <-> RunPod Serverless protocol mismatch noted above — needs a small adapter
  layer before this can go live against a real RunPod endpoint.
- No auth on the gateway WS endpoint — add before exposing publicly.
- "Continuous improvement" loop is latency/regression detection only. There is no
  automated quality (MOS/naturalness) scoring — that needs either human raters or a
  paid judge model, neither wired up here, and deliberately not added without discussing
  the added spend.
