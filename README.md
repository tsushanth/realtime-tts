# realtime-tts

Standalone realtime (streaming, low-latency, barge-in capable) TTS service.
Gateway on Fly.io, GPU inference worker on RunPod Serverless. See `DECISIONS.md` for the
reasoning behind each choice — including a real bug (GPU silently falling back to CPU)
found and partially fixed during this build.

## Architecture

```
client (WS) --> gateway (Fly, server.js) --> RunPod Serverless (HTTP job API, GPU)
                     |                              |
                 /health                    worker/handler.py (Kokoro-82M)

Client-facing protocol (same whether the gateway talks to RunPod or a direct-WS worker):
  -> {"type":"synthesize","text":"...","voice":"af_heart","speed":1.0}
  -> {"type":"stop"}                          # barge-in / cancel
  <- {"type":"chunk_meta","text":..,"gen_ms":..,"audio_s":..}
  <- <binary PCM16LE mono 24kHz>              # immediately follows chunk_meta
  <- {"type":"done"} | {"type":"cancelled"} | {"type":"error","message":".."}
```

`gateway/runpod-adapter.js` translates RunPod's HTTP job API (POST /run, poll /stream,
POST /cancel) to this wire protocol. `gateway/server.js` also supports talking directly
to a WS-speaking worker (`worker/server.py`) for local dev — set `WORKER_WS_URL` instead
of `RUNPOD_ENDPOINT_ID`/`RUNPOD_API_KEY`.

## Current real status

- **Live**: `https://realtime-tts-gateway.fly.dev` (Fly), wired to a real RunPod
  Serverless endpoint running the Kokoro-82M worker. Both were actually deployed and
  tested end-to-end in this build, not just written.
- **Image**: `ghcr.io/tsushanth/realtime-tts-worker:latest`, built via GitHub Actions
  (`.github/workflows/build-worker.yml`) — local Docker builds kept failing (disk space,
  then a corrupted Docker Desktop containerd DB, then network drops pushing a 10GB+ CUDA
  base image). Switched to a slim Python base + explicit `nvidia-*-cu12` pip packages
  instead of a full CUDA image; GHA builds/pushes reliably in under a minute.
- **GPU status: partially verified, not fully resolved.** See DECISIONS.md. First deploy
  silently ran on CPU despite being a "GPU" worker (`onnxruntime-gpu` doesn't pull CUDA
  libs via pip automatically — a wrong assumption baked into an earlier version of this
  file). Fixed the library wiring; a direct RunPod test afterward measured RTF~2.0x, but
  a run minutes later through the full gateway pipeline measured RTF~0.52x on the same
  endpoint — inconsistent, likely tied to the endpoint's 10s idle timeout recycling
  workers faster than CUDA reliably initializes on cold start. Needs more measurement
  before trusting GPU numbers here.
- **Harness**: real, run both locally (against `worker/server.py` on CPU) and against the
  live Fly→RunPod path.
  - Local CPU, concurrency=1: TTFB p50=502ms p95=502ms, RTF=2.77x — **PASS**
  - Local CPU, concurrency=4: TTFB p50=1552ms p95=4106ms, RTF=2.40x — **FAIL** (single
    process serializes synthesis under load — see DECISIONS.md)
  - Live Fly→RunPod, concurrency=1: TTFB ~3.4-4.0s (dominated by RunPod's HTTP job
    polling + cold-start variance, not raw synthesis) — well outside the CPU-tuned
    1200ms budget in `harness/run.py`. That budget needs to be re-tuned for the RunPod
    HTTP path specifically (it has fundamentally different latency characteristics than
    a direct WS connection) — not done yet.

## Run the worker locally (direct WS, no RunPod)

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
# against local direct-WS worker:
worker/.venv/bin/python harness/run.py --endpoint ws://127.0.0.1:8765 --concurrency 4

# against the live gateway (RunPod-backed):
worker/.venv/bin/python harness/run.py --endpoint wss://realtime-tts-gateway.fly.dev/tts --concurrency 1
```

Results accumulate in `harness/history.sqlite3`; regression detection compares each
run's p50 against the rolling baseline of the last 100 successful runs on that endpoint.
The 1200ms TTFB budget in `run.py` was tuned for the CPU baseline — it will (correctly)
fail against the RunPod path until that path's own budget is set.

For continuous running (cron / a scheduled Fly Machine / GH Actions cron):

```bash
ENDPOINT=wss://realtime-tts-gateway.fly.dev/tts CONCURRENCY=4 INTERVAL=300 ./harness/loop.sh
ONE_SHOT=1 ENDPOINT=... ./harness/loop.sh   # single pass for cron
```

## Redeploying

```bash
# worker image (triggers automatically on push to worker/**):
git push

# manually re-run the build:
gh workflow run build-worker.yml -R tsushanth/realtime-tts

# gateway:
cd gateway && flyctl deploy -a realtime-tts-gateway

# point the gateway at a different RunPod endpoint:
flyctl secrets set -a realtime-tts-gateway RUNPOD_ENDPOINT_ID=<id> RUNPOD_API_KEY=<key>
```

RunPod template/endpoint were created via `rest.runpod.io/v1/templates` and
`/v1/endpoints` (see git history for the exact calls). Current live endpoint:
`jyegeieycq613x`, template `r7cttxeun9`, image `ghcr.io/tsushanth/realtime-tts-worker:latest`.

## Known limitations

- **GPU acceleration is unverified/inconsistent** — see "Current real status" above and
  DECISIONS.md. Don't trust RTF numbers from the RunPod path yet without re-checking
  `ort.get_available_providers()` inside a live worker.
- Single-process worker serializes concurrent synthesis on the direct-WS path (see
  DECISIONS.md). RunPod Serverless handles its own worker scaling (`workersMax`), but
  that hasn't been load-tested past concurrency=1 through the full pipeline.
- The harness's 1200ms TTFB budget is CPU-tuned and needs a separate, RunPod-specific
  budget — the HTTP job API's polling overhead means it will never hit CPU-direct-WS
  latency even with a fast GPU.
- No auth on the gateway WS endpoint — add before exposing publicly.
- "Continuous improvement" loop is latency/regression detection only. There is no
  automated quality (MOS/naturalness) scoring — that needs either human raters or a
  paid judge model, neither wired up here, and deliberately not added without discussing
  the added spend.
- `idleTimeout: 10s` on the RunPod endpoint means workers recycle fast between requests,
  which is cheap but may be contributing to the CUDA-init inconsistency noted above —
  worth testing with a longer idle timeout or `workersMin: 1` to keep one warm.
