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
- **GPU status: CONFIRMED not working, root cause identified.** See DECISIONS.md for the
  full investigation. Bottom line: this RunPod account/region is handing out an NVIDIA
  RTX PRO 6000 Blackwell GPU regardless of the `gpuTypeIds` requested (tried restricting
  to L4-only explicitly — still got Blackwell), and `onnxruntime-gpu` 1.29 doesn't have
  compiled kernels for that hardware yet. Confirmed directly via `nvidia-smi` (GPU is
  attached) and `InferenceSession.get_providers()` (only CPUExecutionProvider active) —
  not inferred from timing. The worker runs correctly and reliably, just on CPU, while
  RunPod bills GPU-tier rates. Live endpoint currently pointed at: `u4me5box1h735i`.
- **NOT REALTIME as currently deployed.** RunPod Serverless's own job-dispatch queue adds
  ~7s of latency even to an already-warm, idle worker (confirmed directly — `delayTime`
  in a job's status response, not inferred). Tried switching to an always-on RunPod Pod
  (direct WebSocket, no queue) to fix this — didn't get it connecting (persistent 404 on
  the WS proxy, `runtime: null` from RunPod's own API even after 90+s), tore it down
  rather than keep guessing. Full writeup in DECISIONS.md, including concrete next steps
  for whoever continues this. Current live backend is correct and cheap, just slow.
- **Harness**: real, run both locally (against `worker/server.py` on CPU) and against the
  live Fly→RunPod path.
  - Local CPU, concurrency=1: TTFB p50=502ms p95=502ms, RTF=2.77x — **PASS**
  - Local CPU, concurrency=4: TTFB p50=1552ms p95=4106ms, RTF=2.40x — **FAIL** (single
    process serializes synthesis under load — see DECISIONS.md)
  - Live Fly→RunPod, concurrency=1: TTFB ~3.4-9.6s depending on worker warm/cold state
  - Live Fly→RunPod, concurrency=4: TTFB p50=9.6-16s, p95=14.5-23s — scaling workers
    doesn't fix this; RunPod's queue dispatch overhead dominates regardless of worker count
  - None of the above meet the CPU-tuned 1200ms budget in `harness/run.py` once RunPod is
    in the path — that budget was only ever valid for the direct-WS local path.

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
`u4me5box1h735i`, template `r7cttxeun9`, image `ghcr.io/tsushanth/realtime-tts-worker:latest`.
CPU-only (see "Current real status"), `workersMax: 1`.

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
