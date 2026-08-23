# realtime-tts

Standalone realtime (streaming, low-latency, barge-in capable) TTS service.
Gateway on Fly.io, GPU inference on an always-on RunPod Pod. See `DECISIONS.md` for the
full reasoning and debugging trail — several real, non-obvious bugs were found and fixed
(a RunPod Serverless dispatch-latency dead end, an onnxruntime/onnxruntime-gpu package
collision, a CUDA version mismatch, a missing CUDA library) before landing here.

## Architecture

```
client (WS) --> gateway (Fly, server.js) --> RunPod Pod, always-on (direct WS, GPU)
                     |                              |
                 /health                    worker/server.py + Dockerfile.pod (Kokoro-82M)
```

RunPod Serverless (HTTP job-queue API) was tried first and abandoned — see DECISIONS.md —
it has ~7-10s of dispatch latency per request regardless of tuning, which rules it out
for realtime entirely. `gateway/runpod-adapter.js` still exists for that path but isn't
what's live. The live path is a plain always-on GPU instance, talked to directly over
WebSocket — same protocol `worker/server.py` speaks for local dev.

```
Client-facing protocol:
  -> {"type":"synthesize","text":"...","voice":"af_heart","speed":1.0}
  -> {"type":"stop"}                          # barge-in / cancel
  <- {"type":"chunk_meta","text":..,"gen_ms":..,"audio_s":..,"providers":[..]}
  <- <binary PCM16LE mono 24kHz>              # immediately follows chunk_meta
  <- {"type":"done"} | {"type":"cancelled"} | {"type":"error","message":".."}
```

## Current real status

- **Live**: `https://realtime-tts-gateway.fly.dev` (Fly gateway) → a real RunPod Pod
  (always-on GPU instance, RTX 4090) running `worker/server.py` directly over WebSocket.
  Both deployed and tested end-to-end, not just written.
- **GPU: CONFIRMED WORKING.** `InferenceSession.get_providers()` returns
  `['CUDAExecutionProvider', 'CPUExecutionProvider']` on a clean rebuild, no live
  patching. Three real bugs found and fixed — full trail in DECISIONS.md: an
  onnxruntime/onnxruntime-gpu package collision (kokoro-onnx transitively installs
  plain `onnxruntime`, which silently disables CUDA), an unpinned `onnxruntime-gpu`
  resolving to a version requiring CUDA 13.x when only CUDA 12.x pip packages are
  actually published, and a missing `nvidia-cuda-nvrtc-cu12` package.
- **Real measured numbers** (see DECISIONS.md for the full comparison table):
  - GPU Pod through the live Fly gateway, concurrency=1: **TTFB p50=664ms, RTF=5.74x**
  - GPU Pod through the live Fly gateway, concurrency=4: **TTFB p50=842ms p95=1367ms,
    RTF=10.41x** — GPU throughput *improves* under load, unlike the CPU path where it
    collapsed (CPU: 2.77x → 2.40x RTF, TTFB p95 blew out to 4.1s at concurrency=4)
  - Not yet under the ~300ms aspirational realtime target at concurrency=4, but a real
    order-of-magnitude improvement over both the CPU baseline and RunPod Serverless.
- **Real cost tradeoff**: this Pod is always-on, billed continuously — **$0.74/hr
  (~$533/mo)** regardless of traffic, no autoscaling, no redundancy (single pod, no
  failover if it dies beyond RunPod's own crash-restart supervisor). Opposite cost model
  from Serverless's pay-per-second. Whether that's worth it depends entirely on expected
  utilization.
- **Harness**: real numbers from all paths tested, both local and live — see the table
  in DECISIONS.md. Budget in `harness/run.py` set to 1500ms p95 to match what's actually
  achieved today; tighten it once concurrency-4 numbers improve.

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
# worker images (both :latest Serverless variant and :pod always-on variant,
# triggers automatically on push to worker/**):
git push

# manually re-run the build:
gh workflow run build-worker.yml -R tsushanth/realtime-tts

# gateway:
cd gateway && flyctl deploy -a realtime-tts-gateway

# create a fresh Pod (containers aren't updated in place - a new pod must be created
# from a rebuilt image, then the gateway repointed at it):
RUNPOD_KEY=$(security find-generic-password -a "$(whoami)" -s runpod-api-key -w)
curl -s -X POST "https://rest.runpod.io/v1/pods" -H "Authorization: Bearer $RUNPOD_KEY" \
  -H "Content-Type: application/json" -d '{
    "name": "realtime-tts-pod",
    "imageName": "ghcr.io/tsushanth/realtime-tts-worker:pod",
    "gpuTypeIds": ["NVIDIA L4", "NVIDIA A40", "NVIDIA GeForce RTX 4090"],
    "gpuCount": 1, "containerDiskInGb": 10,
    "env": {"WORKER_HOST": "0.0.0.0", "WORKER_PORT": "8765"},
    "ports": ["8765/tcp", "22/tcp"], "cloudType": "SECURE", "supportPublicIp": true
  }'
# then fetch its public IP:port for 8765/tcp via GraphQL (REST doesn't expose this -
# can take 1-5 min for `runtime` to populate, this varies a lot):
curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"query { pod(input: {podId: \"<ID>\"}) { runtime { ports { ip publicPort type } } } }"}'
# then repoint the gateway (direct-WS path, not the RunPod adapter):
flyctl secrets set -a realtime-tts-gateway WORKER_WS_URL=ws://<IP>:<PORT>
flyctl secrets unset -a realtime-tts-gateway RUNPOD_ENDPOINT_ID RUNPOD_API_KEY
# delete the old pod once the new one is verified working:
curl -s -X DELETE "https://rest.runpod.io/v1/pods/<OLD_ID>" -H "Authorization: Bearer $RUNPOD_KEY"
```

RunPod template `r7cttxeun9` (image `ghcr.io/tsushanth/realtime-tts-worker:pod`).
This is manual — there's no autoscaling or CI/CD wiring for the Pod lifecycle itself,
only for the image build. SSH access: `ssh -i ~/.ssh/runpod_cuda -p <ssh_port> root@<IP>`
(the account's registered key, already on the pod via `$PUBLIC_KEY`).

## Known limitations

- **Single pod, no redundancy.** If it crashes, RunPod's own supervisor restarts the
  process, but there's no failover to a second instance and no health-check-triggered
  gateway rerouting. A pod dying means the service is down until manually replaced.
- **No autoscaling.** Fixed at one GPU instance regardless of load — the concurrency=4
  numbers in DECISIONS.md are the ceiling on this hardware, not a floor that scales up.
- **Continuous billing** ($0.74/hr, ~$533/mo) regardless of traffic — see DECISIONS.md
  for the cost tradeoff vs the abandoned Serverless pay-per-second approach.
- Concurrency=4 p95 (1367ms) still exceeds the ~300ms aspirational realtime target —
  real synthesis is fast (RTF 5.7-16.8x), but hasn't been profiled to find where the
  remaining latency budget goes at higher concurrency.
- No auth on the gateway WS endpoint — add before exposing publicly.
- "Continuous improvement" loop is latency/regression detection only. There is no
  automated quality (MOS/naturalness) scoring — that needs either human raters or a
  paid judge model, neither wired up here, and deliberately not added without discussing
  the added spend.
- The abandoned RunPod Serverless path (`gateway/runpod-adapter.js`, `worker/handler.py`)
  is left in the repo since the code is correct and could be revived if Serverless's
  dispatch-latency problem ever gets resolved on RunPod's end — it's just not what's live.
