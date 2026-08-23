# realtime-tts

Standalone realtime (streaming, low-latency, barge-in capable) TTS service.
Gateway on Fly.io, GPU inference on a RunPod Pod that's provisioned on demand rather
than run continuously — see "API keys and auto-provisioning" below. See `DECISIONS.md`
for the full reasoning and debugging trail — several real, non-obvious bugs were found
and fixed (a RunPod Serverless dispatch-latency dead end, an onnxruntime/onnxruntime-gpu
package collision, a CUDA version mismatch, a missing CUDA library) before landing here.

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

- **GPU Pod is OFF by default.** It was proven working (real numbers below) then
  deliberately torn down after verification — it's a $0.74/hr (~$533/mo) continuous
  cost with no traffic to justify it yet. **Turn it on only when there's real traffic**,
  via the "Redeploying" steps below (`imageName: ghcr.io/tsushanth/realtime-tts-worker:pod`).
  Turn it off the same way (`DELETE /v1/pods/<id>`) once traffic stops.
- **Live right now**: `https://realtime-tts-gateway.fly.dev` (Fly gateway) → RunPod
  Serverless CPU endpoint (`u4me5box1h735i`) — the cheap, pay-per-second fallback.
  Correct, works, multi-second TTFB (see the CPU-Serverless numbers in DECISIONS.md).
  This is the safe default state between traffic bursts.
- **GPU: CONFIRMED WORKING when it's on.** `InferenceSession.get_providers()` returns
  `['CUDAExecutionProvider', 'CPUExecutionProvider']` on a clean rebuild, no live
  patching. Three real bugs found and fixed — full trail in DECISIONS.md: an
  onnxruntime/onnxruntime-gpu package collision (kokoro-onnx transitively installs
  plain `onnxruntime`, which silently disables CUDA), an unpinned `onnxruntime-gpu`
  resolving to a version requiring CUDA 13.x when only CUDA 12.x pip packages are
  actually published, and a missing `nvidia-cuda-nvrtc-cu12` package.
- **Real measured numbers from the verification run** (see DECISIONS.md for the full
  comparison table):
  - GPU Pod through the live Fly gateway, concurrency=1: **TTFB p50=664ms, RTF=5.74x**
  - GPU Pod through the live Fly gateway, concurrency=4: **TTFB p50=842ms p95=1367ms,
    RTF=10.41x** — GPU throughput *improves* under load, unlike the CPU path where it
    collapsed (CPU: 2.77x → 2.40x RTF, TTFB p95 blew out to 4.1s at concurrency=4)
  - Not yet under the ~300ms aspirational realtime target at concurrency=4, but a real
    order-of-magnitude improvement over both the CPU baseline and RunPod Serverless.
- **Harness**: real numbers from all paths tested, both local and live — see the table
  in DECISIONS.md. Budget in `harness/run.py` set to 1500ms p95 to match the GPU-on
  reality; the CPU-Serverless fallback (currently live) won't meet it — that's expected
  when the harness is pointed at the cheap idle state rather than the GPU pod.

## API keys and auto-provisioning

The gateway supports a third mode (`GATEWAY_MODE=auto`, currently **unset in production**
— see "Current real status") that gates GPU access behind API keys and auto-manages the
Pod lifecycle instead of requiring the manual `scripts/gpu-pod-*.sh` calls:

- Every WS connection must include a valid key: `wss://.../tts?key=<key>` (query param —
  browsers' native WebSocket API can't set custom headers, so this is the only option
  that works from a browser client) or `Authorization: Bearer <key>` (for server-to-server
  callers).
- On the **first** authenticated request after the pod is off, the gateway auto-creates
  it (`gateway/pod-manager.js`) and immediately sends the client
  `{"type":"status","state":"provisioning","message":"..."}` — **being upfront that this
  can take up to 5 minutes**, rather than leaving the connection hanging silently. Once
  ready, `{"type":"status","state":"ready"}` is sent and audio streaming proceeds
  normally.
- After **15 minutes** of no requests (`POD_IDLE_TIMEOUT_MS`), the pod is automatically
  torn down to stop billing.
- Keys are issued manually for now via an admin-only HTTP endpoint (no self-serve signup
  flow yet — that's a separate, unbuilt piece):
  ```bash
  curl -X POST https://realtime-tts-gateway.fly.dev/admin/keys \
    -H "Authorization: Bearer $ADMIN_SECRET" -d '{"label":"customer name or email"}'
  # -> {"key":"rtts_..."}
  curl https://realtime-tts-gateway.fly.dev/admin/keys -H "Authorization: Bearer $ADMIN_SECRET"   # list
  curl -X DELETE https://realtime-tts-gateway.fly.dev/admin/keys \
    -H "Authorization: Bearer $ADMIN_SECRET" -d '{"key":"rtts_..."}'                              # revoke
  ```
  `ADMIN_SECRET` is a Fly secret, generated during this build — it is NOT in this repo or
  git history; retrieve it from wherever it was saved when set, or rotate it with
  `flyctl secrets set -a realtime-tts-gateway ADMIN_SECRET=<new value>` (this invalidates
  the old one immediately, no grace period).
- Keys are stored in plaintext in a JSON file (`gateway/keys.js`) on a Fly volume
  (`realtime_tts_data`, mounted at `/data`) — acceptable only because issuance is manual/
  trusted right now. **Hash them before any self-serve signup flow exists.**

**To actually turn this on**: `flyctl secrets set -a realtime-tts-gateway GATEWAY_MODE=auto`.
This changes live behavior immediately — the gateway will stop accepting unauthenticated
connections and start auto-provisioning on the first valid-keyed request. Don't flip this
until there's an actual reason to (a real customer, a real key to issue) — until then the
service should stay on the CPU Serverless fallback, per explicit direction from this
build's requester ("turn it on when there is traffic not before").

**Not built yet, and worth being explicit about the gap**: a self-serve signup page (was
going to be readaloudai.org — verify that's actually where this product's marketing site
should live before building on it, since `readaloud.org` — note the different TLD — is
an unrelated third-party literacy nonprofit, confirmed by fetching it during this build),
billing/plan tiers, and Stripe integration. Those are separate, sizable pieces of work
not started here.

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

# turn the GPU pod ON (creates a pod, waits for it, wires the gateway to it):
./scripts/gpu-pod-on.sh

# turn the GPU pod OFF (deletes it, reverts gateway to the CPU Serverless fallback):
./scripts/gpu-pod-off.sh
```

RunPod template `r7cttxeun9` (image `ghcr.io/tsushanth/realtime-tts-worker:pod`).
Pod lifecycle is manual and on/off by design (see "Current real status" above) — there's
no autoscaling or CI/CD wiring for it, only for the image build. `gpu-pod-off.sh` reverts
to Serverless endpoint `u4me5box1h735i`, hardcoded — update that constant if it's ever
recreated. SSH access: `ssh -i ~/.ssh/runpod_cuda -p <ssh_port> root@<IP>` (the account's
registered key, already on the pod via `$PUBLIC_KEY`).

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
