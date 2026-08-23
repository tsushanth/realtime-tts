# Engineering decisions

**Model: Kokoro-82M (via `kokoro-onnx`).** Already proven in this stack's async TTS
(listenai-tts-worker: ~1.75x realtime on CPU). ONNX runtime means no PyTorch/CUDA
build headaches and it runs fine on CPU for dev.

**GPU status: CONFIRMED CPU-only on this RunPod account, root cause identified but not
fixed.** Long investigation, documented in full because the wrong turns are as
instructive as the answer:

1. First assumption — `onnxruntime-gpu` pulls CUDA runtime libs via pip automatically —
   was wrong. The worker ran "successfully" on RunPod (jobs completed, audio came back)
   while silently running on CPU: `ort.get_available_providers()` inside the built image
   returned only `CPUExecutionProvider`, and RTF measured *worse* than the local CPU
   baseline. Timing-based RTF is not a reliable enough signal for this — a slow "GPU" and
   a CPU fallback look the same from the outside.
2. Added explicit `nvidia-cudnn-cu12`/`nvidia-cublas-cu12`/etc. pip packages plus
   `LD_LIBRARY_PATH`. Timing improved on one test (RTF~2.0x) but regressed on another
   (RTF~0.52x) minutes later on the same endpoint — inconclusive from timing alone.
3. Stopped trusting timing. Added real diagnostics: `chunk["providers"]` now reports
   `InferenceSession.get_providers()` directly (the actual active providers, not just
   what's "available"), and the RunPod handler runs `nvidia-smi -L` and returns it.
   Three separate sequential RunPod jobs, post-fix, all directly confirmed
   `["AzureExecutionProvider", "CPUExecutionProvider"]` — **no CUDAExecutionProvider,
   definitively, not inferred.**
4. `nvidia-smi` confirmed a real GPU device IS attached and visible to the container:
   an **NVIDIA RTX PRO 6000 Blackwell Server Edition** (via MIG 1g.24gb partition) — so
   this isn't a "no GPU attached" problem.
5. That GPU was never requested — the endpoint's `gpuTypeIds` was set to
   `["NVIDIA L4", "NVIDIA A40", "NVIDIA GeForce RTX 4090"]`, none of which is Blackwell.
   Tried pinning `gpuTypeIds: ["NVIDIA L4"]` explicitly on a fresh endpoint — RunPod
   **still assigned the same Blackwell card**. `gpuTypeIds` is not being honored as a
   hard constraint on this account/region; Blackwell may be the only capacity actually
   available right now.
6. Blackwell is new enough hardware (compute capability sm_120-class) that
   `onnxruntime-gpu` 1.29 (the version in use) very likely doesn't have compiled CUDA
   kernels for it yet — a hardware/software version-lag problem, not a config bug this
   codebase can fix by itself.

**RunPod Serverless's HTTP job-queue model has ~7s of dispatch latency even to a warm,
idle worker.** Direct test: `POST /run` against an endpoint reporting 1 idle/1 ready
worker still measured `delayTime: 7290ms` before execution even began (`executionTime`
itself was a fast 1195ms). This is RunPod's own queue/dispatch overhead, not something
fixable by tuning the gateway's poll interval or worker count — it rules out Serverless
entirely for a sub-second realtime latency target, independent of the GPU question above.

**Tried switching to a RunPod Pod (always-on instance, direct WebSocket, no job queue)
to fix the dispatch-latency problem — didn't get it working, reverted.** Created a Pod
running `worker/server.py` directly (`dockerStartCmd` override) with port 8765 exposed.
It was assigned a mature RTX 4090 (good — suggests Pods may dodge the Blackwell
assignment issue Serverless hit), but the WS endpoint returned a persistent 404 through
RunPod's proxy (`https://<pod-id>-8765.proxy.runpod.net`) for 90+ seconds, and a second
pod attempt (with SSH added for debugging) showed `runtime: null` via RunPod's GraphQL
API even after similar wait — meaning the container likely never reached a running state,
not just a slow boot. Deleted both pods to stop the $0.74/hr billing rather than keep
guessing blind. **Root cause not diagnosed** — candidates, untested: RunPod's HTTP-type
port proxy may not support WebSocket upgrade at all (only plain HTTP), the `dockerStartCmd`
override may not actually be honored for pods created without a `templateId`, or there
may be a genuine platform-side provisioning issue independent of anything in this repo.

**Stopped here deliberately** — this GPU/latency investigation is now two independent
dead ends (Serverless: works but ~7-10s dispatch latency; Pods: didn't get a working
connection at all) across many endpoint/pod create-test-destroy cycles in one session.
Further blind iteration wasn't a good use of continued effort. **Current live state**:
worker runs correctly and reliably on CPU via RunPod Serverless (jobs complete, audio is
correct, `workersMax` reset to 1 to minimize idle billing) — this is what the gateway
points at right now. It is NOT realtime by the ~300ms TTFB standard; it's a working,
correct, but slow (multi-second TTFB) synthesis backend.

**Recommendations for whoever picks this up, in priority order**:
1. Before anything else: read RunPod's own docs on Pod HTTP-proxy WebSocket support
   (don't assume, as this session did) — if Pods genuinely don't support WS through their
   proxy, that changes the whole approach (would need a public IP + raw TCP instead, or
   a different exposure method).
2. If Pods do support WS, retry with a `templateId`-based Pod (not templateId-less) in
   case that's why `dockerStartCmd` wasn't honored, and add a startup log/healthcheck
   endpoint so a 404 vs a genuine crash can be told apart without SSH.
3. File a RunPod support ticket about the Blackwell `gpuTypeIds` mismatch on Serverless
   — a straight answer there might make GPU Serverless viable again.
4. Or accept CPU-only Serverless for now: per the ElevenLabs cost comparison, this is
   still ~3-6x cheaper per character than ElevenLabs even at CPU speed and multi-second
   latency, since Kokoro-82M is cheap to run regardless of hardware. It's a real, working,
   cheap backend — just not a realtime one. Whether that's good enough depends entirely
   on the actual use case (see the open question from earlier in this conversation about
   what this service is for).

**Transport: WebSocket, not SSE/HTTP streaming.** Realtime TTS needs bidirectional
control (client sends `stop` mid-stream for barge-in) — SSE is one-directional, and
chunked HTTP has no clean cancellation signal from client to server without a second
request. One WS connection per session carries both JSON control messages and binary
PCM frames.

**Chunking: sentence/clause boundaries, ~90 chars.** Splits on `.!?;:` so synthesis
starts on the first clause without waiting for the full utterance. 90 chars was chosen
empirically — testing showed <60 chars produces too many chunks (per-chunk model
call overhead dominates), and >150 chars merges most short conversational utterances
into one chunk, which defeats streaming. This is a knob, not a law — retune per corpus.

**Gateway is a dumb proxy.** `gateway/server.js` does no batching, buffering, or
protocol translation — it terminates the client WS on Fly and pipes bytes 1:1 to the
worker WS. Any latency added by the gateway is pure network hop overhead, not logic.
This makes it easy to reason about: if TTFB is bad, it's the worker's problem, not the
gateway's.

**Latency budget: p95 TTFB < 1200ms on CPU (not the target for GPU).** The realtime
target most sources cite is <300ms for conversational feel. That number is a GPU-serving
number. This build's actual worker runs Kokoro on CPU (RunPod isn't provisioned — see
README), and measured p50 TTFB there is ~500ms at concurrency=1, ~1.5s at concurrency=4.
1200ms is set as the CI-gate budget for this CPU baseline so the harness has something
real to regress against; when RunPod GPU is live, this constant should drop to ~300ms
and the harness will start failing against the new bar until the GPU worker actually
lands under it — that's intentional, not a bug.

**RTF measured against server-reported generation time, not wall-clock-since-first-byte.**
First attempt computed RTF as `audio_duration / (t_end - t_first_byte)`. This is
mathematically broken for single-chunk utterances: t_end - t_first_byte is near zero
because there's nothing to wait for after the first (only) chunk arrives, producing RTF
values in the thousands. Fixed to `audio_duration / total_server_side_gen_ms`, using the
`gen_ms` the worker reports per chunk. Real-world lesson: validate a derived metric
against a degenerate case (N=1 chunk) before trusting its output.

**Concurrency finding (real, measured):** a single CPU worker process serializes
synthesis across concurrent WS sessions (Python executor threads still contend for the
same onnxruntime CPU inference, which isn't free-threaded). Measured: TTFB p50 502ms at
concurrency=1 vs p50 1552ms / p95 4106ms at concurrency=4 — an ~8x degradation, not a
modest one. This is the load pattern a single RunPod GPU replica will also hit past some
concurrency ceiling; the harness's regression check is what will catch it in prod, and
the fix is horizontal worker scaling, not micro-optimizing one process.
