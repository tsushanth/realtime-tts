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

**RESOLVED — real GPU, real always-on Pod, both root causes found via live SSH
debugging, not guessing.**

**Pod networking (why `runtime: null` / 404 for three straight attempts):** RunPod's
own docs confirm custom Pod images need an SSH daemon for RunPod's infra to report
`runtime` status at all — a plain `python:3.11-slim` base has none. Built
`worker/Dockerfile.pod` + `worker/start-pod.sh` following RunPod's documented pattern
(installs `openssh-server`, wires the account's injected `$PUBLIC_KEY` into
`authorized_keys`, starts `sshd`, then `exec`s the real server). Separately, RunPod's
WSS-over-Cloudflare-proxy (`https://<pod-id>-<port>.proxy.runpod.net`) is independently
unreliable per community reports — the real fix is TCP port exposure
(`ports: ["8765/tcp"]`) plus connecting directly to the pod's public IP:port from
`runtime.ports` (via RunPod's GraphQL API — the REST API doesn't expose this), not the
proxy domain. Both fixes were needed together.

**GPU (why CUDA never activated even once networking worked) — three real bugs, found
by SSHing into a live pod and reading onnxruntime's actual C++ error messages instead
of continuing to guess from Python-level symptoms:**
1. `onnxruntime-gpu` and plain `onnxruntime` install to the same
   `site-packages/onnxruntime/` path and are not safe to have coinstalled —
   `kokoro-onnx` pulls in plain `onnxruntime` transitively, which silently wins the
   file collision and disables CUDA. `kokoro_onnx.resolve_providers()` doesn't just
   check `get_available_providers()`; it checks whether the `onnxruntime-gpu`
   *pip package itself* is still installed (`importlib.metadata.distribution(...)`) —
   so even a `pip uninstall onnxruntime` cleanup after the fact can corrupt
   `onnxruntime-gpu`'s own metadata via the overlapping file paths and make it look
   uninstalled. Fix: install `onnxruntime-gpu` first, then `kokoro-onnx`'s other real
   deps (checked via `pip show kokoro-onnx`), then `kokoro-onnx` itself with `--no-deps`
   — plain `onnxruntime` never touches disk.
2. Unpinned `onnxruntime-gpu` resolved to 1.29.0, which onnxruntime's own runtime error
   states requires **CUDA 13.x** — but real CUDA-13 pip packages
   (`nvidia-cublas-cu13` etc.) are still unpublished stubs (`0.0.1` placeholder
   releases on PyPI as of this build). Pinned to `onnxruntime-gpu==1.20.2`, the newest
   version confirmed (via web research, then verified live) to target CUDA 12.x +
   cuDNN 9.x, which has real, fully-published packages.
3. `nvidia-cuda-nvrtc-cu12` (provides `libnvrtc.so.12`, needed for CUDA JIT
   compilation) was missing from the package list entirely — onnxruntime's CUDA
   provider load error names its missing `.so` files one at a time, so this took two
   rounds of "install the named library, see what's missing next" to fully surface.

Confirmed end to end, clean rebuild, no live patching: `InferenceSession.get_providers()`
→ `['CUDAExecutionProvider', 'CPUExecutionProvider']` on a real RTX 4090 Pod.

**Real measured numbers, GPU Pod vs CPU Serverless:**

| Path | Concurrency | TTFB p50 | TTFB p95 | RTF |
|---|---|---|---|---|
| CPU direct-WS (local) | 1 | 502ms | 502ms | 2.77x |
| CPU direct-WS (local) | 4 | 1552ms | 4106ms | 2.40x |
| GPU Pod, direct (no gateway) | 1 | 594ms | 594ms | 8.28x |
| GPU Pod, direct (no gateway) | 4 | 637ms | 1412ms | 16.82x |
| **GPU Pod, through live Fly gateway** | 1 | **664ms** | **664ms** | **5.74x** |
| **GPU Pod, through live Fly gateway** | 4 | **842ms** | **1367ms** | **10.41x** |

GPU RTF *improves* under concurrency (8.28x → 16.82x direct) instead of collapsing like
CPU did (2.77x → 2.40x, with TTFB p95 blowing out to 4.1s) — real evidence the GPU has
headroom the single CPU process didn't. Still not under the ~300ms aspirational realtime
target at concurrency=4, but a legitimate order-of-magnitude improvement over both the
CPU baseline and RunPod Serverless (~7-10s dispatch latency, ruled out separately above).

**Real operational tradeoff, stated plainly**: this Pod is **always-on and billed
continuously** — $0.74/hr ≈ $533/mo regardless of traffic, no autoscaling, no
redundancy (single pod, single point of failure; RunPod's own container supervisor
restarts a crashed process but there's no failover to a second instance). This is the
opposite cost model from Serverless's pay-per-second. Whether that tradeoff is worth it
depends on expected utilization — cheap at high, sustained traffic; wasteful idle. Not
addressed in this session: horizontal scaling (multiple pods behind the gateway),
autoscaling based on load, or health-check-based failover if the pod dies.

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

**LLM connection reuse & prompt caching: investigated, NOT worth changing (call-loop-poc).**
LLM TTFB in real calls sits at 550-900ms, and the question was whether per-turn
TCP+TLS handshake overhead to `api.anthropic.com` was part of that, and whether
Anthropic prompt caching (`cache_control`) could help turns 2+. Both were measured, both
came back "leave it alone."

1. *Connection reuse already works.* `@anthropic-ai/sdk` 0.32.1 runs on `node-fetch`
   v2 with a module-level `agentkeepalive` HttpsAgent (`keepAlive: true`) — see
   `node_modules/@anthropic-ai/sdk/_shims/node-runtime.js:54`. `server.js` constructs the
   client with no `httpAgent`/`fetch` override and always drains the stream via
   `stream.finalMessage()`, so nothing defeats pooling. A local mock-TLS test (the exact
   SDK version, no key) fired 6 sequential `messages.stream()` calls and observed exactly
   ONE TCP socket — turn 1 handshakes, turns 2..N reuse. Confirmed for the SDK's *default*
   agent too (no `httpAgent` passed).
2. *The handshake is negligible anyway.* Instrumented the deployed Fly app (sjc) with a
   temporary agent that logged NEW-socket vs REUSE plus TCP/TLS timing, then drove
   mixed-gap turns through the `user_text` WS path. Full TCP+TLS handshake to Anthropic
   from Fly measured **~7-14ms**. NEW-socket turns and REUSE turns had statistically
   indistinguishable TTFB (630-935ms across both). So the 550-900ms is essentially all
   Anthropic's own response time — matching the earlier isolated laptop test (530-1009ms)
   — not connection overhead. (Instrumentation was deployed to test, then reverted; prod
   is back on clean HEAD.)
3. *One real-but-tiny nuance:* the SDK sets the socket `timeout` to 5min but leaves
   `freeSocketTimeout` at agentkeepalive's 4000ms default, so a pooled socket idle >4s
   (common between real phone-call turns) is dropped and the next turn re-handshakes.
   Confirmed live (turns fired 7s apart got fresh sockets; back-to-back reused). Bumping
   `freeSocketTimeout` to ~60s via a custom `httpAgent` would keep the socket warm across
   turns — but it only saves that ~10ms handshake, well under the "few tens of ms" bar,
   so it wasn't worth adding a config override for. Documented here in case handshake
   cost ever grows (e.g. a region move putting Anthropic further away network-wise).
4. *Prompt caching doesn't apply.* The model is Claude Haiku 4.5, whose minimum cacheable
   prefix is **4096 tokens** (the highest of any current model). The flat system prompt is
   ~45 tokens; per-node flow prompts ~200-270 tokens; plus a small tool schema and short
   phone-call history — total input per turn is a few hundred tokens, an order of
   magnitude below the minimum, so `cache_control` would silently not cache
   (`cache_creation_input_tokens: 0`) with zero TTFB benefit. Worse, the cacheable prefix
   here isn't even stable: the per-node system prompt is rebuilt every turn with
   interpolated `collectedData`, and KB content is injected mid-`history`, not as a frozen
   prefix. Revisit only if a flow ever front-loads a large (>4k-token) *stable* prefix.

**Fly region: ruled out as a latency factor (call-loop-poc, no code/infra change).**
Same 550-900ms real LLM TTFB question as above, from the network/geography angle instead
of the app-request angle. Ran isolated, timed `POST /v1/messages` requests (Claude Haiku
4.5, streamed, same short prompt) against Anthropic from temporary throwaway machines in
sjc (call-loop-poc's actual region) plus 2-3 other US regions, several trials each, then
tore the throwaway infra down. All four regions clustered within ~40-50ms of each other;
the best case (east-coast iad/ord, ~520ms) beat sjc (~560ms) by only ~40ms — under the
~100ms bar for "worth migrating a live phone-call app over," and well inside per-trial
noise (individual trials ranged 468-1001ms with heavy overlap across regions). There's a
faint, directionally-consistent east-coast edge across both test rounds (mild signal that
Anthropic's serving infra leans US-east), but nowhere near large enough to act on.
Cross-checks cleanly with the connection-reuse finding above: the isolated raw sjc TTFB
(~560ms) sits at the *bottom* of production's measured 550-900ms range, meaning neither
region nor connection overhead explains the range's upper end — whatever pushes a real
turn toward 900ms is Anthropic's own per-request response-time variance, not something
on our side of the wire.
