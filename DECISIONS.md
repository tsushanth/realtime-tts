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

## Can we match ElevenLabs' TTS latency? (2026-09-15/16)

Full investigation, kept in order because several "obvious fix" hypotheses were tested
and killed, and the dead ends are as load-bearing as the one thing that worked.

**Starting point.** Real production numbers, gateway direct-to-Modal: first turn of a
fresh call ~970ms (WS open + first synth), later turns on a warm connection ~408ms.
ElevenLabs' full round trip (network + their compute): ~170-430ms. So we were 2-3x
slower even in our best case, before accounting for the first-turn tax.

**1. Network-layer isolation.** Connection-only (DNS+TCP+TLS+WS-upgrade, no inference)
timing from a US-West vantage point: Modal ~201ms, ElevenLabs ~90ms. Root cause: every
`*.modal.run` endpoint resolves to `us-east-1` (Virginia) — single-region, no anycast.
ElevenLabs/Replicate terminate TLS at an edge PoP near the client; we pay full
transcontinental RTT x2-3 handshake round trips. Confirmed via connection-reuse test:
cold handshake-to-first-byte 267ms, warm/reused 89ms — ~178ms of the cost is pure
handshake, gone on a reused connection. **Fix shipped** (see "Cloudflare edge-termination
proxy" below).

**2. Cold start is real and severe, and scale-to-zero alone can't be the answer for a
competing product.** Measured true cold start (freshly redeployed container, zero warm
state) at **~17.5s** to first audio — 14.8s container boot (image start, CUDA init,
model load) + 2.35s first-hit tax. Unusable for a live caller. This ruled out relying on
Modal's scale-to-zero without at least one warm container (`min_containers`) for a real
product — the fix isn't architectural (our shared-pool Modal app already has the same
shape ElevenLabs' infra does), it's that our current aggregate call volume is too low to
keep the pool naturally warm the way ElevenLabs' volume does. `min_containers=N` is a
paid subsidy for that gap (~$0.59/hr per warm T4, ~$425/mo for one instance 24/7 on
Modal's per-second GPU billing) until real traffic volume closes it on its own.

**3. GPU compute is NOT the bottleneck for Kokoro — ruled out with real measurements, not
assumed.** First profiling pass wrongly suggested GPU compute was ~0ms (a
`torch.cuda.synchronize()` was missing before stopping the timer, so it only measured
async kernel-*dispatch*, not completion) — corrected instrumentation showed the real
model forward pass is **90-185ms/clause on a T4**, and:
- **T4 vs L4 vs L4+fp16: no meaningful difference** (135-191ms / 153-191ms / 113-130ms).
  Kokoro's architecture (StyleTTS2 lineage: BERT encoder → LSTM duration predictor →
  decoder) is kernel-launch-bound on many small sequential ops, not FLOPs-bound — a
  bigger/newer GPU chip doesn't help because there's no large matmul to accelerate.
- **CPU-only (no GPU at all): 900-2500ms/clause, 6-15x slower.** This *does* prove the
  GPU is doing real, necessary work — the earlier "GPU is basically free" read was a
  mislabeling of the instrumentation bug above, not evidence the GPU is idle.
- **`torch.compile()`: hung.** First call ran 70+ seconds and blocked the whole
  container's event loop (had to force-kill it) rather than complete. Root cause: the
  LSTM + data-dependent shapes (`torch.repeat_interleave` sized by predicted durations,
  variable phoneme-length inputs) are a known weak spot for TorchInductor tracing —
  RNNs and dynamic control flow fragment into many small graphs or stall outright. This
  was treated as decisive enough to also rule out CUDA graphs (same static-shape
  requirement) and de-prioritize TensorRT without spending further cycles on it, since
  TensorRT needs the same predictable-shape property torch.compile choked on, and
  `torch.compile` is the cheap/fast way to test that property before committing to the
  much heavier TensorRT export pipeline.
- **ONNX Runtime + CUDA: no win, after fixing three real upstream bugs to get a fair
  test.** Used `kokoro-onnx` (pre-converted v1.0 ONNX weights, no custom export needed).
  (a) the library hardcodes `speed` as `int32` for newer-export models, but the
  published `kokoro-v1.0.onnx` (from the release that added float speed input) expects
  `float32` — worked around with a one-line dtype fix bypassing the buggy method: (b) its
  GPU auto-detection checks `importlib.util.find_spec("onnxruntime-gpu")`, which can
  never succeed (that pip package installs into the `onnxruntime` module namespace, not
  a separately-importable one) — silently runs CPU-only regardless of what's installed;
  worked around via its `ONNX_PROVIDER` env override; (c) even then, `onnxruntime-gpu`'s
  wheel doesn't bundle CUDA/cuDNN like PyTorch's wheels do — needed
  `nvidia-{cublas,cudnn,curand,cufft,cusolver,cusparse,nvjitlink,cuda-nvrtc}-cu12` pip
  packages plus `LD_LIBRARY_PATH` added one missing `.so` at a time (5 rounds) before
  `CUDAExecutionProvider` was actually active (confirmed via
  `session.get_providers()`, not inferred from timing — same lesson as the RunPod
  investigation above). Real result once genuinely GPU-accelerated: **160-270ms/clause —
  same or slightly worse than plain PyTorch eager (90-185ms).** No win available here;
  don't revisit without a materially different approach (e.g. TensorRT execution
  provider specifically, untested).

**Conclusion on Kokoro's floor:** ~90-185ms/clause on a T4 is a real property of this
model's architecture (StyleTTS2/LSTM lineage — see "Architecture-level fix" below), not
a config problem. Every standard inference-optimization lever (GPU tier, fp16, compile,
ONNX) was tried and closed out.

**4. Cloudflare edge-termination proxy — shipped, real win.** Since GPU/model
optimization was closed out, focus shifted to the network-layer finding from step 1.
Fronting `t-sushanth--realtime-tts-worker-web.modal.run` with a Cloudflare Worker
(`worker-cf-edge/`) that terminates the client's WebSocket at Cloudflare's anycast edge
and opens its own outbound connection to Modal — same trick ElevenLabs/Replicate use.
Couldn't use the simpler "just proxy the DNS record" approach: Cloudflare's Origin Rules
SNI-override (needed because Modal's serverless routing depends on SNI matching
`*.modal.run`, not our custom domain) requires a Business-tier+ plan, not available on
this zone's Free plan. The Worker sidesteps that entirely by making its own outbound
fetch with the correct hostname. Two real bugs caught before production: (a) forwarding
the client's full header set (including Cloudflare-injected `cf-*` headers) to the
origin fetch caused a "Network connection lost" failure — fixed with a minimal, explicit
header set; (b) Cloudflare's WebSocket API defaults binary message `event.data` to a
`Blob`, which isn't directly re-sendable and silently stringifies to `"[object Blob]"`
on relay — since our protocol is 100% PCM16 audio over binary frames, this would have
corrupted every reply's audio; fixed with `binaryType = "arraybuffer"` on both sockets,
caught via an authenticated echo test *before* it reached real traffic.
**Measured result: connection-only handshake dropped from 166-193ms to 52-55ms
(~3.3x).** Routed via `tts.readaloudai.org` (Cloudflare-proxied CNAME, zone already on
Cloudflare — no nameserver migration needed) + a Workers Route. `TTS_GATEWAY_WS_URL` on
call-loop-poc now points here. Added a 20s WS ping heartbeat
(`call-loop-poc/server.js`) since this leg introduces a Cloudflare idle-timeout that
direct-to-Modal never had. Verified end-to-end with real production auth and the real
model (`providers: ["modal-t4-cuda"]`, byte count matched exactly).

**5. Alternative model architectures — none beat Kokoro's speed/quality combination.**
- **Chatterbox-Turbo (Resemble AI, MIT, 350M params):** marketed as "75ms, single-step."
  Measured 977-2043ms/clause on T4 — 6-20x slower, and **not GPU-tier-bound** (L4 gave
  no improvement: 949-1982ms), ruling out "just needs a bigger GPU." Root cause found by
  reading `T3.inference_turbo`'s source: "single-step" only describes the vocoder
  (`n_cfm_timesteps=2`, genuinely fast at ~170-225ms) — the text-to-speech-token stage
  is a full autoregressive transformer decode loop (KV-cache, temperature/top-k/top-p
  sampling, one token per forward pass, ~14-15ms/token, 39-97 tokens/utterance),
  architecturally identical to LLM token generation. Built a real streaming-wrapper spike
  (generate only the first ~15-18 tokens, run the vocoder on just that chunk) to test
  whether the 75ms claim was actually a time-to-first-chunk metric rather than
  full-utterance latency: **even the smallest viable first chunk cost 500-750ms warm** —
  the AR decode can't be parallelized away (each token depends on the last) and the
  vocoder has a roughly fixed per-call overhead that doesn't shrink with chunk size.
  Genuine dead end on this hardware tier, not a premature one.
- **FastPitch + HiFi-GAN (the models behind NVIDIA Riva TTS), self-hosted via the
  open-source NeMo toolkit (`nemo_toolkit[tts]`, no Riva/Triton/TensorRT server, no NGC
  login):** this is the "standalone open checkpoint, avoid the Riva/NIM licensing
  question" path — genuinely faster than Kokoro: **77-189ms warm (mostly 91-107ms)**,
  matching NVIDIA's published T4/L4 numbers reasonably well for plain PyTorch eager mode
  (no TensorRT). True cold start ~29.5s (worse than Kokoro's ~17.5s — heavier dependency
  tree, two separate checkpoints). **Killed on voice quality** — confirmed by ear as
  "robotic" compared to Kokoro. Licensing was a non-issue: NVIDIA's Community License
  permits production TTS use free up to 25M characters/day, almost certainly covering
  our volume.
- **MagpieTTS (NVIDIA's newer model, in the same NeMo package):** also autoregressive
  (audio-codec-token decode loop, same family as Chatterbox) despite being the
  higher-quality successor — `list_available_models()` returns empty, no public
  self-hostable checkpoint. Only available as `nvidia/magpie-tts-multilingual` on
  build.nvidia.com (hosted API, gRPC not REST, requires its own account/API key,
  pricing beyond free credits not published). Paused here — real endpoint exists but
  needs an actual account to verify further; not pursued without that.

**Architecture-level fix, three options scoped (2026-09-16):** could Kokoro itself be
modified to have TensorRT/torch.compile-friendly static shapes — e.g. replace the LSTM
duration predictor with a feed-forward alternative? Conceptually yes (this is exactly
the FastPitch/FastSpeech2 design). Three options researched:
- **Option A — full retrain, FastPitch-style architecture, on data that gives Kokoro's
  own voice.** Kokoro's actual training data (paired audio/text) was never released —
  the model card only documents a curation *recipe* ("a few hundred hours" of permissively
  licensed + synthetic audio from unnamed closed commercial TTS providers), and the
  GitHub README thanks "everyone who contributed synthetic training data" — implying a
  community-crowdsourced synthetic corpus (generate audio by calling commercial TTS
  APIs, donate the pairs), not a single documented pipeline. This means the *technique*
  is reproducible even though the exact dataset isn't — no special/restricted access was
  involved. Open substitute datasets exist (LibriTTS-R: 585hrs/2,456 speakers, CC BY 4.0;
  NVIDIA's HiFiTTS-2: ~36,700hrs/5,000 speakers, CC BY 4.0) but are multi-speaker
  audiobook narration — training on them would produce a different-sounding (not
  necessarily worse) voice, not a faster version of Kokoro's specific one. Effort:
  weeks-months. Not started.
- **Option B — distill just the LSTM duration predictor** (run Kokoro on lots of text,
  record its duration outputs as training labels, train a small feed-forward student, no
  original training data needed). Research found the field already tried and moved away
  from this: FastSpeech distilled durations from a teacher this way; FastSpeech 2
  explicitly dropped it as "complicated, time-consuming, and lossy" in favor of training
  on real forced-alignment data. No prior art found for doing this to a StyleTTS2/Kokoro-
  style LSTM specifically. Duration/timing is a highly quality-sensitive submodule. Not
  started — would be a time-boxed spike at most, not a committed plan.
- **Option C — keep FastPitch (already fast, no LSTM), fix its voice instead of Kokoro's
  speed.** Tested by swapping FastPitch's vocoder from HiFi-GAN to BigVGAN (a newer,
  higher-fidelity non-autoregressive vocoder), both the full 112M-param
  `bigvgan_22khz_80band` and the 14M-param `bigvgan_base_22khz_80band`, zero-shot (no
  fine-tuning — FastPitch's exact mel config, 80 mels/22050Hz/fmax 8000Hz, matches these
  checkpoints' expected input directly). Latency: full BigVGAN 374-620ms warm (slower
  than both Kokoro and HiFi-GAN — bigger vocoder, real compute cost); `bigvgan_base`
  216-260ms warm (better, still slower than HiFi-GAN's 77-189ms). **Killed on quality
  regardless of latency: confirmed by ear that BigVGAN "sounds exactly like" HiFi-GAN —
  the vocoder was never the source of the robotic quality.** The problem is upstream, in
  FastPitch's acoustic model (deterministic pitch/duration prediction, a known FastPitch
  weakness in the literature) — no vocoder swap can fix prosody that's already flat in
  the spectrogram it's handed. This closes out the "fix FastPitch's voice" approach
  entirely, not just this specific vocoder choice.

**Net conclusion across all three architecture-level options: none is a quick fix.**
Option C (the one actually built and tested) is fully closed out. Options A and B remain
real but require dedicated ML engineering effort (weeks+) with uncertain payoff, and
weren't started. As of this writing, Kokoro remains the best available combination of
speed and quality for this product.

## Fine-tuning path reopens Option A/B cheaply (2026-09-16)

Revisited after realizing fine-tuning a pretrained checkpoint is a completely
different cost class than the from-scratch training estimated above ($62-195, or
earlier $500-1,500 — both were from-scratch pretraining research, never
recalculated after the pivot to fine-tuning). Real measured fine-tuning cost: see
`training-data/README.md` and `training-data/pilot_finetune.py` — ~$0.01 for 300
steps on a T4, extrapolating to ~$1-5 for a full run. At this cost, doing this
across every viable open-weight candidate and comparing quality + latency head to
head is cheap enough to just do.

**Matcha-TTS: real fine-tuning pilot completed and working**, see
`training-data/pilot_finetune.py` (extensively commented with every environment
gotcha hit — 8 rounds of debugging, don't rediscover these). Pretrained LJSpeech
checkpoint fine-tuned on 279 samples of our own Polly-Joanna corpus; loaded cleanly
(0 missing/unexpected keys), loss decreased and stabilized, synthesized output
confirmed by ear to show real voice adaptation (still artificial at this tiny step
count, expected).

**Kokoro fine-tuning: real but complicated, not officially supported.** hexgrad
never released Kokoro training code — only inference weights (Apache 2.0). The only
path found is loading Kokoro's weights into the separate StyleTTS2 training repo
(yl4579/StyleTTS2), since Kokoro is StyleTTS2-derived (minus diffusion, decoder-
only). Checkpoints are NOT compatible as-is — real community examples
(semidark/kikiri-tts, avri-schneider/kokoro-hebrew) needed custom checkpoint-
conversion scripts and patches to StyleTTS2 itself for stable training. Expect
Matcha-TTS-level debugging risk, possibly worse (no maintained repo, two small
independent community projects instead of one). Not started.

**Piper fine-tuning: real, current, well-documented, low risk — and specifically
relevant to the cold-start/warm-floor cost problem.** Piper (MIT, VITS-based) is
explicitly CPU-optimized. Its training code lives at OHF-Voice/piper1-gpl
(successor to the now-archived rhasspy/piper), with an official, maintained
fine-tuning workflow: `--ckpt_path` against published pretrained checkpoints
(huggingface.co/datasets/rhasspy/piper-checkpoints), explicitly documented to work
"even if the checkpoint is from a different language." Data format is a simple
`filename|text` CSV. Community reports successful fine-tunes on modest consumer
hardware (8GB VRAM). This is the lower-risk second pilot to build — and because
Piper targets CPU serving, a good fine-tune here could let us run an always-on CPU
instance instead of paying for a warm GPU floor (~$425/mo/T4 estimated earlier in
this file) to avoid the ~17.5s cold start — CPU instances are typically far cheaper
to keep resident 24/7. Not started; next planned pilot.

**Recommendation:** build the Piper fine-tuning pilot next (lower risk, official
support, plus the CPU-serving/cold-start angle makes it strategically relevant
beyond just a quality comparison). Revisit Kokoro fine-tuning only if Piper's
output quality/character proves insufficient, since Kokoro is generally regarded as
higher-fidelity but carries real, undocumented integration risk.
