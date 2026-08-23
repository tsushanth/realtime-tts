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

**Stopped here deliberately** rather than keep guessing — five endpoints were created and
torn down chasing this, and further blind iteration wasn't a good use of the session.
**Current state**: worker runs correctly and reliably on CPU via RunPod Serverless (jobs
complete, audio is correct); GPU acceleration does not work despite Serverless billing
GPU-tier rates. Next steps for whoever picks this up: (a) file a RunPod support ticket
asking why `gpuTypeIds` preferences aren't honored and whether non-Blackwell capacity can
be guaranteed; (b) try RunPod's own prebuilt PyTorch/CUDA base image instead of a
from-scratch pip-based CUDA setup, which may bundle Blackwell-compatible kernels; (c) or
just accept CPU-only for now — per the ElevenLabs cost comparison, this is still ~3-6x
cheaper per character even at CPU speed, since Kokoro-82M is small enough that raw
compute cost stays low regardless of GPU/CPU. GPU would mainly buy lower TTFB, not lower
cost, for a model this size.

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
