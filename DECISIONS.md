# Engineering decisions

**Model: Kokoro-82M (via `kokoro-onnx`).** Already proven in this stack's async TTS
(listenai-tts-worker: ~1.75x realtime on CPU). ONNX runtime means no PyTorch/CUDA
build headaches and it runs fine on CPU for dev, which mattered here since RunPod
wasn't provisioned. `onnxruntime-gpu` picks up CUDA automatically on RunPod without
code changes.

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
