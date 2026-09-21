# Realtime (streaming) speech-to-text — DRAFT, not published

Status: draft for internal/owner review. Not a public API reference yet. Numbers below are our own
measurements (see "Measured numbers" and their caveats) — not third-party benchmarks, not vendor claims.

## What this is

`worker-stt-realtime` is a WebSocket streaming speech-to-text service: send audio as it's captured,
get partial and final transcripts back as the caller speaks, instead of waiting for a full recording
and calling batch STT (`worker-stt-prod`, Modal app `realtime-stt-worker`, priced at $0.11/audio-hour).
It's built for voice agents and live captioning, where the loop needs a transcript within a few hundred
milliseconds of the caller stopping talking, not seconds.

Engine: Zipformer streaming transducer (int8), via sherpa-onnx, Apache-2.0, English only right now.
Endpointing: Silero VAD.

## Authenticating and connecting

1. Your backend calls the gateway's `POST /stt/authorize` with `{"key": "<your API key>", "mode": "realtime"}`.
   Same checks as every other authorize call (valid key, free tier/billing status) and returns
   `{"token": "<short-lived session token>", "url": "wss://.../v1/stt/realtime"}`.
2. Open a WebSocket to `url`, passing the token as `?token=...` or an `Authorization: Bearer` header.
3. Stream audio as binary WebSocket frames: PCM16LE mono 16 kHz (default), or G.711 mu-law 8 kHz
   (`encoding=mulaw&sample_rate=8000` on the URL, or a `config` message — see below). Any chunk size;
   20-100 ms per frame is typical.

## Protocol — messages both directions

**Client -> server**

| Message | Shape | Meaning |
|---|---|---|
| binary frame | raw PCM16LE or mu-law bytes | audio |
| `{"type":"config", ...}` | optional, first message | override `encoding`, `sample_rate`, `endpoint_ms`, `semantic_hint`, `partials` |
| `{"type":"commit"}` | | "the caller just stopped talking, finalize now" — this is the fast path |
| `{"type":"end"}` | | flush, get final usage, close the connection cleanly |

**Server -> client**

| Message | Shape | Meaning |
|---|---|---|
| `{"type":"loading"}` | | model still warming (cold start); your audio is being buffered (up to 20 s), not dropped |
| `{"type":"ready","engine":..,"load_s":..}` | | model loaded, buffered audio is about to be replayed |
| `{"type":"partial","text":..,"start_s":..,"audio_s":..}` | | live, in-progress transcript, rate-limited to ~1 per 80 ms |
| `{"type":"speculative_final","text":..,"silence_ms":..}` | | early best guess that the turn just ended, at ~300 ms of silence — start your LLM call now; may be followed by `resume` |
| `{"type":"resume"}` | | the caller kept talking after a `speculative_final` — discard/cancel whatever you started from it |
| `{"type":"final","segment":n,"text":..,"start_s":..,"end_s":..,"reason":"vad"\|"commit"\|"end"\|"max_len"}` | | a finished utterance |
| `{"type":"usage","id":..,"audio_seconds":..,"engine":"stt-realtime"}` | | billing usage for this connection (also POSTed server-side to `/admin/usage/report`) |
| `{"type":"error","message":..}` | | something went wrong; connection closes (4401 = unauthorized, 1013 = at capacity) |

`start_s`/`end_s`/`audio_s` are seconds of audio received on **this connection** (a stream clock
starting at 0 when the socket opens), not wall-clock time. Word-level timestamps and confidence
scores are **not currently emitted** — the underlying sherpa-onnx `OnlineRecognizer` greedy-search
decode used here returns segment-level text only. Getting word timestamps would mean switching
decoding methods or a different model config; not done in this draft.

## Two ways to finalize an utterance — and why the difference matters

**1. Explicit `commit` (fast, recommended).** Your client/telephony layer already usually knows when
a turn ended (button release, VAD in your own stack, a "stop speaking" event) — send `{"type":"commit"}`
the moment you know that, and the server finalizes immediately: no silence wait at all, just the
~10-40 ms it takes to flush the last audio through the model.

**2. Silence-based auto-finalize (fallback, slower).** If you never send `commit`, the server's own
Silero VAD watches for trailing silence and finalizes once it's been quiet for `endpoint_ms`
(default 500 ms, tunable per connection). This exists so a client that can't reliably detect
end-of-turn still gets *a* final transcript — but it is fundamentally a "wait and see" strategy, so
it is slower by roughly the threshold you pick, and it will occasionally cut an utterance mid-sentence
during a natural pause (see the measured cut-rate table below).

**We can only honestly promise sub-100ms latency on the `commit` path.** If your integration can't
signal end-of-turn itself, tell your customers the realistic number is the silence threshold plus
~100 ms, not the `commit` number.

## Measured numbers (ours, from `bench/rtbench.py`)

Measured on a Modal 4-core CPU container as a proxy for Fly `shared-cpu-4x` — **not the same silicon**;
re-measure on the actual Fly target before quoting these externally. Test audio: LibriTTS-R dev-clean
(100 utterances, read audiobook speech, clean + an 8 kHz mu-law/noise "tel" variant) and 48 synthetic
call-center-style turns (macOS `say`, not real speech — used only for endpointing/pause behavior, not WER).

### End-of-speech -> final latency, by finalize strategy (telephone-condition set, n=100)

| Strategy | Median | p90 | Utterances cut mid-sentence (of 100) |
|---|---|---|---|
| `commit` | 48 ms | 91 ms | 0 |
| silence 300 ms | 382 ms | 501 ms | 52 |
| silence 500 ms (default) | 582 ms | 701 ms | 20 |
| silence 700 ms | 781 ms | 901 ms | 10 |
| silence 500 ms + function-word hint | 582 ms | 704 ms | 18 |

On the 48-turn call-style set, 16 turns include a deliberate 800 ms mid-utterance pause (e.g. a caller
reading back an order number); **every silence threshold up to 700 ms cuts all 16 of those**, because
800 ms of silence is by construction longer than any threshold tested. The function-word hint (extend
the wait if the transcript so far ends on "the", "my", "is", etc.) only saves 2 of those 16. This is
the core argument for `commit`, or for `speculative_final` + your own semantic turn detector, over
tuning the silence threshold higher: raising the threshold to catch pauses like that pushes *every*
utterance's latency up by the same amount, and still doesn't reliably catch an 800 ms+ pause.

### WER

**Not measured with a real-speech reference transcript comparison in this draft.** LibriTTS-R has
reference transcripts and was used for the latency numbers above, but a WER pass (decode -> align ->
score) was not run as part of this session's work — see the "What's not verified" section in the
engineering report this draft accompanies. Do not quote a WER number until that's actually been run.
Model provenance for context: this is the stock sherpa-onnx streaming Zipformer (2023-06-26,
LibriSpeech-trained, English, int8), which the sherpa-onnx project reports at 3.4%/4.4% WER on
LibriSpeech test-clean/test-other — that's an upstream number, not one we've reproduced here, and
LibriSpeech is read audiobook speech, not phone calls; expect materially worse WER on real calls.

### Concurrency

**Not load-tested against a running instance in this draft** (no live deployment to point a
concurrent-stream test at yet — see the accompanying engineering report). Design intent: one Zipformer
stream costs ~0.06-0.08 CPU core, so `MAX_CONNECTIONS` (default 6) is meant to comfortably fit a
`shared-cpu-4x` Fly VM math-wise, but Fly's shared-cpu baseline/burst-bucket throttling (documented in
`DESIGN.md`'s Caveats) means sustained load above ~3 concurrent long calls on one shared-cpu-4x machine
is expected to throttle in practice — untested.

### Cold start

Not independently re-measured for this draft; `DESIGN.md` cites ~6-7 s on Modal from earlier bench work
(`bench/modal_cold.py`). The server buffers up to 20 s of audio while loading and replays it once ready,
so a caller connecting during a cold start doesn't lose audio, just gets delayed partials.

## Limits

- English only.
- No word-level timestamps or confidence scores in this build (see protocol table above).
- `MAX_CONNECTIONS` per instance: 6 by default (configurable), enforced server-side (1013 close when full).
- Max single utterance: 30 s (`MAX_UTT_S`) before a forced `max_len` finalize.
- Audio buffered during cold start: 20 s (`MAX_BUFFER_S`) — older audio is dropped past that, not queued forever.

## Pricing — [DECIDE]

Batch STT is $0.11/audio-hour. Realtime is **not priced yet** — `ra-stt-realtime`'s
`realtimeTtsBilling.ts` currently defaults `STT_REALTIME_CHARS_PER_SECOND` to the same rate as batch
so nothing goes unbilled, but that's a placeholder, not a decision. Realtime has a different cost shape
than batch (a stream holds ~0.06-0.08 CPU core for the full call duration, vs. batch's short burst jobs)
and arguably a different value prop (sub-second latency for live agents/captioning vs. offline
transcription). Options to weigh, not a recommendation:

- **Same $0.11/hour as batch.** Simplest, but likely underprices the always-on CPU hold and the
  latency-engineering work (VAD tuning, speculative finalization) that batch doesn't need.
- **A premium over batch** (e.g. 1.5-2x) reflecting the CPU-hold cost and the latency guarantee.
- **Tiered by finalize mode** — a lower rate for `commit`-only integrations (cheap, fast, no VAD
  compute needed server-side) vs. silence-fallback (server holds a VAD running the whole call).

Whoever owns pricing should also decide whether the sub-100ms latency number is marketed at all given
it's conditional on the client sending `commit` — advertising it without that caveat would be
misleading for any integration that relies on the silence fallback.
