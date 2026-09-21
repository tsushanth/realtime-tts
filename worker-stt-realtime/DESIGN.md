# worker-stt-realtime: streaming English STT, CPU-first, scale-to-zero

Measured with `bench/rtbench.py` (audio replayed in 40 ms chunks paced to the wall clock, +0.3 s lead-in, +1.6 s tail) on a Modal
4-core CPU container (proxy for Fly shared-cpu-4x; NOT the same silicon, see caveats). Raw JSON: `bench/results/`, table: `bench/summarize.py`.
Sets: LibriTTS-R dev-clean 100 utts (clean; "tel" = 8 kHz mu-law round trip + ~30 dB noise), and 48 synthetic call-center turns
(macOS `say`, NOT Piper - the Piper endpoint needs a gateway token; 16 of 48 have a deliberate 800 ms mid-utterance pause, digits/names).
TTS turns are cleaner than humans: use them for endpointing behaviour, not for WER.

## Latency definitions
end-of-speech = clip end (ground truth). `commit` = client tells us speech ended (final = tail flush only). `vadN` = silero trailing
silence >= N ms. `+hint` = +400 ms if the hypothesis ends in a function word. cut = strategy fired while the caller was still talking.

## Endpointing (zipformer, telephone LibriTTS n=100; 16 pause-turns of 48 in the call set)
| strategy | e-o-s -> final med/p90 | utterances cut mid-sentence (LibriTTS tel) | cut (call set, 16 pauses) |
|---|---|---|---|
| commit | 48 / 91 ms | 0 | 0 |
| vad300 | 382 / 501 ms | 52 / 100 | 16 |
| vad500 | 582 / 701 ms | 20 / 100 | 16 |
| vad700 | 781 / 901 ms | 10 / 100 | 16 |
| vad500+hint | 582 / 704 ms | 18 / 100 | 14 |
| sherpa native rule (500 ms) | ~1070 / 1250 ms | 10 / 50 | 16 |
Take-aways: latency after speech ~= silence threshold + ~80 ms; the model is not the bottleneck (final decode after the endpoint is 10-40 ms).
LibriTTS read speech contains many natural pauses, so cut rates there are pessimistic for conversation but the 800 ms
in-number pause (a caller reading an order number) is cut by EVERY silence rule up to 700 ms. Function-word hint saves 2 of 16.
Sherpa's built-in endpointer is slower (it counts decoder-frame silence and needs the tail) and no better. Real fix for agents:
speculative end-of-turn (`speculative_final` at 300 ms, `resume` if the caller continues; the LLM starts early and is cancelled on
`resume`) + a client/agent-side semantic turn detector. Not built here.

## Warm on authorize (design; gateway and batch worker untouched)
Batch scheme reused: client POSTs `/stt/authorize {key}`; gateway checks key/402 exactly as today and returns `{token, url}`.
Proposed additive change (not implemented, needs gateway owner): `{key, mode:"realtime"}` -> gateway (a) mints the same HMAC session token,
(b) fires a non-blocking wake: `POST https://api.machines.dev/v1/apps/<app>/machines/<id>/start` (Fly; ~1.1 s VM boot, idempotent on a started machine)
or `Function.spawn()` ping (Modal), (c) returns `{token, url:"wss://.../v1/stt/realtime", ready:false, retry_after_ms:2500}`.
Client waits `retry_after_ms` (or polls `GET <url-host>/warm` -> `{"ready":true}`; a request to a stopped Fly machine also autostarts it via the proxy).
Second safety net, already in server.py: the WebSocket is accepted immediately, server sends `{"type":"loading"}`, buffers up to 20 s of audio,
then `{"type":"ready"}` and replays the buffer, so a caller who connects before ready loses nothing (their first partials are late by the remaining load time).
Usage: same as batch - server POSTs `{id, audio_seconds, engine:"stt-realtime"}` to `/admin/usage/report` (`USAGE_REPORT_SECRET`); the gateway
currently accepts `audio_seconds` with any engine string (gateway/stt.test.js) - confirm "stt-realtime" is priced before launch.
Idle: Fly `auto_stop_machines="stop"`, `min_machines_running=0` (fly.toml); a stopped machine costs only rootfs storage ($0.15/GB-30d).

## Caveats
* Fly shared-cpu has a 6.25%/vCPU baseline (20 ms per 80 ms on 4x) with a 500 s burst bucket: sustained use of >~0.25 core is throttled. One
  zipformer stream uses ~0.06-0.08 core, so >3 concurrent long calls on one shared-cpu-4x will throttle. Use performance-1x/2x for load or several machines.
* Modal vCPU != Fly vCPU; re-measure on Fly with `bench/ws_client.py` before promising anything.
* Zipformer 2023-06-26 is LibriSpeech-only (read audiobooks); expect worse WER on real calls than any number here.
