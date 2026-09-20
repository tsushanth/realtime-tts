# worker-stt: Whisper STT API (vs ElevenLabs Scribe)

Prototype + measurements. Benchmark: 100 LibriTTS-R dev_clean utterances (2-12 s, 544.6 s, ~1.5k words, seed 0),
16 kHz clean and "telephone" (24k->8k, mu-law round trip, ->16k). faster-whisper 1.1.1, greedy (beam 1), language forced "en",
no word timestamps in the WER loop, batch size 1, model load/warm-up excluded. WER = jiwer on lowercased, punctuation-stripped text.
Latency = median wall time of the 25 clips of 4-6 s. Cost = RTF x 3600 x Modal $/s (modal.com/pricing, read 2026-09-20):
T4 $0.000164/s, L4 $0.000222/s, CPU $0.0000131/core/s (4 physical cores = 8 vCPU) + memory $0.00000222/GiB/s (8 GiB).
Raw numbers: bench/results.json. Reproduce: `bench/prep.py`, `modal run --detach bench/bench.py`.

Caveats (be skeptical of the telephone column): LibriTTS-R is restored, studio-clean read speech, and our "telephone" version has
no noise, packet loss or real codec artifacts, so WER barely degrades. Real call audio will be worse; measure on real calls
(see realtime-tts/STT_ACCURACY_INVESTIGATION.md). 1.5k words means roughly +/-0.5-1 pt WER noise. English only.

## API
- `POST /v1/stt` multipart `file`, `format` (auto|wav|flac|mp3|ogg|mulaw), `language` (auto|xx), `word_timestamps`
  -> `{text, language, words:[{word,start,end}], duration}`. Decoding via ffmpeg; mulaw = raw 8 kHz G.711 u-law.
- `WS /v1/stt/stream?token=` JSON config then PCM16 16k (or mulaw 8k) binary frames; server sends `partial` / `final`
  (with words) and a closing `usage` message. VAD endpointing = silero (bundled in faster-whisper), check every 200 ms,
  final when trailing silence >= `endpoint_ms` (default 500). Partials re-decode the growing utterance every ~1 s
  (Whisper is not streaming; partials are best-effort and may revise).
- Auth: Bearer session token, HMAC-SHA256 over base64url payload, identical to gateway/keys.js (`verify_session_token`).
- Usage: audio seconds per key id (`audio_seconds`, engine `whisper-stt`) posted best-effort to USAGE_REPORT_URL and always
  logged. TODO before production: gateway `/admin/usage/report` only accepts `chars`; add `audio_seconds` + a per-second price.
- Scaling: scale-to-zero (`scaledown_window=60`, no min_containers), one decode at a time per GPU container behind a lock.
  Not load-tested for concurrency.
