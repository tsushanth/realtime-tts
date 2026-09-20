# worker-stt-prod: batch speech-to-text (Modal app `realtime-stt-worker`)

Production version of the `worker-stt/` prototype, batch only (streaming lives elsewhere). See `app.py` docstring for the API.
faster-whisper 1.1.1 `BatchedInferencePipeline`, large-v3-turbo baked into the image, L4, VAD filter, word timestamps,
language auto-detect or `language=xx`, beam 1, batch size 16.

## Deploy
Never from a dev machine under the production name for testing. Test: `STT_APP_NAME=stt-prod-test STT_SECRET=stt-prod-test-secret modal deploy app.py`, then `modal app stop stt-prod-test`.
Production: create Modal secret `realtime-stt-secrets` with `MODAL_SESSION_SECRET` (same value as the gateway's), `MODAL_USAGE_REPORT_SECRET`
(same as the gateway's) and `USAGE_REPORT_URL=https://<gateway host>/admin/usage/report`; then `modal deploy app.py`
(app name defaults to `realtime-stt-worker`). Set the gateway's `STT_WORKER_URL` to the printed web URL.

## Knobs
`STT_MAX_CONTAINERS` (default 4, hard GPU spend cap), `scaledown_window=120`, `min_containers=0`, `@modal.concurrent(max_inputs=4)`,
limits in `app.py` (200 MB, 3 h audio, 900 s per request, all enforced in the worker).

## Tests
`tests/e2e.py <url>` (auth, param validation, formats, multipart, limits; needs clips, see file), `tests/client.py` (mint token + POST),
`tests/usage_sink.py` (test-only stand-in for the gateway usage endpoint).
