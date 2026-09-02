"""
Kokoro TTS over WebSocket, on Modal — the replacement for the RunPod
Pod-based worker (see ../worker/, ../gateway/pod-manager.js).

Why this exists: a RunPod Pod is billed by wall-clock time regardless of
whether a call is happening, so a 15-minute (or even 90-second) idle
teardown timer always pays for some idle GPU time after every call, and
guessing a shorter timer risks tearing down between two real calls. Modal
treats one open WebSocket connection as a single unit of billable work — the
container is held for exactly as long as the call lasts and released the
instant it closes, with no timer to tune. See the "Idle GPU Problem"
write-up this migration came out of.

Protocol is identical to worker/server.py (the RunPod-hosted version) on
purpose — call-loop-poc's server.js talks to this exactly the same way, so
switching TTS_GATEWAY_WS_URL is the only change needed there:
  client -> {"type": "synthesize", "text": "...", "voice": "af_heart", "speed": 1.0}
  client -> {"type": "stop"}                      # cancel in-flight synthesis
  server -> {"type": "chunk_meta", "text": "...", "gen_ms": .., "audio_s": ..}
  server -> <binary PCM16LE mono 24kHz frame>      # immediately follows chunk_meta
  server -> {"type": "done"} | {"type": "cancelled"} | {"type": "error", "message": "..."}

Auth: a single shared bearer token (Modal Secret `tts-ws-auth-token`), checked
before accepting the connection — this is one internal client (call-loop-poc),
not a multi-tenant API-key product like the RunPod gateway was, so the
gateway's whole keys.js/billing layer has no equivalent need here.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng", "libsndfile1")
    .pip_install(
        "kokoro==0.9.4",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers==4.44.0",
        "soundfile==0.12.1",
        "numpy>=1.24.0,<2.0.0",
        "fastapi==0.109.0",
        "uvicorn[standard]==0.27.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        # Bake the model + voice used by call-loop-poc into the image so a
        # cold start never has to hit the network for weights — same reason
        # ReadAloud's Modal app pre-caches voices at build time.
        "python3 -c \""
        "from kokoro import KPipeline; "
        "p = KPipeline(lang_code='a'); "
        "next(p('Test.', voice='af_heart'))"
        "\""
    )
)

app = modal.App("realtime-tts-worker", image=image)


def to_pcm16(samples) -> bytes:
    import numpy as np

    # kokoro's KPipeline yields audio as a torch Tensor, not numpy — unlike
    # kokoro_onnx (the RunPod worker's engine), which returns numpy directly.
    # Found live: torch.Tensor has no .astype(), only .numpy() converts it.
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


@app.function(
    gpu="T4",
    # Real calls run seconds to a couple minutes; a gap of a few minutes
    # between calls is a different session, not the same one continuing.
    # This only affects how long a container sticks around *after* the
    # WebSocket that was billing it closes — it does not create an idle
    # billing tail the way the RunPod pod's timer did, since the call
    # itself is the billable unit, not this window.
    scaledown_window=120,
    max_containers=10,
    secrets=[modal.Secret.from_name("tts-ws-auth-token")],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import os
    import time

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
    from kokoro import KPipeline

    web_app = FastAPI()
    pipeline = KPipeline(lang_code="a")
    AUTH_TOKEN = os.environ["TTS_WS_AUTH_TOKEN"]

    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "model": "kokoro", "device": "cuda"}

    @web_app.websocket("/tts")
    async def tts(ws: WebSocket, token: str = Query(default="")):
        auth_header = ws.headers.get("authorization", "")
        bearer = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else ""
        if not (token == AUTH_TOKEN or bearer == AUTH_TOKEN):
            await ws.close(code=4401)
            return

        await ws.accept()
        cancel = False

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    import json

                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    await ws.send_json({"type": "error", "message": "invalid JSON"})
                    continue

                if msg.get("type") == "stop":
                    cancel = True
                    continue

                if msg.get("type") != "synthesize":
                    await ws.send_json({"type": "error", "message": "unknown message type"})
                    continue

                cancel = False
                text = msg.get("text", "")
                voice = msg.get("voice", "af_heart")
                speed = float(msg.get("speed", 1.0))

                try:
                    for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=speed):
                        if cancel:
                            await ws.send_json({"type": "cancelled"})
                            break
                        t0 = time.perf_counter()
                        # KPipeline already did the generation by the time this
                        # tuple is yielded — gen_ms here is ~0, kept only for
                        # log-format parity with worker/synth.py's chunk_meta.
                        gen_ms = (time.perf_counter() - t0) * 1000
                        audio_s = len(audio) / 24000
                        await ws.send_json({
                            "type": "chunk_meta",
                            "text": "",
                            "gen_ms": gen_ms,
                            "audio_s": audio_s,
                            "providers": ["modal-t4-cuda"],
                        })
                        await ws.send_bytes(to_pcm16(audio))
                    else:
                        await ws.send_json({"type": "done"})
                except Exception as e:  # noqa: BLE001 — report to the client, don't crash the container
                    await ws.send_json({"type": "error", "message": str(e)})
        except WebSocketDisconnect:
            pass

    return web_app
