"""
Kokoro TTS over WebSocket, on Modal — the ReadAloud developer-API worker.

This is a deliberately SEPARATE, isolated Modal app from ../worker-modal/app.py
(which serves call-loop-poc). Same protocol, same reasoning for being on Modal
at all (see that file's docstring), but kept as its own deployment rather than
shared, because:
  - cost visibility: ReadAloud's public multi-tenant API traffic shouldn't be
    mixed into call-loop-poc's billing/usage numbers or vice versa
  - blast radius: a bug or abuse pattern from one consumer (e.g. a malicious or
    buggy API key hammering this worker) shouldn't be able to degrade or
    exhaust capacity for the other
  - independent scaling knobs: these two consumers may want different
    max_containers/GPU-tier tuning as traffic patterns diverge

Auth: two accepted methods, checked in this order:
  1. A short-lived, HMAC-signed session token (see gateway/keys.js's
     createSessionToken/verifySessionToken) issued by gateway/server.js's
     POST /tts/authorize after it validates the end-user's API key and
     billing/free-tier status. The CLIENT connects directly here with this
     token — the gateway is out of the data path entirely, which is the
     whole point (avoids the ~350-400ms of relay handshake overhead the old
     gateway-proxied path had, see DECISIONS.md). Since usage is no longer
     metered inline by the gateway inspecting every frame, THIS worker
     reports back what it actually synthesized via POST /admin/usage/report
     after each call — see report_usage() below.
  2. The original static shared bearer token (TTS_WS_AUTH_TOKEN), kept for
     backward compatibility with gateway/server.js's existing relay path
     (AUTO_MODE's proxyToWorker) — unchanged, still metered inline by the
     gateway as before. Nothing using this path needs to change.

Protocol (identical to worker/server.py and ../worker-modal/app.py, so
gateway/server.js's proxy logic doesn't need protocol-aware branching):
  client -> {"type": "synthesize", "text": "...", "voice": "af_heart", "speed": 1.0}
  client -> {"type": "stop"}                      # cancel in-flight synthesis
  server -> {"type": "chunk_meta", "text": "...", "gen_ms": .., "audio_s": ..}
  server -> <binary PCM16LE mono 24kHz frame>      # immediately follows chunk_meta
  server -> {"type": "done"} | {"type": "cancelled"} | {"type": "error", "message": "..."}
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
        # Bake the model + default voice into the image so a cold start never
        # has to hit the network for weights.
        "python3 -c \""
        "from kokoro import KPipeline; "
        "p = KPipeline(lang_code='a'); "
        "next(p('Test.', voice='af_heart'))"
        "\""
    )
)

app = modal.App("realtime-tts-worker-readaloud", image=image)


def to_pcm16(samples) -> bytes:
    import numpy as np

    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


# Same clause/comma chunking as worker/synth.py's chunk_text, kept in sync
# deliberately — this worker exists to serve the same latency profile the
# RunPod Pod path was measured at, not a from-scratch redesign.
import re

_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")
_SUBSPLIT_RE = re.compile(r"(?<=[,])\s+")


def chunk_text(text, max_chars=90, first_chunk_max_chars=35):
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    if not parts:
        parts = [text]
    chunks = []
    buf = ""
    for p in parts:
        if buf and len(buf) + len(p) + 1 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf} {p}".strip()
    if buf:
        chunks.append(buf)

    if chunks and len(chunks[0]) > first_chunk_max_chars:
        sub_parts = [p.strip() for p in _SUBSPLIT_RE.split(chunks[0]) if p.strip()]
        if len(sub_parts) > 1:
            chunks = [sub_parts[0], " ".join(sub_parts[1:])] + chunks[1:]
    return chunks


@app.function(
    gpu="T4",
    # Matches ../worker-modal's rationale exactly — real calls run seconds to
    # a couple minutes; a gap of a few minutes between requests is a new
    # session, not the same one continuing. This only affects how long a
    # container sticks around AFTER a call ends, not while it's active — the
    # call itself is the billable unit, so this doesn't recreate the idle-
    # billing problem the RunPod Pod's 90s/15min timer had.
    scaledown_window=120,
    max_containers=10,
    secrets=[
        modal.Secret.from_name("readaloud-tts-ws-auth-token"),
        modal.Secret.from_name("readaloud-tts-session-secret"),
        modal.Secret.from_name("readaloud-tts-usage-report-secret"),
    ],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import base64
    import hashlib
    import hmac
    import json
    import os
    import time
    import urllib.request

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
    from kokoro import KPipeline

    web_app = FastAPI()
    pipeline = KPipeline(lang_code="a")
    AUTH_TOKEN = os.environ["TTS_WS_AUTH_TOKEN"]
    SESSION_SECRET = os.environ["MODAL_SESSION_SECRET"]
    USAGE_REPORT_SECRET = os.environ["MODAL_USAGE_REPORT_SECRET"]
    USAGE_REPORT_URL = os.environ.get(
        "USAGE_REPORT_URL", "https://realtime-tts-gateway.fly.dev/admin/usage/report"
    )

    def b64url_decode(s: str) -> bytes:
        return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

    def verify_session_token(token: str):
        """Mirrors gateway/keys.js's createSessionToken/verifySessionToken exactly —
        same HMAC-SHA256-over-the-base64url-payload-string construction, same
        base64url (unpadded) encoding. Returns the key id, or None if invalid/expired."""
        try:
            payload_b64, sig_b64 = token.split(".")
        except ValueError:
            return None
        expected_sig = base64.urlsafe_b64encode(
            hmac.new(SESSION_SECRET.encode(), payload_b64.encode(), hashlib.sha256).digest()
        ).decode().rstrip("=")
        if not hmac.compare_digest(sig_b64, expected_sig):
            return None
        try:
            payload = json.loads(b64url_decode(payload_b64))
        except Exception:
            return None
        if "id" not in payload or "exp" not in payload:
            return None
        if time.time() * 1000 > payload["exp"]:
            return None
        return payload["id"]

    def report_usage(key_id: str, chars: int):
        """Fire-and-forget call back to the gateway with what was actually
        synthesized — the authoritative usage number for billing, since it
        reflects real work done, not anything the client declared upfront.
        Called AFTER the client already has its audio, so a slow or failed
        report never adds to the client's perceived latency. Best-effort:
        logged on failure, not retried — acceptable at this volume, revisit
        with a durable queue if reports start being lost at real scale."""
        if not key_id or not chars:
            return
        try:
            req = urllib.request.Request(
                USAGE_REPORT_URL,
                data=json.dumps({"id": key_id, "chars": chars}).encode(),
                headers={
                    "Authorization": f"Bearer {USAGE_REPORT_SECRET}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:  # noqa: BLE001 — never let a reporting failure affect the session
            print(f"usage report failed for key {key_id}: {e}")

    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "model": "kokoro", "device": "cuda"}

    @web_app.websocket("/tts")
    async def tts(ws: WebSocket, token: str = Query(default="")):
        auth_header = ws.headers.get("authorization", "")
        bearer = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else ""
        presented = token or bearer

        session_key_id = verify_session_token(presented)
        if session_key_id is None and presented != AUTH_TOKEN:
            await ws.close(code=4401)
            return

        await ws.accept()
        cancel = False

        try:
            while True:
                raw = await ws.receive_text()
                try:
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
                    for clause in chunk_text(text):
                        if cancel:
                            await ws.send_json({"type": "cancelled"})
                            break
                        t0 = time.perf_counter()
                        gen = pipeline(clause, voice=voice, speed=speed)
                        _gs, _ps, audio = next(gen)
                        gen_ms = (time.perf_counter() - t0) * 1000
                        audio_s = len(audio) / 24000
                        await ws.send_json({
                            "type": "chunk_meta",
                            "text": clause,
                            "gen_ms": gen_ms,
                            "audio_s": audio_s,
                            "providers": ["modal-t4-cuda"],
                        })
                        await ws.send_bytes(to_pcm16(audio))
                    else:
                        await ws.send_json({"type": "done"})
                        # Only session-token clients get metered this way — the
                        # legacy shared-secret path is still metered inline by
                        # the gateway itself, unchanged, per the module docstring.
                        if session_key_id:
                            report_usage(session_key_id, len(text))
                except Exception as e:  # noqa: BLE001 — report to client, don't crash container
                    await ws.send_json({"type": "error", "message": str(e)})
        except WebSocketDisconnect:
            pass

    return web_app
