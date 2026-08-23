"""Standalone local WS server for the TTS worker (used for local dev/testing and as the
reference implementation the RunPod handler wraps). Not what runs on RunPod directly —
see handler.py for the RunPod Serverless entrypoint.

Protocol (JSON control + binary audio frames on one WS connection):
  client -> {"type": "synthesize", "text": "...", "voice": "af_heart", "speed": 1.0}
  client -> {"type": "stop"}                      # cancel in-flight synthesis
  server -> {"type": "chunk_meta", "text": "...", "gen_ms": .., "audio_s": ..}
  server -> <binary PCM16LE mono 24kHz frame>      # immediately follows chunk_meta
  server -> {"type": "done"} | {"type": "cancelled"} | {"type": "error", "message": "..."}
"""
import asyncio
import json
import logging
import os

import numpy as np
import websockets

from synth import synthesize_stream, get_engine, SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")


def to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


async def handle_client(ws):
    cancel_flag = {"cancel": False}

    async for raw in ws:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            await ws.send(json.dumps({"type": "error", "message": "invalid JSON"}))
            continue

        if msg.get("type") == "stop":
            cancel_flag["cancel"] = True
            continue

        if msg.get("type") != "synthesize":
            await ws.send(json.dumps({"type": "error", "message": "unknown message type"}))
            continue

        cancel_flag["cancel"] = False
        text = msg.get("text", "")
        voice = msg.get("voice", "af_heart")
        speed = float(msg.get("speed", 1.0))

        try:
            loop = asyncio.get_event_loop()
            gen = synthesize_stream(text, voice=voice, speed=speed)
            while True:
                if cancel_flag["cancel"]:
                    await ws.send(json.dumps({"type": "cancelled"}))
                    break
                try:
                    chunk = await loop.run_in_executor(None, lambda: next(gen, None))
                except StopIteration:
                    chunk = None
                if chunk is None:
                    await ws.send(json.dumps({"type": "done"}))
                    break
                await ws.send(json.dumps({
                    "type": "chunk_meta",
                    "text": chunk["text"],
                    "gen_ms": chunk["gen_ms"],
                    "audio_s": chunk["audio_s"],
                    "providers": chunk["providers"],
                }))
                await ws.send(to_pcm16(chunk["samples"]))
        except Exception as e:
            log.exception("synthesis failed")
            await ws.send(json.dumps({"type": "error", "message": str(e)}))


async def main():
    host = os.environ.get("WORKER_HOST", "0.0.0.0")
    port = int(os.environ.get("WORKER_PORT", "8765"))
    log.info("warming up Kokoro engine...")
    get_engine().create("warmup", voice="af_heart", speed=1.0, lang="en-us")
    log.info(f"worker ready, listening on ws://{host}:{port}")
    async with websockets.serve(handle_client, host, port, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
