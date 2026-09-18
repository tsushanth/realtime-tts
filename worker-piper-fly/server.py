"""
Piper (our fine-tuned voice) over WebSocket - platform-independent build of
worker-modal-piper/app.py: plain FastAPI + uvicorn, configured by env vars, no
Modal. Meant for an always-on CPU box (Fly Machine next to call-loop-poc, Hetzner, a
home server...). Same protocol as the Kokoro worker, so call-loop-poc switches by
changing TTS_GATEWAY_WS_URL only:
  client -> {"type": "synthesize", "text": "...", "voice": "<ignored>", "speed": 1.0}
  client -> {"type": "stop"}
  server -> {"type": "chunk_meta", "text": "", "gen_ms": .., "audio_s": .., "providers": [..]}
  server -> <binary PCM16LE mono 24kHz>      # immediately follows chunk_meta
  server -> {"type": "done"} | {"type": "cancelled"} | {"type": "error", "message": ".."}
See worker-modal-piper/README.md for the design reasons (24kHz resample, working `stop`,
phonemization serialized, inference off the event loop).

Env: AUTH_TOKEN (required), MODEL_PATH (default /models/full_ft.onnx; the .json config
must sit beside it), ORT_INTRA_THREADS (default 2), PORT (default 8080),
HOST (default "::" - dual-stack/IPv6, which Fly private networking needs; use 0.0.0.0 on IPv4-only hosts).
"""
import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import onnxruntime
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from piper import PiperVoice, SynthesisConfig
from scipy.signal import resample_poly

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/full_ft.onnx")
ORT_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "2"))
AUTH_TOKEN = os.environ["AUTH_TOKEN"]
OUT_SR = 24000
WARMUP_TEXT = "Thanks for calling, I can help you with that. Let me pull up your account details right now."


class PiperEngine:
    def __init__(self, model_path: str, intra_threads: int):
        self.voice = PiperVoice.load(model_path)
        # PiperVoice.load builds a default SessionOptions, which sizes onnxruntime's
        # thread pool from the HOST core count - oversubscribes a small VM/container.
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = 1
        self.voice.session = onnxruntime.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        ratio = Fraction(OUT_SR, self.voice.config.sample_rate).limit_denominator(1000)
        self.up, self.down = ratio.numerator, ratio.denominator
        self._phonemize_lock = threading.Lock()  # espeak-ng has global state

    def sentences(self, text: str) -> list[list[int]]:
        with self._phonemize_lock:
            return [self.voice.phonemes_to_ids(p) for p in self.voice.phonemize(text) if p]

    def synth(self, phoneme_ids: list[int], speed: float = 1.0) -> tuple[bytes, float]:
        cfg = SynthesisConfig(length_scale=self.voice.config.length_scale / max(speed, 0.1))
        audio = self.voice.phoneme_ids_to_audio(phoneme_ids, cfg)
        peak = float(np.max(np.abs(audio)))
        audio = audio / peak if peak > 1e-8 else np.zeros_like(audio)  # piper's normalize_audio
        if self.up != self.down:
            audio = resample_poly(audio, self.up, self.down)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes(), len(pcm) / OUT_SR


engine = PiperEngine(MODEL_PATH, ORT_INTRA_THREADS)
for _ids in engine.sentences(WARMUP_TEXT):  # pay first-inference costs at startup, not on a caller
    engine.synth(_ids)
pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 2)
app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "piper-full-ft", "device": "cpu"}


@app.websocket("/tts")
async def tts(ws: WebSocket, token: str = Query(default="")):
    auth_header = ws.headers.get("authorization", "")
    bearer = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else ""
    if not (token == AUTH_TOKEN or bearer == AUTH_TOKEN):
        await ws.close(code=4401)
        return
    await ws.accept()

    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()
    cancel = asyncio.Event()

    async def reader():
        # Keeps receiving during synthesis so `stop` lands mid-utterance; only the main
        # loop below writes to the socket.
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    await inbox.put({"type": "_invalid"})
                    continue
                if msg.get("type") == "stop":
                    cancel.set()
                else:
                    await inbox.put(msg)
        except WebSocketDisconnect:
            cancel.set()
            await inbox.put(None)

    reader_task = asyncio.create_task(reader())
    try:
        while True:
            msg = await inbox.get()
            if msg is None:
                break
            if msg.get("type") == "_invalid":
                await ws.send_json({"type": "error", "message": "invalid JSON"})
                continue
            if msg.get("type") != "synthesize":
                await ws.send_json({"type": "error", "message": "unknown message type"})
                continue

            cancel.clear()
            text = msg.get("text", "")
            speed = float(msg.get("speed", 1.0))
            try:
                sentences = await loop.run_in_executor(pool, engine.sentences, text)
                cancelled = False
                for ids in sentences:
                    if cancel.is_set():
                        cancelled = True
                        break
                    t0 = time.perf_counter()
                    pcm, audio_s = await loop.run_in_executor(pool, engine.synth, ids, speed)
                    await ws.send_json({
                        "type": "chunk_meta", "text": "",
                        "gen_ms": (time.perf_counter() - t0) * 1000,
                        "audio_s": audio_s, "providers": ["cpu-onnx"],
                    })
                    await ws.send_bytes(pcm)
                await ws.send_json({"type": "cancelled" if cancelled else "done"})
            except WebSocketDisconnect:
                break
            except Exception as e:  # noqa: BLE001 - report to client, keep the process alive
                await ws.send_json({"type": "error", "message": str(e)})
    finally:
        reader_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "::"), port=int(os.environ.get("PORT", "8080")), log_level="info")
