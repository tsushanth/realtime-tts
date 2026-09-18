"""
Piper (our fine-tuned voice) over WebSocket on CPU-only Modal - the always-on,
no-GPU, no-cold-start alternative to worker-modal/app.py (Kokoro on a T4).

Drop-in protocol (identical to worker-modal/app.py, so call-loop-poc switches by
changing TTS_GATEWAY_WS_URL only):
  client -> {"type": "synthesize", "text": "...", "voice": "<ignored>", "speed": 1.0}
  client -> {"type": "stop"}
  server -> {"type": "chunk_meta", "text": "", "gen_ms": .., "audio_s": .., "providers": [..]}
  server -> <binary PCM16LE mono 24kHz>      # immediately follows chunk_meta
  server -> {"type": "done"} | {"type": "cancelled"} | {"type": "error", "message": ".."}

Differences from the Kokoro worker, all deliberate:
- Piper natively outputs 22,050Hz; the contract is 24kHz, so each chunk is resampled
  (polyphase, exact rational ratio) before sending.
- `stop` actually works mid-utterance (checked between sentences): a reader task
  keeps receiving while synthesis runs. The Kokoro worker only reads the socket
  between requests, so its `stop` can't land until synthesis is already over.
- Synthesis is CPU/thread-pool work kept OFF the event loop; one blocking
  onnxruntime call would otherwise stall every other call's socket.
- espeak-ng (Piper's phonemizer) is a C library with global state: phonemization is
  serialized under a lock, only the ONNX inference runs in parallel. `bench` below
  measures whether that lock is actually required.

Model files come from the tts-checkpoints Volume (written by export_onnx.py) and
are copied to local disk at container start.

`modal run app.py::bench` sweeps onnxruntime thread count x concurrent callers to
find the real capacity of one container; the constants below come from that.
"""
import modal

app = modal.App("realtime-tts-worker-piper")

vol = modal.Volume.from_name("tts-checkpoints")
MODEL_DIR = "/checkpoints/piper_full_ft/serving"
MODEL_NAME = "full_ft.onnx"

CPU = 4
MEMORY_MB = 2048
ORT_INTRA_THREADS = 2   # set from bench results, see README.md in this directory
MAX_CONCURRENT_CALLS = 8
MIN_CONTAINERS = 0      # 0 = scale to zero (no standing cost). Set to 1 for always-on.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "piper-tts==1.8.0",
        "numpy<2",
        "scipy",
        "soundfile",
        "fastapi==0.109.0",
        "uvicorn[standard]==0.27.0",
    )
)

TEST_TEXTS = [
    "Thanks for calling, I can help you with that. Let me pull up your account details right now.",
    "Your order should arrive within three to five business days, and I will send a confirmation email shortly.",
    "I understand your frustration, let me see what I can do to make this right.",
    "Is there anything else I can help you with today?",
    "Your extension is six six three five.",
]


class PiperEngine:
    """Owns the loaded voice. Lazy imports: this module is also imported on the
    developer's machine (which has no piper installed) when Modal builds the app."""

    OUT_SR = 24000

    def __init__(self, intra_threads: int):
        import os
        import shutil
        import threading
        from fractions import Fraction

        import onnxruntime
        from piper import PiperVoice, SynthesisConfig

        os.makedirs("/tmp/model", exist_ok=True)
        for f in (MODEL_NAME, MODEL_NAME + ".json"):
            shutil.copy(f"{MODEL_DIR}/{f}", f"/tmp/model/{f}")
        path = f"/tmp/model/{MODEL_NAME}"

        self.voice = PiperVoice.load(path)
        # PiperVoice.load builds a default SessionOptions (onnxruntime then
        # sizes its thread pool from the HOST's core count, which oversubscribes
        # a CPU-limited container). Swap in a session with explicit threads.
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = 1
        self.voice.session = onnxruntime.InferenceSession(
            path, sess_options=so, providers=["CPUExecutionProvider"]
        )

        self.SynthesisConfig = SynthesisConfig
        ratio = Fraction(self.OUT_SR, self.voice.config.sample_rate).limit_denominator(1000)
        self.up, self.down = ratio.numerator, ratio.denominator
        self._phonemize_lock = threading.Lock()

    def sentences(self, text: str) -> list[list[int]]:
        """Text -> one phoneme-id list per sentence. Serialized (espeak global state)."""
        with self._phonemize_lock:
            return [
                self.voice.phonemes_to_ids(p)
                for p in self.voice.phonemize(text)
                if p
            ]

    def synth(self, phoneme_ids: list[int], speed: float = 1.0) -> tuple[bytes, float]:
        """One sentence -> (PCM16LE mono 24kHz bytes, audio seconds). Thread-safe."""
        import numpy as np
        from scipy.signal import resample_poly

        cfg = self.SynthesisConfig(
            length_scale=self.voice.config.length_scale / max(speed, 0.1)
        )
        audio = self.voice.phoneme_ids_to_audio(phoneme_ids, cfg)
        peak = float(np.max(np.abs(audio)))
        audio = audio / peak if peak > 1e-8 else np.zeros_like(audio)  # same as piper's normalize_audio
        if self.up != self.down:
            audio = resample_poly(audio, self.up, self.down)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes(), len(pcm) / self.OUT_SR


@app.function(
    image=image,
    cpu=CPU,
    memory=MEMORY_MB,
    min_containers=MIN_CONTAINERS,
    max_containers=5,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("tts-ws-auth-token")],
    volumes={"/checkpoints": vol},
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_CALLS)
@modal.asgi_app()
def web():
    import asyncio
    import json
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

    engine = PiperEngine(ORT_INTRA_THREADS)
    pool = ThreadPoolExecutor(max_workers=CPU)
    AUTH_TOKEN = os.environ["TTS_WS_AUTH_TOKEN"]
    web_app = FastAPI()

    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "model": "piper-full-ft", "device": "cpu"}

    @web_app.websocket("/tts")
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
            # Keeps receiving while synthesis runs, so `stop` lands mid-utterance.
            # Everything except `stop` goes through the inbox so only the main
            # loop below ever writes to the socket.
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
                            "type": "chunk_meta",
                            "text": "",
                            "gen_ms": (time.perf_counter() - t0) * 1000,
                            "audio_s": audio_s,
                            "providers": ["modal-cpu-onnx"],
                        })
                        await ws.send_bytes(pcm)
                    await ws.send_json({"type": "cancelled" if cancelled else "done"})
                except WebSocketDisconnect:
                    break
                except Exception as e:  # noqa: BLE001 - report to client, keep container alive
                    await ws.send_json({"type": "error", "message": str(e)})
        finally:
            reader_task.cancel()

    return web_app


# ---------------------------------------------------------------------------
# Capacity benchmark: `modal run app.py::bench`
# ---------------------------------------------------------------------------

@app.function(image=image, cpu=CPU, memory=MEMORY_MB, timeout=1800, volumes={"/checkpoints": vol})
def bench(threads_list: list[int], concurrency_list: list[int], reps: int = 6):
    import statistics
    import subprocess
    import sys
    import textwrap
    import time
    from concurrent.futures import ThreadPoolExecutor

    rows = []
    for threads in threads_list:
        engine = PiperEngine(threads)
        # warmup
        for t in TEST_TEXTS:
            for ids in engine.sentences(t):
                engine.synth(ids)

        for conc in concurrency_list:
            def one_request(i):
                text = TEST_TEXTS[i % len(TEST_TEXTS)]
                t0 = time.perf_counter()
                audio_s = 0.0
                for ids in engine.sentences(text):
                    _, a = engine.synth(ids)
                    audio_s += a
                return time.perf_counter() - t0, audio_s

            n = conc * reps
            wall0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=conc) as ex:
                out = list(ex.map(one_request, range(n)))
            wall = time.perf_counter() - wall0
            lat = sorted(o[0] for o in out)
            total_audio = sum(o[1] for o in out)
            rows.append({
                "threads": threads, "concurrent": conc,
                "p50_ms": statistics.median(lat) * 1000,
                "p95_ms": lat[int(len(lat) * 0.95) - 1] * 1000,
                "audio_s_per_wall_s": total_audio / wall,
            })

    # Is the phonemize lock actually needed? Run espeak from many threads with NO
    # lock in a subprocess (a segfault would kill this function otherwise) and
    # compare against a serial reference.
    probe = textwrap.dedent(f"""
        import sys, json
        from concurrent.futures import ThreadPoolExecutor
        sys.path.insert(0, "/")
        from piper import PiperVoice
        v = PiperVoice.load("/tmp/model/{MODEL_NAME}")
        texts = {TEST_TEXTS!r} * 40
        ref = [v.phonemize(t) for t in {TEST_TEXTS!r}]
        ref = ref * 40
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(v.phonemize, texts))
        bad = sum(1 for a, b in zip(ref, got) if a != b)
        print(json.dumps({{"total": len(texts), "mismatches": bad}}))
    """)
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=300)
    espeak = {"returncode": r.returncode, "stdout": r.stdout.strip()[-300:], "stderr": r.stderr.strip()[-300:]}
    return {"rows": rows, "espeak_unlocked_probe": espeak}


@app.local_entrypoint()
def bench_main(threads: str = "1,2,4", concurrency: str = "1,2,4,8,12"):
    res = bench.remote([int(x) for x in threads.split(",")], [int(x) for x in concurrency.split(",")])
    print(f"{'threads':>7} {'callers':>7} {'p50 ms':>8} {'p95 ms':>8} {'audio-s/wall-s':>15}")
    for r in res["rows"]:
        print(f"{r['threads']:>7} {r['concurrent']:>7} {r['p50_ms']:>8.0f} {r['p95_ms']:>8.0f} {r['audio_s_per_wall_s']:>15.1f}")
    print("espeak unlocked probe:", res["espeak_unlocked_probe"])


@app.function(image=image, cpu=CPU, memory=MEMORY_MB, timeout=600, volumes={"/checkpoints": vol})
def phonemize_probe():
    """Isolates per-request fixed costs from synthesis: text->phoneme-ids time."""
    import statistics
    import time

    engine = PiperEngine(ORT_INTRA_THREADS)
    engine.sentences(TEST_TEXTS[0])  # warmup
    out = {}
    for t in TEST_TEXTS:
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            engine.sentences(t)
            ts.append((time.perf_counter() - t0) * 1000)
        out[t[:32]] = round(statistics.median(ts), 1)
    return out


@app.local_entrypoint()
def phonemize_main():
    print(phonemize_probe.remote())
