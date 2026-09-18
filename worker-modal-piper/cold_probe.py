"""Measures connect -> first audio on a FRESH connection after the service has been
idle, N cycles, subtracting the network RTT (measured with a zero-synthesis error
reply on the same connection) so the number is the service's, not the test topology's.
Run inside Modal so the auth secret stays there: modal run cold_probe.py --cycles 3 --idle 150
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("websockets")
app = modal.App("piper-cold-probe", image=image)
URL = "wss://t-sushanth--realtime-tts-worker-piper-web.modal.run/tts"


@app.function(secrets=[modal.Secret.from_name("tts-ws-auth-token")], timeout=1800)
def probe(cycles: int, idle_s: int):
    import asyncio, json, os, time, websockets
    token = os.environ["TTS_WS_AUTH_TOKEN"]

    async def one():
        t0 = time.perf_counter()
        async with websockets.connect(f"{URL}?token={token}", open_timeout=120) as ws:
            t_conn = time.perf_counter()
            await ws.send(json.dumps({"type": "synthesize", "text": "Thanks for calling, how can I help you today?"}))
            while True:
                m = await ws.recv()
                if isinstance(m, bytes):
                    t_audio = time.perf_counter()
                    break
            while True:
                m = await ws.recv()
                if not isinstance(m, bytes) and json.loads(m)["type"] == "done":
                    break
            r0 = time.perf_counter()
            await ws.send("not json")
            await ws.recv()
            rtt = time.perf_counter() - r0
        return {
            "connect_ms": round((t_conn - t0) * 1000),
            "first_audio_after_connect_ms": round((t_audio - t_conn) * 1000),
            "connect_to_first_audio_ms": round((t_audio - t0) * 1000),
            "rtt_ms": round(rtt * 1000),
        }

    out = []
    for i in range(cycles):
        out.append(asyncio.run(one()))
        if i < cycles - 1:
            time.sleep(idle_s)
    return out


@app.local_entrypoint()
def main(cycles: int = 3, idle: int = 150):
    for i, r in enumerate(probe.remote(cycles, idle)):
        svc = r["connect_to_first_audio_ms"] - r["rtt_ms"] * 2
        print(f"cycle {i}: {r}  ~service-only (minus 2 RTT): {svc}ms")
