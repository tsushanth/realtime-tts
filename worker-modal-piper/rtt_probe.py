import modal
image = modal.Image.debian_slim(python_version="3.11").pip_install("websockets")
app = modal.App("piper-rtt-probe", image=image)
URL = "wss://t-sushanth--realtime-tts-worker-piper-web.modal.run/tts"

@app.function(secrets=[modal.Secret.from_name("tts-ws-auth-token")], timeout=300)
def probe():
    import asyncio, os, statistics, time, websockets
    token = os.environ["TTS_WS_AUTH_TOKEN"]
    async def go():
        async with websockets.connect(f"{URL}?token={token}", open_timeout=120) as ws:
            ts = []
            for _ in range(40):
                t0 = time.perf_counter()
                await ws.send("not json")
                await ws.recv()
                ts.append((time.perf_counter() - t0) * 1000)
            ts = ts[5:]
            return {"rtt_p50_ms": round(statistics.median(ts), 1), "rtt_max_ms": round(max(ts), 1)}
    return asyncio.run(go())

@app.local_entrypoint()
def main():
    print(probe.remote())
