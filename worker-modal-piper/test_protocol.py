"""End-to-end protocol test against the DEPLOYED Piper worker. Runs inside Modal
so the shared auth token (Modal Secret tts-ws-auth-token) is injected as an env
var instead of being read/printed locally. Latency here is Modal->Modal, so it
EXCLUDES the client's own internet path and any Cloudflare edge hop - it measures
the service, not the whole call path.

Run: modal run test_protocol.py
"""
import modal

URL = "wss://t-sushanth--realtime-tts-worker-piper-web.modal.run/tts"

image = modal.Image.debian_slim(python_version="3.11").pip_install("websockets")
app = modal.App("piper-protocol-test", image=image)

LONG_TEXT = " ".join([
    "Thank you for holding while I looked into that for you.",
    "I can see your account and the recent order you mentioned.",
    "It shipped on Tuesday and should arrive by Friday afternoon.",
    "I have also added a note so the carrier calls before delivery.",
    "If anything changes I will send you a text message right away.",
    "Would you like me to go over the return policy as well.",
    "There is no charge for returns within thirty days of delivery.",
    "Is there anything else I can help you with today.",
])


@app.function(secrets=[modal.Secret.from_name("tts-ws-auth-token")], timeout=600)
def run_tests():
    import asyncio
    import json
    import os
    import statistics
    import time

    import websockets

    token = os.environ["TTS_WS_AUTH_TOKEN"]
    results = {}

    async def synth(ws, text, speed=1.0, stop_after_first=False):
        """Returns dict with framing checks + timings for one request."""
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "synthesize", "text": text, "voice": "x", "speed": speed}))
        ttfb = None
        chunks = []           # (audio_s, nbytes)
        pending_meta = None
        terminal = None
        framing_ok = True
        while True:
            m = await asyncio.wait_for(ws.recv(), timeout=60)
            if isinstance(m, bytes):
                if pending_meta is None:
                    framing_ok = False
                else:
                    if ttfb is None:
                        ttfb = time.perf_counter() - t0
                    chunks.append((pending_meta["audio_s"], len(m), pending_meta["gen_ms"]))
                    pending_meta = None
                    if stop_after_first and len(chunks) == 1:
                        await ws.send(json.dumps({"type": "stop"}))
            else:
                j = json.loads(m)
                if j["type"] == "chunk_meta":
                    if pending_meta is not None:
                        framing_ok = False
                    pending_meta = j
                else:
                    terminal = j
                    break
        # bytes must equal round(audio_s*24000)*2 (PCM16 mono 24kHz)
        bytes_ok = all(abs(nb - round(a * 24000) * 2) <= 2 for a, nb, _ in chunks)
        return {
            "ttfb_ms": None if ttfb is None else ttfb * 1000,
            "total_ms": (time.perf_counter() - t0) * 1000,
            "n_chunks": len(chunks),
            "audio_s": sum(a for a, _, _ in chunks),
            "server_gen_ms": sum(g for _, _, g in chunks),
            "terminal": terminal["type"],
            "framing_ok": framing_ok and pending_meta is None,
            "bytes_match_duration": bytes_ok,
        }

    async def main():
        # 1. auth rejection
        try:
            async with websockets.connect(URL + "?token=wrong", open_timeout=90) as ws:
                await ws.recv()
            results["bad_token"] = "UNEXPECTED: connection accepted"
        except Exception as e:  # noqa: BLE001
            results["bad_token"] = f"rejected ({type(e).__name__}: {str(e)[:80]})"

        # 2. cold start (container is scaled to zero right after deploy / idle)
        t0 = time.perf_counter()
        async with websockets.connect(f"{URL}?token={token}", open_timeout=120) as ws:
            connect_ms = (time.perf_counter() - t0) * 1000
            first = await synth(ws, "Thanks for calling, how can I help you today?")
            results["cold_connect_ms"] = round(connect_ms)
            results["cold_first_request"] = first
            cold_total = (time.perf_counter() - t0) * 1000
            results["cold_connect_to_first_audio_ms"] = round(connect_ms + first["ttfb_ms"])

            # 3. warm, same connection, several requests
            warm = []
            for t in ["Your extension is six six three five.",
                      "Is there anything else I can help you with today?",
                      "I understand your frustration, let me see what I can do to make this right."]:
                warm.append(await synth(ws, t))
            results["warm_requests"] = warm

            # 4. speed control changes duration
            slow = await synth(ws, "Is there anything else I can help you with today?", speed=0.8)
            fast = await synth(ws, "Is there anything else I can help you with today?", speed=1.25)
            results["speed_audio_s"] = {"0.8x": round(slow["audio_s"], 2), "1.25x": round(fast["audio_s"], 2)}

            # 5. stop mid-utterance
            full = await synth(ws, LONG_TEXT)
            stopped = await synth(ws, LONG_TEXT, stop_after_first=True)
            results["stop_test"] = {
                "full_chunks": full["n_chunks"], "stopped_chunks": stopped["n_chunks"],
                "terminal": stopped["terminal"],
            }

            # 6. bad input keeps the connection usable
            await ws.send("not json")
            e1 = json.loads(await ws.recv())
            await ws.send(json.dumps({"type": "bogus"}))
            e2 = json.loads(await ws.recv())
            after = await synth(ws, "Still working.")
            results["bad_input"] = {"invalid_json": e1, "unknown_type": e2, "works_after": after["terminal"]}

        # 7. concurrency: N callers x 3 requests each
        async def caller(i):
            out = []
            async with websockets.connect(f"{URL}?token={token}", open_timeout=120) as ws:
                for t in ["Thanks for calling, I can help you with that. Let me pull up your account details right now.",
                          "Your order should arrive within three to five business days.",
                          "Is there anything else I can help you with today?"]:
                    out.append(await synth(ws, t))
            return out

        for n in (4, 8, 16):
            t0 = time.perf_counter()
            allr = await asyncio.gather(*[caller(i) for i in range(n)])
            wall = time.perf_counter() - t0
            flat = [r for c in allr for r in c]
            tt = sorted(r["ttfb_ms"] for r in flat)
            results[f"concurrent_{n}"] = {
                "requests": len(flat),
                "ttfb_p50_ms": round(statistics.median(tt)),
                "ttfb_p95_ms": round(tt[int(len(tt) * 0.95) - 1]),
                "all_done": all(r["terminal"] == "done" for r in flat),
                "all_framing_ok": all(r["framing_ok"] and r["bytes_match_duration"] for r in flat),
                "wall_s": round(wall, 1),
            }

    asyncio.run(main())
    return results


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run_tests.remote(), indent=2))
