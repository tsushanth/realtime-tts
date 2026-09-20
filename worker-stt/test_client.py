"""Test client: mints a session token (same scheme as gateway/keys.js), hits POST /v1/stt and streams clips in real time over the WS.
usage: STT_SECRET_FILE=... python test_client.py https://<app>.modal.run  clip1.wav [clip2.wav ...]"""
import base64, hashlib, hmac, json, os, sys, time, asyncio, urllib.request, uuid

def mint(secret, key_id="test-key", ttl_ms=600000):
    payload = base64.urlsafe_b64encode(json.dumps({"id": key_id, "exp": int(time.time() * 1000) + ttl_ms}).encode()).decode().rstrip("=")
    sig = base64.urlsafe_b64encode(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    return f"{payload}.{sig}"

def post_file(base, token, path, fmt="auto"):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="format"\r\n\r\n{fmt}\r\n'
            f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a"\r\nContent-Type: application/octet-stream\r\n\r\n').encode() \
           + open(path, "rb").read() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(base + "/v1/stt", data=body, headers={"Authorization": "Bearer " + token, "Content-Type": "multipart/form-data; boundary=" + b})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r, time.perf_counter() - t

async def stream_clips(base, token, paths, tail_s=1.5, endpoint_ms=500):
    import websockets, soundfile as sf, numpy as np
    url = base.replace("https", "wss") + "/v1/stt/stream?token=" + token
    out = []
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"encoding": "pcm16", "endpoint_ms": endpoint_ms}))
        for p in paths:
            x, sr = sf.read(p, dtype="int16"); assert sr == 16000
            finals = asyncio.Queue()
            async def reader():
                async for m in ws:
                    j = json.loads(m)
                    if j["type"] == "final": finals.put_nowait((time.perf_counter(), j))
                    elif j["type"] == "usage": return
            rd = asyncio.create_task(reader())
            step = 320   # 20 ms
            t0 = time.perf_counter()
            for i in range(0, len(x), step):
                await ws.send(x[i:i + step].tobytes())
                await asyncio.sleep(max(0, t0 + (i + step) / 16000 - time.perf_counter()))
            t_end = time.perf_counter()               # last speech sample sent, real-time paced
            sil = np.zeros(step, dtype=np.int16).tobytes()
            for i in range(int(tail_s * 50)):
                await ws.send(sil); await asyncio.sleep(0.02)
                if not finals.empty(): break
            try:
                tf, j = await asyncio.wait_for(finals.get(), 10)
                out.append({"clip": os.path.basename(p), "eos_to_final_ms": round((tf - t_end) * 1000), "decode_ms": j.get("decode_ms"), "text": j["text"]})
            except asyncio.TimeoutError:
                out.append({"clip": os.path.basename(p), "error": "no final"})
            rd.cancel()
        await ws.send(json.dumps({"type": "end"}))
    return out

if __name__ == "__main__":
    base, paths = sys.argv[1].rstrip("/"), sys.argv[2:]
    tok = mint(open(os.environ["STT_SECRET_FILE"]).read().strip())
    print(json.dumps(post_file(base, tok, paths[0]), indent=1))
    res = asyncio.run(stream_clips(base, tok, paths))
    for r in res: print(r)
    lat = sorted(r["eos_to_final_ms"] for r in res if "eos_to_final_ms" in r)
    if lat: print("median eos->final ms", lat[len(lat) // 2], "p90", lat[int(len(lat) * .9) - 1], "n", len(lat))
