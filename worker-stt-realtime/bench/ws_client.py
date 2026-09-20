"""Real-time replay client for server.py: paces PCM16 (or mu-law 8k) chunks at wall-clock speed over the WebSocket and
reports end-of-speech -> final latency INCLUDING the network/WebSocket/server stack.
  python bench/ws_client.py --url ws://localhost:8080/v1/stt/realtime --secret test --set tel --n 30 [--mulaw] [--commit]
"""
import argparse, asyncio, base64, hashlib, hmac, json, os, sys, time
import numpy as np, websockets, jiwer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtbench import load, norm

def token(secret, key_id="k_test", ttl=300):
    p = base64.urlsafe_b64encode(json.dumps({"id": key_id, "exp": int((time.time() + ttl) * 1000)}).encode()).decode().rstrip("=")
    s = base64.urlsafe_b64encode(hmac.new(secret.encode(), p.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    return f"{p}.{s}"

def to_mulaw(x16):
    x = np.clip(x16, -1, 1); mu = 255.0
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return ((np.round((y + 1) / 2 * 255)).astype(np.uint8) ^ 0xFF).tobytes() if False else _enc(x)

def _enc(x):
    # G.711 mu-law encode (8 kHz), inverse of the server table
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int32); sign = (pcm < 0) * 0x80
    pcm = np.minimum(np.abs(pcm), 32635) + 0x84
    exp = np.floor(np.log2(pcm)).astype(np.int32) - 7; man = (pcm >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | man) & 0xFF).astype(np.uint8).tobytes()

async def one(url, tok, clip, mulaw, commit, chunk_ms=40, extra=""):
    q = f"?token={tok}" + ("&encoding=mulaw&sample_rate=8000" if mulaw else "") + extra
    a = clip["audio"]
    if mulaw:
        from scipy.signal import resample_poly
        a = resample_poly(a, 1, 2).astype("float32")
    sr = 8000 if mulaw else 16000
    step = int(sr * chunk_ms / 1000)
    async with websockets.connect(url + q, max_size=None) as ws:
        first = json.loads(await ws.recv())   # ready or loading
        while first.get("type") == "loading":
            first = json.loads(await ws.recv())
        finals, partials, specs = [], [], []
        async def rx():
            try:
                async for m in ws:
                    e = json.loads(m); e["_t"] = time.perf_counter()
                    (finals if e["type"] in ("final", "usage") else partials if e["type"] == "partial" else specs if e["type"] in ("speculative_final", "resume") else []).append(e)
            except Exception:
                pass
        r = asyncio.ensure_future(rx())
        t0 = time.perf_counter(); dur = clip["dur"]
        committed = False
        for i in range(0, len(a), step):
            due = t0 + (i + step) / sr
            d = due - time.perf_counter()
            if d > 0: await asyncio.sleep(d)
            x = a[i:i + step]
            await ws.send(_enc(x) if mulaw else (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
            if commit and not committed and (i + step) / sr >= dur:
                committed = True; await ws.send(json.dumps({"type": "commit"}))
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "end"}))
        await asyncio.wait_for(r, 15)
    sp_after = [f for f in specs if f['type'] == 'speculative_final' and f['_t'] - t0 >= dur - 0.05]
    fin = [f for f in finals if f["type"] == "final"]
    text = " ".join(f["text"] for f in fin)
    after = [f for f in fin if f["_t"] - t0 >= dur - 0.05]
    lat = (after[0]["_t"] - t0 - dur) if after else None
    fp = min((p["_t"] - t0 for p in partials), default=None)
    usage = next((f for f in finals if f["type"] == "usage"), None)
    return dict(id=clip["id"], lat=lat, spec_lat=(sp_after[0]['_t'] - t0 - dur) if sp_after else None, resumes=sum(1 for s in specs if s['type'] == 'resume'), first_partial=fp, cuts=len([f for f in fin if f["_t"] - t0 < dur - 0.05]), text=text, ref=clip["ref"], usage=usage and usage["audio_seconds"])

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8080/v1/stt/realtime"); ap.add_argument("--secret", default="test")
    ap.add_argument("--set", default="tel"); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mulaw", action="store_true"); ap.add_argument("--commit", action="store_true")
    ap.add_argument("--concurrency", type=int, default=1); ap.add_argument("--endpoint-ms", type=int, default=500)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    clips = load(a.set, a.n); tok = token(a.secret)
    sem = asyncio.Semaphore(a.concurrency); res = []
    async def go(c):
        async with sem:
            res.append(await one(a.url, tok, c, a.mulaw, a.commit, extra=f"&endpoint_ms={a.endpoint_ms}"))
    await asyncio.gather(*[go(c) for c in clips])
    lats = [r["lat"] for r in res if r["lat"] is not None]; fps = [r["first_partial"] for r in res if r["first_partial"] is not None]
    err = sum((lambda o: o.substitutions + o.deletions + o.insertions)(jiwer.process_words(norm(r["ref"]), norm(r["text"]) or "x")) for r in res)
    words = sum(len(norm(r["ref"]).split()) for r in res)
    out = dict(n=len(res), concurrency=a.concurrency, mulaw=a.mulaw, commit=a.commit, endpoint_ms=a.endpoint_ms, wer=err / words,
               lat_med=float(np.median(lats)), lat_p90=float(np.percentile(lats, 90)), lat_max=max(lats), missing_final=len(res) - len(lats),
               spec_lat_med=float(np.median([r['spec_lat'] for r in res if r['spec_lat'] is not None] or [0])), resumes=sum(r['resumes'] for r in res), first_partial_med=float(np.median(fps)), cut_utts=sum(1 for r in res if r["cuts"]), usage_total_s=sum(r["usage"] or 0 for r in res))
    print(json.dumps(out))
    if a.out: json.dump({"summary": out, "rows": res}, open(a.out, "w"))
asyncio.run(main())
