"""Runs INSIDE Modal (fast network) to exercise upload limits and the 3 h long-file path against a deployed TEST app,
which a home connection cannot do (uploads of 100+ MB take minutes; Modal answers requests running >150 s with a 303 redirect).
  modal run tests/cloud_client.py::main --url https://<test-app>.modal.run --base /tmp/sttprod/base.wav
Benchmark (cold start + latency; run after >120 s idle): modal run tests/cloud_client.py::bench_main --url ... --base ...
Uses secret stt-prod-test-secret to mint tokens. Prints status/timing only."""
import modal

app = modal.App("stt-prod-test-client")
img = modal.Image.debian_slim().apt_install("ffmpeg").pip_install("requests")

@app.function(image=img, secrets=[modal.Secret.from_name("stt-prod-test-secret")], timeout=1800, cpu=2)
def run(url: str, base_wav: bytes):
    import base64, hashlib, hmac, json, os, subprocess, time, requests
    def mint(ttl=3600000):
        p = base64.urlsafe_b64encode(json.dumps({"id": "cloud-test", "exp": int(time.time() * 1000) + ttl}).encode()).decode().rstrip("=")
        s = base64.urlsafe_b64encode(hmac.new(os.environ["MODAL_SESSION_SECRET"].encode(), p.encode(), hashlib.sha256).digest()).decode().rstrip("=")
        return f"{p}.{s}"
    def post(data, q, label):
        t = time.time()
        try:
            r = requests.post(f"{url}/v1/stt{q}", data=data, headers={"Authorization": f"Bearer {mint()}"}, timeout=1500)
            try:
                j = r.json()
            except ValueError:
                j = {"raw": r.text[:200]}
            print(f"{label}: HTTP {r.status_code} in {time.time() - t:.1f}s",
                  {k: (v if k in ("error", "duration", "language") else len(str(v))) for k, v in j.items()}, flush=True)
            return j
        except Exception as e:
            print(f"{label}: EXC {type(e).__name__} after {time.time() - t:.1f}s {str(e)[:150]}", flush=True)
    open("/tmp/base.wav", "wb").write(base_wav)
    n = 240   # 44.2 s * 240 = 2.95 h (just under the limit)
    open("/tmp/l.txt", "w").write("file '/tmp/base.wav'\n" * n)
    subprocess.run("ffmpeg -v error -y -f concat -safe 0 -i /tmp/l.txt -c:a libmp3lame -b:a 64k -ac 1 /tmp/3h.mp3", shell=True, check=True)
    size = os.path.getsize("/tmp/3h.mp3")
    dur = float(subprocess.check_output("ffprobe -v error -show_entries format=duration -of csv=p=0 /tmp/3h.mp3", shell=True))
    print(f"3h clip: {size / 1e6:.1f} MB, {dur / 3600:.3f} h", flush=True)
    post(open("/tmp/3h.mp3", "rb").read(), "?format=mp3", "near-3h mp3 (should succeed if <=3h)")
    subprocess.run("ffmpeg -v error -y -f concat -safe 0 -i /tmp/l.txt -c:a libmp3lame -b:a 64k -ac 1 /tmp/x.mp3; "
                   "printf \"file '/tmp/base.wav'\\n\" > /tmp/l2.txt; for i in 1 2 3 4; do cat /tmp/l2.txt >> /tmp/l3.txt; done; "
                   "ffmpeg -v error -y -f concat -safe 0 -i /tmp/l3.txt -c copy /tmp/short.wav", shell=True)
    post(b"\xff" * (8000 * (3 * 3600 + 120)), "?format=mulaw_8000", "3h02m mulaw (expect 413)")
    post(b"\0" * (201 * 1024 * 1024), "?format=pcm_16000", "201 MB body (expect 413)")
    post(b"\0" * (150 * 1024 * 1024), "?format=pcm_16000", "150 MB pcm silence 78 min (expect 200, empty text)")
    # multipart 60 MB: header check + streaming parser
    files = {"file": ("a.mp3", open("/tmp/3h.mp3", "rb"))}
    t = time.time(); r = requests.post(f"{url}/v1/stt?format=mp3", files=files, headers={"Authorization": f"Bearer {mint()}"}, timeout=1500)
    print(f"multipart near-3h: HTTP {r.status_code} in {time.time() - t:.1f}s", str(r.json())[:80] if r.status_code != 200 else r.json()["duration"], flush=True)

@app.local_entrypoint()
def main(url: str, base: str):
    run.remote(url, open(base, "rb").read())


@app.function(image=img, secrets=[modal.Secret.from_name("stt-prod-test-secret")], timeout=1800)
def bench(url: str, base_wav: bytes):
    """Cold start + latency for 10 s / 60 s / 10 min clips (fast in-cloud network, so upload time is ~0).
    Run after the app has been idle > 120 s so the first request is a true cold start."""
    import base64, hashlib, hmac, json, os, subprocess, time, requests
    def mint():
        p = base64.urlsafe_b64encode(json.dumps({"id": "cloud-bench", "exp": int(time.time() * 1000) + 3600000}).encode()).decode().rstrip("=")
        s = base64.urlsafe_b64encode(hmac.new(os.environ["MODAL_SESSION_SECRET"].encode(), p.encode(), hashlib.sha256).digest()).decode().rstrip("=")
        return f"{p}.{s}"
    open("/tmp/base.wav", "wb").write(base_wav)
    open("/tmp/l.txt", "w").write("file '/tmp/base.wav'\n" * 14)
    subprocess.run("ffmpeg -v error -y -f concat -safe 0 -i /tmp/l.txt -c copy /tmp/long.wav && "
                   "ffmpeg -v error -y -i /tmp/long.wav -t 10 /tmp/c10.wav && ffmpeg -v error -y -i /tmp/long.wav -t 60 /tmp/c60.wav && "
                   "ffmpeg -v error -y -i /tmp/long.wav -t 600 /tmp/c600.wav", shell=True, check=True)
    def go(name):
        d = open(f"/tmp/{name}.wav", "rb").read(); t = time.time()
        r = requests.post(f"{url}/v1/stt", data=d, headers={"Authorization": f"Bearer {mint()}"}, timeout=1200)
        print(f"{name}: {r.status_code} {time.time() - t:.2f}s", flush=True)
    print("COLD:", flush=True); go("c10")
    for n in ("c10", "c10", "c10", "c60", "c60", "c60", "c600", "c600", "c600"):
        go(n)

@app.local_entrypoint()
def bench_main(url: str, base: str):
    bench.remote(url, open(base, "rb").read())
