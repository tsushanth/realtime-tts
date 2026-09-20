"""Test client: mints a session token like gateway/keys.js and POSTs audio to /v1/stt.
usage: STT_SESSION_SECRET_FILE=... python client.py https://<app>.modal.run FILE [FORMAT] [LANG]"""
import base64, hashlib, hmac, json, os, sys, time, urllib.request, urllib.error

def mint(secret, key_id="test-key", ttl_ms=600000):
    p = base64.urlsafe_b64encode(json.dumps({"id": key_id, "exp": int(time.time() * 1000) + ttl_ms}).encode()).decode().rstrip("=")
    s = base64.urlsafe_b64encode(hmac.new(secret.encode(), p.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    return f"{p}.{s}"

def post(base, token, data, query="", ctype="application/octet-stream", timeout=900):
    import requests   # urllib breaks when the server answers (401/413) before the upload finishes
    h = {**({"Authorization": f"Bearer {token}"} if token else {}), "Content-Type": ctype}
    t = time.perf_counter()
    try:
        r = requests.post(f"{base}/v1/stt{query}", data=data, headers=h, timeout=timeout)   # follows Modal's 303 polling redirects
    except requests.exceptions.ConnectionError as e:
        return None, {"error": f"connection error {type(e).__name__}"}, time.perf_counter() - t
    try:
        body = r.json()
    except ValueError:
        body = {"error": r.text[:200]}
    return r.status_code, body, time.perf_counter() - t

if __name__ == "__main__":
    secret = open(os.environ["STT_SESSION_SECRET_FILE"]).read().strip()
    base, path = sys.argv[1], sys.argv[2]
    q = "?" + "&".join(f"{k}={v}" for k, v in (("format", sys.argv[3] if len(sys.argv) > 3 else None),
                                               ("language", sys.argv[4] if len(sys.argv) > 4 else None)) if v)
    st, body, dt = post(base, mint(secret), open(path, "rb").read(), q if len(q) > 1 else "")
    print(st, f"{dt:.2f}s", json.dumps({k: (v if k in ("language", "duration", "error") else str(v)[:120]) for k, v in body.items()}))
