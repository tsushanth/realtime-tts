#!/usr/bin/env python3
"""Pull a house voice from the `house-voices` Modal Volume and PUT it to the Piper server.

    PIPER_ADMIN_URL=https://<piper-host> VOICES_ADMIN_TOKEN=... \
        python3 voices/publish_voice.py en-us-kristin [--dry-run]

Needs the `modal` CLI (authenticated) and `requests`. Token is read from env only and never printed.
The tar (gzip) holds model.onnx, model.onnx.json, owner.json as flat members, which is what
PUT /admin/voices/<id> expects. The server must support public voices (owner.json {"public": true}).
"""
import io, json, os, re, subprocess, sys, tarfile, tempfile

FILES = ("model.onnx", "model.onnx.json", "owner.json")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if len(args) != 1 or not re.fullmatch(r"[a-z0-9_-]{1,64}", args[0]):
        sys.exit("usage: publish_voice.py <voice_id> [--dry-run]   (id: [a-z0-9_-], max 64 chars)")
    vid = args[0]
    url = os.environ.get("PIPER_ADMIN_URL", "").rstrip("/")
    token = os.environ.get("VOICES_ADMIN_TOKEN", "")
    if not dry and not (url and token):
        sys.exit("set PIPER_ADMIN_URL and VOICES_ADMIN_TOKEN")
    with tempfile.TemporaryDirectory() as d:
        for f in FILES:
            subprocess.run(["modal", "volume", "get", "house-voices", f"{vid}/{f}", os.path.join(d, f)],
                           check=True, capture_output=True)
        owner = json.load(open(os.path.join(d, "owner.json")))
        if owner.get("public") is not True:
            sys.exit("owner.json is not public:true; refusing")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for f in FILES:
                tf.add(os.path.join(d, f), arcname=f)
    print(f"{vid}: {len(buf.getvalue())/1e6:.1f} MB tar, license={owner['license']}")
    if dry:
        return
    import requests
    r = requests.put(f"{url}/admin/voices/{vid}", data=buf.getvalue(), timeout=300,
                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/gzip"})
    print(r.status_code, r.text[:300])
    sys.exit(0 if r.ok else 1)


if __name__ == "__main__":
    main()
