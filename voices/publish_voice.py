#!/usr/bin/env python3
"""Pull house voices from the `house-voices` Modal Volume and PUT them to the Piper server.

    PIPER_ADMIN_URL=https://<piper-host> VOICES_ADMIN_TOKEN=... \
        python3 voices/publish_voice.py en-us-kristin [--dry-run]
    python3 voices/publish_voice.py --tier A                 # bulk: every catalog.json voice of tier A
    python3 voices/publish_voice.py --tier A,B --lang es,en --allow-tier-b
    python3 voices/publish_voice.py --ids a,b,c --dry-run    # explicit list
Bulk selection reads voices/catalog.json. Tier B (Lessac lineage, see LICENSES.md) additionally requires
--allow-tier-b; tier C is never published. Voices already live on production (ALREADY_PUBLISHED) are skipped unless
--republish. Failures are reported at the end (non-zero exit); other voices are still attempted.

Needs the `modal` CLI (authenticated) and `requests`. Token is read from env only and never printed.
The tar (gzip) holds model.onnx, model.onnx.json, owner.json as flat members, which is what
PUT /admin/voices/<id> expects. The server must support public voices (owner.json {"public": true}).
"""
import io, json, os, re, subprocess, sys, tarfile, tempfile

FILES = ("model.onnx", "model.onnx.json", "owner.json")
# Already live on production as of 2026-09-20 (unpinned/speaker 0). Bulk selection skips them unless --republish.
ALREADY_PUBLISHED = {"en-us-ljspeech", "en-us-kristin", "en-us-john", "it-it-serena", "de-de-mls", "fr-fr-mls", "nl-nl-mls"}


def load_catalog():
    here = os.path.dirname(os.path.abspath(__file__))
    return json.load(open(os.path.join(here, "catalog.json")))["voices"]


def select(argv):
    """Returns the ordered list of voice ids for the given CLI args (pure; unit-testable)."""
    def opt(name):
        return next((a.split("=", 1)[1] for a in argv if a.startswith(name + "=")), None) or (
            argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None)
    ids = [a for a in argv if not a.startswith("--") and a not in (opt("--tier"), opt("--lang"), opt("--ids"))]
    if opt("--ids"):
        ids += opt("--ids").split(",")
    if opt("--tier") or opt("--lang"):
        tiers = set(opt("--tier").split(",")) if opt("--tier") else {"A"}
        langs = set(opt("--lang").split(",")) if opt("--lang") else None
        for v in load_catalog():
            if v["tier"] in tiers and (langs is None or v["language"].split("_")[0] in langs or v["language"] in langs):
                ids.append(v["id"])
    if "C" in (opt("--tier") or "").split(","):
        sys.exit("tier C voices are never published")
    tiers_of = {v["id"]: v["tier"] for v in load_catalog()} if os.path.exists(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")) else {}
    for i in ids:
        t = tiers_of.get(i)
        if t == "C":
            sys.exit(f"{i} is tier C; refusing")
        if t == "B" and "--allow-tier-b" not in argv:
            sys.exit(f"{i} is tier B (Lessac-derived lineage); pass --allow-tier-b to publish it")
    ids = list(dict.fromkeys(ids))
    if "--republish" not in argv:
        skipped = [i for i in ids if i in ALREADY_PUBLISHED]
        if skipped:
            print("skipping already-published:", ", ".join(skipped), "(use --republish to force)")
        ids = [i for i in ids if i not in ALREADY_PUBLISHED]
    return ids


def publish(vid, url, token, dry):
    with tempfile.TemporaryDirectory() as d:
        for f in FILES:
            subprocess.run(["modal", "volume", "get", "house-voices", f"{vid}/{f}", os.path.join(d, f)],
                           check=True, capture_output=True)
        owner = json.load(open(os.path.join(d, "owner.json")))
        if owner.get("public") is not True:
            raise RuntimeError("owner.json is not public:true; refusing")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for f in FILES:
                tf.add(os.path.join(d, f), arcname=f)
    print(f"{vid}: {len(buf.getvalue())/1e6:.1f} MB tar, license={owner['license']}")
    if dry:
        return True
    import requests
    r = requests.put(f"{url}/admin/voices/{vid}", data=buf.getvalue(), timeout=300,
                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/gzip"})
    print(r.status_code, r.text[:300])
    return r.ok


def main():
    argv = sys.argv[1:]
    dry = "--dry-run" in argv
    ids = select(argv)
    for vid in ids:
        if not re.fullmatch(r"[a-z0-9_-]{1,64}", vid):
            sys.exit(f"bad voice id {vid!r} (id: [a-z0-9_-], max 64 chars)")
    if not ids:
        sys.exit("usage: publish_voice.py <voice_id>... | --tier A[,B] [--lang es,en] | --ids a,b  [--dry-run] [--allow-tier-b]")
    url = os.environ.get("PIPER_ADMIN_URL", "").rstrip("/")
    token = os.environ.get("VOICES_ADMIN_TOKEN", "")
    if not dry and not (url and token):
        sys.exit("set PIPER_ADMIN_URL and VOICES_ADMIN_TOKEN")
    failed = []
    for vid in ids:
        try:
            if not publish(vid, url, token, dry):
                failed.append(vid)
        except Exception as e:  # noqa: BLE001
            print(f"{vid}: ERROR {type(e).__name__}: {str(e)[:200]}")
            failed.append(vid)
    print(f"{len(ids) - len(failed)}/{len(ids)} ok" + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
