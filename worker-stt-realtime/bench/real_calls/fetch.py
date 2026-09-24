"""screen: show how many recorded calls exist and which masked numbers they involve.
fetch:  download poc-engine calls to data/real_raw/ as mono 16 kHz wav of one channel (default: far end,
        channel 0). Audio comes through calldesktech's call-loop-poc recording proxy
        (CALL_LOOP_POC_BASE_URL / CALL_LOOP_POC_TEST_CALL_SECRET from --calldesk-env). Only calls whose
        counterparty is in data/own_numbers.txt unless --include-unmatched.
Secrets are read from env files at run time and never printed."""
import argparse
import json
import os
import subprocess
from collections import Counter

import requests

from .numbers import counterparty, mask_number, parse_env, recording_wav_url, select_own_calls

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("STT_DATA", os.path.abspath(os.path.join(HERE, "..", "..", "data")))
COLS = "id,voice_engine,direction,caller_phone,to_number,recording_url,created_at,duration_seconds"


def load_env(path):
    with open(os.path.expanduser(path)) as f:
        return parse_env(f.read())


def fetch_rows(env, limit):
    key = env["SUPABASE_SERVICE_ROLE_KEY"]
    r = requests.get(
        f"{env['NEXT_PUBLIC_SUPABASE_URL']}/rest/v1/calldesk_call_logs",
        params={"select": COLS, "recording_url": "not.is.null", "voice_engine": "eq.poc",
                "order": "created_at.desc", "limit": str(limit)},
        headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return r.json()


def read_own_numbers(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _channels(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=channels", "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def _duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


TRUNCATION_RATIO = 0.7  # downloaded audio shorter than this fraction of the logged duration = truncated


def _is_fresh(dest, caller_channel):
    """A cached wav is reusable only if its sidecar records the same channel; no sidecar = stale."""
    if not os.path.exists(dest):
        return False
    try:
        with open(dest + ".meta") as f:
            return json.load(f).get("caller_channel") == caller_channel
    except (OSError, ValueError):
        return False


def download_call(row, base_url, secret, raw_dir, caller_channel=0):
    """Fetch via calldesktech's recording proxy; keep one channel as mono 16 kHz PCM_16."""
    dest = os.path.join(raw_dir, row["id"] + ".wav")
    meta = dest + ".meta"
    if _is_fresh(dest, caller_channel):
        return dest
    for p in (dest, meta):
        if os.path.exists(p):
            os.remove(p)
    src, part = dest + ".src.wav", dest + ".part.wav"
    try:
        with requests.get(base_url.rstrip("/") + "/recording-audio",
                          params={"url": recording_wav_url(row["recording_url"])},
                          headers={"Authorization": f"Bearer {secret}"}, stream=True, timeout=120) as r:
            r.raise_for_status()
            written = 0
            with open(src, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
                    written += len(chunk)
            cl = r.headers.get("Content-Length")
            if cl and not r.headers.get("Content-Encoding") and int(cl) != written:
                raise RuntimeError("downloaded audio incomplete (Content-Length mismatch)")
        logged = row.get("duration_seconds")
        if logged and _duration(src) < TRUNCATION_RATIO * float(logged):
            raise RuntimeError("downloaded audio much shorter than logged duration")
        af = f"pan=mono|c0=c{caller_channel}" if _channels(src) >= 2 else "anull"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", af, "-ar", "16000",
                        "-c:a", "pcm_s16le", part], check=True)
        os.replace(part, dest)
        with open(meta, "w") as f:
            json.dump({"caller_channel": caller_channel, "sr": 16000, "source": "proxy-wav"}, f)
    finally:
        for p in (src, part):
            if os.path.exists(p):
                os.remove(p)
    return dest


def cmd_screen(a):
    rows = fetch_rows(load_env(a.calldesk_env), a.limit)
    print(f"{len(rows)} recorded poc-engine calls")
    counts = Counter((r["direction"], mask_number(counterparty(r))) for r in rows)
    for (direction, masked), n in counts.most_common(15):
        print(f"  {direction:8s} {masked:16s} {n}")
    own_path = os.path.join(DATA, "own_numbers.txt")
    if os.path.exists(own_path):
        own = read_own_numbers(own_path)
        print(f"{len(select_own_calls(rows, own))} of {len(rows)} match data/own_numbers.txt")
    else:
        print("data/own_numbers.txt not found: write your own phone numbers there, one per line")


def cmd_fetch(a):
    env = load_env(a.calldesk_env)
    try:
        base_url, secret = env["CALL_LOOP_POC_BASE_URL"], env["CALL_LOOP_POC_TEST_CALL_SECRET"]
    except KeyError:
        raise SystemExit(f"missing CALL_LOOP_POC_BASE_URL / CALL_LOOP_POC_TEST_CALL_SECRET in {a.calldesk_env}") from None
    rows = fetch_rows(env, a.limit)
    if a.include_unmatched:
        picked = [r for r in rows if r.get("voice_engine") == "poc" and r.get("recording_url")]
        if a.max_calls:
            picked = picked[:a.max_calls]
        print(f"including {len(picked)} poc-engine calls without matching data/own_numbers.txt")
    else:
        own_path = os.path.join(DATA, "own_numbers.txt")
        if not os.path.exists(own_path):
            raise SystemExit("data/own_numbers.txt is required (your own phone numbers, one per line)")
        picked = select_own_calls(rows, read_own_numbers(own_path), limit=a.max_calls)
    print(f"downloading {len(picked)} calls")
    raw_dir = os.path.join(DATA, "real_raw")
    os.makedirs(raw_dir, exist_ok=True)
    stale = sum(1 for r in picked if os.path.exists(os.path.join(raw_dir, r["id"] + ".wav"))
                and not _is_fresh(os.path.join(raw_dir, r["id"] + ".wav"), a.caller_channel))
    if stale:
        print(f"refetching {stale} files (channel/format unknown or different)")
    for r in picked:
        download_call(r, base_url, secret, raw_dir, caller_channel=a.caller_channel)
    with open(os.path.join(raw_dir, "calls.json"), "w") as f:
        json.dump(picked, f, indent=1)
    print(f"done: {raw_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "fetch"])
    ap.add_argument("--calldesk-env", default="~/Documents/GitHub/calldesktech/.env")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--caller-channel", type=int, choices=[0, 1], default=0,
                    help="0 = far end (caller/callee/test persona), 1 = our agent; verified 2026-09-24")
    ap.add_argument("--include-unmatched", action="store_true",
                    help="also download poc-engine calls whose phone columns do not match data/own_numbers.txt")
    a = ap.parse_args()
    {"screen": cmd_screen, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
