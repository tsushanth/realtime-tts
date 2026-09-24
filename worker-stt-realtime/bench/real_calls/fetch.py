"""screen: show how many recorded calls exist and which masked numbers they involve.
fetch:  download calls whose counterparty is in data/own_numbers.txt (one number per line) to data/real_raw/.
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


def download_call(row, sid, token, raw_dir):
    dest = os.path.join(raw_dir, row["id"] + ".wav")
    if os.path.exists(dest):
        return dest
    tmp = dest + ".src.wav"
    part = dest + ".part.wav"
    try:
        with requests.get(recording_wav_url(row["recording_url"]), auth=(sid, token), stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-ar", "16000", part], check=True)
        os.replace(part, dest)
    finally:
        for p in (tmp, part):
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
    own_path = os.path.join(DATA, "own_numbers.txt")
    if not os.path.exists(own_path):
        raise SystemExit("data/own_numbers.txt is required (your own phone numbers, one per line)")
    rows = fetch_rows(load_env(a.calldesk_env), a.limit)
    picked = select_own_calls(rows, read_own_numbers(own_path), limit=a.max_calls)
    print(f"downloading {len(picked)} own calls ({len(rows) - len(picked)} excluded)")
    tw = load_env(a.twilio_env)
    raw_dir = os.path.join(DATA, "real_raw")
    os.makedirs(raw_dir, exist_ok=True)
    for r in picked:
        download_call(r, tw["TWILIO_ACCOUNT_SID"], tw["TWILIO_AUTH_TOKEN"], raw_dir)
    with open(os.path.join(raw_dir, "calls.json"), "w") as f:
        json.dump(picked, f, indent=1)
    print(f"done: {raw_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "fetch"])
    ap.add_argument("--calldesk-env", default="~/Documents/GitHub/calldesktech/.env")
    ap.add_argument("--twilio-env", default="~/Documents/GitHub/realtime-tts/call-loop-poc/.env")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--max-calls", type=int, default=None)
    a = ap.parse_args()
    {"screen": cmd_screen, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
