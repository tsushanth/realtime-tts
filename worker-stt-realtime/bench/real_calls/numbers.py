import re


def parse_env(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", line)
        if not m:
            continue
        v = m.group(2)
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[m.group(1)] = v
    return out


def normalize_number(n):
    if not n:
        return None
    digits = re.sub(r"\D", "", n)
    if not digits:
        return None
    if len(digits) == 10:
        digits = "1" + digits
    return "+" + digits


def mask_number(n):
    n = normalize_number(n)
    if not n:
        return "unknown"
    return n[:2] + "*" * (len(n) - 4) + n[-2:]


def counterparty(row):
    raw = row.get("caller_phone") if row.get("direction") == "inbound" else row.get("to_number")
    return normalize_number(raw)


def select_own_calls(rows, own_numbers, engine="poc", limit=None):
    own = {x for x in map(normalize_number, own_numbers) if x}
    picked = [r for r in rows
              if r.get("voice_engine") == engine and r.get("recording_url") and counterparty(r) in own]
    picked.sort(key=lambda r: r["created_at"], reverse=True)
    return picked[:limit] if limit else picked


def recording_wav_url(url):
    return re.sub(r"\.(mp3|wav|json)$", "", url.split("?")[0]) + ".wav"
