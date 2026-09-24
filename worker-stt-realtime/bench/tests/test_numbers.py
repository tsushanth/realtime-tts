from real_calls.numbers import (
    counterparty, mask_number, normalize_number, parse_env, recording_wav_url, select_own_calls,
)


def test_parse_env_handles_export_quotes_and_comments():
    text = 'A=1\n# B=2\nexport C="x y"\nD=\'z\'\n  E = 5 \n'
    assert parse_env(text) == {"A": "1", "C": "x y", "D": "z", "E": "5"}


def test_normalize_number():
    assert normalize_number("(555) 123-4567") == "+15551234567"
    assert normalize_number("+44 20 7946 0958") == "+442079460958"
    assert normalize_number("") is None
    assert normalize_number(None) is None


def test_mask_number():
    assert mask_number("+1 (555) 123-4567") == "+1********67"
    assert mask_number(None) == "unknown"


def test_counterparty_is_direction_aware():
    inbound = {"direction": "inbound", "caller_phone": "(555) 123-4567", "to_number": "+18005550100"}
    outbound = {"direction": "outbound", "caller_phone": "+18005550100", "to_number": "(555) 123-4567"}
    assert counterparty(inbound) == "+15551234567"
    assert counterparty(outbound) == "+15551234567"


def _row(i, direction, caller, to, engine="poc", url="https://x/RE1", created="2026-09-20T10:00:00Z"):
    return {"id": i, "direction": direction, "caller_phone": caller, "to_number": to,
            "voice_engine": engine, "recording_url": url, "created_at": created}


def test_select_own_calls_filters_orders_and_limits():
    own = ["+15551234567"]
    rows = [
        _row("old", "inbound", "+15551234567", "+18005550100", created="2026-09-20T10:00:00Z"),
        _row("new", "outbound", "+18005550100", "+15551234567", created="2026-09-21T10:00:00Z"),
        _row("stranger", "inbound", "+15559999999", "+18005550100"),
        _row("retell", "inbound", "+15551234567", "+18005550100", engine="retell"),
        _row("norec", "inbound", "+15551234567", "+18005550100", url=None),
    ]
    assert [r["id"] for r in select_own_calls(rows, own)] == ["new", "old"]
    assert [r["id"] for r in select_own_calls(rows, own, limit=1)] == ["new"]


def test_recording_wav_url():
    base = "https://api.twilio.com/2010-04-01/Accounts/AC1/Recordings/RE1"
    assert recording_wav_url(base) == base + ".wav"
    assert recording_wav_url(base + ".mp3") == base + ".wav"
    assert recording_wav_url(base + ".json?x=1") == base + ".wav"
