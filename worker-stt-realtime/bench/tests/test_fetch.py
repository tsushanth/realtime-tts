import os
import subprocess

import pytest

from real_calls import fetch


def test_read_own_numbers(tmp_path):
    p = tmp_path / "own.txt"
    p.write_text("# mine\n+15551234567\n\n  (555) 987-6543  \n# x\n+442079460958\n")
    assert fetch.read_own_numbers(str(p)) == ["+15551234567", "(555) 987-6543", "+442079460958"]


class _Resp:
    def __init__(self, status_error=None):
        self.status_error = status_error

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def iter_content(self, n):
        yield b"data"


ROW = {"id": "c1", "recording_url": "https://x/RE1"}
BASE, SECRET = "https://poc.example", "s3cret"


def _fake_run(channels, calls):
    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{channels}\n", stderr="")
        open(cmd[-1], "wb").write(b"wav")
        return subprocess.CompletedProcess(cmd, 0)
    return run


def test_download_call_uses_proxy(tmp_path, monkeypatch):
    seen = {}

    def get(url, **k):
        seen["url"], seen["k"] = url, k
        return _Resp()

    monkeypatch.setattr(fetch.requests, "get", get)
    monkeypatch.setattr(fetch.subprocess, "run", _fake_run(2, []))
    fetch.download_call(ROW, BASE + "/", SECRET, str(tmp_path))
    assert seen["url"] == "https://poc.example/recording-audio"
    assert seen["k"]["params"] == {"url": fetch.recording_wav_url(ROW["recording_url"])}
    assert seen["k"]["headers"] == {"Authorization": "Bearer s3cret"}
    assert seen["k"]["stream"] is True


@pytest.mark.parametrize("ch", [0, 1])
def test_download_call_two_channel_pans_far_end(tmp_path, monkeypatch, ch):
    calls = []
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(fetch.subprocess, "run", _fake_run(2, calls))
    fetch.download_call(ROW, BASE, SECRET, str(tmp_path), caller_channel=ch)
    ff = [c for c in calls if c[0] == "ffmpeg"][0]
    assert ff[ff.index("-af") + 1] == f"pan=mono|c0=c{ch}"
    assert ff[ff.index("-ar") + 1] == "16000"
    assert ff[ff.index("-c:a") + 1] == "pcm_s16le"


def test_download_call_mono_uses_anull(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(fetch.subprocess, "run", _fake_run(1, calls))
    dest = fetch.download_call(ROW, BASE, SECRET, str(tmp_path))
    ff = [c for c in calls if c[0] == "ffmpeg"][0]
    assert ff[ff.index("-af") + 1] == "anull"
    assert os.listdir(tmp_path) == ["c1.wav"] and os.path.exists(dest)


def test_download_call_ffmpeg_failure_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp())

    def run(cmd, **kw):
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, 0, stdout="2\n", stderr="")
        open(cmd[-1], "wb").write(b"trunc")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(fetch.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        fetch.download_call(ROW, BASE, SECRET, str(tmp_path))
    assert os.listdir(tmp_path) == []


def test_download_call_success_leaves_only_dest(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(fetch.subprocess, "run", _fake_run(2, []))
    dest = fetch.download_call(ROW, BASE, SECRET, str(tmp_path))
    assert os.listdir(tmp_path) == ["c1.wav"]
    assert open(dest, "rb").read() == b"wav"


def test_download_call_http_500_does_not_leak_secret(tmp_path, monkeypatch, capsys):
    err = fetch.requests.HTTPError("500 Server Error for url: https://poc.example/recording-audio?url=x")
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp(err))
    monkeypatch.setattr(fetch.subprocess, "run", _fake_run(2, []))
    with pytest.raises(fetch.requests.HTTPError) as ei:
        fetch.download_call(ROW, BASE, SECRET, str(tmp_path))
    out = capsys.readouterr()
    assert SECRET not in str(ei.value) and SECRET not in out.out + out.err
    assert os.listdir(tmp_path) == []


def _wire(tmp_path, monkeypatch, own_lines, env=None):
    monkeypatch.setattr(fetch, "DATA", str(tmp_path))
    (tmp_path / "own_numbers.txt").write_text(own_lines)
    rows = [
        {"id": "a", "voice_engine": "poc", "direction": "inbound", "caller_phone": "+15550001111",
         "to_number": "+15559990000", "recording_url": "https://x/a", "created_at": "2026-01-02T00:00:00Z", "duration_seconds": 30},
        {"id": "b", "voice_engine": "poc", "direction": "inbound", "caller_phone": "+15552223333",
         "to_number": "+15559990000", "recording_url": "https://x/b", "created_at": "2026-01-01T00:00:00Z", "duration_seconds": 30},
        {"id": "c", "voice_engine": "poc", "direction": "inbound", "caller_phone": "+15552223333",
         "to_number": "+15559990000", "recording_url": None, "created_at": "2026-01-01T00:00:00Z", "duration_seconds": 30},
    ]
    monkeypatch.setattr(fetch, "fetch_rows", lambda env, limit: rows)
    monkeypatch.setattr(fetch, "load_env", lambda p: env if env is not None else {
        "CALL_LOOP_POC_BASE_URL": BASE, "CALL_LOOP_POC_TEST_CALL_SECRET": SECRET})
    got = []
    monkeypatch.setattr(fetch, "download_call",
                        lambda r, base, sec, d, caller_channel=0: got.append((r["id"], base, sec, caller_channel)))
    return got


def _args(**kw):
    import argparse
    d = dict(calldesk_env="e", limit=10, max_calls=None, caller_channel=0, include_unmatched=False)
    d.update(kw)
    return argparse.Namespace(**d)


def test_cmd_fetch_matching_only(tmp_path, monkeypatch):
    got = _wire(tmp_path, monkeypatch, "+15550001111\n")
    fetch.cmd_fetch(_args())
    assert [g[0] for g in got] == ["a"]


def test_cmd_fetch_include_unmatched(tmp_path, monkeypatch, capsys):
    got = _wire(tmp_path, monkeypatch, "+19998887777\n")
    fetch.cmd_fetch(_args(include_unmatched=True, caller_channel=1))
    assert sorted(g[0] for g in got) == ["a", "b"]
    assert all(g[1:] == (BASE, SECRET, 1) for g in got)
    assert SECRET not in capsys.readouterr().out
    got.clear()
    fetch.cmd_fetch(_args())
    assert got == []


def test_cmd_fetch_missing_env_var_no_leak(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, "+15550001111\n", env={"CALL_LOOP_POC_TEST_CALL_SECRET": SECRET})
    with pytest.raises(SystemExit) as ei:
        fetch.cmd_fetch(_args())
    assert "CALL_LOOP_POC_BASE_URL" in str(ei.value) and SECRET not in str(ei.value)
