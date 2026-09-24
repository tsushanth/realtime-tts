import os
import subprocess

import pytest

from real_calls import fetch


def test_read_own_numbers(tmp_path):
    p = tmp_path / "own.txt"
    p.write_text("# mine\n+15551234567\n\n  (555) 987-6543  \n# x\n+442079460958\n")
    assert fetch.read_own_numbers(str(p)) == ["+15551234567", "(555) 987-6543", "+442079460958"]


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        yield b"data"


ROW = {"id": "c1", "recording_url": "https://x/RE1"}


def test_download_call_ffmpeg_failure_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp())

    def boom(cmd, check):
        open(cmd[-1], "wb").write(b"trunc")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(fetch.subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        fetch.download_call(ROW, "s", "t", str(tmp_path))
    assert os.listdir(tmp_path) == []


def test_download_call_success_leaves_only_dest(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(fetch.subprocess, "run",
                        lambda cmd, check: open(cmd[-1], "wb").write(b"wav"))
    dest = fetch.download_call(ROW, "s", "t", str(tmp_path))
    assert os.listdir(tmp_path) == ["c1.wav"]
    assert open(dest, "rb").read() == b"wav"
