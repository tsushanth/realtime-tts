import json
import sys

import numpy as np
import pytest
import soundfile as sf

import rtbench


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["rtbench.py", *args])


@pytest.mark.parametrize("out", ["results/x.json", "results/out.json", "/tmp/whatever.json"])
def test_real_set_refuses_non_real_out_before_loading(monkeypatch, out):
    _argv(monkeypatch, "--engine", "does-not-exist", "--set", "real", "--out", out)
    monkeypatch.setattr(rtbench, "load", lambda *a, **k: pytest.fail("loaded data before validating --out"))
    with pytest.raises(SystemExit) as e:
        rtbench.main()
    assert e.value.code == 2


def test_real_set_allows_real_prefixed_out():
    rtbench.check_out_name("real", "results/real__x__real.json")


def test_other_sets_unaffected():
    rtbench.check_out_name("clean", "results/x.json")


def test_summary_records_only_verified(tmp_path, monkeypatch):
    import engines
    d = tmp_path / "real"
    d.mkdir()
    sf.write(str(d / "a_000.wav"), np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
    (d / "manifest.json").write_text(json.dumps([{"id": "a_000", "call_id": "a", "ref": "hello", "verified": True, "keyterms": []}]))
    monkeypatch.setattr(rtbench, "DATA", str(tmp_path))

    class St:
        def final(self):
            return "hello"

    monkeypatch.setattr(engines, "is_third_party", lambda n: False)
    monkeypatch.setattr(engines, "build", lambda *a, **k: object())
    monkeypatch.setattr(rtbench, "SilenceVAD", lambda: object())
    monkeypatch.setattr(rtbench, "_close", lambda st: None)
    monkeypatch.setattr(rtbench, "run_clip", lambda eng, vad, c, strategies, nm: (St(), {s: [] for s in strategies}, 0.1, 0.01, 1.0, 0.0))
    out = tmp_path / "res" / "real__f__real.json"
    _argv(monkeypatch, "--engine", "f", "--set", "real", "--only-verified", "--out", str(out))
    rtbench.main()
    assert json.load(open(out))["summary"]["only_verified"] is True
