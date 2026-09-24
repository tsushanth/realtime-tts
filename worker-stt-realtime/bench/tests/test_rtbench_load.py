import json

import numpy as np
import soundfile as sf

import rtbench


def _make_set(tmp_path):
    d = tmp_path / "real"
    d.mkdir()
    man = [
        {"id": "a_000", "call_id": "a", "ref": "hello", "verified": True, "keyterms": ["hello"]},
        {"id": "b_000", "call_id": "b", "ref": "world", "verified": False, "keyterms": []},
    ]
    for m in man:
        sf.write(str(d / (m["id"] + ".wav")), np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
    (d / "manifest.json").write_text(json.dumps(man))


def test_load_real_set_carries_metadata_and_filters_verified(tmp_path, monkeypatch):
    _make_set(tmp_path)
    monkeypatch.setattr(rtbench, "DATA", str(tmp_path))
    clips = rtbench.load("real", 10)
    assert [c["call_id"] for c in clips] == ["a", "b"]
    assert clips[0]["keyterms"] == ["hello"]
    only = rtbench.load("real", 10, verified_only=True)
    assert [c["id"] for c in only] == ["a_000"]


def _mid(clip):
    lead = int(rtbench.LEAD_S * 16000)
    return clip["audio"][lead:lead + 16000]


def test_real_set_injects_no_noise_but_tel_does(tmp_path, monkeypatch):
    _make_set(tmp_path)
    tel = tmp_path / "tel"
    tel.mkdir()
    sf.write(str(tel / "a_000.wav"), np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
    (tel / "manifest.json").write_text(json.dumps([{"id": "a_000", "ref": "hello"}]))
    monkeypatch.setattr(rtbench, "DATA", str(tmp_path))
    assert not _mid(rtbench.load("real", 1)[0]).any()
    assert _mid(rtbench.load("tel", 1)[0]).any()


def test_require_clips_exits_with_clear_message():
    import pytest
    with pytest.raises(SystemExit) as e:
        rtbench._require_clips([], "real", True)
    assert "no clips to run" in str(e.value) and "'real'" in str(e.value)
    rtbench._require_clips([{"id": "x"}], "real", True)
