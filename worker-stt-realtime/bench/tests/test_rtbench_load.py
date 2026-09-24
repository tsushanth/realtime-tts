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
