import json

import numpy as np
import pytest
import soundfile as sf

from real_calls import segment
from real_calls.segment import FRAME_S, build_manifest, split_speech_ranges

F = 0.1


def test_two_blocks_separated_by_long_silence_stay_separate():
    flags = [False] * 5 + [True] * 20 + [False] * 10 + [True] * 20 + [False] * 5
    got = split_speech_ranges(flags, F)
    assert got == [pytest.approx((0.25, 2.75)), pytest.approx((3.25, 5.75))]


def test_short_gap_is_merged_and_padding_is_clipped():
    flags = [True] * 20 + [False] * 3 + [True] * 20
    assert split_speech_ranges(flags, F) == [pytest.approx((0.0, 4.3))]


def test_blip_shorter_than_min_segment_is_dropped():
    assert split_speech_ranges([False] * 10 + [True] * 5 + [False] * 10, F) == []


def test_long_continuous_speech_is_hard_split():
    got = split_speech_ranges([True] * 400, F)
    assert len(got) == 3
    assert all(e - s <= 15.0 + 1e-9 for s, e in got)


def test_merged_group_over_max_is_split_at_run_boundary():
    flags = [True] * 100 + [False] * 3 + [True] * 100
    assert len(split_speech_ranges(flags, F)) == 2


def test_all_silence_and_empty_give_no_segments():
    assert split_speech_ranges([False] * 50, F) == []
    assert split_speech_ranges([], F) == []


def test_gap_exactly_min_silence_stays_separate():
    flags = [True] * 20 + [False] * 6 + [True] * 20
    got = split_speech_ranges(flags, F, min_silence_s=0.6)
    assert len(got) == 2


def _setup_build(tmp_path, monkeypatch, drafts):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    sf.write(str(raw / "callA.wav"), np.zeros(16000 * 15, dtype="float32"), 16000, subtype="PCM_16")
    n = int(15 * 16000 / 512)
    flags = [False] * n
    for a in (10, 130, 250):
        for i in range(a, a + 60):
            flags[i] = True
    monkeypatch.setattr(segment, "speech_flags", lambda x: flags)
    monkeypatch.setattr(segment, "draft_refs", lambda paths, model_size="x": iter(drafts))
    return raw, out


def test_build_manifest_aligns_drafts_and_drops_empty_only(tmp_path, monkeypatch):
    raw, out = _setup_build(tmp_path, monkeypatch, ["first", "", "third"])
    kept = build_manifest(str(raw), str(out))
    assert [(e["id"], e["ref"]) for e in kept] == [("callA_000", "first"), ("callA_002", "third")]
    assert sorted(p.name for p in out.glob("*.wav")) == ["callA_000.wav", "callA_002.wav"]
    assert [e["id"] for e in json.load(open(out / "manifest.json"))] == ["callA_000", "callA_002"]


def test_build_manifest_refuses_to_clobber_verified(tmp_path, monkeypatch):
    raw, out = _setup_build(tmp_path, monkeypatch, ["a", "b", "c"])
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps([{"id": "x", "verified": True}, {"id": "y", "verified": False}]))
    with pytest.raises(RuntimeError, match="1"):
        build_manifest(str(raw), str(out))
    assert json.load(open(out / "manifest.json"))[0]["id"] == "x"
    kept = build_manifest(str(raw), str(out), force=True)
    assert len(kept) == 3
