import io
import json
import os
import sys
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engines_cloud as ec  # noqa: E402
import public_sets  # noqa: E402
import rtbench  # noqa: E402


def _noise(sr, dur, seed):
    return (np.random.default_rng(seed).normal(0, 0.1, int(sr * dur))).astype("float32")


def _fake_fleurs(tmp_path):
    durs = [1.0, 4.0, 5.0, 6.0, 20.0, 7.0]   # rows 0 and 4 are outside 3..15 s
    tar_path = tmp_path / "test.tar.gz"
    rows = []
    with tarfile.open(tar_path, "w:gz") as tf:
        for i, d in enumerate(durs):
            fn = f"f{i}.wav"
            buf = io.BytesIO()
            sf.write(buf, _noise(16000, d, i), 16000, format="WAV", subtype="PCM_16")
            data = buf.getvalue()
            ti = tarfile.TarInfo("test/" + fn)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
            rows.append("\t".join([str(i), fn, f"Raw {i}.", f"text number {i}", "c", str(int(d * 16000)), "MALE"]))
    tsv = tmp_path / "test.tsv"
    tsv.write_text("\n".join(rows) + "\n")
    return str(tar_path), str(tsv)


def _hf_energy(x):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / 16000)
    return spec[freqs > 4000].sum() / spec.sum()


def test_build_fleurs(tmp_path):
    tar, tsv = _fake_fleurs(tmp_path)
    out = tmp_path / "out"
    info = public_sets.build_fleurs(str(out), n=10, seed=0, tar_path=tar, tsv_path=tsv)
    assert set(info) == {"pub_fleurs", "pub_fleurs_tel"} and info["pub_fleurs"]["n"] == 4
    clean = json.load(open(out / "pub_fleurs" / "manifest.json"))
    tel = json.load(open(out / "pub_fleurs_tel" / "manifest.json"))
    assert len(clean) == 4   # only rows 1,2,3,5 are inside the window
    assert [(m["id"], m["ref"]) for m in clean] == [(m["id"], m["ref"]) for m in tel]
    assert {m["ref"] for m in clean} == {"text number 1", "text number 2", "text number 3", "text number 5"}
    for m in clean:
        for s in ("pub_fleurs", "pub_fleurs_tel"):
            p = str(out / s / (m["id"] + ".wav"))
            i = sf.info(p)
            assert (i.samplerate, i.channels, i.subtype) == (16000, 1, "PCM_16")
        xc, _ = sf.read(str(out / "pub_fleurs" / (m["id"] + ".wav")))
        xt, _ = sf.read(str(out / "pub_fleurs_tel" / (m["id"] + ".wav")))
        assert _hf_energy(xt) < _hf_energy(xc)


def test_build_fleurs_deterministic_and_n(tmp_path):
    tar, tsv = _fake_fleurs(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    public_sets.build_fleurs(str(a), n=2, seed=3, tar_path=tar, tsv_path=tsv)
    public_sets.build_fleurs(str(b), n=2, seed=3, tar_path=tar, tsv_path=tsv)
    ma = json.load(open(a / "pub_fleurs" / "manifest.json"))
    assert len(ma) == 2 and ma == json.load(open(b / "pub_fleurs" / "manifest.json"))
    for m in ma:
        assert (a / "pub_fleurs_tel" / (m["id"] + ".wav")).read_bytes() == (b / "pub_fleurs_tel" / (m["id"] + ".wav")).read_bytes()


def _enc(x, sr, fmt):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format=fmt)
    return buf.getvalue()


def _fake_parquet(tmp_path):
    rows = [
        (_enc(_noise(16000, 5.0, 1), 16000, "WAV"), "good one"),
        (_enc(_noise(16000, 1.0, 2), 16000, "WAV"), "too short"),
        (_enc(_noise(16000, 5.0, 3), 16000, "WAV"), "   "),
        (_enc(_noise(24000, 6.0, 4), 24000, "FLAC"), "resampled row"),
        (_enc(_noise(16000, 4.0, 5), 16000, "WAV"), "extra"),
    ]
    tbl = pa.table({
        "audio": pa.array([{"bytes": b, "path": None} for b, _ in rows]),
        "transcription": [t for _, t in rows],
    })
    p = tmp_path / "e.parquet"
    pq.write_table(tbl, p, row_group_size=2)
    return str(p)


def test_build_earnings(tmp_path):
    src = _fake_parquet(tmp_path)
    out = tmp_path / "out"
    info = public_sets.build_earnings(str(out), n=10, parquet_source=lambda: [src])
    man = json.load(open(out / "pub_earnings" / "manifest.json"))
    assert [m["ref"] for m in man] == ["good one", "resampled row", "extra"]
    assert info["set"] == "pub_earnings" and info["n"] == 3 and info["hours"] > 0
    for m in man:
        i = sf.info(str(out / "pub_earnings" / (m["id"] + ".wav")))
        assert (i.samplerate, i.channels, i.subtype) == (16000, 1, "PCM_16")
    assert abs(sf.info(str(out / "pub_earnings" / (man[1]["id"] + ".wav"))).duration - 6.0) < 0.01


def test_build_earnings_stops_early(tmp_path):
    src = _fake_parquet(tmp_path)
    out = tmp_path / "out"
    public_sets.build_earnings(str(out), n=1, parquet_source=lambda: [src])
    assert len(json.load(open(out / "pub_earnings" / "manifest.json"))) == 1


def test_is_public_set():
    assert ec.is_public_set("pub_x") and ec.is_public_set("clean") and ec.is_public_set("tel")
    assert not ec.is_public_set("real") and not ec.is_public_set("weird") and not ec.is_public_set("xpub_")


def test_assert_allowed_pub(tmp_path):
    c = tmp_path / "c.txt"
    c.write_text("a\n")
    ec.assert_allowed([{"call_id": None}], str(c), "pub_fleurs_tel")
    with pytest.raises(PermissionError):
        ec.assert_allowed([{"call_id": None}], str(c), "real")
    with pytest.raises(PermissionError):
        ec.assert_allowed([{"call_id": None}], str(c), "weird")


def test_load_pub_set_no_noise(tmp_path, monkeypatch):
    d = tmp_path / "pub_fleurs_tel"
    d.mkdir()
    sf.write(str(d / "x.wav"), np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
    (d / "manifest.json").write_text(json.dumps([{"id": "x", "ref": "r"}]))
    monkeypatch.setattr(rtbench, "DATA", str(tmp_path))
    c = rtbench.load("pub_fleurs_tel", 1)[0]
    lead = int(rtbench.LEAD_S * 16000)
    assert np.all(c["audio"][lead:lead + 16000] == 0)


def test_real_out_guard_unchanged():
    with pytest.raises(SystemExit):
        rtbench.check_out_name("real", "results/x.json")
    rtbench.check_out_name("real", "results/real__x.json")
    rtbench.check_out_name("pub_fleurs", "results/x.json")
