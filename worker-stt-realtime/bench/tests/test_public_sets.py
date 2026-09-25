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
    info = public_sets.build_fleurs(str(out), n=4, seed=0, tar_path=tar, tsv_path=tsv)
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
    pq.write_table(tbl, p, row_group_size=5)
    return str(p)


def _multi_shards(tmp_path, n_shards=4, groups=3, per=6):
    paths = []
    for s in range(n_shards):
        rows, fids = [], []
        for g in range(groups):
            for r in range(per):
                rows.append((_enc(_noise(16000, 4.0, s * 100 + g * 10 + r), 16000, "WAV"), f"s{s} g{g} r{r}"))
                fids.append(f"call_{s}_{g}")
        tbl = pa.table({"audio": pa.array([{"bytes": b, "path": None} for b, _ in rows]),
                        "transcription": [t for _, t in rows], "file_id": fids})
        p = tmp_path / f"shard{s}.parquet"
        pq.write_table(tbl, p, row_group_size=per)
        paths.append(str(p))
    return paths


def test_build_earnings_filters_and_resamples(tmp_path):
    src = _fake_parquet(tmp_path)
    out = tmp_path / "out"
    info = public_sets.build_earnings(str(out), n=3, parquet_source=lambda: [src])
    man = json.load(open(out / "pub_earnings" / "manifest.json"))
    assert sorted(m["ref"] for m in man) == ["extra", "good one", "resampled row"]   # short + blank rows dropped
    assert info["set"] == "pub_earnings" and info["n"] == len(man) and info["hours"] > 0
    for m in man:
        i = sf.info(str(out / "pub_earnings" / (m["id"] + ".wav")))
        assert (i.samplerate, i.channels, i.subtype) == (16000, 1, "PCM_16")
        assert m["source"] == "public"
    r = [m for m in man if m["ref"] == "resampled row"]
    assert r
    if True:
        assert abs(sf.info(str(out / "pub_earnings" / (r[0]["id"] + ".wav"))).duration - 6.0) < 0.01


def test_build_earnings_spans_calls_and_is_seeded(tmp_path):
    shards = _multi_shards(tmp_path)
    a, a2, b = tmp_path / "a", tmp_path / "a2", tmp_path / "b"
    public_sets.build_earnings(str(a), n=8, seed=1, parquet_source=lambda: list(shards))
    public_sets.build_earnings(str(a2), n=8, seed=1, parquet_source=lambda: list(shards))
    public_sets.build_earnings(str(b), n=8, seed=2, parquet_source=lambda: list(shards))
    ma = json.load(open(a / "pub_earnings" / "manifest.json"))
    assert len(ma) == 8 and len({m["file_id"] for m in ma}) >= 3
    strip = lambda m: [(x["id"], x["ref"]) for x in m]   # noqa: E731
    assert strip(ma) == strip(json.load(open(a2 / "pub_earnings" / "manifest.json")))
    assert strip(ma) != strip(json.load(open(b / "pub_earnings" / "manifest.json")))


def test_build_earnings_short_raises(tmp_path):
    src = _fake_parquet(tmp_path)
    with pytest.raises(ValueError, match="requested 10.*available"):
        public_sets.build_earnings(str(tmp_path / "o"), n=10, parquet_source=lambda: [src])
    assert not (tmp_path / "o" / "pub_earnings").exists()


def test_build_fleurs_short_raises(tmp_path):
    tar, tsv = _fake_fleurs(tmp_path)
    with pytest.raises(ValueError, match="requested 10.*only 4 available"):
        public_sets.build_fleurs(str(tmp_path / "o"), n=10, tar_path=tar, tsv_path=tsv, allow_short=False)
    assert not (tmp_path / "o" / "pub_fleurs").exists()


def test_fleurs_dedupes_transcriptions(tmp_path):
    tar, tsv = _fake_fleurs(tmp_path)
    lines = open(tsv).read().splitlines()
    # make rows 1,2,3 share one transcription -> only 2 unique eligible texts remain (rows 1 and 5)
    for k in (1, 2, 3):
        c = lines[k].split("\t")
        c[3] = "same text"
        lines[k] = "\t".join(c)
    open(tsv, "w").write("\n".join(lines) + "\n")
    public_sets.build_fleurs(str(tmp_path / "o"), n=10, tar_path=tar, tsv_path=tsv, allow_short=True)
    man = json.load(open(tmp_path / "o" / "pub_fleurs" / "manifest.json"))
    assert sorted(m["ref"] for m in man) == ["same text", "text number 5"]


def test_fleurs_atomic_when_tel_fails(tmp_path, monkeypatch):
    tar, tsv = _fake_fleurs(tmp_path)
    out = tmp_path / "o"
    calls = {"n": 0}
    real = public_sets.to_telephone

    def flaky(x, seed=0, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return real(x, seed=seed, **k)
    monkeypatch.setattr(public_sets, "to_telephone", flaky)
    with pytest.raises(RuntimeError):
        public_sets.build_fleurs(str(out), n=4, tar_path=tar, tsv_path=tsv)
    assert os.listdir(out) == [] if out.exists() else True


def test_rebuild_removes_stale_files(tmp_path):
    tar, tsv = _fake_fleurs(tmp_path)
    out = tmp_path / "o"
    public_sets.build_fleurs(str(out), n=4, seed=0, tar_path=tar, tsv_path=tsv)
    stale = out / "pub_fleurs" / "stale.wav"
    stale.write_bytes(b"x")
    public_sets.build_fleurs(str(out), n=2, seed=0, tar_path=tar, tsv_path=tsv)
    assert not stale.exists()
    assert len(json.load(open(out / "pub_fleurs" / "manifest.json"))) == 2
    assert sorted(os.listdir(out)) == ["pub_fleurs", "pub_fleurs_tel"]


def test_is_public_set():
    for ok in ("pub_x", "clean", "tel", "call", "pub_fleurs_tel"):
        assert ec.is_public_set(ok)
    for bad in ("real", "weird", "xpub_", "pub_", "pub_real", "pub_real_x", "pub_Real"):
        assert not ec.is_public_set(bad)


GOOD = {"call_id": None, "source": "public", "ref": "SECRET WORDS"}


def test_assert_allowed_pub_two_key(tmp_path):
    c = tmp_path / "c.txt"
    c.write_text("a\n")
    ec.assert_allowed([dict(GOOD)], str(c), "pub_fleurs_tel")
    for clip in (dict(GOOD, call_id="a"), {"call_id": None}, dict(GOOD, source="real"), dict(GOOD, source=None)):
        with pytest.raises(PermissionError, match="pub_") as ei:
            ec.assert_allowed([clip], str(c), "pub_fleurs_tel")
        assert "SECRET" not in str(ei.value)
    for name in ("pub_", "pub_real", "pub_real_x", "real", "weird"):
        with pytest.raises(PermissionError):
            ec.assert_allowed([dict(GOOD)], str(c), name)
    ec.assert_allowed([{"call_id": None}], str(c), "tel")   # old behaviour intact


def test_main_refuses_pub_set_without_source(tmp_path, monkeypatch):
    import engines
    d = tmp_path / "pub_x"
    d.mkdir()
    sf.write(str(d / "a.wav"), np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
    (d / "manifest.json").write_text(json.dumps([{"id": "a", "ref": "hi"}]))
    monkeypatch.setattr(rtbench, "DATA", str(tmp_path))
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    def boom(*a, **k):
        raise AssertionError("engines.build called before the gate")
    monkeypatch.setattr(engines, "build", boom)
    monkeypatch.setattr("sys.argv", ["rtbench", "--engine", "dg-flux", "--set", "pub_x", "--out", str(tmp_path / "o.json")])
    with pytest.raises(PermissionError):
        rtbench.main()


def test_load_passes_source_through(tmp_path, monkeypatch):
    d = tmp_path / "pub_x"
    d.mkdir()
    sf.write(str(d / "a.wav"), np.zeros(1600, dtype="float32"), 16000, subtype="PCM_16")
    (d / "manifest.json").write_text(json.dumps([{"id": "a", "ref": "hi", "source": "public"}]))
    monkeypatch.setattr(rtbench, "DATA", str(tmp_path))
    assert rtbench.load("pub_x", 1)[0]["source"] == "public"


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


def test_normalize_peak():
    x = (np.random.default_rng(0).normal(0, 0.003, 8000)).astype("float64")
    y = public_sets.normalize_peak(x, peak=0.7)
    assert y.dtype == np.float32 and abs(float(np.max(np.abs(y))) - 0.7) < 1e-6
    z = np.zeros(100, dtype="float32")
    assert np.array_equal(public_sets.normalize_peak(z), z)
    with pytest.raises(ValueError):
        public_sets.normalize_peak(np.array([0.1, np.nan], dtype="float32"))
    with pytest.raises(ValueError):
        public_sets.normalize_peak(np.array([0.1, np.inf], dtype="float32"))


def test_derive_normalized_set(tmp_path):
    tar, tsv = _fake_fleurs(tmp_path)
    out = tmp_path / "o"
    public_sets.build_fleurs(str(out), n=4, tar_path=tar, tsv_path=tsv)
    before = {f: (out / "pub_fleurs" / f).read_bytes() for f in os.listdir(out / "pub_fleurs")}
    info = public_sets.derive_normalized_set(str(out))
    assert info["set"] == "pub_fleurs_norm" and info["n"] == 4
    src = json.load(open(out / "pub_fleurs" / "manifest.json"))
    dst = json.load(open(out / "pub_fleurs_norm" / "manifest.json"))
    assert src == dst and all(m["source"] == "public" for m in dst)
    for m in dst:
        x, sr = sf.read(str(out / "pub_fleurs_norm" / (m["id"] + ".wav")), dtype="float32")
        assert sr == 16000 and abs(float(np.max(np.abs(x))) - 0.7) < 1e-3
    assert before == {f: (out / "pub_fleurs" / f).read_bytes() for f in os.listdir(out / "pub_fleurs")}


def test_derive_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="pub_fleurs"):
        public_sets.derive_normalized_set(str(tmp_path))
    tar, tsv = _fake_fleurs(tmp_path)
    public_sets.build_fleurs(str(tmp_path / "o"), n=4, tar_path=tar, tsv_path=tsv)
    with pytest.raises(ValueError, match="public"):
        public_sets.derive_normalized_set(str(tmp_path / "o"), dst_set="real_x")
    assert not (tmp_path / "o" / "real_x").exists()


def test_build_fleurs_reports_peak_median(tmp_path):
    tar, tsv = _fake_fleurs(tmp_path)
    info = public_sets.build_fleurs(str(tmp_path / "o"), n=4, tar_path=tar, tsv_path=tsv)
    for k in ("pub_fleurs", "pub_fleurs_tel"):
        assert 0 < info[k]["peak_median"] <= 1.0
