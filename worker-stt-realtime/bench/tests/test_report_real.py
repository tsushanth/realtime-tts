import json
import math
import subprocess
import sys
from pathlib import Path

from report_real import gate, load_results, price_per_hour, render_table


def _s(engine, wer, kr, commit_lat, cpu=0.1, native=0.6, n=40, kt=30, ff=1.0, ov=True):
    return {"engine": engine, "set": "real", "only_verified": ov, "n": n, "wer": wer, "keyterm_recall": kr, "keyterm_total": kt,
            "first_partial_med_s": 0.5, "cpu_per_audio_s": cpu,
            "strat": {"commit": {"lat_med": commit_lat, "lat_p90": commit_lat * 2, "cut_utts": 0, "fired_frac": ff},
                      "native": {"lat_med": native, "lat_p90": native * 1.3, "cut_utts": 3}}}


def test_price_per_hour_cloud_is_fixed_and_local_uses_cpu():
    assert price_per_hour(_s("dg-flux", 0.1, 0.9, 0.1)) == 0.39
    assert abs(price_per_hour(_s("nemo-480", 0.1, 0.9, 0.05, cpu=0.1)) - 0.1 * 3600 * 0.0000131) < 1e-9


def test_gate_passes_when_local_is_close_enough():
    res = gate([_s("nemo-480", 0.12, 0.85, 0.06), _s("zip-en-int8", 0.20, 0.7, 0.05),
                _s("dg-flux", 0.09, 0.90, 0.2), _s("el-scribe", 0.10, 0.88, 0.2)])
    assert res["best_ours"]["engine"] == "nemo-480" and res["best_cloud"]["engine"] == "dg-flux"
    assert res["pass"] is True and all(v == "PASS" for v in res["checks"].values())


def test_gate_fails_on_wer_and_reports_which_check():
    res = gate([_s("nemo-480", 0.30, 0.85, 0.06), _s("dg-flux", 0.09, 0.90, 0.2)])
    assert res["pass"] is False and res["checks"]["wer_within_1.5x"] == "FAIL"


def test_gate_needs_both_kinds_of_engine():
    assert "error" in gate([_s("nemo-480", 0.1, 0.9, 0.05)])


def test_render_table_lists_every_engine_and_price():
    out = render_table([_s("nemo-480", 0.12, 0.85, 0.06), _s("dg-flux", 0.09, 0.9, 0.2)])
    assert "nemo-480" in out and "dg-flux" in out and "0.39" in out and "12.0%" in out


def test_render_table_without_native_strategy_or_nan_partial():
    s = _s("zip-en-int8", 0.2, None, 0.05)
    del s["strat"]["native"]
    s["first_partial_med_s"] = float("nan")
    out = render_table([s])
    assert "zip-en-int8" in out and "nan" not in out.lower()


def test_gate_zero_cloud_wer_does_not_crash_and_fails_local():
    res = gate([_s("nemo-480", 0.05, 0.9, 0.06), _s("dg-flux", 0.0, 0.9, 0.2)])
    assert res["pass"] is False and res["checks"]["wer_within_1.5x"] == "FAIL"


def test_gate_zero_wer_both_and_missing_keyterms_fail_keyterm_check():
    res = gate([_s("nemo-480", 0.0, None, 0.06), _s("dg-flux", 0.0, 0.9, 0.2)])
    assert res["checks"]["keyterm_within_10pts"] == "INSUFFICIENT" and res["pass"] is None
    res = gate([_s("nemo-480", 0.1, 0.9, 0.06), _s("dg-flux", 0.1, None, 0.2)])
    assert res["checks"]["keyterm_within_10pts"] == "INSUFFICIENT"


def test_gate_missing_or_nan_wer_or_cpu_fails_not_crashes():
    res = gate([_s("nemo-480", None, 0.9, 0.06), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert "error" in res or res["pass"] is not True
    res = gate([_s("nemo-480", 0.1, 0.9, 0.06, cpu=None), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert res["pass"] is None and res["checks"]["cpu_le_0.15"] == "INSUFFICIENT"
    res = gate([_s("nemo-480", float("nan"), 0.9, 0.06), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert "error" in res or res["pass"] is not True


def test_load_results_reads_summary_only(tmp_path):
    (tmp_path / "real__nemo-480__real.json").write_text(json.dumps(
        {"summary": _s("nemo-480", 0.1, 0.9, 0.05), "rows": [{"hyp": "SECRET TRANSCRIPT"}]}))
    (tmp_path / "clean__x__clean.json").write_text(json.dumps({"summary": _s("x", 0.1, 0.9, 0.05)}))
    out = load_results(str(tmp_path))
    assert [s["engine"] for s in out] == ["nemo-480"] and "SECRET" not in json.dumps(out)


def test_cli_no_results_gives_clear_message(tmp_path):
    out = tmp_path / "o.md"
    p = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "report_real.py"),
                        "--results", str(tmp_path / "nope"), "--out", str(out)], capture_output=True, text=True)
    assert p.returncode != 0 and "Traceback" not in p.stderr and "no results" in (p.stderr + p.stdout).lower()
    assert not out.exists()


# ---- fix round 1: per-metric baselines, tri-state, per-local evaluation ----
import pytest
import report_real


def _chk(local, cloud, name):
    return gate([local, cloud])["checks"][name]


def test_thresholds_are_module_constants():
    assert (report_real.MIN_UTTS, report_real.MIN_KEYTERMS, report_real.MIN_FIRED_FRAC) == (30, 30, 0.9)


def test_wer_boundary():
    cl = _s("dg-flux", 0.10, 0.9, 0.2)
    assert _chk(_s("nemo-480", 0.15, 0.9, 0.05), cl, "wer_within_1.5x") == "PASS"
    assert _chk(_s("nemo-480", 0.1501, 0.9, 0.05), cl, "wer_within_1.5x") == "FAIL"


def test_latency_cpu_keyterm_boundaries():
    cl = _s("dg-flux", 0.10, 0.90, 0.2)
    r = gate([_s("nemo-480", 0.10, 0.80, 0.150, cpu=0.15), cl])
    assert r["checks"]["commit_final_le_150ms"] == "PASS" and r["checks"]["cpu_le_0.15"] == "PASS"
    assert r["checks"]["keyterm_within_10pts"] == "PASS"
    r = gate([_s("nemo-480", 0.10, 0.79, 0.151, cpu=0.151), cl])
    assert r["checks"]["commit_final_le_150ms"] == "FAIL" and r["checks"]["cpu_le_0.15"] == "FAIL"
    assert r["checks"]["keyterm_within_10pts"] == "FAIL"


def test_keyterm_baseline_is_best_keyterm_cloud_not_best_wer_cloud():
    a = _s("dg-flux", 0.08, 0.70, 0.2)     # best WER
    b = _s("el-scribe", 0.12, 0.95, 0.2)   # best keyterm
    r = gate([_s("nemo-480", 0.10, 0.80, 0.05), a, b])
    assert r["best_cloud"]["engine"] == "dg-flux" and r["best_cloud_keyterm"]["engine"] == "el-scribe"
    assert r["checks"]["keyterm_within_10pts"] == "FAIL"     # 0.80 < 0.95 - 0.10
    assert r["checks"]["wer_within_1.5x"] == "PASS"


@pytest.mark.parametrize("kw,check", [
    ({"kt": 29}, "keyterm_within_10pts"), ({"n": 29}, "wer_within_1.5x"),
    ({"n": 35}, "wer_within_1.5x"), ({"ff": 0.89}, "commit_final_le_150ms")])
def test_insufficient_data(kw, check):
    r = gate([_s("nemo-480", 0.1, 0.9, 0.05, **kw), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert r["checks"][check] == "INSUFFICIENT" and r["pass"] is None


def test_insufficient_when_cloud_side_is_thin():
    r = gate([_s("nemo-480", 0.1, 0.9, 0.05), _s("dg-flux", 0.1, 0.9, 0.2, n=10, kt=5)])
    assert r["checks"]["wer_within_1.5x"] == "INSUFFICIENT" and r["checks"]["keyterm_within_10pts"] == "INSUFFICIENT"


def test_fired_frac_boundary_and_fail_beats_insufficient():
    cl = _s("dg-flux", 0.1, 0.9, 0.2)
    assert gate([_s("nemo-480", 0.1, 0.9, 0.05, ff=0.9), cl])["checks"]["commit_final_le_150ms"] == "PASS"
    r = gate([_s("nemo-480", 0.5, 0.9, 0.05, ff=0.5), cl])      # WER FAIL + latency INSUFFICIENT
    assert r["locals"][0]["verdict"] == "FAIL" and r["pass"] is False


def test_each_local_engine_evaluated_and_recommended_is_the_passer():
    cl = _s("dg-flux", 0.10, 0.9, 0.2)
    r = gate([_s("nemo-480", 0.08, 0.9, 0.5), _s("zip-en-int8", 0.12, 0.85, 0.05), cl])
    v = {x["engine"]: x["verdict"] for x in r["locals"]}
    assert v == {"nemo-480": "FAIL", "zip-en-int8": "PASS"}
    assert r["pass"] is True and r["recommended"] == "zip-en-int8" and r["best_ours"]["engine"] == "zip-en-int8"
    assert set(r["locals"][0]) >= {"engine", "checks", "verdict"}


def test_no_passer_all_fail_is_false_and_recommended_none():
    r = gate([_s("nemo-480", 0.5, 0.9, 0.05), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert r["pass"] is False and r["recommended"] is None


def test_advisory_when_cloud_wer_below_2pct():
    assert "degenerate" in gate([_s("nemo-480", 0.02, 0.9, 0.05), _s("dg-flux", 0.019, 0.9, 0.2)])["advisory"]
    assert not gate([_s("nemo-480", 0.02, 0.9, 0.05), _s("dg-flux", 0.02, 0.9, 0.2)]).get("advisory")


def test_unknown_cloud_prefixed_engine_raises_and_unknown_other_is_local():
    with pytest.raises(ValueError):
        gate([_s("nemo-480", 0.1, 0.9, 0.05), _s("dg-nova", 0.1, 0.9, 0.2)])
    with pytest.raises(ValueError):
        price_per_hour(_s("el-other", 0.1, 0.9, 0.2))


def test_tristate_and_footnotes_rendered_in_report(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    for s in [_s("nemo-480", 0.1, 0.9, 0.05, kt=10), _s("dg-flux", 0.1, 0.9, 0.2)]:
        (d / f"real__{s['engine']}__real.json").write_text(json.dumps({"summary": s}))
    out = tmp_path / "o.md"
    subprocess.run([sys.executable, str(Path(__file__).parent.parent / "report_real.py"),
                    "--results", str(d), "--out", str(out)], check=True, capture_output=True)
    t = out.read_text()
    assert "INSUFFICIENT" in t and "PASS" in t and "dg-flux" in t
    assert "30" in t and "0.9" in t and "compute-only, 100% packed" in t and "INSUFFICIENT DATA" in t


# ---- fix wave: unverified references, unknown-cloud CLI error ----
def _run_cli(tmp_path, summaries):
    d = tmp_path / "r"
    d.mkdir()
    for s in summaries:
        (d / f"real__{s['engine']}__real.json").write_text(json.dumps({"summary": s}))
    out = tmp_path / "o.md"
    p = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "report_real.py"),
                        "--results", str(d), "--out", str(out)], capture_output=True, text=True)
    return p, out


def test_unverified_local_engine_makes_wer_and_keyterm_insufficient():
    r = gate([_s("nemo-480", 0.1, 0.9, 0.05, ov=False), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert r["checks"]["wer_within_1.5x"] == "INSUFFICIENT" and r["checks"]["keyterm_within_10pts"] == "INSUFFICIENT"
    assert r["pass"] is None


def test_missing_only_verified_counts_as_unverified_and_cloud_side_too():
    s = _s("nemo-480", 0.1, 0.9, 0.05)
    del s["only_verified"]
    assert gate([s, _s("dg-flux", 0.1, 0.9, 0.2)])["pass"] is None
    assert gate([_s("nemo-480", 0.1, 0.9, 0.05), _s("dg-flux", 0.1, 0.9, 0.2, ov=False)])["pass"] is None


def test_other_sets_are_unaffected_by_verified_rule():
    s = _s("nemo-480", 0.1, 0.9, 0.05, ov=False)
    s["set"] = "clean"
    assert gate([s, _s("dg-flux", 0.1, 0.9, 0.2)])["pass"] is True


def test_report_prints_warning_for_unverified_engine(tmp_path):
    p, out = _run_cli(tmp_path, [_s("nemo-480", 0.1, 0.9, 0.05, ov=False), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert p.returncode == 0
    assert "WARNING: nemo-480 was scored on unverified draft references" in out.read_text()
    assert "WARNING: dg-flux" not in out.read_text()


def test_cli_unknown_cloud_engine_clear_error(tmp_path):
    p, out = _run_cli(tmp_path, [_s("nemo-480", 0.1, 0.9, 0.05), _s("dg-nova", 0.1, 0.9, 0.2)])
    assert p.returncode == 1 and "Traceback" not in p.stderr and "dg-nova" in p.stderr
    assert not out.exists()


# ---- task 3: public sets (--tag/--set/--title) ----
def _pub(engine, set_name, wer=0.1):
    s = _s(engine, wer, None, 0.05, kt=0)
    s.update({"set": set_name, "keyterm_total": 0})
    s.pop("only_verified")
    return s


def _pub_dir(tmp_path):
    d = tmp_path / "pr"
    d.mkdir()
    for st in ("pub_fleurs_tel", "pub_other"):
        for e in ("nemo-480", "dg-flux"):
            (d / f"pub__{e}__{st}.json").write_text(json.dumps({"summary": _pub(e, st)}))
    return d


def _run_pub(tmp_path, *extra):
    d = _pub_dir(tmp_path)
    out = tmp_path / "o.md"
    p = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "report_real.py"),
                        "--results", str(d), "--out", str(out), *extra], capture_output=True, text=True)
    return p, out


def test_load_results_set_filter(tmp_path):
    d = _pub_dir(tmp_path)
    assert len(load_results(str(d), tag="pub")) == 4
    got = load_results(str(d), tag="pub", set_name="pub_fleurs_tel")
    assert len(got) == 2 and {s["set"] for s in got} == {"pub_fleurs_tel"}


def test_cli_pub_set_report(tmp_path):
    p, out = _run_pub(tmp_path, "--tag", "pub", "--set", "pub_fleurs_tel", "--title", "FLEURS tel eval")
    assert p.returncode == 0, p.stderr
    t = out.read_text()
    assert t.startswith("# FLEURS tel eval")
    assert "unverified draft references" not in t
    assert "Keyterm check not applicable (public sets have no keyterms)" in t
    assert "ship-worthy" not in t and "not yet" not in t
    assert t.count("| nemo-480 |") == 1


def test_cli_no_match_for_set_or_tag(tmp_path):
    p, out = _run_pub(tmp_path, "--tag", "pub", "--set", "pub_missing")
    assert p.returncode != 0 and "Traceback" not in p.stderr and "no results" in p.stderr.lower()
    assert not out.exists()


def test_unverified_warning_still_applies_to_real_and_title_rendered(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    for x in [_s("nemo-480", 0.1, 0.9, 0.05, ov=False), _s("dg-flux", 0.1, 0.9, 0.2)]:
        (d / f"real__{x['engine']}__real.json").write_text(json.dumps({"summary": x}))
    out = tmp_path / "o.md"
    subprocess.run([sys.executable, str(Path(__file__).parent.parent / "report_real.py"), "--results", str(d),
                    "--out", str(out), "--title", "My Title"], check=True, capture_output=True)
    t = out.read_text()
    assert "unverified draft references" in t and t.startswith("# My Title")
    assert "## Decision gate" in t and "Cloud baselines" in t and "wer_within_1.5x" in t
    assert "Verified utterances only; paced replay" in t


def test_pub_report_has_no_gate_section_or_check_lines(tmp_path):
    p, out = _run_pub(tmp_path, "--tag", "pub", "--set", "pub_fleurs_tel")
    assert p.returncode == 0, p.stderr
    t = out.read_text()
    for bad in ("Decision gate", "Cloud baselines", "wer_within_1.5x", "commit_final_le_150ms", "cpu_le_0.15",
                "keyterm_within_10pts", "### ", "Verified utterances only"):
        assert bad not in t, bad
    assert "Keyterm check not applicable (public sets have no keyterms); the WER/latency table is the result." in t
    assert "Reference transcripts come from the public dataset; paced replay as in bench/rtbench.py; local engines measured on the dev machine's CPU (a proxy, not Fly)." in t
    assert "| nemo-480 |" in t


def test_pub_report_unknown_cloud_engine_still_exits_1(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    for e in ("nemo-480", "dg-nova"):
        (d / f"pub__{e}__pub_x.json").write_text(json.dumps({"summary": _pub(e, "pub_x")}))
    p = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "report_real.py"), "--results", str(d),
                        "--out", str(tmp_path / "o.md"), "--tag", "pub"], capture_output=True, text=True)
    assert p.returncode == 1 and "dg-nova" in p.stderr and "Traceback" not in p.stderr


def test_mixed_real_and_pub_selection_errors(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "pub__nemo-480__real.json").write_text(json.dumps({"summary": _s("nemo-480", 0.1, 0.9, 0.05)}))
    (d / "pub__dg-flux__pub_x.json").write_text(json.dumps({"summary": _pub("dg-flux", "pub_x")}))
    p = subprocess.run([sys.executable, str(Path(__file__).parent.parent / "report_real.py"), "--results", str(d),
                        "--out", str(tmp_path / "o.md"), "--tag", "pub"], capture_output=True, text=True)
    assert p.returncode != 0 and "mix" in p.stderr.lower() and "Traceback" not in p.stderr
