import json
import math
import subprocess
import sys
from pathlib import Path

from report_real import gate, load_results, price_per_hour, render_table


def _s(engine, wer, kr, commit_lat, cpu=0.1, native=0.6):
    return {"engine": engine, "n": 40, "wer": wer, "keyterm_recall": kr, "keyterm_total": 30,
            "first_partial_med_s": 0.5, "cpu_per_audio_s": cpu,
            "strat": {"commit": {"lat_med": commit_lat, "lat_p90": commit_lat * 2, "cut_utts": 0},
                      "native": {"lat_med": native, "lat_p90": native * 1.3, "cut_utts": 3}}}


def test_price_per_hour_cloud_is_fixed_and_local_uses_cpu():
    assert price_per_hour(_s("dg-flux", 0.1, 0.9, 0.1)) == 0.39
    assert abs(price_per_hour(_s("nemo-480", 0.1, 0.9, 0.05, cpu=0.1)) - 0.1 * 3600 * 0.0000131) < 1e-9


def test_gate_passes_when_local_is_close_enough():
    res = gate([_s("nemo-480", 0.12, 0.85, 0.06), _s("zip-en-int8", 0.20, 0.7, 0.05),
                _s("dg-flux", 0.09, 0.90, 0.2), _s("el-scribe", 0.10, 0.88, 0.2)])
    assert res["best_ours"]["engine"] == "nemo-480" and res["best_cloud"]["engine"] == "dg-flux"
    assert res["pass"] is True and all(res["checks"].values())


def test_gate_fails_on_wer_and_reports_which_check():
    res = gate([_s("nemo-480", 0.30, 0.85, 0.06), _s("dg-flux", 0.09, 0.90, 0.2)])
    assert res["pass"] is False and res["checks"]["wer_within_1.5x"] is False


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
    assert res["pass"] is False and res["checks"]["wer_within_1.5x"] is False


def test_gate_zero_wer_both_and_missing_keyterms_fail_keyterm_check():
    res = gate([_s("nemo-480", 0.0, None, 0.06), _s("dg-flux", 0.0, 0.9, 0.2)])
    assert res["checks"]["keyterm_within_10pts"] is False and res["pass"] is False
    res = gate([_s("nemo-480", 0.1, 0.9, 0.06), _s("dg-flux", 0.1, None, 0.2)])
    assert res["checks"]["keyterm_within_10pts"] is False


def test_gate_missing_or_nan_wer_or_cpu_fails_not_crashes():
    res = gate([_s("nemo-480", None, 0.9, 0.06), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert "error" in res or res["pass"] is False
    res = gate([_s("nemo-480", 0.1, 0.9, 0.06, cpu=None), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert res["pass"] is False and res["checks"]["cpu_le_0.15"] is False
    res = gate([_s("nemo-480", float("nan"), 0.9, 0.06), _s("dg-flux", 0.1, 0.9, 0.2)])
    assert "error" in res or res["pass"] is False


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
