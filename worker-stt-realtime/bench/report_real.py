"""Render the real-call comparison and the Phase 1 decision gate from bench/results/real__*.json.
Aggregates only: reads each file's `summary`, never `rows` (which hold transcripts).
  python report_real.py [--results results] [--out ../REAL_CALL_EVAL.md]"""
import argparse
import glob
import json
import math
import os
import sys

CLOUD_PRICE = {"dg-flux": 0.39, "el-scribe": 0.39}     # $/audio-hour, published rates (2026-09)
MODAL_CPU_PER_CORE_S = 0.0000131                      # $/core-second, worker-stt/DESIGN.md (compute only, packed)


def _ok(x):
    return isinstance(x, (int, float)) and not math.isnan(x)


def load_results(results_dir, tag="real"):
    out = []
    for f in sorted(glob.glob(os.path.join(results_dir, f"{tag}__*.json"))):
        with open(f) as fh:
            out.append(json.load(fh)["summary"])
    return out


def price_per_hour(s):
    if s["engine"] in CLOUD_PRICE:
        return CLOUD_PRICE[s["engine"]]
    cpu = s.get("cpu_per_audio_s")
    return cpu * 3600 * MODAL_CPU_PER_CORE_S if _ok(cpu) else float("nan")


def _ms(x):
    return "-" if not _ok(x) else f"{x * 1000:.0f}"


def _pct(x):
    return "-" if not _ok(x) else f"{x * 100:.1f}%"


def _price(s):
    p = price_per_hour(s)
    return "-" if not _ok(p) else f"{p:.4f}"


def render_table(summaries):
    lines = ["| engine | n utts | WER | keyterm recall | first partial ms | commit final ms med/p90 | native final ms med/p90 | cpu-s per audio-s | $/audio-hr |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        strat = s.get("strat", {})
        c, n = strat.get("commit", {}), strat.get("native", {})
        cpu = s.get("cpu_per_audio_s")
        cpu_txt = "-" if s["engine"] in CLOUD_PRICE or not _ok(cpu) else round(cpu, 3)
        lines.append(f"| {s['engine']} | {s.get('n', '-')} | {_pct(s.get('wer'))} | {_pct(s.get('keyterm_recall'))} | "
                     f"{_ms(s.get('first_partial_med_s'))} | {_ms(c.get('lat_med'))}/{_ms(c.get('lat_p90'))} | "
                     f"{_ms(n.get('lat_med'))}/{_ms(n.get('lat_p90'))} | "
                     f"{cpu_txt} | {_price(s)} |")
    return "\n".join(lines)


def gate(summaries):
    usable = [s for s in summaries if _ok(s.get("wer"))]     # an engine with no valid WER cannot be ranked
    ours = [s for s in usable if s["engine"] not in CLOUD_PRICE]
    cloud = [s for s in usable if s["engine"] in CLOUD_PRICE]
    if not ours or not cloud:
        return {"error": "need at least one local and one cloud result with a valid WER"}
    bo, bc = min(ours, key=lambda s: s["wer"]), min(cloud, key=lambda s: s["wer"])
    ko, kc = bo.get("keyterm_recall"), bc.get("keyterm_recall")
    lat = bo.get("strat", {}).get("commit", {}).get("lat_med")
    cpu = bo.get("cpu_per_audio_s")
    checks = {
        "wer_within_1.5x": bo["wer"] <= 1.5 * bc["wer"],
        "keyterm_within_10pts": _ok(ko) and _ok(kc) and ko >= kc - 0.10,
        "commit_final_le_150ms": _ok(lat) and lat <= 0.150,
        "cpu_le_0.15": _ok(cpu) and cpu <= 0.15,
    }
    return {"best_ours": bo, "best_cloud": bc, "checks": checks, "pass": all(checks.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "REAL_CALL_EVAL.md"))
    a = ap.parse_args()
    summ = load_results(a.results)
    if not summ:
        sys.exit(f"no results found: expected {os.path.join(a.results, 'real__*.json')} (run rtbench.py --set real first)")
    g = gate(summ)
    body = ["# Real-call STT evaluation (Phase 1)", "",
            "Verified utterances only; paced replay as in `bench/rtbench.py`; local engines measured on the dev machine's CPU (a proxy, not Fly).",
            "", render_table(summ), "", "## Decision gate", ""]
    if "error" in g:
        body.append(g["error"])
    else:
        body += [f"Best local: **{g['best_ours']['engine']}**, best cloud: **{g['best_cloud']['engine']}**.", ""]
        body += [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in g["checks"].items()]
        body += ["", f"**Result: {'ship-worthy, proceed to Phase 2 with ' + g['best_ours']['engine'] if g['pass'] else 'not yet, Phase 2 becomes an accuracy investigation'}**"]
    text = "\n".join(body) + "\n"
    with open(a.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
