"""Render the real-call comparison and the Phase 1 decision gate from bench/results/real__*.json.
Aggregates only: reads each file's `summary`, never `rows` (which hold transcripts).
  python report_real.py [--results results] [--out ../REAL_CALL_EVAL.md]"""
import argparse
import glob
import json
import math
import os
import sys

CLOUD_PREFIXES = ("dg-", "el-")
MIN_UTTS = 30            # WER check needs >= this many utterances on both engines, and equal n
MIN_KEYTERMS = 30        # keyterm check needs >= this many keyterms on both engines
MIN_FIRED_FRAC = 0.9     # commit latency check needs the strategy to have fired on >= this fraction
EPS = 1e-9
CLOUD_PRICE = {"dg-flux": 0.39, "el-scribe": 0.39}     # $/audio-hour, published rates (2026-09)
MODAL_CPU_PER_CORE_S = 0.0000131                      # $/core-second, worker-stt/DESIGN.md (compute only, packed)


def _ok(x):
    return isinstance(x, (int, float)) and not math.isnan(x)


def load_results(results_dir, tag="real", set_name=None):
    out = []
    for f in sorted(glob.glob(os.path.join(results_dir, f"{tag}__*.json"))):
        with open(f) as fh:
            summ = json.load(fh)["summary"]
        if set_name is None or summ.get("set") == set_name:
            out.append(summ)
    return out


def is_public(s):
    return str(s.get("set", "")).startswith("pub_")


def price_per_hour(s):
    if is_cloud(s["engine"]):
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
    lines = ["| engine | n utts | WER | keyterm recall | first partial ms | commit final ms med/p90 | native final ms med/p90 | cpu-s per audio-s | $/audio-hr (compute-only, 100% packed, dev-machine CPU proxy, Modal CPU rate; not like-for-like with per-hour cloud pricing) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        strat = s.get("strat", {})
        c, n = strat.get("commit", {}), strat.get("native", {})
        cpu = s.get("cpu_per_audio_s")
        cpu_txt = "-" if is_cloud(s["engine"]) or not _ok(cpu) else round(cpu, 3)
        lines.append(f"| {s['engine']} | {s.get('n', '-')} | {_pct(s.get('wer'))} | {_pct(s.get('keyterm_recall'))} | "
                     f"{_ms(s.get('first_partial_med_s'))} | {_ms(c.get('lat_med'))}/{_ms(c.get('lat_p90'))} | "
                     f"{_ms(n.get('lat_med'))}/{_ms(n.get('lat_p90'))} | "
                     f"{cpu_txt} | {_price(s)} |")
    return "\n".join(lines)


def is_cloud(engine):
    if engine in CLOUD_PRICE:
        return True
    if engine.startswith(CLOUD_PREFIXES):
        raise ValueError(f"cloud-prefixed engine {engine!r} has no entry in CLOUD_PRICE")
    return False


def _tri(ok):
    return "PASS" if ok else "FAIL"


def is_unverified(s):
    """A real-set summary not scored on hand-verified references (missing flag counts as unverified)."""
    return s.get("set") == "real" and s.get("only_verified") is not True


def _eval_local(o, bw, bk):
    """Tri-state checks for one local engine against the WER baseline `bw` and keyterm baseline `bk`."""
    wer_c, wer_o = bw.get("wer"), o.get("wer")
    if not (_ok(wer_c) and _ok(wer_o)) or min(o.get("n") or 0, bw.get("n") or 0) < MIN_UTTS or o.get("n") != bw.get("n"):
        wer = "INSUFFICIENT"
    else:
        wer = _tri(wer_o <= 1.5 * wer_c + EPS)
    ko, kc = o.get("keyterm_recall"), bk.get("keyterm_recall")
    unver = is_unverified(o) or is_unverified(bw)
    if unver:
        wer = "INSUFFICIENT"
    if unver or is_unverified(bk):
        kt = "INSUFFICIENT"
    elif not (_ok(ko) and _ok(kc)) or min(o.get("keyterm_total") or 0, bk.get("keyterm_total") or 0) < MIN_KEYTERMS:
        kt = "INSUFFICIENT"
    else:
        kt = _tri(ko >= kc - 0.10 - EPS)
    c = o.get("strat", {}).get("commit", {})
    lat, ff = c.get("lat_med"), c.get("fired_frac")
    lat_r = "INSUFFICIENT" if not (_ok(lat) and _ok(ff)) or ff < MIN_FIRED_FRAC else _tri(lat <= 0.150 + EPS)
    cpu = o.get("cpu_per_audio_s")
    cpu_r = "INSUFFICIENT" if not _ok(cpu) else _tri(cpu <= 0.15 + EPS)
    return {"wer_within_1.5x": wer, "keyterm_within_10pts": kt, "commit_final_le_150ms": lat_r, "cpu_le_0.15": cpu_r}


def _verdict(checks):
    v = list(checks.values())
    return "PASS" if all(x == "PASS" for x in v) else "FAIL" if "FAIL" in v else "INSUFFICIENT"


def gate(summaries):
    ours = [s for s in summaries if not is_cloud(s["engine"])]
    cloud = [s for s in summaries if is_cloud(s["engine"])]
    if not ours or not cloud:
        return {"error": "need at least one local and one cloud result"}
    cw = [s for s in cloud if _ok(s.get("wer"))]
    ck = [s for s in cloud if _ok(s.get("keyterm_recall"))]
    bc = min(cw, key=lambda s: s["wer"]) if cw else cloud[0]                          # WER baseline: lowest WER
    bk = max(ck, key=lambda s: s["keyterm_recall"]) if ck else cloud[0]               # keyterm baseline: highest recall
    locals_ = []
    for o in ours:
        checks = _eval_local(o, bc, bk)
        locals_.append({"engine": o["engine"], "checks": checks, "verdict": _verdict(checks)})
    passers = [x for x in locals_ if x["verdict"] == "PASS"]
    by_eng = {s["engine"]: s for s in ours}
    if passers:
        rec = min(passers, key=lambda x: by_eng[x["engine"]]["wer"])["engine"]
        overall = True
    else:
        rec = None
        overall = None if any(x["verdict"] == "INSUFFICIENT" for x in locals_) else False
    valid = [s for s in ours if _ok(s.get("wer"))]
    bo = by_eng[rec] if rec else (min(valid, key=lambda s: s["wer"]) if valid else ours[0])
    res = {"best_ours": bo, "best_cloud": bc, "best_cloud_keyterm": bk, "locals": locals_,
           "checks": next(x["checks"] for x in locals_ if x["engine"] == bo["engine"]),
           "pass": overall, "recommended": rec}
    if _ok(bc.get("wer")) and bc["wer"] < 0.02:
        res["advisory"] = "advisory: cloud WER very low, 1.5x rule is degenerate"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "REAL_CALL_EVAL.md"))
    ap.add_argument("--tag", default="real")
    ap.add_argument("--set", dest="set_name", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    summ = load_results(a.results, a.tag, a.set_name)
    if not summ:
        sys.exit(f"no results found: expected {os.path.join(a.results, a.tag + '__*.json')}"
                 + (f" with set {a.set_name!r}" if a.set_name else "") + " (run rtbench.py first)")
    public = all(is_public(s) for s in summ)
    try:
        g = gate(summ)
        table = render_table(summ)
    except ValueError as e:
        sys.exit(f"error: {e}")
    warns = [f"**WARNING: {s['engine']} was scored on unverified draft references**" for s in summ if is_unverified(s)]
    body = [f"# {a.title or 'Real-call STT evaluation (Phase 1)'}", "",
            "Verified utterances only; paced replay as in `bench/rtbench.py`; local engines measured on the dev machine's CPU (a proxy, not Fly).",
            "", *([w for w in warns] + [""] if warns else []), table, "", "## Decision gate", ""]
    if "error" in g:
        body.append(g["error"])
    else:
        bc, bk = g["best_cloud"], g["best_cloud_keyterm"]
        body += [f"Cloud baselines: WER vs **{bc['engine']}** (lowest WER), keyterm recall vs **{bk['engine']}** (highest keyterm recall).", ""]
        if g.get("advisory"):
            body += [f"- {g['advisory']}", ""]
        for x in g["locals"]:
            body += [f"### {x['engine']}: {x['verdict']}"] + [f"- {v}: {k}" for k, v in x["checks"].items()] + [""]
        if public:
            res = ("Keyterm check not applicable to public sets (no keyterms); WER table is the result. "
                   "Gate verdict: INSUFFICIENT")
        elif g["pass"]:
            res = "ship-worthy, proceed to Phase 2 with " + g["recommended"]
        elif g["pass"] is None:
            res = "INSUFFICIENT DATA, no local engine passed and some checks could not be evaluated; collect more data before deciding"
        else:
            res = "not yet, Phase 2 becomes an accuracy investigation"
        body += [f"**Result: {res}**"]
    body += ["", f"Footnotes: a check is INSUFFICIENT (not FAIL) when data is missing or thin: WER needs n >= {MIN_UTTS} on both engines and equal n; "
             f"keyterm needs keyterm_total >= {MIN_KEYTERMS} on both; commit latency needs fired_frac >= {MIN_FIRED_FRAC}. "
             "Price is compute-only, 100% packed, dev-machine CPU proxy at the Modal CPU rate; not like-for-like with per-hour cloud pricing."]
    text = "\n".join(body) + "\n"
    with open(a.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
