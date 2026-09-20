import json, glob, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results", "*.json"))):
    d = json.load(open(f)); s = d["summary"]; tag = os.path.basename(f).split("__")[0]
    nat = "_n" in os.path.basename(f).split("__")[1]
    rows.append((tag, s, nat, os.path.basename(f)))
def fm(x): return "-" if x is None else f"{x*1000:.0f}"
print("== main pass (all strategies), latency ms med/p90 [cut utts of N]  ==")
for tag, s, nat, fn in rows:
    if nat: continue
    st = s["strat"]
    cells = " | ".join(f"{k}:{fm(v['lat_med'])}/{fm(v['lat_p90'])}[{v['cut_utts']}]" for k, v in st.items())
    print(f"{tag:10s} {s['engine']:28s} {s['set']:8s} n={s['n']:3d} WER={s['wer']*100:5.1f}% fp={s['first_partial_med_s']*1000:5.0f}ms cpu/s={s['cpu_per_audio_s']:.3f} rss={s['rss_peak_mb']:.0f}MB | {cells}")
print("== native endpointing passes ==")
for tag, s, nat, fn in rows:
    if not nat: continue
    v = s["strat"].get("native")
    print(f"{fn:60s} native {fm(v['lat_med'])}/{fm(v['lat_p90'])} cut {v['cut_utts']}/{s['n']} fired={v['fired_frac']:.2f}")
