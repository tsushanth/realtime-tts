"""Real-time replay benchmark. Usage:
  python rtbench.py --engine zip-en-int8 --set clean --n 100 --out results/x.json [--native-ms 500]

Each clip is fed in 40 ms chunks paced to the wall clock (chunk i is delivered at (i+1)*40 ms, like a live
phone stream), followed by 1.6 s of trailing silence/noise (the stream keeps flowing after the caller stops).
Clip end == end of speech (ground truth from clip length). Several endpoint strategies are evaluated on the
SAME pass (they only decide WHEN the engine's current hypothesis is declared final):
  commit     client signals end of speech exactly at clip end (push-to-talk / explicit commit)
  vad300/500/700   silero VAD trailing silence >= N ms
  vad500+hint      vad500, but +400 ms extra wait if the hypothesis ends in a function word
  native     engine's own endpointer (sherpa rule2, or Moonshine line completion) if present
Latency = wall time of the final result minus end of speech. `cut` = the strategy fired while speech was still
ongoing (a mid-sentence cut-off that a voice agent would act on).
"""
import argparse, json, os, re, resource, sys, time
import numpy as np, soundfile as sf, jiwer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from textnorm import norm            # noqa: E402  (ws_client.py imports norm from here; keep it exported)
from metrics import keyterm_recall   # noqa: E402
DATA = os.environ.get("STT_DATA", os.path.join(HERE, "..", "data"))
CH = 0.04
TAIL_S = 1.6
LEAD_S = 0.3   # a live stream starts before the caller speaks
FUNC = set("a an the and or but so of to in on at for with from by as that which who whom whose if then than because "
           "my your his her their our its is are was were be been am will would can could should i we you they he she it "
           "um uh like about into over".split())
STRATS = ["commit", "vad300", "vad500", "vad700", "vad500+hint"]


def load(set_name, n, seed=0, verified_only=False):
    mp = os.path.join(DATA, set_name, "manifest.json")
    man = json.load(open(mp if os.path.exists(mp) else os.path.join(DATA, "manifest.json")))
    if verified_only:
        man = [m for m in man if m.get("verified")]
    man = man[:n]
    rng = np.random.default_rng(seed)
    out = []
    for m in man:
        x, sr = sf.read(os.path.join(DATA, set_name, m["id"] + ".wav"), dtype="float32")
        assert sr == 16000
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        sigma = 0.0 if set_name in ("clean", "call", "real") else rms * 10 ** (-30 / 20)   # tel: ~30 dB SNR mild noise floor
        x = x + rng.normal(0, sigma, len(x)).astype("float32") if sigma else x
        tail = rng.normal(0, sigma if sigma else 1e-4, int(TAIL_S * 16000)).astype("float32")
        lead = rng.normal(0, sigma if sigma else 1e-4, int(LEAD_S * 16000)).astype("float32")
        out.append({"id": m["id"], "ref": m["ref"], "call_id": m.get("call_id"), "keyterms": m.get("keyterms", []),
                    "dur": (len(x) + len(lead)) / 16000, "audio": np.concatenate([lead, x, tail]).astype("float32")})
    return out


class SilenceVAD:
    """silero (via sherpa-onnx) -> trailing silence in ms of *audio time*."""
    def __init__(self):
        import sherpa_onnx as so
        c = so.VadModelConfig()
        c.silero_vad.model = os.path.join(os.environ.get("STT_MODELS", os.path.join(HERE, "..", "models")), "silero_vad.onnx")
        c.silero_vad.min_silence_duration = 0.05
        c.silero_vad.min_speech_duration = 0.1
        c.silero_vad.threshold = 0.5
        c.sample_rate = 16000
        self.so, self.c = so, c
        self.reset()

    def reset(self):
        self.v = self.so.VoiceActivityDetector(self.c, buffer_size_in_seconds=60)
        self.t = 0.0; self.was = False; self.speech_seen = False; self.sil_start = None

    def push(self, x):
        self.v.accept_waveform(x); self.t += len(x) / 16000
        while not self.v.empty():
            self.v.pop()
        sp = self.v.is_speech_detected()
        if sp:
            self.speech_seen = True; self.sil_start = None
        elif self.speech_seen and self.sil_start is None:
            self.sil_start = self.t - 0.05
        return sp

    def silence_ms(self):
        return 0.0 if self.sil_start is None else (self.t - self.sil_start) * 1000


def run_clip(eng, vad, clip, strategies, native_ms):
    st = eng.new_stream()
    vad.reset()
    a, dur = clip["audio"], clip["dur"]
    step = int(CH * 16000)
    t0 = time.perf_counter(); c0 = time.process_time()
    fires = {s: [] for s in strategies}     # (fire_wall_time_rel, audio_time, text)
    armed = {s: True for s in strategies}
    first_partial = None
    committed = False
    n_chunks = (len(a) + step - 1) // step
    last_txt = ""
    for i in range(n_chunks):
        due = t0 + (i + 1) * CH
        now = time.perf_counter()
        if due > now:
            time.sleep(due - now)
        x = a[i * step:(i + 1) * step]
        st.push(x)
        at = min((i + 1) * step, len(a)) / 16000     # audio time delivered so far
        txt = st.text()
        if txt.strip() and first_partial is None:
            first_partial = time.perf_counter() - t0
        if txt != last_txt:
            last_txt = txt
            for s in strategies:                      # new words => re-arm (a later fire may be needed)
                armed[s] = True
        vad.push(x)
        sil = vad.silence_ms()
        ends_func = bool(norm(txt).split()) and norm(txt).split()[-1] in FUNC
        for s in strategies:
            if not armed[s] or not txt.strip():
                continue
            fire = False
            if s == "commit":
                fire = at >= dur - 1e-6 and not committed
            elif s.startswith("vad"):
                thr = int(re.match(r"vad(\d+)", s).group(1))
                if s.endswith("+hint") and ends_func:
                    thr += 400
                fire = sil >= thr
            elif s == "native":
                fire = bool(st.native_eos())
            if fire:
                armed[s] = False
                fires[s].append([time.perf_counter() - t0, at, txt])
        if "commit" in strategies and at >= dur - 1e-6 and not committed:
            committed = True
    return st, fires, first_partial, time.process_time() - c0, len(a) / 16000, t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--set", default="clean")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--native-ms", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--only-verified", action="store_true")
    a = ap.parse_args()
    import engines
    eng = engines.build(a.engine, a.threads) if not a.native_ms else _native_engine(a.engine, a.threads, a.native_ms / 1000)
    rss_loaded = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_loaded = rss_loaded / (1024 * 1024) if sys.platform == "darwin" else rss_loaded / 1024   # MB
    vad = SilenceVAD()
    clips = load(a.set, a.n, verified_only=a.only_verified)
    strategies = ["commit", "vad300", "vad500", "vad700", "vad500+hint"] + (["native"] if a.native_ms or a.engine.startswith("moonshine") else [])
    # warm-up one short pass so first-call JIT / allocation does not pollute the numbers
    run_clip(eng, vad, {"audio": clips[0]["audio"][:16000], "dur": 1.0, "ref": ""}, ["vad500"], 0)
    rows = []
    for k, c in enumerate(clips):
        st, fires, fp, cpu, alen, t0 = run_clip(eng, vad, c, strategies, a.native_ms)
        # the harness cannot call final() mid-stream on a shared stream, so final text per strategy is the
        # hypothesis at firing time + the cost of the closing call measured separately on the finished stream
        tf0 = time.perf_counter(); final_text = st.final(); close_cost = time.perf_counter() - tf0
        ref = norm(c["ref"]); hyp = norm(final_text)
        wer_edits = jiwer.process_words(ref, hyp if hyp else "x")
        row = {"id": c["id"], "dur": c["dur"], "words": len(ref.split()), "wer_err": wer_edits.substitutions + wer_edits.deletions + wer_edits.insertions,
               "hyp": hyp, "ref": ref, "first_partial_s": fp, "cpu_s": cpu, "audio_s": alen, "close_cost_s": close_cost, "strat": {}}
        kh, kt = keyterm_recall(c.get("keyterms", []), final_text)
        row["keyterm_hits"], row["keyterm_total"] = kh, kt
        for s, fl in fires.items():
            after = [f for f in fl if f[1] >= c["dur"] - 0.05]      # fires at/after end of speech
            before = [f for f in fl if f[1] < c["dur"] - 0.05]      # fires while still speaking = cut-off
            lat = None
            if after:
                f = after[0]
                lat = f[0] - c["dur"]
                if s == "commit" or a.engine.startswith("fw-"):
                    lat += close_cost      # closing call (tail flush / full whisper re-decode) happens after the trigger
            row["strat"][s] = {"lat": lat, "cuts": len(before), "fired": bool(after)}
        rows.append(row)
        print(f"[{k+1}/{len(clips)}] {c['id']} wer_err={row['wer_err']}/{row['words']} " +
              " ".join(f"{s}={(v['lat'] if v['lat'] is not None else -1):.2f}/{v['cuts']}" for s, v in row["strat"].items()), flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    tot_err = sum(r["wer_err"] for r in rows); tot_w = sum(r["words"] for r in rows)
    summ = {"engine": a.engine, "set": a.set, "label": a.label, "n": len(rows), "threads": a.threads, "wer": tot_err / tot_w,
            "cpu_per_audio_s": sum(r["cpu_s"] for r in rows) / sum(r["audio_s"] for r in rows),
            "rss_loaded_mb": rss_loaded, "rss_peak_mb": peak,
            "first_partial_med_s": float(np.median([r["first_partial_s"] for r in rows if r["first_partial_s"] is not None])),
            "close_cost_med_s": float(np.median([r["close_cost_s"] for r in rows])), "strat": {},
            "keyterm_total": sum(r["keyterm_total"] for r in rows),
            "keyterm_recall": (sum(r["keyterm_hits"] for r in rows) / sum(r["keyterm_total"] for r in rows))
                              if sum(r["keyterm_total"] for r in rows) else None}
    for s in strategies:
        lats = [r["strat"][s]["lat"] for r in rows if r["strat"][s]["lat"] is not None]
        summ["strat"][s] = {"lat_med": float(np.median(lats)) if lats else None, "lat_p90": float(np.percentile(lats, 90)) if lats else None,
                            "fired_frac": len(lats) / len(rows), "cut_utts": sum(1 for r in rows if r["strat"][s]["cuts"] > 0),
                            "cuts_total": sum(r["strat"][s]["cuts"] for r in rows)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


def _native_engine(name, threads, rule2):
    import engines
    if name == "zip-en-int8":
        s = "chunk-16-left-128.int8.onnx"
        return engines.SherpaEngine(name, "sherpa-onnx-streaming-zipformer-en-2023-06-26", f"encoder-epoch-99-avg-1-{s}",
                                    f"decoder-epoch-99-avg-1-{s}", f"joiner-epoch-99-avg-1-{s}", threads, native_rule2=rule2)
    if name.startswith("nemo-"):
        ms = name.split("-")[1]
        return engines.SherpaEngine(name, f"sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-{ms}ms-int8",
                                    "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", threads, native_rule2=rule2)
    raise ValueError("native endpoint override only for sherpa engines")


if __name__ == "__main__":
    main()
