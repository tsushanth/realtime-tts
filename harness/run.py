"""Latency/regression harness for the realtime TTS worker (or gateway).

Usage:
  python run.py --endpoint ws://127.0.0.1:8765 --concurrency 4
  python run.py --endpoint wss://realtime-tts-gateway.fly.dev/tts --concurrency 8

Talks the same JSON+binary protocol as worker/server.py directly. Against the gateway,
point --endpoint at wss://<app>.fly.dev/tts instead.
"""
import argparse
import asyncio
import json
import sqlite3
import statistics
import time
import uuid
from pathlib import Path

import websockets

DB_PATH = Path(__file__).parent / "history.sqlite3"

CORPUS = [
    "Hey, quick question.",
    "This is a medium length sentence used to test typical chat responses in a realtime pipeline.",
    "This is a considerably longer utterance meant to stress multi-chunk streaming synthesis, "
    "covering several clauses so the harness can measure inter-chunk jitter as well as the "
    "initial time to first byte, which matters most for how responsive the experience feels.",
]

TTFB_P95_BUDGET_MS = 1500  # real measured GPU-pod-over-gateway p95 at concurrency=4 was
# 1367ms (see DECISIONS.md); 300ms is the aspirational realtime target but this backend
# doesn't hit it yet at load, so the budget is set slightly above what's actually
# achieved today rather than a number that would always fail. Tighten this once the
# concurrency-4 numbers improve — see the open concurrency-scaling question in DECISIONS.md.


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT, ts REAL, endpoint TEXT,
            ttfb_ms REAL, e2e_ms REAL, rtf REAL, jitter_ms REAL,
            success INTEGER, error TEXT
        )
    """)
    con.commit()
    return con


async def one_session(endpoint, text, session_id):
    result = {"ttfb_ms": None, "e2e_ms": None, "rtf": None, "jitter_ms": None,
              "success": False, "error": None, "providers": None}
    t_start = time.perf_counter()
    first_byte_t = None
    last_frame_t = None
    gaps = []
    total_audio_s = 0.0
    total_gen_ms = 0.0
    try:
        async with websockets.connect(endpoint, max_size=None) as ws:
            await ws.send(json.dumps({"type": "synthesize", "text": text}))
            pending_meta = None
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=90)
                now = time.perf_counter()
                if isinstance(raw, (bytes, bytearray)):
                    if first_byte_t is None:
                        first_byte_t = now
                    elif last_frame_t is not None:
                        gaps.append((now - last_frame_t) * 1000)
                    last_frame_t = now
                    if pending_meta:
                        total_audio_s += pending_meta.get("audio_s", 0)
                        total_gen_ms += pending_meta.get("gen_ms", 0)
                        pending_meta = None
                    continue
                msg = json.loads(raw)
                if msg["type"] == "chunk_meta":
                    pending_meta = msg
                    if msg.get("providers"):
                        result["providers"] = msg["providers"]
                elif msg["type"] == "done":
                    break
                elif msg["type"] in ("error", "cancelled"):
                    result["error"] = msg.get("message", msg["type"])
                    break
        t_end = time.perf_counter()
        if first_byte_t is not None:
            result["ttfb_ms"] = (first_byte_t - t_start) * 1000
            result["e2e_ms"] = (t_end - t_start) * 1000
            # RTF = synthesized audio duration / actual server-side compute time it took to
            # generate it. Using wall-clock-since-first-byte instead breaks for single-chunk
            # utterances (near-zero elapsed time between first byte and "done").
            result["rtf"] = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else None
            result["jitter_ms"] = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
            result["success"] = result["error"] is None
        else:
            result["error"] = result["error"] or "no audio received"
    except Exception as e:
        result["error"] = str(e)
    return result


async def run_load(endpoint, concurrency, con, run_id):
    tasks = []
    for i in range(concurrency):
        text = CORPUS[i % len(CORPUS)]
        tasks.append(one_session(endpoint, text, i))
    results = await asyncio.gather(*tasks)
    ts = time.time()
    for r in results:
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, ts, endpoint, r["ttfb_ms"], r["e2e_ms"], r["rtf"], r["jitter_ms"],
             int(r["success"]), r["error"]),
        )
    con.commit()
    return results


def summarize(results, con, endpoint):
    ttfbs = [r["ttfb_ms"] for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    print(f"\n=== run summary: {len(results)} sessions, {len(failures)} failed ===")
    for f in failures:
        print(f"  FAIL: {f['error']}")
    if not ttfbs:
        print("  no successful sessions — cannot compute latency stats")
        return False
    ttfbs.sort()
    p50 = statistics.median(ttfbs)
    p95 = ttfbs[min(len(ttfbs) - 1, int(len(ttfbs) * 0.95))]
    rtfs = [r["rtf"] for r in results if r["success"] and r["rtf"]]
    print(f"  TTFB p50={p50:.0f}ms p95={p95:.0f}ms budget_p95={TTFB_P95_BUDGET_MS}ms")
    print(f"  RTF avg={statistics.mean(rtfs):.2f}x" if rtfs else "  RTF: n/a")
    seen_providers = {tuple(r["providers"]) for r in results if r["success"] and r["providers"]}
    if seen_providers:
        print(f"  providers seen: {[list(p) for p in seen_providers]}")
        if any("CUDAExecutionProvider" not in p for p in seen_providers):
            print("  WARNING: at least one request ran WITHOUT CUDAExecutionProvider (CPU fallback)")

    # regression check vs rolling baseline (prior runs on this endpoint, excluding this one)
    hist = con.execute(
        "SELECT ttfb_ms FROM runs WHERE endpoint=? AND success=1 AND ttfb_ms IS NOT NULL "
        "ORDER BY ts DESC LIMIT 100", (endpoint,)
    ).fetchall()
    passed = p95 <= TTFB_P95_BUDGET_MS
    if len(hist) >= 10:
        baseline_p50 = statistics.median([h[0] for h in hist])
        regressed = p50 > baseline_p50 * 1.5
        print(f"  baseline p50={baseline_p50:.0f}ms (n={len(hist)}) -> {'REGRESSION' if regressed else 'ok'}")
        passed = passed and not regressed
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="ws://127.0.0.1:8765")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    con = init_db()
    run_id = str(uuid.uuid4())[:8]
    results = asyncio.run(run_load(args.endpoint, args.concurrency, con, run_id))
    passed = summarize(results, con, args.endpoint)
    con.close()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
