"""Load test for worker-demucs-fly's /v1/isolate.

Fires N concurrent requests at a running instance (local or remote) using the committed
verification/noisy_mix.wav clip, and reports per-request status/latency plus a summary. Used to
validate the MAX_CONNECTIONS=2 concurrency limit's actual behavior under load (README's "What's
left" flagged this as unmeasured - a guess based on Piper's fly.toml as a reference point).

Usage:
    .venv/bin/python verification/load_test.py --url http://localhost:8080 --token devtoken -n 2
    .venv/bin/python verification/load_test.py --url http://localhost:8080 --token devtoken -n 4
    # against the live dev worker (be conservative - small -n only, see README caveats):
    .venv/bin/python verification/load_test.py --url https://demucs-isolation-dev.fly.dev \
        --token "$SESSION_TOKEN" -n 3

Prints one line per request (status, latency) plus a summary (success/503/error counts, latency
percentiles for successful requests). Does not assert/fail - this is a measurement tool, not a
pass/fail gate (there's no agreed SLA yet to gate against).
"""
import argparse
import concurrent.futures
import pathlib
import statistics
import time

import requests

CLIP = pathlib.Path(__file__).parent / "noisy_mix.wav"


def one_request(url: str, token: str) -> dict:
    t0 = time.time()
    try:
        with open(CLIP, "rb") as f:
            resp = requests.post(
                f"{url}/v1/isolate",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("noisy_mix.wav", f, "audio/wav")},
                data={"stem": "vocals"},
                timeout=120,
            )
        return {"status": resp.status_code, "latency": time.time() - t0, "body": resp.text[:200] if resp.status_code != 200 else None}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "latency": time.time() - t0, "body": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("-n", "--concurrency", type=int, required=True, help="number of concurrent requests fired at once")
    args = ap.parse_args()

    print(f"Firing {args.concurrency} concurrent requests at {args.url}/v1/isolate ...")
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, args.url, args.token) for _ in range(args.concurrency)]
        results = [f.result() for f in futures]
    t_total = time.time() - t_start

    for i, r in enumerate(results):
        extra = f" body={r['body']!r}" if r["body"] else ""
        print(f"  req {i}: status={r['status']} latency={r['latency']:.2f}s{extra}")

    ok = [r["latency"] for r in results if r["status"] == 200]
    capacity = sum(1 for r in results if r["status"] == 503)
    errors = sum(1 for r in results if r["status"] not in (200, 503))
    print(f"\nSummary (n={args.concurrency}, wall={t_total:.2f}s):")
    print(f"  200 OK:        {len(ok)}")
    print(f"  503 at-capacity: {capacity}")
    print(f"  other/error:   {errors}")
    if ok:
        ok_sorted = sorted(ok)
        p50 = statistics.median(ok_sorted)
        p95 = ok_sorted[min(len(ok_sorted) - 1, int(len(ok_sorted) * 0.95))]
        print(f"  successful latency: min={min(ok):.2f}s p50={p50:.2f}s p95={p95:.2f}s max={max(ok):.2f}s")


if __name__ == "__main__":
    main()
