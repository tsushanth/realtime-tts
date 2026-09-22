"""Drives N concurrent /tts WebSocket connections against a running worker (local or
deployed) and reports latency + how many machines Fly actually ran during the test.
Not a pytest test (it needs a live server and real wall-clock time) - a standalone
script, run manually per LOAD_TEST.md.

Usage: python3 tests/load_test.py <ws-url> <session-token> --concurrency 10 --requests 30
"""
import argparse
import asyncio
import json
import time

import websockets


async def one_synthesis(ws_url: str, token: str, text: str) -> float:
    start = time.monotonic()
    async with websockets.connect(f"{ws_url}?token={token}") as ws:
        await ws.send(json.dumps({"type": "synthesize", "text": text, "voice": "custom:en-us-john", "speed": 1.0}))
        first_byte_at = None
        async for message in ws:
            if isinstance(message, bytes):
                if first_byte_at is None:
                    first_byte_at = time.monotonic()
            else:
                payload = json.loads(message)
                if payload.get("type") == "done":
                    break
                if payload.get("type") == "error":
                    raise RuntimeError(f"synthesis error: {payload.get('message')}")
    return (first_byte_at or time.monotonic()) - start


async def run_load_test(ws_url: str, token: str, concurrency: int, total_requests: int):
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = []

    async def bounded_request(i: int):
        async with sem:
            try:
                latencies.append(await one_synthesis(ws_url, token, f"Load test request number {i}."))
            except Exception as e:  # noqa: BLE001 - report every failure, don't let one kill the run
                errors.append(str(e))

    await asyncio.gather(*(bounded_request(i) for i in range(total_requests)))

    print(f"completed: {len(latencies)}/{total_requests}, errors: {len(errors)}")
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p90 = latencies[int(len(latencies) * 0.9)]
        print(f"time-to-first-byte: p50={p50*1000:.0f}ms p90={p90*1000:.0f}ms")
    if errors:
        print("errors:", errors[:5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ws_url")
    parser.add_argument("token")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(run_load_test(args.ws_url, args.token, args.concurrency, args.requests))
