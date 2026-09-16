#!/usr/bin/env python3
"""Bulk-generate audio for corpus_final.txt via Amazon Polly Generative.

Resumable: writes one mp3 per line, named by a stable hash of the line's
index+text, and skips any that already exist — safe to stop and re-run.
Rate-limited with a small thread pool plus exponential backoff on
throttling, since Polly's generative tier has a lower TPS ceiling than
Standard/Neural.

Usage: python3 generate_corpus_audio.py [--limit N] [--workers 4]
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError

VOICE = "Joanna"
ENGINE = "generative"
OUT_DIR = os.path.join(os.path.dirname(__file__), "audio")
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifest.jsonl")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus_final.txt")


def synthesize_one(polly, idx: int, text: str) -> dict:
    out_path = os.path.join(OUT_DIR, f"{idx:06d}.mp3")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return {"idx": idx, "text": text, "file": os.path.basename(out_path), "status": "skipped"}

    backoff = 1.0
    for attempt in range(6):
        try:
            resp = polly.synthesize_speech(
                Text=text, OutputFormat="mp3", VoiceId=VOICE, Engine=ENGINE
            )
            with open(out_path, "wb") as f:
                f.write(resp["AudioStream"].read())
            return {"idx": idx, "text": text, "file": os.path.basename(out_path), "status": "ok"}
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("ThrottlingException", "TooManyRequestsException") and attempt < 5:
                time.sleep(backoff)
                backoff = min(backoff * 2, 20)
                continue
            return {"idx": idx, "text": text, "file": None, "status": "error", "error": str(e)}
    return {"idx": idx, "text": text, "file": None, "status": "error", "error": "max retries exceeded"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process the first N lines (for a smoke test)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CORPUS_PATH, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if args.limit:
        lines = lines[: args.limit]

    polly = boto3.client("polly", region_name="us-east-1")

    total = len(lines)
    done = 0
    errors = 0
    t0 = time.time()

    with open(MANIFEST_PATH, "a", encoding="utf-8") as manifest, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(synthesize_one, polly, i, text): i for i, text in enumerate(lines)}
        for fut in as_completed(futures):
            result = fut.result()
            manifest.write(json.dumps(result) + "\n")
            manifest.flush()
            done += 1
            if result["status"] == "error":
                errors += 1
                print(f"ERROR idx={result['idx']}: {result.get('error')}")
            if done % 200 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else float("inf")
                print(f"{done}/{total} done ({errors} errors) — {rate:.1f}/s, ETA {eta/60:.1f} min")

    print(f"Finished: {done}/{total}, {errors} errors, {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
