#!/usr/bin/env bash
# Continuous harness runner. Intended for cron / a scheduled Fly Machine / GitHub Actions
# cron. Runs the latency harness on an interval and appends a human-readable report.
#
# Usage: ENDPOINT=wss://realtime-tts-gateway.fly.dev/tts CONCURRENCY=8 INTERVAL=300 ./loop.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINT="${ENDPOINT:-ws://127.0.0.1:8765}"
CONCURRENCY="${CONCURRENCY:-4}"
INTERVAL="${INTERVAL:-300}"
REPORT="$DIR/last_report.txt"
PYTHON="${PYTHON:-$DIR/../worker/.venv/bin/python}"

run_once() {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) endpoint=$ENDPOINT concurrency=$CONCURRENCY ===" | tee "$REPORT"
    if "$PYTHON" "$DIR/run.py" --endpoint "$ENDPOINT" --concurrency "$CONCURRENCY" 2>&1 | tee -a "$REPORT"; then
        echo "harness: PASS" | tee -a "$REPORT"
    else
        echo "harness: FAIL (see above)" | tee -a "$REPORT"
    fi
}

if [ "${ONE_SHOT:-0}" = "1" ]; then
    run_once
    exit 0
fi

while true; do
    run_once
    sleep "$INTERVAL"
done
