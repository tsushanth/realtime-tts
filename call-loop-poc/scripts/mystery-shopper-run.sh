#!/usr/bin/env bash
# Mystery-shopper full A/B cycle, fully automated (see MYSTERY_SHOPPER_DECISIONS.md).
# Swaps the shared Retell number to the benchmark agent, places both shopper
# calls, waits for completion, ALWAYS restores the original number even on
# failure (trap on EXIT), pulls both transcripts AND both Twilio recordings,
# derives objective audio latency metrics with analyze-call-ttfb.py, hands
# those to the blind judge alongside the transcripts, and prints a summary.
#
# The number swap happens ONCE at the start (shared across all rounds) and is
# restored on any exit -- see MYSTERY_SHOPPER_DECISIONS.md. Running with
# --rounds N repeats the call+judge cycle N times so win rates and latency
# spread become statistically meaningful.
#
# Requires: TEST_CALL_SECRET (call-loop-poc's own), RETELL_API_KEY,
# TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN in env or sourced from a local file.
set -euo pipefail

CALL_LOOP_URL="https://call-loop-poc.fly.dev"
OUR_NUMBER="+12245061194"
RETELL_NUMBER="+16506755852"
RETELL_BENCHMARK_AGENT="agent_0e5e7627ca1e12392f39457013"
AUDEXA_DJ_AGENT="agent_50299fb2ef4f6de4a60de2faf7"
AUDEXA_DJ_VERSION="3"

: "${TEST_CALL_SECRET:?set TEST_CALL_SECRET}"
: "${RETELL_API_KEY:?set RETELL_API_KEY}"
: "${TWILIO_ACCOUNT_SID:?set TWILIO_ACCOUNT_SID}"
: "${TWILIO_AUTH_TOKEN:?set TWILIO_AUTH_TOKEN}"

ROUNDS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --rounds)
      ROUNDS="${2:?--rounds needs a number}"; shift 2 ;;
    *)
      echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(mktemp -d /tmp/mystery-shopper.XXXX)"
echo "[mystery-shopper] run dir: $RUN_DIR"

restore_number() {
  echo "[mystery-shopper] restoring $RETELL_NUMBER to Audexa DJ (agent_id=$AUDEXA_DJ_AGENT)"
  curl -s -X PATCH "https://api.retellai.com/update-phone-number/%2B16506755852" \
    -H "Content-Type: application/json" -H "Authorization: Bearer $RETELL_API_KEY" \
    -d "{\"inbound_agents\":[{\"agent_id\":\"$AUDEXA_DJ_AGENT\",\"weight\":1,\"agent_version\":$AUDEXA_DJ_VERSION}]}" \
    > /dev/null
}
trap restore_number EXIT

echo "[mystery-shopper] pointing $RETELL_NUMBER at benchmark agent ($RETELL_BENCHMARK_AGENT)"
curl -s -X PATCH "https://api.retellai.com/update-phone-number/%2B16506755852" \
  -H "Content-Type: application/json" -H "Authorization: Bearer $RETELL_API_KEY" \
  -d "{\"inbound_agents\":[{\"agent_id\":\"$RETELL_BENCHMARK_AGENT\",\"weight\":1}]}" \
  > /dev/null

# --- helpers ---------------------------------------------------------------

tw() { curl -s -u "$TWILIO_ACCOUNT_SID:$TWILIO_AUTH_TOKEN" "$@"; }

# Poll Twilio until the given call has a finalized recording and echo its
# RecordingSid. Recordings lag call completion by seconds-to-minutes.
wait_for_recording() {
  local sid="$1" recs st rsid=""
  for i in $(seq 1 24); do
    recs=$(tw "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Recordings.json?CallSid=$sid")
    rsid=$(echo "$recs" | python3 -c "import json,sys; d=json.load(sys.stdin)['recordings']; print(d[0]['sid'] if d else '')" 2>/dev/null || true)
    if [ -n "$rsid" ]; then
      st=$(tw "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Recordings/$rsid.json" \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('status'))" 2>/dev/null || true)
      if [ "$st" = "completed" ] || [ "$st" = "processing-complete" ]; then
        echo "$rsid"; return 0
      fi
    fi
    sleep 5
  done
  echo ""; return 1
}

# Resolve the business-flow inbound CallSid for the leg placed by OUR_SID
# (a locally-originated shopper call answered by our own app shows up twice
# in Twilio: the outbound-api leg == OUR_SID, then a separate inbound leg
# tagged with the business session). Echo the inbound sid if found.
resolve_business_sid() {
  local ours_sid="$1"
  local since since_enc bounds call sid
  since=$(date -u -v-10M +"%Y-%m-%d %H:%M:%S")
  since_enc=$(python3 -c "import urllib.parse;print(urllib.parse.quote('$since'))")
  bounds=$(tw "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Calls.json?To=$(python3 -c "import urllib.parse;print(urllib.parse.quote('$OUR_NUMBER'))")&Direction=inbound&StartTime%3E=$since_enc&PageSize=100")
  echo "$bounds" | python3 -c "
import json,sys
try:
    calls = json.load(sys.stdin)['calls']
except Exception:
    sys.exit(0)
for c in sorted(calls, key=lambda c: c.get('date_created') or '', reverse=True):
    if c['sid'] != '$ours_sid':
        print(c['sid'])
        break
"
}

# Grab [latency] lines (server.js's per-user-turn TTFB log) for a business
# session from the live Fly console log stream.
latency_lines() {
  flyctl logs -a call-loop-poc --no-tail 2>&1 | grep "\[call $1\]" | grep -oE '\[latency\].*' || true
}

echo "[mystery-shopper] running $ROUNDS round(s)"
for r in $(seq 1 "$ROUNDS"); do
  echo "=== round $r ==="

  WINDOW_START=$(($(date +%s) * 1000))

  OURS_SID=$(curl -s -X POST "$CALL_LOOP_URL/place-test-call" \
    -H "Authorization: Bearer $TEST_CALL_SECRET" -H "Content-Type: application/json" \
    -d "{\"toNumber\":\"$OUR_NUMBER\",\"shopper\":true,\"record\":true}" | python3 -c "import json,sys; print(json.load(sys.stdin)['sid'])")

  RETELL_SID=$(curl -s -X POST "$CALL_LOOP_URL/place-test-call" \
    -H "Authorization: Bearer $TEST_CALL_SECRET" -H "Content-Type: application/json" \
    -d "{\"toNumber\":\"$RETELL_NUMBER\",\"shopper\":true,\"record\":true}" | python3 -c "import json,sys; print(json.load(sys.stdin)['sid'])")

  # The Twilio number the shopper dials out from -- used to single out our
  # Retell call later.
  FROM_NUMBER=$(tw "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Calls/$OURS_SID.json" \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('from'))")
  echo "[mystery-shopper] round $r ours=$OURS_SID retell=$RETELL_SID shopper-from=$FROM_NUMBER"

  for sid in "$OURS_SID" "$RETELL_SID"; do
    while true; do
      st=$(tw "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Calls/$sid.json" \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('status'))")
      case "$st" in
        completed|failed|busy|no-answer|canceled) break ;;
      esac
      sleep 15
    done
    echo "[mystery-shopper] $sid -> $st"
  done

  WINDOW_END=$(($(date +%s) * 1000))

  # --- transcripts ---

  echo "[mystery-shopper] pulling Retell transcript"
  RETELL_CALL=$(curl -s -X POST https://api.retellai.com/v2/list-calls \
    -H "Content-Type: application/json" -H "Authorization: Bearer $RETELL_API_KEY" \
    -d "{\"filter_criteria\":{\"agent_id\":[\"$RETELL_BENCHMARK_AGENT\"],\"from_number\":[\"$FROM_NUMBER\"],\"start_timestamp\":{\"lower_threshold\":$WINDOW_START,\"upper_threshold\":$WINDOW_END}},\"sort_order\":\"descending\",\"limit\":1}")
  echo "$RETELL_CALL" | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)[0]
except Exception:
    sys.exit(1)
t = d['transcript'].replace('Agent:', 'Business:').replace('User:', 'Customer:')
print(t)
" > "$RUN_DIR/round-$r-retell.txt"

  echo "[mystery-shopper] pulling 'ours' transcript (tagged by CallSid $OURS_SID)"
  # Tagged session pull -- see server.js's [call <CallSid>] log tag. The
  # SHOPPER's session reverses roles vs the business session; keep untangled.
  flyctl logs -a call-loop-poc --no-tail 2>&1 \
    | grep "\[call $OURS_SID\]" \
    | grep -oE '(user|assistant): ".*"$' \
    | sed -E 's/^user:/Business:/; s/^assistant:/Customer:/' \
    > "$RUN_DIR/round-$r-ours.txt" || true

  # --- recordings + objective latency metrics ---

  OURS_REC=$(wait_for_recording "$OURS_SID") || OURS_REC=""
  RETELL_REC=$(wait_for_recording "$RETELL_SID") || RETELL_REC=""

  if [ -n "$OURS_REC" ]; then
    python3 "$SCRIPT_DIR/analyze-call-ttfb.py" --recording "$OURS_REC" --json \
      > "$RUN_DIR/round-$r-ours-metrics.json" \
      || echo "[mystery-shopper] analyzer failed for ours recording $OURS_REC" >&2
  else
    echo "[mystery-shopper] WARN: no finalized recording for ours call $OURS_SID" >&2
  fi
  if [ -n "$RETELL_REC" ]; then
    python3 "$SCRIPT_DIR/analyze-call-ttfb.py" --recording "$RETELL_REC" --json \
      > "$RUN_DIR/round-$r-retell-metrics.json" \
      || echo "[mystery-shopper] analyzer failed for retell recording $RETELL_REC" >&2
  else
    echo "[mystery-shopper] WARN: no finalized recording for retell call $RETELL_SID" >&2
  fi

  # --- business-flow server-side latency (ours only) ---

  BIZ_SID=$(resolve_business_sid "$OURS_SID")
  if [ -n "$BIZ_SID" ]; then
    latency_lines "$BIZ_SID" > "$RUN_DIR/round-$r-biz-latency.txt"
    echo "[mystery-shopper] business inbound sid: $BIZ_SID ($(wc -l < "$RUN_DIR/round-$r-biz-latency.txt" | tr -d ' ') [latency] lines)"
  else
    echo "[mystery-shopper] WARN: could not resolve business inbound call for $OURS_SID" >&2
  fi

  # --- blind judge ---

  if [ ! -s "$RUN_DIR/round-$r-ours.txt" ]; then
    echo "[mystery-shopper] ERROR: ours transcript empty for round $r" >&2
    continue
  fi
  if [ ! -s "$RUN_DIR/round-$r-retell.txt" ]; then
    echo "[mystery-shopper] ERROR: retell transcript empty for round $r" >&2
    continue
  fi

  echo "[mystery-shopper] running blind judge"
  node "$SCRIPT_DIR/mystery-shopper-judge.mjs" \
    --ours "$RUN_DIR/round-$r-ours.txt" \
    --retell "$RUN_DIR/round-$r-retell.txt" \
    --ours-metrics "$RUN_DIR/round-$r-ours-metrics.json" \
    --retell-metrics "$RUN_DIR/round-$r-retell-metrics.json" \
    > "$RUN_DIR/round-$r-verdict.txt" 2>&1

  WINNER=$(grep -E '^WINNER_SYSTEM:' "$RUN_DIR/round-$r-verdict.txt" | tail -1 | awk '{print $2}' || true)
  echo "[mystery-shopper] round $r winner: ${WINNER:-n/a}"
done

# --- summary ---

echo
echo "=================================================================="
echo " MYSTERY-SHOPPER SUMMARY ($ROUNDS round(s))"
echo "=================================================================="
printf "%-5s %-8s %-16s %-16s %-16s\n" "round" "winner" "ours p50/p95" "retell p50/p95" "ours llm/ttfb"
for r in $(seq 1 "$ROUNDS"); do
  ours_w=$(python3 -c "
import json
try:
    d=json.load(open('$RUN_DIR/round-$r-ours-metrics.json'))
    l=d['response_latency_ms']
    print(f\"{l['median_ms']}/{l['p95_ms']}\" if l.get('n') else 'n/a')
except Exception:
    print('n/a')")
  retell_w=$(python3 -c "
import json
try:
    d=json.load(open('$RUN_DIR/round-$r-retell-metrics.json'))
    l=d['response_latency_ms']
    print(f\"{l['median_ms']}/{l['p95_ms']}\" if l.get('n') else 'n/a')
except Exception:
    print('n/a')")
  winner=$(grep -E '^WINNER_SYSTEM:' "$RUN_DIR/round-$r-verdict.txt" 2>/dev/null | tail -1 | awk '{print $2}' || true)
  winner=${winner:-n/a}
  printf "%-5s %-8s %-16s %-16s\n" "$r" "$winner" "$ours_w" "$retell_w"
  if [ -s "$RUN_DIR/round-$r-biz-latency.txt" ]; then
    echo "  -> biz server latency (round $r):"
    sed 's/^/     /' "$RUN_DIR/round-$r-biz-latency.txt"
  fi
done
echo "=================================================================="
echo " reports: $RUN_DIR"
echo " restore note: number $RETELL_NUMBER is restored by the EXIT trap."