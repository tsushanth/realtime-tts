#!/usr/bin/env bash
# Mystery-shopper full A/B cycle, fully automated (see MYSTERY_SHOPPER_DECISIONS.md).
# Swaps the shared Retell number to the benchmark agent, places both shopper
# calls, waits for completion, ALWAYS restores the original number even on
# failure (trap on EXIT), pulls both transcripts, and runs the blind judge.
#
# Explicit user authorization on record for the swap-reuse-restore pattern
# running unattended going forward (no more per-run confirmation) -- see
# conversation history around 2026-09-08 ("it should run whenever.. reuse
# then put it back").
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

restore_number() {
  echo "[mystery-shopper] restoring +16506755852 to Audexa DJ (agent_id=$AUDEXA_DJ_AGENT)"
  curl -s -X PATCH "https://api.retellai.com/update-phone-number/%2B16506755852" \
    -H "Content-Type: application/json" -H "Authorization: Bearer $RETELL_API_KEY" \
    -d "{\"inbound_agents\":[{\"agent_id\":\"$AUDEXA_DJ_AGENT\",\"weight\":1,\"agent_version\":$AUDEXA_DJ_VERSION}]}" \
    > /dev/null
}
# Runs on ANY exit (success, error, or interrupt) -- the whole point of the
# trap is that a mid-script failure must never leave Audexa DJ's live number
# pointed at our benchmark agent.
trap restore_number EXIT

echo "[mystery-shopper] pointing +16506755852 at the benchmark agent"
curl -s -X PATCH "https://api.retellai.com/update-phone-number/%2B16506755852" \
  -H "Content-Type: application/json" -H "Authorization: Bearer $RETELL_API_KEY" \
  -d "{\"inbound_agents\":[{\"agent_id\":\"$RETELL_BENCHMARK_AGENT\",\"weight\":1}]}" \
  > /dev/null

echo "[mystery-shopper] placing shopper -> ours"
OURS_SID=$(curl -s -X POST "$CALL_LOOP_URL/place-test-call" \
  -H "Authorization: Bearer $TEST_CALL_SECRET" -H "Content-Type: application/json" \
  -d "{\"toNumber\":\"$OUR_NUMBER\",\"shopper\":true,\"record\":true}" | python3 -c "import json,sys; print(json.load(sys.stdin)['sid'])")

echo "[mystery-shopper] placing shopper -> retell"
RETELL_SID=$(curl -s -X POST "$CALL_LOOP_URL/place-test-call" \
  -H "Authorization: Bearer $TEST_CALL_SECRET" -H "Content-Type: application/json" \
  -d "{\"toNumber\":\"$RETELL_NUMBER\",\"shopper\":true,\"record\":true}" | python3 -c "import json,sys; print(json.load(sys.stdin)['sid'])")

echo "[mystery-shopper] ours=$OURS_SID retell=$RETELL_SID -- waiting for both to complete"
for sid in "$OURS_SID" "$RETELL_SID"; do
  while true; do
    st=$(curl -s -u "$TWILIO_ACCOUNT_SID:$TWILIO_AUTH_TOKEN" \
      "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Calls/$sid.json" \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('status'))")
    case "$st" in
      completed|failed|busy|no-answer|canceled) break ;;
    esac
    sleep 15
  done
  echo "[mystery-shopper] $sid -> $st"
done

echo "[mystery-shopper] pulling Retell transcript"
RETELL_CALL=$(curl -s -X POST https://api.retellai.com/v2/list-calls \
  -H "Content-Type: application/json" -H "Authorization: Bearer $RETELL_API_KEY" \
  -d "{\"filter_criteria\":{\"agent_id\":[\"$RETELL_BENCHMARK_AGENT\"]},\"sort_order\":\"descending\",\"limit\":1}")
echo "$RETELL_CALL" | python3 -c "
import json,sys
d = json.load(sys.stdin)[0]
t = d['transcript'].replace('Agent:', 'Business:').replace('User:', 'Customer:')
print(t)
" > /tmp/mystery-shopper-retell.txt

echo "[mystery-shopper] pulling 'ours' transcript (tagged by CallSid $OURS_SID)"
# Every turn log line now carries [call <CallSid>] (see server.js's
# adapter.on('start') handler) specifically so a concurrent run's two (or
# three, when a shopper call and its answering leg are both ours) sessions'
# interleaved turn counters can be pulled apart automatically instead of by
# hand -- the real gap found running this framework manually the first few
# times.
# Pulling the SHOPPER's own session (tagged by its own CallSid, $OURS_SID)
# -- in ITS session, "user" is whatever it heard (the business's voice) and
# "assistant" is its own reply, which is the reverse of the business-flow
# session's labeling. Don't swap in the business-flow session's CallSid here
# by mistake -- they're two different local sessions bridged by one PSTN call.
flyctl logs -a call-loop-poc --no-tail 2>&1 \
  | grep "\[call $OURS_SID\]" \
  | grep -oE '(user|assistant): ".*"$' \
  | sed -E 's/^user:/Business:/; s/^assistant:/Customer:/' \
  > /tmp/mystery-shopper-ours.txt

echo "[mystery-shopper] running blind judge"
node "$(dirname "$0")/mystery-shopper-judge.mjs" \
  --ours /tmp/mystery-shopper-ours.txt \
  --retell /tmp/mystery-shopper-retell.txt
