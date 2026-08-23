#!/usr/bin/env bash
# Turn off the GPU Pod and revert the gateway to the cheap CPU Serverless fallback.
# Pairs with gpu-pod-on.sh. Run this once traffic stops to avoid paying ~$0.74/hr idle.
set -euo pipefail

RUNPOD_KEY=$(security find-generic-password -a "$(whoami)" -s runpod-api-key -w)
SERVERLESS_ENDPOINT_ID="u4me5box1h735i"

POD_ID="${1:-}"
if [ -z "$POD_ID" ] && [ -f /tmp/realtime-tts-active-pod-id ]; then
  POD_ID=$(cat /tmp/realtime-tts-active-pod-id)
fi
if [ -z "$POD_ID" ]; then
  echo "No pod ID given and none cached. Listing pods:"
  curl -s "https://rest.runpod.io/v1/pods" -H "Authorization: Bearer $RUNPOD_KEY" | python3 -m json.tool
  echo "Pass the pod ID explicitly: ./gpu-pod-off.sh <pod-id>"
  exit 1
fi

echo "Deleting pod $POD_ID..."
curl -s -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $RUNPOD_KEY" -w " -> %{http_code}\n"
rm -f /tmp/realtime-tts-active-pod-id

echo "Reverting gateway to CPU Serverless fallback..."
flyctl secrets set -a realtime-tts-gateway \
  RUNPOD_ENDPOINT_ID="$SERVERLESS_ENDPOINT_ID" \
  RUNPOD_API_KEY="$RUNPOD_KEY" \
  WORKER_WS_URL=ws://127.0.0.1:8765

sleep 3
echo "Gateway health:"
curl -s https://realtime-tts-gateway.fly.dev/health; echo
