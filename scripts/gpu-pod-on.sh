#!/usr/bin/env bash
# Turn on the GPU Pod (real cost: ~$0.74/hr, ~$533/mo if left running) and point the
# live Fly gateway at it. Only run this when there's actual traffic to serve — see
# README.md "Current real status". Pairs with gpu-pod-off.sh.
set -euo pipefail

RUNPOD_KEY=$(security find-generic-password -a "$(whoami)" -s runpod-api-key -w)
TEMPLATE_ID="r7cttxeun9"

echo "Creating GPU pod..."
CREATE=$(curl -s -X POST "https://rest.runpod.io/v1/pods" -H "Authorization: Bearer $RUNPOD_KEY" \
  -H "Content-Type: application/json" -d "{
    \"name\": \"realtime-tts-pod\",
    \"imageName\": \"ghcr.io/tsushanth/realtime-tts-worker:pod\",
    \"gpuTypeIds\": [\"NVIDIA L4\", \"NVIDIA A40\", \"NVIDIA GeForce RTX 4090\"],
    \"gpuCount\": 1, \"containerDiskInGb\": 10,
    \"env\": {\"WORKER_HOST\": \"0.0.0.0\", \"WORKER_PORT\": \"8765\"},
    \"ports\": [\"8765/tcp\", \"22/tcp\"], \"cloudType\": \"SECURE\", \"supportPublicIp\": true
  }")
POD_ID=$(echo "$CREATE" | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
echo "Pod created: $POD_ID (this can take 1-5 minutes to boot, RunPod scheduling varies)"

echo "Waiting for runtime info..."
for i in $(seq 1 40); do
  RUNTIME=$(curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"query { pod(input: {podId: \\\"$POD_ID\\\"}) { runtime { ports { ip isIpPublic privatePort publicPort type } } } }\"}")
  if echo "$RUNTIME" | grep -q '"runtime":{"'; then
    break
  fi
  sleep 10
done

# Match on privatePort==8765 explicitly, not array position — position isn't a
# documented guarantee even though it happened to be stable during testing.
IP_PORT=$(echo "$RUNTIME" | python3 -c "
import json,sys
d = json.load(sys.stdin)['data']['pod']['runtime']['ports']
p = [x for x in d if x['type']=='tcp' and x['isIpPublic'] and x['privatePort']==8765][0]
print(p['ip'], p['publicPort'])
" 2>/dev/null) || { echo "FAILED: pod never reported runtime, or no 8765/tcp mapping found. Check RunPod console. Pod ID: $POD_ID"; exit 1; }
IP=$(echo "$IP_PORT" | cut -d' ' -f1)
PORT=$(echo "$IP_PORT" | cut -d' ' -f2)

echo "Pod reachable at $IP:$PORT"
echo "Wiring gateway..."
flyctl secrets set -a realtime-tts-gateway WORKER_WS_URL="ws://$IP:$PORT"
flyctl secrets unset -a realtime-tts-gateway RUNPOD_ENDPOINT_ID RUNPOD_API_KEY

sleep 3
echo "Gateway health:"
curl -s https://realtime-tts-gateway.fly.dev/health; echo
echo ""
echo "Pod ID for teardown: $POD_ID (save this, or use gpu-pod-off.sh which looks it up)"
echo "$POD_ID" > /tmp/realtime-tts-active-pod-id
