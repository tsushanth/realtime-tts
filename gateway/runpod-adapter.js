// Adapts RunPod Serverless's HTTP job API (POST /run, poll /stream/{id}, POST /cancel/{id})
// to the same client-facing WS wire protocol worker/server.py speaks directly, so the
// harness and any client code work unmodified regardless of which backend is behind the
// gateway. See ../worker/handler.py for the RunPod-side handler this talks to.
const RUNPOD_API_BASE = "https://api.runpod.ai/v2";

export function runpodConfigured() {
  return !!(process.env.RUNPOD_ENDPOINT_ID && process.env.RUNPOD_API_KEY);
}

async function rpFetch(path, opts = {}) {
  const endpointId = process.env.RUNPOD_ENDPOINT_ID;
  const res = await fetch(`${RUNPOD_API_BASE}/${endpointId}${path}`, {
    ...opts,
    headers: {
      Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
      "Content-Type": "application/json",
      ...(opts.headers || {}),
    },
  });
  if (!res.ok) throw new Error(`RunPod ${path} -> ${res.status} ${await res.text()}`);
  return res.json();
}

// Handles one client WS connection end-to-end against a RunPod Serverless endpoint,
// translating streamed handler outputs into the same {type:"chunk_meta"} + binary-frame
// pairs worker/server.py sends, so client.js/harness code paths are identical either way.
export async function handleClientOverRunpod(client) {
  let cancelled = false;
  let jobId = null;

  client.on("message", async (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      client.send(JSON.stringify({ type: "error", message: "invalid JSON" }));
      return;
    }

    if (msg.type === "stop") {
      cancelled = true;
      if (jobId) rpFetch(`/cancel/${jobId}`, { method: "POST" }).catch(() => {});
      return;
    }

    if (msg.type !== "synthesize") return;
    cancelled = false;

    try {
      const started = await rpFetch("/run", {
        method: "POST",
        body: JSON.stringify({ input: { text: msg.text, voice: msg.voice, speed: msg.speed } }),
      });
      jobId = started.id;

      let seen = 0;
      while (!cancelled) {
        const status = await rpFetch(`/stream/${jobId}`);
        const outputs = status.stream || [];
        for (; seen < outputs.length; seen++) {
          const item = outputs[seen].output;
          if (!item) continue;
          client.send(JSON.stringify({
            type: "chunk_meta",
            text: item.text,
            gen_ms: item.gen_ms,
            audio_s: item.audio_s,
          }));
          client.send(Buffer.from(item.pcm16_b64, "base64"));
        }
        if (status.status === "COMPLETED") {
          client.send(JSON.stringify({ type: cancelled ? "cancelled" : "done" }));
          break;
        }
        if (status.status === "FAILED" || status.status === "CANCELLED") {
          client.send(JSON.stringify({ type: cancelled ? "cancelled" : "error", message: status.error || status.status }));
          break;
        }
        await new Promise((r) => setTimeout(r, 150));
      }
    } catch (err) {
      client.send(JSON.stringify({ type: "error", message: `runpod: ${err.message}` }));
    }
  });
}
