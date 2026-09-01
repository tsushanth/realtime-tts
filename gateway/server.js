// Realtime TTS gateway: terminates client WebSockets on Fly, proxies each session 1:1 to
// a GPU inference worker. Three backend modes, selected by env vars:
//   GATEWAY_MODE=auto       -> pod-manager.js auto-provisions/tears-down a RunPod Pod on
//                              real traffic, requires a valid API key per connection
//   RUNPOD_ENDPOINT_ID set  -> RunPod Serverless (runpod-adapter.js), CPU-only fallback,
//                              requires a valid, billing-enabled API key per connection
//                              (same key store as AUTO_MODE)
//   otherwise                -> static WORKER_WS_URL (local dev against worker/server.py)
// See ../DECISIONS.md.
import { WebSocketServer, WebSocket } from "ws";
import http from "node:http";
import { URL } from "node:url";
import { runpodConfigured, handleClientOverRunpod } from "./runpod-adapter.js";
import * as podManager from "./pod-manager.js";
import * as keys from "./keys.js";

const PORT = process.env.PORT || 8080;
const WORKER_URL = process.env.WORKER_WS_URL || "ws://127.0.0.1:8765";
const AUTO_MODE = process.env.GATEWAY_MODE === "auto";
const USE_RUNPOD = !AUTO_MODE && runpodConfigured();
const ADMIN_SECRET = process.env.ADMIN_SECRET;

function requireAdmin(req, res) {
  const auth = req.headers["authorization"] || "";
  if (!ADMIN_SECRET || auth !== `Bearer ${ADMIN_SECRET}`) {
    res.writeHead(401, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "unauthorized" }));
    return false;
  }
  return true;
}

function readBody(req) {
  return new Promise((resolve) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => resolve(body));
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");

  if (url.pathname === "/health") {
    res.writeHead(200, { "content-type": "application/json" });
    if (AUTO_MODE) {
      res.end(JSON.stringify({ status: "ok", backend: "auto", ...podManager.getState() }));
    } else {
      res.end(JSON.stringify({
        status: "ok",
        backend: USE_RUNPOD ? "runpod" : "direct-ws",
        worker: USE_RUNPOD ? process.env.RUNPOD_ENDPOINT_ID : WORKER_URL,
      }));
    }
    return;
  }

  if (url.pathname === "/admin/keys" && req.method === "POST") {
    if (!requireAdmin(req, res)) return;
    const body = await readBody(req);
    const { label } = body ? JSON.parse(body) : {};
    const { id, key } = keys.issueKey(label);
    res.writeHead(200, { "content-type": "application/json" });
    // `key` is the raw secret, returned ONLY here — callers must persist it
    // themselves (or discard it and let the end user see it once); `id` is safe
    // to store long-term and is what DELETE takes for revocation.
    res.end(JSON.stringify({ id, key }));
    return;
  }

  if (url.pathname === "/admin/keys" && req.method === "GET") {
    if (!requireAdmin(req, res)) return;
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(keys.listKeys()));
    return;
  }

  if (url.pathname === "/admin/keys" && req.method === "DELETE") {
    if (!requireAdmin(req, res)) return;
    const body = await readBody(req);
    const { id } = body ? JSON.parse(body) : {};
    const ok = keys.revokeKeyById(id);
    res.writeHead(ok ? 200 : 404, { "content-type": "application/json" });
    res.end(JSON.stringify({ revoked: ok }));
    return;
  }

  if (url.pathname === "/admin/keys/billing" && req.method === "POST") {
    if (!requireAdmin(req, res)) return;
    const body = await readBody(req);
    const { id, enabled } = body ? JSON.parse(body) : {};
    const ok = keys.setBillingEnabledById(id, enabled);
    res.writeHead(ok ? 200 : 404, { "content-type": "application/json" });
    res.end(JSON.stringify({ billingEnabled: ok ? !!enabled : undefined }));
    return;
  }

  // Called by the backend's usage-reporting job. Returns accumulated character usage
  // per key since the last call and resets the counters — the backend is responsible
  // for translating this into Stripe usage records, so a failed report on the backend
  // side would lose that batch. Acceptable for now at this volume; revisit with a
  // durable outbox if usage volume/reliability requirements grow.
  if (url.pathname === "/admin/usage/drain" && req.method === "POST") {
    if (!requireAdmin(req, res)) return;
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(keys.drainUsage()));
    return;
  }

  res.writeHead(404);
  res.end();
});

const wss = new WebSocketServer({ server, path: "/tts" });

function recordUsageFromFrame(key, data, isBinary) {
  if (!key || isBinary) return;
  try {
    const msg = JSON.parse(data.toString());
    if (msg.type === "synthesize" && typeof msg.text === "string") {
      keys.recordUsage(key, msg.text.length);
    }
  } catch {
    // not JSON / not a synthesize frame — nothing to meter
  }
}

function proxyToWorker(client, workerUrl, bufferedFrames = [], key = null) {
  const worker = new WebSocket(workerUrl);
  let workerOpen = false;
  // Frames the client already sent while we were waiting on the pod (see the
  // bufferListener below) — replayed here so the caller's very first message
  // (typically sent immediately on open, before any "ready" handshake) isn't
  // silently lost. No `message` listener was attached to `client` until now,
  // so anything sent earlier had nowhere to go.
  const pending = bufferedFrames.map((f) => f.data);
  for (const f of bufferedFrames) recordUsageFromFrame(key, f.data, f.isBinary);

  worker.on("open", () => {
    workerOpen = true;
    for (const frame of pending.splice(0)) worker.send(frame);
  });
  worker.on("message", (data, isBinary) => {
    if (client.readyState === WebSocket.OPEN) client.send(data, { binary: isBinary });
  });
  worker.on("error", (err) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ type: "error", message: `worker: ${err.message}` }));
    }
  });
  worker.on("close", () => {
    if (client.readyState === WebSocket.OPEN) client.close();
  });
  client.on("message", (data, isBinary) => {
    recordUsageFromFrame(key, data, isBinary);
    if (workerOpen) worker.send(data, { binary: isBinary });
    else pending.push(data);
  });
  client.on("close", () => {
    if (worker.readyState === WebSocket.OPEN || worker.readyState === WebSocket.CONNECTING) {
      worker.close();
    }
  });
  client.on("error", () => worker.close());
}

wss.on("connection", async (client, req) => {
  if (AUTO_MODE) {
    const url = new URL(req.url, "http://x");
    const headerKey = (req.headers["authorization"] || "").replace(/^Bearer /, "");
    const key = headerKey || url.searchParams.get("key");
    if (!keys.isValidKey(key)) {
      client.send(JSON.stringify({ type: "error", message: "invalid or missing API key" }));
      client.close();
      return;
    }

    if (!keys.isBillingEnabled(key)) {
      client.send(JSON.stringify({
        type: "error",
        message: "This key has no payment method on file. Add one in your dashboard to use the API.",
      }));
      client.close();
      return;
    }

    const status = podManager.touchAndGetStatus();
    if (status.state === "ready") {
      proxyToWorker(client, status.workerUrl, [], key);
      return;
    }

    // Upfront about latency, not silent: first request after idle can take minutes
    // while the GPU pod boots (observed 90s-280s in testing) — tell the client instead
    // of just hanging.
    client.send(JSON.stringify({
      type: "status",
      state: "provisioning",
      message: "GPU worker is starting — this can take up to 5 minutes on the first request after idle. Please wait.",
    }));
    // A client that sends its synthesize request immediately on open (the
    // documented, expected pattern — e.g. call-loop-poc never waits for a
    // handshake) would otherwise have that message dropped: nothing is
    // listening for it until proxyToWorker() attaches its own listener after
    // the pod is ready. Buffer here, replay there.
    const bufferedFrames = [];
    const bufferListener = (data, isBinary) => bufferedFrames.push({ data, isBinary });
    client.on("message", bufferListener);
    try {
      const ready = await podManager.waitUntilReady();
      if (ready.state !== "ready") throw new Error("pod failed to become ready");
      client.removeListener("message", bufferListener);
      client.send(JSON.stringify({ type: "status", state: "ready" }));
      proxyToWorker(client, ready.workerUrl, bufferedFrames, key);
    } catch (err) {
      client.removeListener("message", bufferListener);
      client.send(JSON.stringify({ type: "error", message: `provisioning failed: ${err.message}` }));
      client.close();
    }
    return;
  }

  if (USE_RUNPOD) {
    const url = new URL(req.url, "http://x");
    const headerKey = (req.headers["authorization"] || "").replace(/^Bearer /, "");
    const key = headerKey || url.searchParams.get("key");
    if (!keys.isValidKey(key)) {
      client.send(JSON.stringify({ type: "error", message: "invalid or missing API key" }));
      client.close();
      return;
    }
    if (!keys.isBillingEnabled(key)) {
      client.send(JSON.stringify({
        type: "error",
        message: "This key has no payment method on file. Add one in your dashboard to use the API.",
      }));
      client.close();
      return;
    }
    handleClientOverRunpod(client, (chars) => keys.recordUsage(key, chars));
    return;
  }

  proxyToWorker(client, WORKER_URL);
});

server.listen(PORT, async () => {
  const target = AUTO_MODE ? "auto (pod-manager)" : USE_RUNPOD ? `runpod:${process.env.RUNPOD_ENDPOINT_ID}` : WORKER_URL;
  console.log(`gateway listening on :${PORT}, backend=${target}`);
  if (AUTO_MODE) {
    await podManager.reconcileOnStartup();
  }
});
