// Realtime TTS gateway: terminates client WebSockets on Fly, proxies each session 1:1 to
// a GPU inference worker. Backend modes, selected by env vars:
//   GATEWAY_MODE=auto       -> proxies to the ReadAloud Modal deployment
//                              (worker-modal-readaloud/), requires a valid,
//                              billing-enabled API key per connection. Was RunPod
//                              Pod + pod-manager.js until 2026-09-06 — moved off it
//                              because a Pod's idle-teardown timer always loses money
//                              on sporadic traffic (see DECISIONS.md); Modal bills per
//                              actual connection duration instead, with no timer to
//                              tune or orphan-pod risk to reconcile.
//   RUNPOD_ENDPOINT_ID set  -> RunPod Serverless (runpod-adapter.js), CPU-only fallback,
//                              requires a valid, billing-enabled API key per connection
//                              (same key store as AUTO_MODE)
//   otherwise                -> static WORKER_WS_URL (local dev against worker/server.py)
// See ../DECISIONS.md.
import { WebSocketServer, WebSocket } from "ws";
import http from "node:http";
import { URL } from "node:url";
import { runpodConfigured, handleClientOverRunpod } from "./runpod-adapter.js";
import * as keys from "./keys.js";

const PORT = process.env.PORT || 8080;
const WORKER_URL = process.env.WORKER_WS_URL || "ws://127.0.0.1:8765";
const AUTO_MODE = process.env.GATEWAY_MODE === "auto";
const USE_RUNPOD = !AUTO_MODE && runpodConfigured();
const ADMIN_SECRET = process.env.ADMIN_SECRET;
const MODAL_WORKER_URL = process.env.MODAL_READALOUD_WS_URL;
const MODAL_AUTH_TOKEN = process.env.MODAL_READALOUD_AUTH_TOKEN;
// Separate from ADMIN_SECRET on purpose — this only lets the holder report
// usage numbers for a key it already has the ID for, not manage keys at all.
const MODAL_USAGE_REPORT_SECRET = process.env.MODAL_USAGE_REPORT_SECRET;

function requireAdmin(req, res) {
  const auth = req.headers["authorization"] || "";
  if (!ADMIN_SECRET || auth !== `Bearer ${ADMIN_SECRET}`) {
    res.writeHead(401, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "unauthorized" }));
    return false;
  }
  return true;
}

function requireUsageReportSecret(req, res) {
  const auth = req.headers["authorization"] || "";
  if (!MODAL_USAGE_REPORT_SECRET || auth !== `Bearer ${MODAL_USAGE_REPORT_SECRET}`) {
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
      res.end(JSON.stringify({ status: "ok", backend: "modal" }));
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

  // Fast path: authorize a key for a direct client<->Modal connection instead of
  // proxying bytes through this gateway (see DECISIONS.md — the relay adds
  // ~350-400ms of pure handshake overhead per session on top of Modal's own
  // latency). Only does the cheap checks (valid key, not revoked, free-tier
  // flag / billing enabled) — actual usage is metered by Modal reporting back
  // to /admin/usage/report after synthesis, not by this endpoint.
  if (url.pathname === "/tts/authorize" && req.method === "POST") {
    const body = await readBody(req);
    const { key } = body ? JSON.parse(body) : {};
    if (!keys.isValidKey(key)) {
      res.writeHead(401, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "invalid or missing API key" }));
      return;
    }
    const access = keys.checkAccess(key);
    if (!access.allowed) {
      res.writeHead(402, { "content-type": "application/json" });
      res.end(JSON.stringify({
        error: "Free tier exhausted for this key. Add a payment method in your dashboard to continue.",
      }));
      return;
    }
    const id = keys.getIdForKey(key);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ token: keys.createSessionToken(id), url: MODAL_WORKER_URL }));
    return;
  }

  // Called by worker-modal-readaloud/app.py after a synthesize call completes —
  // fire-and-forget from Modal's side, doesn't block the client's response.
  // This is the authoritative usage number (what Modal actually generated),
  // not anything a client declared upfront.
  if (url.pathname === "/admin/usage/report" && req.method === "POST") {
    if (!requireUsageReportSecret(req, res)) return;
    const body = await readBody(req);
    const { id, chars } = body ? JSON.parse(body) : {};
    const ok = keys.recordUsageById(id, chars);
    res.writeHead(ok ? 200 : 404, { "content-type": "application/json" });
    res.end(JSON.stringify({ recorded: ok }));
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

// Checks a free-tier key's remaining budget BEFORE forwarding a synthesize
// request (not after, and not by truncating mid-stream) — a billing-enabled
// key is always allowed through untouched. Returns { allow, deny } where
// `deny`, if present, is the error message to send the client instead of
// proxying the frame. Non-synthesize frames (e.g. "stop") and unparseable
// frames are always allowed through — there's nothing to meter or gate there.
function checkFrame(key, data, isBinary) {
  if (!key || isBinary) return { allow: true };
  let msg;
  try {
    msg = JSON.parse(data.toString());
  } catch {
    return { allow: true };
  }
  if (msg.type !== "synthesize" || typeof msg.text !== "string") return { allow: true };
  if (!keys.canAffordRequest(key, msg.text.length)) {
    return {
      allow: false,
      deny: "Free tier exhausted for this key. Add a payment method in your dashboard to continue.",
    };
  }
  keys.recordUsage(key, msg.text.length);
  return { allow: true };
}

function proxyToWorker(client, workerUrl, bufferedFrames = [], key = null, headers = undefined) {
  const worker = new WebSocket(workerUrl, headers ? { headers } : undefined);
  let workerOpen = false;
  // Frames the client already sent while we were waiting on the pod (see the
  // bufferListener below) — replayed here so the caller's very first message
  // (typically sent immediately on open, before any "ready" handshake) isn't
  // silently lost. No `message` listener was attached to `client` until now,
  // so anything sent earlier had nowhere to go.
  // Preserve isBinary through the buffer — ws.send() defaults an unmarked Buffer to a
  // BINARY frame, which silently broke this exact path: a JSON control message queued
  // here (because it arrived before the outbound worker connection finished its
  // handshake) would get relayed as binary instead of text. worker/server.py's Python
  // json.loads() tolerates that (it accepts bytes), which is why this went unnoticed
  // against the RunPod worker — but Modal's FastAPI worker calls ws.receive_text(),
  // which strictly rejects a binary frame. Found live via the Modal migration.
  const pending = [];
  for (const f of bufferedFrames) {
    const result = checkFrame(key, f.data, f.isBinary);
    if (!result.allow) {
      client.send(JSON.stringify({ type: "error", message: result.deny }));
      client.close();
      worker.close();
      return;
    }
    pending.push({ data: f.data, isBinary: f.isBinary });
  }

  worker.on("open", () => {
    workerOpen = true;
    for (const frame of pending.splice(0)) worker.send(frame.data, { binary: frame.isBinary });
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
    const result = checkFrame(key, data, isBinary);
    if (!result.allow) {
      client.send(JSON.stringify({ type: "error", message: result.deny }));
      return;
    }
    if (workerOpen) worker.send(data, { binary: isBinary });
    else pending.push({ data, isBinary });
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

    const access = keys.checkAccess(key);
    if (!access.allowed) {
      client.send(JSON.stringify({
        type: "error",
        message: "Free tier exhausted for this key. Add a payment method in your dashboard to continue.",
      }));
      client.close();
      return;
    }

    // No provisioning wait needed — Modal is always available; a cold container
    // just makes the first chunk of THIS request slower (measured ~4s for a cold
    // model load vs ~100-200ms warm), not a separate multi-minute wait before the
    // connection is even usable the way a RunPod Pod boot was.
    proxyToWorker(client, MODAL_WORKER_URL, [], key, { Authorization: `Bearer ${MODAL_AUTH_TOKEN}` });
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
    const access = keys.checkAccess(key);
    if (!access.allowed) {
      client.send(JSON.stringify({
        type: "error",
        message: "Free tier exhausted for this key. Add a payment method in your dashboard to continue.",
      }));
      client.close();
      return;
    }
    handleClientOverRunpod(client, (chars) => keys.recordUsage(key, chars));
    return;
  }

  proxyToWorker(client, WORKER_URL);
});

server.listen(PORT, () => {
  const target = AUTO_MODE ? `modal:${MODAL_WORKER_URL}` : USE_RUNPOD ? `runpod:${process.env.RUNPOD_ENDPOINT_ID}` : WORKER_URL;
  console.log(`gateway listening on :${PORT}, backend=${target}`);
});
