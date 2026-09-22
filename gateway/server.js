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
import https from "node:https";
import { URL } from "node:url";
import { runpodConfigured, handleClientOverRunpod } from "./runpod-adapter.js";
import * as keys from "./keys.js";
import { handleVoiceApi } from "./voiceApiProxy.js";
import { auditLog } from "./audit.js";

// Best-effort caller IP for audit records — trusts XFF from Fly's proxy in front of
// this gateway; fine for an audit trail (not used for any access-control decision).
function clientIp(req) {
  const xff = req.headers["x-forwarded-for"];
  if (typeof xff === "string" && xff.length) return xff.split(",")[0].trim();
  return req.socket?.remoteAddress || null;
}

const PORT = process.env.PORT || 8080;
const WORKER_URL = process.env.WORKER_WS_URL || "ws://127.0.0.1:8765";
const AUTO_MODE = process.env.GATEWAY_MODE === "auto";
const USE_RUNPOD = !AUTO_MODE && runpodConfigured();
const ADMIN_SECRET = process.env.ADMIN_SECRET;
const MODAL_WORKER_URL = process.env.MODAL_READALOUD_WS_URL;
// Optional CPU Piper engine (worker-piper-fly). Opt-in per request via {engine:"piper"}; unset => engine unavailable.
const PIPER_WORKER_URL = process.env.PIPER_WORKER_URL;
// Batch speech-to-text worker (worker-stt/, Modal app realtime-stt-worker). Unset => /stt/authorize returns 501.
const STT_WORKER_URL = process.env.STT_WORKER_URL;
// Streaming/realtime STT worker (worker-stt-realtime/, deployed separately from the batch worker - see DESIGN.md).
// Unset => /stt/authorize with mode:"realtime" returns 501, same as the batch worker being unconfigured.
const STT_REALTIME_WORKER_URL = process.env.STT_REALTIME_WORKER_URL;
// Audio isolation / denoising worker (worker-demucs-fly/, Demucs htdemucs). Unset => /audio/authorize returns 501,
// same "opt-in, absent-by-default" shape as PIPER_WORKER_URL/STT_WORKER_URL above.
const DEMUCS_WORKER_URL = process.env.DEMUCS_WORKER_URL;
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

// Best-effort wake-up ping for a scale-to-zero realtime STT worker: fired from /stt/authorize (mode:"realtime")
// so the Fly machine (or Modal container) is warming while the client sets up its mic/socket, instead of paying
// full cold start on the first WebSocket frame. Never blocks or fails the authorize response - errors/timeouts
// are swallowed; the worker's own WebSocket handler has a second safety net (buffers audio while loading).
function warmRealtimeWorker() {
  if (!STT_REALTIME_WORKER_URL) return;
  try {
    const u = new URL(STT_REALTIME_WORKER_URL);
    u.protocol = u.protocol === "wss:" ? "https:" : u.protocol === "ws:" ? "http:" : u.protocol;
    u.pathname = "/health";
    const mod = u.protocol === "https:" ? https : http;
    const r = mod.get(u, { timeout: 3000 }, (resp) => resp.resume());
    r.on("error", () => {});
    r.on("timeout", () => r.destroy());
  } catch { /* malformed STT_REALTIME_WORKER_URL: skip the ping */ }
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
    const { label, owner } = body ? JSON.parse(body) : {};
    const { id, key } = keys.issueKey(label, typeof owner === "string" ? owner : undefined);
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

  if (url.pathname === "/admin/keys/owner" && req.method === "POST") {
    if (!requireAdmin(req, res)) return;
    const body = await readBody(req);
    const { id, owner } = body ? JSON.parse(body) : {};
    const r = typeof owner === "string" ? keys.setOwnerById(id, owner) : null;
    res.writeHead(r === null ? 404 : r === "conflict" ? 409 : 200, { "content-type": "application/json" });
    res.end(JSON.stringify({ owner: r === "set" || r === "unchanged" ? owner : undefined, result: r }));
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
  // Customer voice cloning via API key (no browser/Supabase login) - validated here the same way
  // /tts/authorize validates a key, then forwarded to the ReadAloudAI backend's voice-studio logic.
  // See voiceApiProxy.js and ../VOICE_API_DRAFT.md.
  if (url.pathname.startsWith("/v1/voices")) {
    if (await handleVoiceApi(req, res, url)) return;
  }

  if (url.pathname === "/tts/authorize" && req.method === "POST") {
    const body = await readBody(req);
    const { key, engine } = body ? JSON.parse(body) : {};
    if (engine !== undefined && engine !== "kokoro" && engine !== "piper") {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: 'engine must be "kokoro" or "piper"' }));
      return;
    }
    if (engine === "piper" && !PIPER_WORKER_URL) {
      res.writeHead(501, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "piper engine not available" }));
      return;
    }
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
    const out = { token: keys.createSessionToken(id, keys.getOwnerForKey(key)), url: engine === "piper" ? PIPER_WORKER_URL : MODAL_WORKER_URL };
    if (engine === "piper") {
      // wss://host/tts -> https://host/v1/tts/stream (HTTP streaming endpoint on the same worker)
      try {
        const u = new URL(PIPER_WORKER_URL);
        u.protocol = u.protocol === "ws:" ? "http:" : "https:";
        out.http_url = `${u.protocol}//${u.host}/v1/tts/stream`;
      } catch { /* malformed PIPER_WORKER_URL: omit http_url */ }
    }
    res.end(JSON.stringify(out));
    return;
  }

  // Speech-to-text: same checks as /tts/authorize; the client then POSTs audio to `url` + "/v1/stt"
  // with the token as a Bearer header (batch), or opens a WebSocket at `url` (realtime, mode:"realtime").
  // Usage is reported by the worker to /admin/usage/report, tagged with engine "stt" (batch) or "stt-realtime".
  if (url.pathname === "/stt/authorize" && req.method === "POST") {
    let key, mode, engineReq;
    try { ({ key, mode, engine: engineReq } = JSON.parse((await readBody(req)) || "{}")); } catch { key = undefined; }
    const realtime = mode === "realtime" || engineReq === "realtime"; // `engine` accepted as an alias of `mode` for forward-compat
    const workerUrl = realtime ? STT_REALTIME_WORKER_URL : STT_WORKER_URL;
    if (!workerUrl) {
      res.writeHead(501, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: realtime ? "realtime speech-to-text not available" : "speech-to-text not available" }));
      return;
    }
    if (!keys.isValidKey(key)) {
      res.writeHead(401, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "invalid or missing API key" }));
      return;
    }
    if (!keys.checkAccess(key).allowed) {
      res.writeHead(402, { "content-type": "application/json" });
      res.end(JSON.stringify({
        error: "Free tier exhausted for this key. Add a payment method in your dashboard to continue.",
      }));
      return;
    }
    if (realtime) warmRealtimeWorker(); // fire-and-forget; never blocks or fails this response
    const sttId = keys.getIdForKey(key);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ token: keys.createSessionToken(sttId), url: workerUrl }));
    return;
  }

  // Audio isolation / denoising (worker-demucs-fly, Demucs htdemucs): same checks as
  // /tts/authorize and /stt/authorize. The client then POSTs the clip as multipart/form-data
  // to `url` + "/v1/isolate" with the token as a Bearer header. Usage is reported by the worker
  // to /admin/usage/report tagged with engine "denoise" (audio_seconds), once a per-second price
  // is set - see worker-demucs-fly/README.md "What's left".
  if (url.pathname === "/audio/authorize" && req.method === "POST") {
    const body = await readBody(req);
    const { key, engine } = body ? JSON.parse(body) : {};
    if (engine !== undefined && engine !== "denoise") {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: 'engine must be "denoise"' }));
      return;
    }
    if (!DEMUCS_WORKER_URL) {
      res.writeHead(501, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "audio isolation not available" }));
      return;
    }
    if (!keys.isValidKey(key)) {
      auditLog("auth_failure", { surface: "audio_authorize", reason: "invalid_key", ip: clientIp(req) });
      res.writeHead(401, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "invalid or missing API key" }));
      return;
    }
    const access = keys.checkAccess(key);
    if (!access.allowed) {
      auditLog("auth_failure", { surface: "audio_authorize", reason: "free_tier_exhausted", id: keys.getIdForKey(key), ip: clientIp(req) });
      res.writeHead(402, { "content-type": "application/json" });
      res.end(JSON.stringify({
        error: "Free tier exhausted for this key. Add a payment method in your dashboard to continue.",
      }));
      return;
    }
    const audioId = keys.getIdForKey(key);
    auditLog("auth_success", { surface: "audio_authorize", id: audioId, engine: "denoise", ip: clientIp(req) });
    res.writeHead(200, { "content-type": "application/json" });
    // wss-style url isn't applicable here (plain HTTP worker) - `url` is the base the client
    // POSTs `<url>/v1/isolate` to, exactly like the piper http_url shape in /tts/authorize.
    res.end(JSON.stringify({ token: keys.createSessionToken(audioId), url: DEMUCS_WORKER_URL }));
    return;
  }

  // Same key-validity + billing gate as /audio/authorize and /tts/authorize, but for
  // dubbing/job_server.py (Python, stdlib HTTP, no shared process with this gateway) to call
  // synchronously before it queues a job - see dubbing/gateway_auth.py and dubbing/README.md
  // "Consent / ownership gating". Unlike /tts/authorize and /audio/authorize this returns no
  // token/url: dubbing's actual TTS/STT calls happen from Python, not from a client the gateway
  // hands a URL to - this endpoint only answers "is this a real, billing-enabled key".
  if (url.pathname === "/dubbing/authorize" && req.method === "POST") {
    const body = await readBody(req);
    const { key } = body ? JSON.parse(body) : {};
    if (!keys.isValidKey(key)) {
      auditLog("auth_failure", { surface: "dubbing_authorize", reason: "invalid_key", ip: clientIp(req) });
      res.writeHead(401, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "invalid or missing API key" }));
      return;
    }
    const access = keys.checkAccess(key);
    if (!access.allowed) {
      auditLog("auth_failure", { surface: "dubbing_authorize", reason: "free_tier_exhausted", id: keys.getIdForKey(key), ip: clientIp(req) });
      res.writeHead(402, { "content-type": "application/json" });
      res.end(JSON.stringify({
        error: "Free tier exhausted for this key. Add a payment method in your dashboard to continue.",
      }));
      return;
    }
    const dubbingId = keys.getIdForKey(key);
    auditLog("auth_success", { surface: "dubbing_authorize", id: dubbingId, engine: "dubbing", ip: clientIp(req) });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ authorized: true, id: dubbingId }));
    return;
  }

  // Called by worker-modal-readaloud/app.py after a synthesize call completes —
  // fire-and-forget from Modal's side, doesn't block the client's response.
  // This is the authoritative usage number (what Modal actually generated),
  // not anything a client declared upfront.
  if (url.pathname === "/admin/usage/report" && req.method === "POST") {
    if (!requireUsageReportSecret(req, res)) return;
    const body = await readBody(req);
    const { id, chars, engine, audio_seconds } = body ? JSON.parse(body) : {};
    const ok = keys.recordUsageById(id, chars, engine, audio_seconds);
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
