// API-key front door for Orpheus streaming voice cloning. Same shape as voiceApiProxy.js (this gateway
// owns API-key validation and billing state via keys.js; the ReadAloudAI backend owns the Orpheus
// training/synthesis proxying to the Modal service). Unlike voiceApiProxy.js this is a brand-new,
// isolated, paid-only feature end to end: no built-in-voice catalog, no chunked-upload-parts routes
// (single-shot zip upload only), and no free-tier discovery exemption — every route, including GET
// polling, requires a billing-enabled key.
//
// Handles two prefixes in one module:
//   /v1/orpheus-voices/...   training lifecycle (create, dataset upload, commit, status, delete)
//   /v1/orpheus-tts/...      synthesis
// Everything forwards byte-for-byte to the backend's /internal/orpheus-clone-api router.
import http from "node:http";
import https from "node:https";
import { URL } from "node:url";
import * as keys from "./keys.js";

const BACKEND_URL = process.env.READALOUD_BACKEND_URL;
const GATEWAY_FORWARD_SECRET = process.env.GATEWAY_FORWARD_SECRET;
const VOICES_PREFIX = "/v1/orpheus-voices";
const TTS_PREFIX = "/v1/orpheus-tts";

export function orpheusApiConfigured() {
  return !!(BACKEND_URL && GATEWAY_FORWARD_SECRET);
}

export function readRawBody(req, maxBytes) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > maxBytes) {
        reject(Object.assign(new Error("payload too large"), { statusCode: 413 }));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

// Large enough for a single-shot dataset zip (the documented 8-20 min recommendation can exceed
// 32MB at higher sample rates/bit depths); this feature has no chunked-parts route.
const MAX_BODY_BYTES = 100 * 1024 * 1024;

export function forward(backendPath, method, headers, body) {
  return new Promise((resolve, reject) => {
    const target = new URL(backendPath, BACKEND_URL);
    const mod = target.protocol === "https:" ? https : http;
    const upstream = mod.request(
      target,
      { method, headers },
      (upRes) => {
        const chunks = [];
        upRes.on("data", (c) => chunks.push(c));
        upRes.on("end", () => resolve({ status: upRes.statusCode, headers: upRes.headers, body: Buffer.concat(chunks) }));
      }
    );
    upstream.on("error", reject);
    if (body && body.length) upstream.write(body);
    upstream.end();
  });
}

// Handles one request if it's under /v1/orpheus-voices or /v1/orpheus-tts; returns true if handled
// (caller should not continue routing), false otherwise.
export async function handleOrpheusVoiceApi(req, res, url) {
  let prefix;
  // Exact match or a "/" boundary, so e.g. /v1/orpheus-voicesX is not treated as ours.
  const under = (p) => url.pathname === p || url.pathname.startsWith(p + "/");
  if (under(VOICES_PREFIX)) prefix = VOICES_PREFIX;
  else if (under(TTS_PREFIX)) prefix = TTS_PREFIX;
  else return false;

  if (!orpheusApiConfigured()) {
    res.writeHead(501, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "Orpheus voice cloning API not configured on this gateway" }));
    return true;
  }

  const auth = req.headers["authorization"] || "";
  const key = auth.startsWith("Bearer ") ? auth.slice(7) : undefined;
  if (!keys.isValidKey(key)) {
    res.writeHead(401, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "invalid or missing API key" }));
    return true;
  }

  // No free-tier discovery exemption for this feature: every route, including GET polling,
  // requires a billing-enabled key.
  if (!keys.isBillingEnabled(key)) {
    res.writeHead(402, { "content-type": "application/json" });
    res.end(JSON.stringify({
      error: "Orpheus voice cloning requires a billing-enabled API key (training/synthesis spend real GPU cost). Enable billing on this key in your dashboard.",
    }));
    return true;
  }

  const id = keys.getIdForKey(key);
  const uid = keys.getOwnerForKey(key) || "";

  let body;
  try {
    body = await readRawBody(req, MAX_BODY_BYTES);
  } catch (e) {
    res.writeHead(e.statusCode || 400, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: e.statusCode === 413 ? "request body too large" : "bad request" }));
    return true;
  }

  // TTS has no sub-resource path shape (unlike voices, which has /:id, /:id/dataset, etc.), so it
  // must be pinned to the backend's dedicated /tts route rather than the mount root — otherwise a
  // bare POST to either prefix collides on the same backendPath.
  const backendPath = prefix === TTS_PREFIX
    ? "/internal/orpheus-clone-api/tts" + url.search
    : "/internal/orpheus-clone-api" + url.pathname.slice(prefix.length) + url.search;
  let upstream;
  try {
    upstream = await forward(backendPath, req.method, {
      "content-type": req.headers["content-type"] || "application/octet-stream",
      "content-length": String(body.length),
      "x-gateway-admin-secret": GATEWAY_FORWARD_SECRET,
      "x-gateway-key-id": id || "",
      "x-gateway-uid": uid,
    }, body);
  } catch (e) {
    res.writeHead(502, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "voice service unavailable, please retry" }));
    return true;
  }

  res.writeHead(upstream.status, { "content-type": upstream.headers["content-type"] || "application/json" });
  res.end(upstream.body);
  return true;
}
