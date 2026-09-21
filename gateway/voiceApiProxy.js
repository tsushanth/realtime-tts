// API-key front door for customer voice cloning. Design: this gateway already owns API-key validation
// and billing state (keys.js); the ReadAloudAI backend already owns the voice-studio business logic,
// consent records and intake calls (backend/src/routes/voiceStudioRouter.ts). Duplicating either side
// would be the real drift risk, so this module does the minimum: validate the key here (same checks as
// /tts/authorize), then forward the request byte-for-byte to the backend's API-key voice-studio router,
// which runs the exact same createVoiceStudioRouter logic the web flow uses. See DECISIONS.md.
//
// Routes handled (all require `Authorization: Bearer <api key>`):
//   POST   /v1/voices                          create (consent body, see VOICE_API_DRAFT.md)
//   GET    /v1/voices                           list my voices
//   GET    /v1/voices/:id                        status
//   PUT    /v1/voices/:id/dataset                single-shot zip upload
//   PUT    /v1/voices/:id/dataset/parts/:n       chunked part upload
//   GET    /v1/voices/:id/dataset/parts          list uploaded parts
//   POST   /v1/voices/:id/dataset/commit         join parts + start training
//   GET    /v1/voices/:id/samples/:n             fixed preview sample
//   POST   /v1/voices/:id/preview                custom text preview
//   POST   /v1/voices/:id/deploy                 deploy -> {voice: "custom:<id>"}
//   DELETE /v1/voices/:id                        revoke
import http from "node:http";
import https from "node:https";
import { URL } from "node:url";
import * as keys from "./keys.js";

const BACKEND_URL = process.env.READALOUD_BACKEND_URL; // e.g. https://api-internal.readaloudai.org
const GATEWAY_FORWARD_SECRET = process.env.GATEWAY_FORWARD_SECRET;
const PREFIX = "/v1/voices";

export function voiceApiConfigured() {
  return !!(BACKEND_URL && GATEWAY_FORWARD_SECRET);
}

function readRawBody(req, maxBytes) {
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

// Large enough for a single-shot small dataset zip; big uploads should use the chunked parts route
// (<=16 MB/part), same guidance as the web client and documented in VOICE_API_DRAFT.md.
const MAX_BODY_BYTES = 32 * 1024 * 1024;

function forward(backendPath, method, headers, body) {
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

// Handles one request if it's under /v1/voices; returns true if handled (caller should not continue
// routing), false otherwise.
export async function handleVoiceApi(req, res, url) {
  if (!url.pathname.startsWith(PREFIX)) return false;

  if (!voiceApiConfigured()) {
    res.writeHead(501, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "voice cloning API not configured on this gateway" }));
    return true;
  }

  const auth = req.headers["authorization"] || "";
  const key = auth.startsWith("Bearer ") ? auth.slice(7) : undefined;
  if (!keys.isValidKey(key)) {
    res.writeHead(401, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "invalid or missing API key" }));
    return true;
  }
  // Safety default (not the owner's final pricing decision - see VOICE_API_DRAFT.md): voice training
  // spends real GPU money per job, so only billing-enabled keys may use this surface at all. A free-tier
  // key gets a clear, actionable error rather than a silent 404 or a vague 401.
  if (!keys.isBillingEnabled(key)) {
    res.writeHead(402, { "content-type": "application/json" });
    res.end(JSON.stringify({
      error: "Voice cloning requires a billing-enabled API key (training spends real GPU cost per voice). Enable billing on this key in your dashboard.",
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
    res.end(JSON.stringify({ error: e.statusCode === 413 ? "request body too large; use the chunked parts upload for large datasets" : "bad request" }));
    return true;
  }

  const backendPath = "/internal/voice-studio-api" + url.pathname.slice(PREFIX.length) + url.search;
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
