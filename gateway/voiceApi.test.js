// node --test : the /v1/voices API-key auth path (voiceApiProxy.js). Mocks the backend so no real
// intake/training happens - this only exercises key validation, billing gate, and header forwarding.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gw-voiceapi-"));
process.env.KEYS_PATH = path.join(dir, "keys.json");
process.env.MODAL_SESSION_SECRET = "test-session-secret";
process.env.FREE_TIER_CHARS = "10000";
process.env.GATEWAY_FORWARD_SECRET = "test-forward-secret";

// Fake backend: records the last request it saw and echoes a canned response.
let lastReq = null;
const backend = http.createServer((req, res) => {
  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    lastReq = { method: req.method, url: req.url, headers: req.headers, body: Buffer.concat(chunks).toString() };
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
  });
});
await new Promise((resolve) => backend.listen(0, resolve));
process.env.READALOUD_BACKEND_URL = `http://127.0.0.1:${backend.address().port}`;

const keys = await import("./keys.js");
const { handleVoiceApi } = await import("./voiceApiProxy.js");

function fakeReqRes(method, urlPath, { authorization, body = "" } = {}) {
  const chunks = body ? [Buffer.from(body)] : [];
  const req = {
    method,
    headers: authorization ? { authorization, "content-type": "application/json" } : { "content-type": "application/json" },
    on(evt, cb) {
      if (evt === "data") chunks.forEach((c) => cb(c));
      if (evt === "end") cb();
      return req;
    },
  };
  const res = {
    statusCode: null,
    headers: null,
    body: null,
    writeHead(code, headers) { this.statusCode = code; this.headers = headers; },
    end(b) { this.body = b; },
  };
  const url = new URL(urlPath, "http://x");
  return { req, res, url };
}

test("missing API key -> 401, backend never called", async () => {
  lastReq = null;
  const { req, res, url } = fakeReqRes("GET", "/v1/voices");
  const handled = await handleVoiceApi(req, res, url);
  assert.equal(handled, true);
  assert.equal(res.statusCode, 401);
  assert.equal(lastReq, null);
});

test("invalid API key -> 401", async () => {
  const { req, res, url } = fakeReqRes("GET", "/v1/voices", { authorization: "Bearer not-a-real-key" });
  await handleVoiceApi(req, res, url);
  assert.equal(res.statusCode, 401);
});

test("valid key without billing enabled -> 402, backend never called", async () => {
  lastReq = null;
  const { key } = keys.issueKey("free-tier-key");
  const { req, res, url } = fakeReqRes("GET", "/v1/voices", { authorization: `Bearer ${key}` });
  await handleVoiceApi(req, res, url);
  assert.equal(res.statusCode, 402);
  assert.match(JSON.parse(res.body).error, /billing-enabled/);
  assert.equal(lastReq, null);
});

test("billing-enabled key -> forwarded to backend with resolved identity headers", async () => {
  const { id, key } = keys.issueKey("paid-key");
  keys.setBillingEnabledById(id, true);
  keys.setOwnerById(id, "user-42");
  const { req, res, url } = fakeReqRes("POST", "/v1/voices", {
    authorization: `Bearer ${key}`,
    body: JSON.stringify({ speaker_name: "Test Speaker" }),
  });
  await handleVoiceApi(req, res, url);
  assert.equal(res.statusCode, 200);
  assert.equal(lastReq.method, "POST");
  assert.equal(lastReq.url, "/internal/voice-studio-api");
  assert.equal(lastReq.headers["x-gateway-admin-secret"], "test-forward-secret");
  assert.equal(lastReq.headers["x-gateway-key-id"], id);
  assert.equal(lastReq.headers["x-gateway-uid"], "user-42");
  assert.equal(JSON.parse(lastReq.body).speaker_name, "Test Speaker");
});

test("billing-enabled key with no bound owner -> uid header empty, key id still forwarded", async () => {
  const { id, key } = keys.issueKey("paid-key-no-owner");
  keys.setBillingEnabledById(id, true);
  const { req, res, url } = fakeReqRes("GET", "/v1/voices/v-deadbeef00", { authorization: `Bearer ${key}` });
  await handleVoiceApi(req, res, url);
  assert.equal(res.statusCode, 200);
  assert.equal(lastReq.headers["x-gateway-key-id"], id);
  assert.equal(lastReq.headers["x-gateway-uid"], "");
  assert.equal(lastReq.url, "/internal/voice-studio-api/v-deadbeef00");
});

test("non /v1/voices path is not handled", async () => {
  const { req, res, url } = fakeReqRes("GET", "/health");
  const handled = await handleVoiceApi(req, res, url);
  assert.equal(handled, false);
});

test.after(() => backend.close());
