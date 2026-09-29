// node --test : the /v1/orpheus-voices and /v1/orpheus-tts API-key auth path (orpheusApiProxy.js).
// Mocks the backend so no real training/synthesis happens - this only exercises key validation, the
// billing gate (unconditional, unlike voiceApiProxy.js), and header forwarding.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gw-orpheusapi-"));
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
const { handleOrpheusVoiceApi } = await import("./orpheusApiProxy.js");

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
  const { req, res, url } = fakeReqRes("GET", "/v1/orpheus-voices");
  const handled = await handleOrpheusVoiceApi(req, res, url);
  assert.equal(handled, true);
  assert.equal(res.statusCode, 401);
  assert.equal(lastReq, null);
});

test("invalid API key -> 401", async () => {
  const { req, res, url } = fakeReqRes("GET", "/v1/orpheus-voices", { authorization: "Bearer not-a-real-key" });
  await handleOrpheusVoiceApi(req, res, url);
  assert.equal(res.statusCode, 401);
});

test("valid key without billing enabled -> 402 on GET too, backend never called", async () => {
  lastReq = null;
  const { key } = keys.issueKey("free-tier-orpheus-key");
  const { req, res, url } = fakeReqRes("GET", "/v1/orpheus-voices/v-deadbeef00", { authorization: `Bearer ${key}` });
  await handleOrpheusVoiceApi(req, res, url);
  assert.equal(res.statusCode, 402);
  assert.match(JSON.parse(res.body).error, /billing-enabled/);
  assert.equal(lastReq, null);
});

test("billing-enabled key -> POST /v1/orpheus-voices forwarded with resolved identity headers", async () => {
  const { id, key } = keys.issueKey("paid-orpheus-key");
  keys.setBillingEnabledById(id, true);
  keys.setOwnerById(id, "user-77");
  const { req, res, url } = fakeReqRes("POST", "/v1/orpheus-voices", {
    authorization: `Bearer ${key}`,
    body: JSON.stringify({ speaker_name: "Test Speaker" }),
  });
  await handleOrpheusVoiceApi(req, res, url);
  assert.equal(res.statusCode, 200);
  assert.equal(lastReq.method, "POST");
  assert.equal(lastReq.url, "/internal/orpheus-clone-api");
  assert.equal(lastReq.headers["x-gateway-admin-secret"], "test-forward-secret");
  assert.equal(lastReq.headers["x-gateway-key-id"], id);
  assert.equal(lastReq.headers["x-gateway-uid"], "user-77");
  assert.equal(JSON.parse(lastReq.body).speaker_name, "Test Speaker");
});

test("billing-enabled key -> GET /v1/orpheus-voices/:id forwarded under orpheus-clone-api", async () => {
  const { id, key } = keys.issueKey("paid-orpheus-key-2");
  keys.setBillingEnabledById(id, true);
  const { req, res, url } = fakeReqRes("GET", "/v1/orpheus-voices/v-deadbeef00", { authorization: `Bearer ${key}` });
  await handleOrpheusVoiceApi(req, res, url);
  assert.equal(res.statusCode, 200);
  assert.equal(lastReq.url, "/internal/orpheus-clone-api/v-deadbeef00");
  assert.equal(lastReq.headers["x-gateway-key-id"], id);
});

test("billing-enabled key -> /v1/orpheus-tts forwarded under orpheus-clone-api", async () => {
  const { id, key } = keys.issueKey("paid-orpheus-key-3");
  keys.setBillingEnabledById(id, true);
  const { req, res, url } = fakeReqRes("POST", "/v1/orpheus-tts/synthesize", {
    authorization: `Bearer ${key}`,
    body: JSON.stringify({ text: "hello" }),
  });
  await handleOrpheusVoiceApi(req, res, url);
  assert.equal(res.statusCode, 200);
  assert.equal(lastReq.url, "/internal/orpheus-clone-api/synthesize");
  assert.equal(lastReq.headers["x-gateway-key-id"], id);
});

test("non-matching path is not handled", async () => {
  const { req, res, url } = fakeReqRes("GET", "/v1/voices");
  const handled = await handleOrpheusVoiceApi(req, res, url);
  assert.equal(handled, false);
});

test("unrelated path is not handled", async () => {
  const { req, res, url } = fakeReqRes("GET", "/health");
  const handled = await handleOrpheusVoiceApi(req, res, url);
  assert.equal(handled, false);
});

test.after(() => backend.close());
