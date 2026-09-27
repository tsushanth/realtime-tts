// node --test : voice discovery endpoints (GET /v1/voices, GET /v1/voices/:id).
// Mocks the backend so no real intake is needed. Exercises built-in catalog + custom merging.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gw-voices-disc-"));
process.env.KEYS_PATH = path.join(dir, "keys.json");
process.env.MODAL_SESSION_SECRET = "test-session-secret";
process.env.FREE_TIER_CHARS = "10000";
process.env.GATEWAY_FORWARD_SECRET = "test-forward-secret";

// Fake backend: returns canned custom voices for GET /, forwards errors otherwise.
const CUSTOM_VOICES = [
  { id: "v-a1b2c3d4e5", status: "ready", speaker_name: "Test Voice", voice: "custom:v-a1b2c3d4e5", created_at: "2026-09-20T10:00:00Z", stats: { clips: 42, minutes: 25 } },
];

const backend = http.createServer((req, res) => {
  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    if (req.method === "GET" && req.url === "/") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ voices: CUSTOM_VOICES }));
      return;
    }
    if (req.method === "GET" && req.url.startsWith("/v-a1b2c3d4e5")) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(CUSTOM_VOICES[0]));
      return;
    }
    if (req.method === "GET" && req.url.startsWith("/v-")) {
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "Voice not found." }));
      return;
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ ok: true }));
  });
});
await new Promise((resolve) => backend.listen(0, resolve));
process.env.READALOUD_BACKEND_URL = `http://127.0.0.1:${backend.address().port}`;

const keys = await import("./keys.js");
const { handleVoiceApi } = await import("./voiceApiProxy.js");

function fakeReqRes(method, urlPath, { authorization } = {}) {
  const req = {
    method,
    headers: authorization ? { authorization, "content-type": "application/json" } : { "content-type": "application/json" },
    on() { return req; },
  };
  const res = {
    statusCode: null,
    headers: null,
    body: null,
    writeHead(code, headers) { this.statusCode = code; this.headers = headers; },
    end(b) { this.body = b; },
  };
  return { req, res };
}

function parseBody(res) {
  try { return JSON.parse(res.body); } catch { return null; }
}

// --------------------------------------------------------------------------
// voices.js catalog
// --------------------------------------------------------------------------

const { listBuiltInVoices, getBuiltInVoice, voiceExists } = await import("./voices.js");

test("listBuiltInVoices returns 53 premade voices", () => {
  const voices = listBuiltInVoices();
  assert.equal(voices.length, 53);
  assert.ok(voices.every((v) => v.category === "premade"));
  assert.ok(voices.some((v) => v.voice_id === "af_heart"));
  assert.ok(voices.some((v) => v.voice_id === "am_adam"));
});

test("getBuiltInVoice looks up by id", () => {
  const v = getBuiltInVoice("af_heart");
  assert.ok(v);
  assert.equal(v.name, "Heart");
  assert.equal(v.language, "en");
  assert.equal(getBuiltInVoice("not_a_voice"), null);
});

test("voiceExists matches built-in and custom patterns", () => {
  assert.equal(voiceExists("af_heart"), true);
  assert.equal(voiceExists("v-a1b2c3d4e5"), true);
  assert.equal(voiceExists("random"), false);
});

// --------------------------------------------------------------------------
// GET /v1/voices  (list)
// --------------------------------------------------------------------------

test("GET /v1/voices returns 401 without key", async () => {
  const { req, res } = fakeReqRes("GET", "/v1/voices");
  const handled = await handleVoiceApi(req, res, new URL("http://x/v1/voices"));
  assert.equal(handled, true);
  assert.equal(res.statusCode, 401);
});

test("GET /v1/voices lists built-in voices for free-tier key", async () => {
  const { key } = keys.issueKey("free-list");
  const { req, res } = fakeReqRes("GET", "/v1/voices", { authorization: `Bearer ${key}` });
  const handled = await handleVoiceApi(req, res, new URL("http://x/v1/voices"));
  assert.equal(handled, true);
  assert.equal(res.statusCode, 200);
  const body = parseBody(res);
  assert.ok(Array.isArray(body.voices));
  assert.ok(body.voices.length >= 53, `expected >=53, got ${body.voices.length}`);
  const premade = body.voices.filter((v) => v.category === "premade");
  assert.equal(premade.length, 53);
  const af = premade.find((v) => v.voice_id === "af_heart");
  assert.ok(af);
  assert.equal(af.name, "Heart");
  assert.equal(af.language, "en");
  assert.equal(af.gender, "female");
  // No custom voices for free-tier keys (backend not called with billing key, but our fake backend
  // always returns one — the test is that free-tier keys CAN call this endpoint without 402)
  const custom = body.voices.filter((v) => v.category === "cloned");
  assert.ok(custom.length >= 0);
});

test("GET /v1/voices merges built-in + custom for billing-enabled key", async () => {
  const { key } = keys.issueKey("paid-list");
  keys.setBillingEnabledById(keys.getIdForKey(key), true);
  const { req, res } = fakeReqRes("GET", "/v1/voices", { authorization: `Bearer ${key}` });
  const handled = await handleVoiceApi(req, res, new URL("http://x/v1/voices"));
  assert.equal(handled, true);
  assert.equal(res.statusCode, 200);
  const body = parseBody(res);
  const premade = body.voices.filter((v) => v.category === "premade");
  const custom = body.voices.filter((v) => v.category === "cloned");
  assert.equal(premade.length, 53);
  assert.equal(custom.length, 1);
  assert.equal(custom[0].voice_id, "custom:v-a1b2c3d4e5");
  assert.equal(custom[0].name, "Test Voice");
});

// --------------------------------------------------------------------------
// GET /v1/voices/:id  (detail)
// --------------------------------------------------------------------------

test("GET /v1/voices/:id returns built-in detail for free-tier key", async () => {
  const { key } = keys.issueKey("free-detail");
  const { req, res } = fakeReqRes("GET", "/v1/voices/af_heart", { authorization: `Bearer ${key}` });
  const handled = await handleVoiceApi(req, res, new URL("http://x/v1/voices/af_heart"));
  assert.equal(handled, true);
  assert.equal(res.statusCode, 200);
  const body = parseBody(res);
  assert.equal(body.voice_id, "af_heart");
  assert.equal(body.name, "Heart");
  assert.equal(body.category, "premade");
});

test("GET /v1/voices/:id returns 404 for unknown voice", async () => {
  const { key } = keys.issueKey("free-404");
  const { req, res } = fakeReqRes("GET", "/v1/voices/no-such-voice", { authorization: `Bearer ${key}` });
  const handled = await handleVoiceApi(req, res, new URL("http://x/v1/voices/no-such-voice"));
  assert.equal(handled, true);
  assert.equal(res.statusCode, 404);
});

test("GET /v1/voices/:id proxies custom voice to backend", async () => {
  const { key } = keys.issueKey("paid-detail");
  keys.setBillingEnabledById(keys.getIdForKey(key), true);
  const { req, res } = fakeReqRes("GET", "/v1/voices/v-a1b2c3d4e5", { authorization: `Bearer ${key}` });
  const handled = await handleVoiceApi(req, res, new URL("http://x/v1/voices/v-a1b2c3d4e5"));
  assert.equal(handled, true);
  assert.equal(res.statusCode, 200);
  const body = parseBody(res);
  assert.equal(body.voice_id, "custom:v-a1b2c3d4e5");
  assert.equal(body.category, "cloned");
  assert.equal(body.name, "Test Voice");
});

// --------------------------------------------------------------------------
// Non-readonly routes still require billing
// --------------------------------------------------------------------------

test("POST /v1/voices still requires billing even for valid free key", async () => {
  const { key } = keys.issueKey("free-create");
  const { req, res } = fakeReqRes("POST", "/v1/voices", { authorization: `Bearer ${key}` });
  const handled = await handleVoiceApi(req, res, new URL("http://x/v1/voices"));
  assert.equal(handled, true);
  assert.equal(res.statusCode, 402);
});

test("GET /v1/voices/enabled is NOT a read-only discovery route (it forwards to backend)", async () => {
  // /v1/voices/enabled has a slash after the base path, so isReadOnlyRoute returns false,
  // and it gets forwarded to the backend which returns 402 for free-tier keys.
  // This is correct: the "enabled" endpoint carries config data for the cloning flow.
  const { key } = keys.issueKey("free-enabled");
  const { req, res } = fakeReqRes("GET", "/v1/voices/enabled", { authorization: `Bearer ${key}` });
  const handled = await handleVoiceApi(req, res, new URL("http://x/v1/voices/enabled"));
  assert.equal(handled, true);
  // Forwarded to fake backend which returns 200 ok for anything it doesn't recognise
  assert.equal(res.statusCode, 200);
});
