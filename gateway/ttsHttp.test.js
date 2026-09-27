// node --test : POST /v1/text-to-speech (one-shot HTTP TTS) gateway validation gates.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gw-tts-http-"));
process.env.KEYS_PATH = path.join(dir, "keys.json");
process.env.MODAL_SESSION_SECRET = "test-session-secret";
process.env.FREE_TIER_CHARS = "10000";
const keys = await import("./keys.js");

async function startServer(env) {
  const port = 20000 + Math.floor(Math.random() * 20000);
  const p = spawn(process.execPath, ["server.js"], {
    cwd: path.dirname(new URL(import.meta.url).pathname),
    env: { ...process.env, PORT: String(port), ADMIN_SECRET: "adm", MODAL_USAGE_REPORT_SECRET: "usg", ...env },
    stdio: ["ignore", "pipe", "inherit"],
  });
  await new Promise((r) => p.stdout.on("data", (d) => String(d).includes("listening") && r()));
  return { p, base: `http://127.0.0.1:${port}` };
}
const post = (base, path, body, auth) => fetch(base + path, {
  method: "POST", headers: { "content-type": "application/json", ...(auth ? { authorization: `Bearer ${auth}` } : {}) },
  body: JSON.stringify(body),
});

test("/v1/text-to-speech returns 401 without a valid key", async () => {
  const { p, base } = await startServer({});
  try {
    assert.equal((await post(base, "/v1/text-to-speech", { text: "hello" })).status, 401);
    assert.equal((await post(base, "/v1/text-to-speech", { text: "hello" }, "bad-key")).status, 401);
  } finally { p.kill(); }
});

test("/v1/text-to-speech returns 400 for invalid JSON or missing text", async () => {
  const { p, base } = await startServer({});
  try {
    const { key } = keys.issueKey("val");
    keys.setBillingEnabledById(keys.getIdForKey(key), true);
    assert.equal((await fetch(base + "/v1/text-to-speech", {
      method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
      body: "not-json",
    })).status, 400);
    assert.equal((await post(base, "/v1/text-to-speech", {}, key)).status, 400);
    assert.equal((await post(base, "/v1/text-to-speech", { text: "" }, key)).status, 400);
    assert.equal((await post(base, "/v1/text-to-speech", { text: "   " }, key)).status, 400);
  } finally { p.kill(); }
});

test("/v1/text-to-speech returns 413 for text over 5000 chars", async () => {
  const { p, base } = await startServer({});
  try {
    const { key } = keys.issueKey("long");
    keys.setBillingEnabledById(keys.getIdForKey(key), true);
    const r = await post(base, "/v1/text-to-speech", { text: "a".repeat(5001) }, key);
    assert.equal(r.status, 413);
  } finally { p.kill(); }
});

test("/v1/text-to-speech returns 501 when engine worker is unavailable", async () => {
  const { p, base } = await startServer({ MODAL_WORKER_URL: "", PIPER_WORKER_URL: "" });
  try {
    const { key } = keys.issueKey("no-worker");
    keys.setBillingEnabledById(keys.getIdForKey(key), true);
    const r = await post(base, "/v1/text-to-speech", { text: "hello" }, key);
    assert.equal(r.status, 501);
  } finally { p.kill(); }
});

test("/v1/text-to-speech returns 402 when free tier is exhausted", async () => {
  const { p, base } = await startServer({});
  try {
    const { id, key } = keys.issueKey("free-exhaust");
    keys.recordUsageById(id, 10000); // exhaust free tier
    const r = await post(base, "/v1/text-to-speech", { text: "hello" }, key);
    assert.equal(r.status, 402);
  } finally { p.kill(); }
});

test("/v1/text-to-speech returns 402 when free tier can't afford request", async () => {
  const { p, base } = await startServer({ MODAL_READALOUD_WS_URL: "ws://127.0.0.1:1" });
  try {
    const { id, key } = keys.issueKey("free-small");
    keys.recordUsageById(id, 9996); // 4 chars remaining
    const r = await post(base, "/v1/text-to-speech", { text: "hello" }, key); // 5 chars
    assert.equal(r.status, 402);
  } finally { p.kill(); }
});

test("/v1/text-to-speech proxies to worker and returns upstream status", async () => {
  // Point the gateway at an unreachable port so the proxy fails fast.
  // This proves the gateway is forwarding the request; a real worker is not needed.
  const { p, base } = await startServer({ MODAL_READALOUD_WS_URL: "ws://127.0.0.1:1" });
  try {
    const { key } = keys.issueKey("proxy");
    keys.setBillingEnabledById(keys.getIdForKey(key), true);
    const r = await post(base, "/v1/text-to-speech", { text: "hello" }, key);
    // Connection refused -> gateway returns 502
    assert.equal(r.status, 502);
    const j = await r.json();
    assert.equal(j.error, "TTS worker unavailable, please retry");
  } finally { p.kill(); }
});

test("/v1/text-to-speech selects piper engine when requested", async () => {
  const { p, base } = await startServer({ PIPER_WORKER_URL: "ws://127.0.0.1:1" });
  try {
    const { key } = keys.issueKey("piper-req");
    keys.setBillingEnabledById(keys.getIdForKey(key), true);
    const r = await post(base, "/v1/text-to-speech", { text: "hello", engine: "piper" }, key);
    // Connection refused -> 502, proving it tried the Piper worker
    assert.equal(r.status, 502);
  } finally { p.kill(); }
});

test("/v1/text-to-speech defaults to kokoro engine", async () => {
  const { p, base } = await startServer({ MODAL_READALOUD_WS_URL: "ws://127.0.0.1:1" });
  try {
    const { key } = keys.issueKey("kokoro-default");
    keys.setBillingEnabledById(keys.getIdForKey(key), true);
    const r = await post(base, "/v1/text-to-speech", { text: "hello" }, key);
    assert.equal(r.status, 502);
  } finally { p.kill(); }
});
