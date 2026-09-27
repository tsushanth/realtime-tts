// node --test : GET /v1/user/subscription (quota/subscription status endpoint)
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gw-sub-"));
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
const get = (base, path, auth) => fetch(base + path, {
  headers: { ...(auth ? { authorization: `Bearer ${auth}` } : {}) },
});

const post = (base, path, body, auth) => fetch(base + path, {
  method: "POST", headers: { "content-type": "application/json", ...(auth ? { authorization: `Bearer ${auth}` } : {}) },
  body: JSON.stringify(body),
});

test("/v1/user/subscription returns 401 without a valid key", async () => {
  const { p, base } = await startServer({});
  try {
    assert.equal((await get(base, "/v1/user/subscription")).status, 401);
    assert.equal((await get(base, "/v1/user/subscription", "bad-key")).status, 401);
  } finally { p.kill(); }
});

test("/v1/user/subscription returns free-tier info for a fresh key", async () => {
  const { p, base } = await startServer({});
  try {
    const { key } = keys.issueKey("free-sub");
    const r = await get(base, "/v1/user/subscription", key);
    assert.equal(r.status, 200);
    const j = await r.json();
    assert.equal(j.tier, "free");
    assert.equal(j.billing_enabled, false);
    assert.equal(j.free_chars_limit, 10000);
    assert.equal(j.free_chars_used, 0);
    assert.equal(j.free_chars_remaining, 10000);
    assert.ok(j.created_at);
    assert.equal(j.usage_reported_since_drain.chars, 0);
  } finally { p.kill(); }
});

test("/v1/user/subscription returns paid-tier info for a billing-enabled key", async () => {
  const { p, base } = await startServer({});
  try {
    const { id, key } = keys.issueKey("paid-sub");
    keys.setBillingEnabledById(id, true);
    const r = await get(base, "/v1/user/subscription", key);
    assert.equal(r.status, 200);
    const j = await r.json();
    assert.equal(j.tier, "paid");
    assert.equal(j.billing_enabled, true);
    assert.equal(j.free_chars_remaining, null);
    assert.equal(j.usage_reported_since_drain.chars, 0);
  } finally { p.kill(); }
});

test("/v1/user/subscription reflects usage after reporting", async () => {
  const { p, base } = await startServer({});
  try {
    const { id, key } = keys.issueKey("used-sub");
    keys.setBillingEnabledById(id, true);
    // Simulate worker reporting usage
    await post(base, "/admin/usage/report", { id, chars: 500, engine: "kokoro" }, "usg");
    const r = await get(base, "/v1/user/subscription", key);
    assert.equal(r.status, 200);
    const j = await r.json();
    assert.equal(j.usage_reported_since_drain.chars, 500);
  } finally { p.kill(); }
});

test("/v1/user/subscription reflects exhausted free tier", async () => {
  const { p, base } = await startServer({});
  try {
    const { id, key } = keys.issueKey("exhausted-sub");
    keys.recordUsageById(id, 10000); // exhaust free tier
    const r = await get(base, "/v1/user/subscription", key);
    assert.equal(r.status, 200);
    const j = await r.json();
    assert.equal(j.tier, "free");
    assert.equal(j.billing_enabled, false);
    assert.equal(j.free_chars_used, 10000);
    assert.equal(j.free_chars_remaining, 0);
  } finally { p.kill(); }
});
