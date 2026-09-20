// node --test : STT usage accounting (keys.js) and /stt/authorize + /admin/usage/report (server.js).
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gw-stt-"));
process.env.KEYS_PATH = path.join(dir, "keys.json");
process.env.MODAL_SESSION_SECRET = "test-session-secret";
process.env.FREE_TIER_CHARS = "10000";
const keys = await import("./keys.js");

test("audio seconds accumulate; chars/piperChars semantics unchanged", () => {
  const { id } = keys.issueKey("t1");
  keys.setBillingEnabledById(id, true);
  assert.ok(keys.recordUsageById(id, 100, undefined, undefined));
  assert.ok(keys.recordUsageById(id, 50, "piper"));
  assert.ok(keys.recordUsageById(id, undefined, "stt", 12.5));
  assert.ok(keys.recordUsageById(id, 0, "stt", 7.5));
  const d = keys.drainUsage().find((x) => x.id === id);
  assert.deepEqual(d, { id, chars: 150, piperChars: 50, audioSeconds: 20 });
  assert.equal(keys.drainUsage().find((x) => x.id === id), undefined); // reset
});

test("chars-only key drains with audioSeconds 0; audio-only key is included", () => {
  const a = keys.issueKey("a"), b = keys.issueKey("b");
  keys.setBillingEnabledById(a.id, true); keys.setBillingEnabledById(b.id, true);
  keys.recordUsageById(a.id, 10);
  keys.recordUsageById(b.id, undefined, "stt", 3);
  const out = keys.drainUsage();
  assert.deepEqual(out.find((x) => x.id === a.id), { id: a.id, chars: 10, piperChars: 0, audioSeconds: 0 });
  assert.deepEqual(out.find((x) => x.id === b.id), { id: b.id, chars: 0, piperChars: 0, audioSeconds: 3 });
});

test("invalid audio seconds are ignored", () => {
  const { id } = keys.issueKey("bad");
  keys.setBillingEnabledById(id, true);
  assert.equal(keys.recordUsageById(id, undefined, "stt", -5), false);
  assert.equal(keys.recordUsageById(id, undefined, "stt", NaN), false);
  assert.equal(keys.recordUsageById(id, undefined, "stt", "9"), false);
  assert.equal(keys.recordUsageById(id, undefined, "stt", 0), false);
});

test("free tier: audio seconds convert to char-equivalents on the shared allowance", () => {
  const { id, key } = keys.issueKey("free");
  // 1000 s * 3.0556 = 3055.6 chars
  keys.recordUsageById(id, undefined, "stt", 1000);
  assert.equal(Math.round(keys.listKeys().find((k) => k.id === id).freeCharsRemaining), 10000 - 3056);
  keys.recordUsageById(id, 4000);                       // TTS shares the same pool
  assert.equal(keys.checkAccess(key).allowed, true);
  keys.recordUsageById(id, undefined, "stt", 1000);     // 3055.6 more -> 10111 >= 10000
  const acc = keys.checkAccess(key);
  assert.equal(acc.allowed, false);
  assert.equal(keys.drainUsage().find((x) => x.id === id), undefined); // free usage never billed
});

// ---- HTTP ----
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

test("/stt/authorize returns 501 when STT_WORKER_URL unset", async () => {
  const { p, base } = await startServer({ STT_WORKER_URL: "" });
  try {
    const { key } = keys.issueKey("x");
    assert.equal((await post(base, "/stt/authorize", { key })).status, 501);
  } finally { p.kill(); }
});

test("/stt/authorize + usage report end to end", async () => {
  const { p, base } = await startServer({ STT_WORKER_URL: "https://stt.example" });
  try {
    assert.equal((await post(base, "/stt/authorize", { key: "nope" })).status, 401);
    assert.equal((await post(base, "/stt/authorize", {})).status, 401);
    const { id, key } = keys.issueKey("e2e");
    keys.setBillingEnabledById(id, true);
    const r = await post(base, "/stt/authorize", { key });
    assert.equal(r.status, 200);
    const j = await r.json();
    assert.equal(j.url, "https://stt.example");
    assert.equal(keys.verifySessionToken(j.token), id);
    assert.equal((await post(base, "/admin/usage/report", { id, audio_seconds: 42, engine: "stt" })).status, 401);
    const rep = await post(base, "/admin/usage/report", { id, audio_seconds: 42, engine: "stt" }, "usg");
    assert.deepEqual(await rep.json(), { recorded: true });
    const dr = await (await post(base, "/admin/usage/drain", {}, "adm")).json();
    assert.deepEqual(dr.find((x) => x.id === id), { id, chars: 0, piperChars: 0, audioSeconds: 42 });
    // exhausted free key => 402
    const f = keys.issueKey("free2");
    keys.recordUsageById(f.id, undefined, "stt", 4000);
    assert.equal((await post(base, "/stt/authorize", { key: f.key })).status, 402);
  } finally { p.kill(); }
});
