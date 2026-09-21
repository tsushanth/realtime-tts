// node --test : /audio/authorize (server.js) - audio isolation (worker-demucs-fly) authorize route.
// Mirrors stt.test.js's HTTP-level pattern for /stt/authorize.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gw-audio-"));
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

test("/audio/authorize returns 501 when DEMUCS_WORKER_URL unset", async () => {
  const { p, base } = await startServer({ DEMUCS_WORKER_URL: "" });
  try {
    const { key } = keys.issueKey("x");
    assert.equal((await post(base, "/audio/authorize", { key })).status, 501);
  } finally { p.kill(); }
});

test("/audio/authorize rejects an unknown engine value", async () => {
  const { p, base } = await startServer({ DEMUCS_WORKER_URL: "https://demucs.example" });
  try {
    const { key } = keys.issueKey("bad-engine");
    const r = await post(base, "/audio/authorize", { key, engine: "kokoro" });
    assert.equal(r.status, 400);
  } finally { p.kill(); }
});

test("/audio/authorize + usage report end to end", async () => {
  const { p, base } = await startServer({ DEMUCS_WORKER_URL: "https://demucs.example" });
  try {
    assert.equal((await post(base, "/audio/authorize", { key: "nope" })).status, 401);
    assert.equal((await post(base, "/audio/authorize", {})).status, 401);
    const { id, key } = keys.issueKey("e2e");
    keys.setBillingEnabledById(id, true);
    const r = await post(base, "/audio/authorize", { key, engine: "denoise" });
    assert.equal(r.status, 200);
    const j = await r.json();
    assert.equal(j.url, "https://demucs.example");
    assert.equal(keys.verifySessionToken(j.token), id);
    assert.equal((await post(base, "/admin/usage/report", { id, audio_seconds: 12, engine: "denoise" })).status, 401);
    const rep = await post(base, "/admin/usage/report", { id, audio_seconds: 12, engine: "denoise" }, "usg");
    assert.deepEqual(await rep.json(), { recorded: true });
    const dr = await (await post(base, "/admin/usage/drain", {}, "adm")).json();
    assert.deepEqual(dr.find((x) => x.id === id), { id, chars: 0, piperChars: 0, audioSeconds: 12, realtimeAudioSeconds: 0 });
    // exhausted free key => 402
    const f = keys.issueKey("free-audio");
    keys.recordUsageById(f.id, undefined, "stt", 4000);
    assert.equal((await post(base, "/audio/authorize", { key: f.key })).status, 402);
  } finally { p.kill(); }
});

test("/audio/authorize omits engine (defaults accepted, not rejected)", async () => {
  const { p, base } = await startServer({ DEMUCS_WORKER_URL: "https://demucs.example" });
  try {
    const { id, key } = keys.issueKey("no-engine");
    keys.setBillingEnabledById(id, true);
    const r = await post(base, "/audio/authorize", { key });
    assert.equal(r.status, 200);
  } finally { p.kill(); }
});
