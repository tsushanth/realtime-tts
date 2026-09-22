// node --test : /dubbing/authorize (server.js) - the key-validity + billing gate
// dubbing/job_server.py (dubbing/gateway_auth.py) calls before queuing a dubbing job.
// Mirrors audio.test.js's HTTP-level pattern for /audio/authorize.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "gw-dubbing-"));
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
const post = (base, path, body) => fetch(base + path, {
  method: "POST", headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

test("/dubbing/authorize rejects a missing/unknown key", async () => {
  const { p, base } = await startServer({});
  try {
    assert.equal((await post(base, "/dubbing/authorize", {})).status, 401);
    assert.equal((await post(base, "/dubbing/authorize", { key: "nope" })).status, 401);
  } finally { p.kill(); }
});

test("/dubbing/authorize accepts a valid, billing-enabled key and returns its id (no token/url)", async () => {
  const { p, base } = await startServer({});
  try {
    const { id, key } = keys.issueKey("dub-e2e");
    keys.setBillingEnabledById(id, true);
    const r = await post(base, "/dubbing/authorize", { key });
    assert.equal(r.status, 200);
    const j = await r.json();
    assert.deepEqual(j, { authorized: true, id });
    assert.equal(j.token, undefined);
  } finally { p.kill(); }
});

test("/dubbing/authorize returns 402 once a free-tier key is exhausted", async () => {
  const { p, base } = await startServer({});
  try {
    const { id, key } = keys.issueKey("dub-free-exhausted");
    keys.recordUsageById(id, undefined, "dubbing", 4000);
    const r = await post(base, "/dubbing/authorize", { key });
    assert.equal(r.status, 402);
  } finally { p.kill(); }
});
