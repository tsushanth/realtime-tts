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

// ---------------------------------------------------------------------------------------------
// Compressed output (mp3 / opus) against a mock worker. Needs ffmpeg on PATH (or FFMPEG_PATH).
// ---------------------------------------------------------------------------------------------
import http from "node:http";
import { spawnSync } from "node:child_process";

const HAVE_FFMPEG = spawnSync(process.env.FFMPEG_PATH || "ffmpeg", ["-version"]).status === 0;
const ffTest = HAVE_FFMPEG ? test : test.skip;

// 0.5 s of a 440 Hz sine, s16le mono 24 kHz, sent in small chunks with a small delay (streaming).
function sinePcm(seconds) {
  const n = Math.floor(24000 * seconds);
  const b = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(8000 * Math.sin((2 * Math.PI * 440 * i) / 24000)), i * 2);
  return b;
}

async function startMockWorker({ seconds = 0.5, failAfterBytes = null, status = 200, hang = false } = {}) {
  const seen = { bodies: [], closed: 0 };
  const srv = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (d) => (raw += d));
    req.on("end", () => {
      seen.bodies.push(JSON.parse(raw));
      if (status !== 200) { res.writeHead(status, { "content-type": "application/json" }); return res.end('{"error":"bad format"}'); }
      res.writeHead(200, { "content-type": "audio/pcm", "x-sample-rate": "24000", "x-audio-format": "pcm_24000" });
      res.flushHeaders();
      const pcm = sinePcm(seconds);
      let off = 0;
      const tick = setInterval(() => {
        if (failAfterBytes !== null && off >= failAfterBytes) { clearInterval(tick); return res.destroy(); }
        if (off >= pcm.length) { clearInterval(tick); return hang ? undefined : res.end(); }
        res.write(pcm.subarray(off, off + 4800)); off += 4800;
      }, 10);
      res.on("close", () => { clearInterval(tick); seen.closed++; });
    });
  });
  await new Promise((r) => srv.listen(0, "127.0.0.1", r));
  return { srv, seen, wsUrl: `ws://127.0.0.1:${srv.address().port}` };
}

async function withGateway(workerOpts, fn) {
  const w = await startMockWorker(workerOpts);
  const { p, base } = await startServer({ MODAL_READALOUD_WS_URL: w.wsUrl });
  const { key } = keys.issueKey("enc-" + Math.random());
  keys.setBillingEnabledById(keys.getIdForKey(key), true);
  try { await fn({ base, key, w, gw: p }); } finally { p.kill(); w.srv.closeAllConnections(); w.srv.close(); }
}

const isMp3Sync = (b, i = 0) => b[i] === 0xff && (b[i + 1] & 0xe0) === 0xe0;
const skipId3 = (b) => (b.subarray(0, 3).toString() === "ID3" ? 10 + ((b[6] << 21) | (b[7] << 14) | (b[8] << 7) | b[9]) : 0);

for (const [fmt, kbps] of [["mp3_24000_64", 64], ["mp3_24000_128", 128]]) {
  ffTest(`/v1/text-to-speech ${fmt}: audio/mpeg, headers, valid MP3 frame sync, worker asked for pcm_24000`, async () => {
    await withGateway({ seconds: 1 }, async ({ base, key, w }) => {
      const r = await post(base, "/v1/text-to-speech", { text: "hello", format: fmt, voice: "af_heart" }, key);
      assert.equal(r.status, 200);
      assert.equal(r.headers.get("content-type"), "audio/mpeg");
      assert.equal(r.headers.get("x-audio-format"), fmt);
      assert.equal(r.headers.get("x-sample-rate"), "24000");
      assert.equal(r.headers.get("cache-control"), "no-store");
      assert.equal(r.headers.get("content-length"), null); // chunked, not buffered
      const buf = Buffer.from(await r.arrayBuffer());
      const start = skipId3(buf);
      assert.ok(isMp3Sync(buf, start), "first frame has sync word");
      assert.equal((buf[start + 1] >> 1) & 3, 1, "layer III");
      assert.equal((buf[start + 1] >> 3) & 3, 2, "MPEG-2 (24 kHz)");
      // ~1 s of audio at kbps => roughly kbps*125 bytes (allow wide tolerance for padding/priming)
      assert.ok(buf.length > kbps * 125 * 0.7 && buf.length < kbps * 125 * 1.6, `size ${buf.length}`);
      // the worker only ever sees PCM and the other body fields untouched
      assert.equal(w.seen.bodies[0].format, "pcm_24000");
      assert.equal(w.seen.bodies[0].voice, "af_heart");
      assert.equal(w.seen.bodies[0].text, "hello");
    });
  });
}

ffTest("/v1/text-to-speech opus_24000: audio/ogg with OggS + OpusHead", async () => {
  await withGateway({}, async ({ base, key }) => {
    const r = await post(base, "/v1/text-to-speech", { text: "hello", format: "opus_24000" }, key);
    assert.equal(r.status, 200);
    assert.equal(r.headers.get("content-type"), "audio/ogg");
    assert.equal(r.headers.get("x-audio-format"), "opus_24000");
    assert.equal(r.headers.get("x-sample-rate"), "24000");
    const buf = Buffer.from(await r.arrayBuffer());
    assert.equal(buf.subarray(0, 4).toString(), "OggS");
    assert.ok(buf.includes("OpusHead"));
  });
});

ffTest("/v1/text-to-speech mp3 output is streamed: first bytes arrive before the worker finishes", async () => {
  await withGateway({ seconds: 3, hang: false }, async ({ base, key }) => {
    const t0 = Date.now();
    const r = await post(base, "/v1/text-to-speech", { text: "hello", format: "mp3_24000_64" }, key);
    const reader = r.body.getReader();
    const first = await reader.read();
    const tFirst = Date.now() - t0;
    assert.ok(first.value.length > 0);
    let total = first.value.length;
    for (;;) { const { done, value } = await reader.read(); if (done) break; total += value.length; }
    const tAll = Date.now() - t0;
    assert.ok(tFirst < tAll - 100, `first chunk at ${tFirst}ms should precede completion at ${tAll}ms`);
    assert.ok(total > first.value.length);
  });
});

ffTest("/v1/text-to-speech mp3: client disconnect kills ffmpeg and aborts the worker request", async () => {
  await withGateway({ seconds: 30, hang: true }, async ({ base, key, w, gw }) => {
    const ac = new AbortController();
    const r = await fetch(base + "/v1/text-to-speech", {
      method: "POST", signal: ac.signal,
      headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
      body: JSON.stringify({ text: "hello", format: "mp3_24000_64" }),
    });
    const reader = r.body.getReader();
    await reader.read();
    const ffCount = () => spawnSync("pgrep", ["-P", String(gw.pid), "ffmpeg"]).stdout.toString().trim().split("\n").filter(Boolean).length;
    assert.equal(ffCount(), 1, "one ffmpeg child while streaming");
    ac.abort();
    const deadline = Date.now() + 3000;
    while ((ffCount() > 0 || w.seen.closed === 0) && Date.now() < deadline) await new Promise((r) => setTimeout(r, 50));
    assert.equal(ffCount(), 0, "ffmpeg child reaped after disconnect");
    assert.ok(w.seen.closed >= 1, "upstream worker connection closed");
  });
});

ffTest("/v1/text-to-speech mp3: worker dies before any audio -> 502 JSON", async () => {
  await withGateway({ failAfterBytes: 0 }, async ({ base, key }) => {
    const r = await post(base, "/v1/text-to-speech", { text: "hello", format: "mp3_24000_128" }, key);
    assert.equal(r.status, 502);
    assert.match(r.headers.get("content-type"), /json/);
    assert.match((await r.json()).error, /encoding failed/);
  });
});

ffTest("/v1/text-to-speech mp3: worker dies mid-stream -> connection is aborted, not ended as if complete", async () => {
  await withGateway({ seconds: 3, failAfterBytes: 48000 }, async ({ base, key }) => {
    const r = await post(base, "/v1/text-to-speech", { text: "hello", format: "mp3_24000_64" }, key);
    assert.equal(r.status, 200);
    await assert.rejects(async () => { await r.arrayBuffer(); });
  });
});

ffTest("/v1/text-to-speech mp3: worker error status is passed through, not masked", async () => {
  await withGateway({ status: 400 }, async ({ base, key }) => {
    const r = await post(base, "/v1/text-to-speech", { text: "hello", format: "mp3_24000_64", speed: 99 }, key);
    assert.equal(r.status, 400);
    assert.equal((await r.json()).error, "bad format");
  });
});

test("/v1/text-to-speech mp3: encoder binary missing -> 502 JSON (headers not yet sent)", async () => {
  const w = await startMockWorker({});
  const { p, base } = await startServer({ MODAL_READALOUD_WS_URL: w.wsUrl, FFMPEG_PATH: "/nonexistent/ffmpeg" });
  try {
    const { key } = keys.issueKey("noff");
    keys.setBillingEnabledById(keys.getIdForKey(key), true);
    const r = await post(base, "/v1/text-to-speech", { text: "hello", format: "mp3_24000_64" }, key);
    assert.equal(r.status, 502);
    assert.equal((await r.json()).error, "audio encoder unavailable");
  } finally { p.kill(); w.srv.closeAllConnections(); w.srv.close(); }
});

test("/v1/text-to-speech mp3: 503 when encoder concurrency cap is reached", async () => {
  const w = await startMockWorker({ seconds: 30, hang: true });
  const { p, base } = await startServer({ MODAL_READALOUD_WS_URL: w.wsUrl, MAX_CONCURRENT_ENCODERS: "1" });
  try {
    const { key } = keys.issueKey("cap");
    keys.setBillingEnabledById(keys.getIdForKey(key), true);
    if (!HAVE_FFMPEG) return; // needs a live first stream to hold the slot
    const ac = new AbortController();
    const r1 = await fetch(base + "/v1/text-to-speech", { method: "POST", signal: ac.signal,
      headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
      body: JSON.stringify({ text: "hello", format: "mp3_24000_64" }) });
    await r1.body.getReader().read();
    const r2 = await post(base, "/v1/text-to-speech", { text: "hello", format: "mp3_24000_64" }, key);
    assert.equal(r2.status, 503);
    ac.abort();
  } finally { p.kill(); w.srv.closeAllConnections(); w.srv.close(); }
});

test("/v1/text-to-speech pcm_24000 / mulaw_8000 are forwarded unchanged (no encoder, body untouched)", async () => {
  for (const format of ["pcm_24000", "mulaw_8000"]) {
    const w = await startMockWorker({});
    const { p, base } = await startServer({ MODAL_READALOUD_WS_URL: w.wsUrl, FFMPEG_PATH: "/nonexistent/ffmpeg" });
    try {
      const { key } = keys.issueKey("pcm-" + format);
      keys.setBillingEnabledById(keys.getIdForKey(key), true);
      const r = await post(base, "/v1/text-to-speech", { text: "hello", format }, key);
      assert.equal(r.status, 200);
      assert.equal(r.headers.get("content-type"), "audio/pcm"); // mock worker's passthrough header
      assert.equal(r.headers.get("x-audio-format"), "pcm_24000");
      assert.equal(Buffer.from(await r.arrayBuffer()).length, 24000); // 0.5 s * 24000 * 2 bytes, byte-exact
      assert.equal(w.seen.bodies[0].format, format);
    } finally { p.kill(); w.srv.closeAllConnections(); w.srv.close(); }
  }
});

test("/v1/text-to-speech unknown formats still reach the worker (which owns validation)", async () => {
  const w = await startMockWorker({ status: 400 });
  const { p, base } = await startServer({ MODAL_READALOUD_WS_URL: w.wsUrl });
  try {
    const { key } = keys.issueKey("unk");
    keys.setBillingEnabledById(keys.getIdForKey(key), true);
    const r = await post(base, "/v1/text-to-speech", { text: "hello", format: "mp3_44100_128" }, key);
    assert.equal(r.status, 400);
    assert.equal(w.seen.bodies[0].format, "mp3_44100_128");
  } finally { p.kill(); w.srv.closeAllConnections(); w.srv.close(); }
});
