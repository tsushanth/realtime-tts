// End-to-end tests against a running container. Needs Node >= 22 (global WebSocket + fetch).
//   docker run --rm -e HOST=0.0.0.0 -e SESSION_SECRET=testsecret -e AUTH_TOKEN=static -e VOICES_ADMIN_TOKEN=adm \
//     -e USAGE_REPORT_URL=http://host.docker.internal:9111/admin/usage/report -e USAGE_REPORT_SECRET=us \
//     -v $(mktemp -d):/voices -p 8099:8080 piper-tts-feat
//   node tests/integration.mjs            (BASE=localhost:8099 by default)
// The script mints session tokens with gateway/keys.js, runs the usage-report listener on :9111,
// and uploads two custom voices (a copy of the base model) through the admin API.
import http from "node:http";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.env.BASE || "localhost:8099";
const HERE = path.dirname(fileURLToPath(import.meta.url));
process.env.MODAL_SESSION_SECRET = "testsecret";
process.env.KEYS_PATH = path.join(os.tmpdir(), "rt-k.json");
const keys = await import(path.join(HERE, "../../gateway/keys.js"));
const OWNER = keys.createSessionToken("owner-id");
const OTHER = keys.createSessionToken("other-id");
const STATIC = "static";
const auth = (t) => ({ authorization: `Bearer ${t}`, "content-type": "application/json" });

let failures = 0;
const ok = (name, cond, extra = "") => { console.log(`${cond ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`); if (!cond) failures++; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- usage-report listener ----
const reports = [];
http.createServer((req, res) => {
  let b = ""; req.on("data", (c) => (b += c)); req.on("end", () => {
    reports.push({ auth: req.headers.authorization, ...JSON.parse(b) }); res.end("{}");
  });
}).listen(9111);

// ---- helpers ----
const SHORT = "Thanks for calling, I can help you with that.";
const LONG = Array.from({ length: 14 }, (_, i) => `This is sentence number ${i + 1} of a long paragraph that keeps the synthesizer busy for a while.`).join(" ");

function wsSession(token, query = "") {
  const ws = new WebSocket(`ws://${BASE}/tts?token=${token}${query}`);
  ws.binaryType = "arraybuffer";
  const q = []; let waiter = null;
  ws.onmessage = (e) => { q.push(e.data); waiter?.(); };
  ws.onclose = (e) => { q.push({ closed: e.code }); waiter?.(); };
  const next = async () => { while (!q.length) await new Promise((r) => (waiter = r)); return q.shift(); };
  const opened = new Promise((r, j) => { ws.onopen = r; setTimeout(() => j(new Error("ws open timeout")), 5000); });
  // synth: returns {chunks:[{meta, bytes}], end: msg}
  async function synth(msg) {
    ws.send(JSON.stringify({ type: "synthesize", ...msg }));
    const chunks = []; let meta = null;
    for (;;) {
      const m = await next();
      if (m.closed) return { chunks, end: { type: "closed", code: m.closed } };
      if (typeof m === "string") {
        const j = JSON.parse(m);
        if (j.type === "chunk_meta") meta = j; else return { chunks, end: j };
      } else { chunks.push({ meta, bytes: new Uint8Array(m) }); }
    }
  }
  return { ws, opened, synth, next };
}
const concat = (chunks) => { const n = chunks.reduce((a, c) => a + c.bytes.length, 0), o = new Uint8Array(n); let p = 0; for (const c of chunks) { o.set(c.bytes, p); p += c.bytes.length; } return o; };
const i16 = (u8) => new Int16Array(u8.buffer, u8.byteOffset, u8.length >> 1);
function ulaw(b) { const o = new Int16Array(b.length); for (let i = 0; i < b.length; i++) { const u = ~b[i] & 0xff; let t = (((u & 15) << 3) + 0x84) << ((u & 0x70) >> 4); o[i] = ((u & 0x80) ? 0x84 - t : t - 0x84); } return o; }
function alaw(b) { const o = new Int16Array(b.length); for (let i = 0; i < b.length; i++) { const a = b[i] ^ 0x55, s = (a & 0x70) >> 4; const t = s === 0 ? ((a & 15) << 4) + 8 : (((a & 15) << 4) + 0x108) << (s - 1); o[i] = ((a & 0x80) ? t : -t); } return o; }
const rmsDb = (x) => { let s = 0; for (const v of x) s += v * v; return 10 * Math.log10(s / x.length + 1e-9); };
const WIDTH = { pcm_24000: 2, pcm_8000: 2, mulaw_8000: 1, alaw_8000: 1 };
const SR = { pcm_24000: 24000, pcm_8000: 8000, mulaw_8000: 8000, alaw_8000: 8000 };

async function stream(token, body, opts = {}) {
  return fetch(`http://${BASE}/v1/tts/stream`, { method: "POST", headers: token ? auth(token) : { "content-type": "application/json" }, body: JSON.stringify(body), ...opts });
}
async function active() { return (await (await fetch(`http://${BASE}/health`)).json()).active; }

// ---- admin: upload voices ----
async function putVoice(id, owner, expectStatus = 200) {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "voice-"));
  const models = path.join(HERE, "../models");
  fs.copyFileSync(path.join(models, "full_ft.onnx"), path.join(d, "model.onnx"));
  fs.copyFileSync(path.join(models, "full_ft.onnx.json"), path.join(d, "model.onnx.json"));
  fs.writeFileSync(path.join(d, "owner.json"), JSON.stringify(owner));
  execFileSync("tar", ["-cf", path.join(d, "v.tar"), "-C", d, "model.onnx", "model.onnx.json", "owner.json"]);
  const r = await fetch(`http://${BASE}/admin/voices/${id}`, { method: "PUT", headers: { authorization: "Bearer adm" }, body: fs.readFileSync(path.join(d, "v.tar")) });
  fs.rmSync(d, { recursive: true });
  return { status: r.status, body: await r.text() };
}

// =============== tests ===============
console.log("--- admin owner.json validation");
for (const [label, owner, want] of [
  ["key_ids ok", { key_ids: ["owner-id"] }, 200], ["public ok", { public: true }, 200],
  ["empty key_ids rejected", { key_ids: [] }, 400], ["public:false rejected", { public: false }, 400],
  ["public:'yes' rejected", { public: "yes" }, 400], ["empty object rejected", {}, 400], ["array rejected", [1], 400],
]) {
  const id = label.startsWith("key_ids ok") ? "ownerv" : label === "public ok" ? "pubv" : "badv";
  const r = await putVoice(id, owner);
  ok(`PUT owner.json ${label}`, r.status === want, `status ${r.status} ${want === 400 ? r.body : ""}`);
}

console.log("--- websocket: formats");
{
  const s = wsSession(STATIC); await s.opened;
  const r = await s.synth({ text: SHORT, voice: "x", speed: 1.0 });
  const b = concat(r.chunks);
  ok("default: no format => pcm_24000 meta", r.chunks.every((c) => c.meta.format === "pcm_24000" && c.meta.sample_rate === 24000) && r.end.type === "done");
  ok("default: bytes == audio_s*24000*2", Math.abs(b.length / 2 - r.chunks.reduce((a, c) => a + c.meta.audio_s, 0) * 24000) < 2 * r.chunks.length);
  const rms = {};
  for (const fmt of Object.keys(WIDTH)) {
    const r = await s.synth({ text: SHORT, format: fmt });
    const b = concat(r.chunks), secs = r.chunks.reduce((a, c) => a + c.meta.audio_s, 0);
    ok(`${fmt}: meta format/sample_rate`, r.chunks.every((c) => c.meta.format === fmt && c.meta.sample_rate === SR[fmt]) && r.end.type === "done");
    ok(`${fmt}: byte length matches audio_s`, Math.abs(b.length / WIDTH[fmt] / SR[fmt] - secs) < 0.001 * r.chunks.length + 1e-3, `${b.length} B, ${secs.toFixed(2)} s`);
    const pcm = fmt.startsWith("pcm") ? i16(b.slice()) : fmt.startsWith("mulaw") ? ulaw(b) : alaw(b);
    rms[fmt] = rmsDb(pcm);
  }
  const spread = Math.max(...Object.values(rms)) - Math.min(...Object.values(rms));
  ok("decoded RMS level agrees across formats (<3 dB)", spread < 3, JSON.stringify(Object.fromEntries(Object.entries(rms).map(([k, v]) => [k, +v.toFixed(1)]))));
  const bad = await s.synth({ text: SHORT, format: "opus" });
  ok("bad format => error message", bad.end.type === "error" && /unknown format/.test(bad.end.message) && bad.chunks.length === 0, bad.end.message);
  const again = await s.synth({ text: SHORT, format: "mulaw_8000" });
  ok("connection still usable after bad format", again.end.type === "done" && again.chunks.length > 0);
  s.ws.close();
}

console.log("--- http: auth, validation, cors");
{
  let r = await stream(null, { text: SHORT }); ok("401 without token", r.status === 401); await r.text();
  r = await stream("garbage", { text: SHORT }); ok("401 with bad token", r.status === 401); await r.text();
  r = await stream(STATIC, { text: "" }); ok("400 empty text", r.status === 400); await r.text();
  r = await stream(STATIC, { text: "x".repeat(5001) }); ok("413 text too long", r.status === 413); await r.text();
  r = await stream(STATIC, { text: SHORT, format: "opus" }); ok("400 bad format", r.status === 400, await r.text());
  r = await fetch(`http://${BASE}/v1/tts/stream`, { method: "OPTIONS", headers: { origin: "https://app.example", "access-control-request-method": "POST", "access-control-request-headers": "authorization,content-type" } });
  ok("OPTIONS preflight 204 + CORS", r.status === 204 && r.headers.get("access-control-allow-origin") === "*" && /authorization/i.test(r.headers.get("access-control-allow-headers")) && /content-type/i.test(r.headers.get("access-control-allow-headers")), r.headers.get("access-control-allow-headers"));
  r = await fetch(`http://${BASE}/tts`, { method: "OPTIONS" });
  ok("no CORS on other paths", !r.headers.get("access-control-allow-origin"));
}

console.log("--- http: happy path, all formats");
for (const fmt of [undefined, ...Object.keys(WIDTH)]) {
  const f = fmt ?? "pcm_24000";
  const r = await stream(STATIC, { text: SHORT, ...(fmt ? { format: fmt } : {}) });
  const b = new Uint8Array(await r.arrayBuffer());
  const ct = r.headers.get("content-type");
  ok(`http ${fmt ?? "(default)"}: 200, headers, body`, r.status === 200 && r.headers.get("x-sample-rate") === String(SR[f]) && r.headers.get("x-audio-format") === f && r.headers.get("access-control-allow-origin") === "*" && b.length > 1000,
    `${ct} ${b.length} B (${(b.length / WIDTH[f] / SR[f]).toFixed(2)} s)`);
}

console.log("--- http: first byte before completion");
{
  const t0 = performance.now();
  const r = await stream(STATIC, { text: LONG });
  const rd = r.body.getReader(); const first = await rd.read(); const tFirst = performance.now() - t0;
  let n = first.value.length, chunks = 1;
  for (;;) { const { done, value } = await rd.read(); if (done) break; n += value.length; chunks++; }
  const tAll = performance.now() - t0;
  ok("first byte well before full completion", tFirst < 0.5 * tAll, `first ${tFirst.toFixed(0)} ms, total ${tAll.toFixed(0)} ms, ${(n / 48000).toFixed(1)} s audio, ${chunks} reads`);
}

console.log("--- http: billing");
reports.length = 0;
{
  let r = await stream(OWNER, { text: SHORT }); await r.arrayBuffer(); await sleep(1500);
  ok("completed request billed once with len(text)", reports.length === 1 && reports[0].id === "owner-id" && reports[0].chars === SHORT.length && reports[0].engine === "piper" && reports[0].auth === "Bearer us", JSON.stringify(reports));
  reports.length = 0;
  r = await stream(STATIC, { text: SHORT }); await r.arrayBuffer(); await sleep(1000);
  ok("static token not billed", reports.length === 0);
  const ac = new AbortController();
  r = await stream(OWNER, { text: LONG }, { signal: ac.signal });
  const rd = r.body.getReader(); await rd.read(); ac.abort(); await rd.cancel().catch(() => {});
  await sleep(4000);
  ok("client abort mid-stream => not billed", reports.length === 0, JSON.stringify(reports));
  let a = 0; for (let i = 0; i < 20 && (a = await active()) !== 0; i++) await sleep(500);
  ok("abort released the capacity slot (health.active == 0)", a === 0, `active=${a}`);
}

console.log("--- http: capacity");
{
  const max = (await (await fetch(`http://${BASE}/health`)).json()).max;
  const big = LONG.repeat(3);
  const acs = [], rds = [];
  for (let i = 0; i < max; i++) { const ac = new AbortController(); acs.push(ac); const r = await stream(STATIC, { text: big }, { signal: ac.signal }); rds.push(r); }
  ok(`${max} concurrent streams accepted`, rds.every((r) => r.status === 200));
  const r = await stream(STATIC, { text: SHORT });
  ok("next request => 503 + Retry-After", r.status === 503 && r.headers.get("retry-after") === "1", `status ${r.status}`); await r.text();
  const s = wsSession(STATIC); await s.opened; const m = await s.next();
  ok("websocket sees the same capacity (HTTP streams counted)", typeof m === "string" && /capacity/.test(m) || m.closed === 1013, String(m).slice(0, 80));
  for (const ac of acs) ac.abort();
  for (const r of rds) r.body.cancel().catch(() => {});
  let a = 1; for (let i = 0; i < 30 && (a = await active()) !== 0; i++) await sleep(500);
  ok("slots released after aborts", a === 0, `active=${a}`);
  const r2 = await stream(STATIC, { text: SHORT }); ok("accepts again after release", r2.status === 200); await r2.arrayBuffer();
}

console.log("--- voices: public vs owner");
{
  const post = async (tok, voice) => { const r = await stream(tok, { text: SHORT, voice }); const t = await r.arrayBuffer(); return { status: r.status, len: t.byteLength, txt: r.status === 200 ? "" : new TextDecoder().decode(t) }; };
  let r = await post(OWNER, "custom:ownerv"); ok("owner can use owner voice", r.status === 200 && r.len > 1000);
  r = await post(OTHER, "custom:ownerv"); const forbidden = r; ok("other key cannot use owner voice", r.status === 404 && /unknown voice/.test(r.txt), r.txt);
  r = await post(OTHER, "custom:doesnotexist"); ok("missing voice: identical error to forbidden", r.status === 404 && r.txt === forbidden.txt, r.txt);
  r = await post(OTHER, "custom:pubv"); ok("public voice usable by any authenticated key", r.status === 200 && r.len > 1000);
  r = await post(OWNER, "custom:pubv"); ok("public voice usable by owner-key too", r.status === 200);
  r = await post(STATIC, "custom:ownerv"); ok("static token may use any voice", r.status === 200);
  r = await post(null, "custom:pubv"); ok("public voice still needs auth", r.status === 401);
  const s = wsSession(OTHER); await s.opened;
  let w = await s.synth({ text: SHORT, voice: "custom:pubv" }); ok("ws: public voice ok for other key", w.end.type === "done" && w.chunks.length > 0);
  w = await s.synth({ text: SHORT, voice: "custom:ownerv" }); ok("ws: owner voice denied for other key (same error)", w.end.type === "error" && w.end.message === "unknown voice");
  w = await s.synth({ text: SHORT }); ok("ws: still usable after voice error", w.end.type === "done");
  s.ws.close();
}

console.log("--- websocket: billing unchanged");
{
  reports.length = 0;
  const s = wsSession(OWNER); await s.opened;
  const w = await s.synth({ text: SHORT, format: "mulaw_8000" }); await sleep(1500);
  ok("ws done => billed len(text)", w.end.type === "done" && reports.length === 1 && reports[0].chars === SHORT.length);
  s.ws.close();
}

console.log(failures ? `\n${failures} FAILED` : "\nALL PASSED");
process.exit(failures ? 1 : 0);
