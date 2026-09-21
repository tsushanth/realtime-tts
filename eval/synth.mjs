#!/usr/bin/env node
// Eval framework, step 1: synthesize the whole test set (eval/testset.json) through every
// engine/config, measuring time-to-first-audio-byte the same way benchmarks/latency_probe.js
// does (same vantage, interleaved across engines to share network conditions, warm connection
// per engine reused across sentences). Saves audio to eval/audio/<engine>/<lang>-<id>.wav and
// raw latency numbers to eval/results/latency.json. Re-run this after any engine change to get
// a fresh comparison - this is a framework, not a one-shot script.
//
// Env: TTS_GATEWAY_API_KEY (our Piper/Kokoro key, billing-enabled), ELEVENLABS_API_KEY.
// Usage: node eval/synth.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const testset = JSON.parse(fs.readFileSync(path.join(HERE, "testset.json"), "utf8"));
const AUDIO_DIR = path.join(HERE, "audio");
const RESULTS_DIR = path.join(HERE, "results");
fs.mkdirSync(RESULTS_DIR, { recursive: true });

const GATEWAY_KEY = process.env.TTS_GATEWAY_API_KEY;
const EL_KEY = process.env.ELEVENLABS_API_KEY;
if (!GATEWAY_KEY) throw new Error("set TTS_GATEWAY_API_KEY");
if (!EL_KEY) throw new Error("set ELEVENLABS_API_KEY");

// Piper voice per language - the best DEPLOYED voice we have for each (see voices/catalog.json
// and voices/LICENSES.md for tier/licence detail). Kokoro is English-only (its published voice
// set), so it's only run for lang "en".
const PIPER_VOICE = { en: "custom:en-us-john", es: "custom:es-es-davefx", de: "custom:de-de-mls-f", fr: "custom:fr-fr-mls-f" };
const KOKORO_VOICE = "af_heart";
const EL_VOICE = process.env.ELEVENLABS_VOICE_ID || "JBFqnCBsd6RMkjVDRZzb"; // multilingual voice, same id used elsewhere in this repo
const EL_MODELS = { "elevenlabs-flash": "eleven_flash_v2_5", "elevenlabs-multilingual": "eleven_multilingual_v2" };

async function authorize(engine) {
  const r = await fetch("https://api.readaloudai.org/tts/authorize", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ key: GATEWAY_KEY, engine }),
  });
  if (!r.ok) throw new Error(`authorize ${engine} failed: ${r.status} ${await r.text()}`);
  return r.json();
}

// One persistent WebSocket per engine, opened once and reused for every sentence - this is what
// "warm connection" means in benchmarks/latency_probe.js's methodology (a real customer holds one
// connection open across a call's turns, not a fresh connection per utterance).
function openPersistent(url, token) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${url}?token=${token}`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => resolve(ws);
    ws.onerror = () => reject(new Error("ws open error"));
  });
}

// The persistent connection can go idle for tens of seconds between turns while other engines in the
// interleaved run take their turn (ElevenLabs Multilingual alone averages ~1-1.5s/request), and it was
// observed to silently die mid-run (found by this eval run itself, not anticipated in advance) - reopen
// transparently rather than let one dead socket fail every remaining job for that engine.
async function ensureOpen(ws, authorize, engine) {
  if (ws.readyState === WebSocket.OPEN) return ws;
  const auth = await authorize(engine);
  return openPersistent(auth.url, auth.token);
}

function wsSynthOn(ws, text, voice) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let t0, firstMs;
    const timer = setTimeout(() => { cleanup(); reject(new Error("timeout")); }, 30000);
    const onMessage = async (e) => {
      try {
        if (typeof e.data !== "string") {
          // Node's global WebSocket delivers binary frames as a Blob regardless of binaryType, so
          // normalize explicitly instead of assuming ArrayBuffer (Buffer.from(Blob) silently produces
          // garbage/throws, which previously hung this promise forever with no rejection).
          const ab = e.data instanceof Blob ? await e.data.arrayBuffer() : e.data;
          firstMs ??= performance.now() - t0;
          chunks.push(Buffer.from(ab));
          return;
        }
        const j = JSON.parse(e.data);
        if (j.type === "done") { clearTimeout(timer); cleanup(); resolve({ pcm: Buffer.concat(chunks), firstMs, sr: 24000 }); }
        if (j.type === "error") { clearTimeout(timer); cleanup(); reject(new Error(j.message)); }
      } catch (err) { clearTimeout(timer); cleanup(); reject(err); }
    };
    function cleanup() { ws.removeEventListener("message", onMessage); }
    ws.addEventListener("message", onMessage);
    t0 = performance.now();
    ws.send(JSON.stringify({ type: "synthesize", text, voice, speed: 1.0 }));
  });
}

async function elevenSynth(text, model) {
  const t0 = performance.now();
  const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${EL_VOICE}/stream?output_format=pcm_24000`, {
    method: "POST",
    headers: { "xi-api-key": EL_KEY, "content-type": "application/json" },
    body: JSON.stringify({ text, model_id: model }),
  });
  if (!r.ok) throw new Error(`elevenlabs ${model} failed: ${r.status} ${await r.text()}`);
  const reader = r.body.getReader();
  const chunks = [];
  let firstMs;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    firstMs ??= performance.now() - t0;
    chunks.push(Buffer.from(value));
  }
  return { pcm: Buffer.concat(chunks), firstMs, sr: 24000 };
}

function pcmToWav(pcm, sr) {
  const header = Buffer.alloc(44);
  header.write("RIFF", 0); header.writeUInt32LE(36 + pcm.length, 4); header.write("WAVE", 8);
  header.write("fmt ", 12); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sr, 24); header.writeUInt32LE(sr * 2, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
  header.write("data", 36); header.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([header, pcm]);
}

async function main() {
  let piperAuth = await authorize("piper");
  let kokoroAuth = await authorize("kokoro");
  let piperWs = await openPersistent(piperAuth.url, piperAuth.token);
  let kokoroWs = await openPersistent(kokoroAuth.url, kokoroAuth.token);
  console.log("authorized + connected piper + kokoro (persistent, reused for every sentence; auto-reconnect on drop)");

  const jobs = [];
  for (const s of testset.sentences) {
    jobs.push({ engine: "piper", lang: s.lang, id: s.id, text: s.text, voice: PIPER_VOICE[s.lang] });
    if (s.lang === "en") jobs.push({ engine: "kokoro", lang: s.lang, id: s.id, text: s.text, voice: KOKORO_VOICE });
    for (const engine of Object.keys(EL_MODELS)) jobs.push({ engine, lang: s.lang, id: s.id, text: s.text });
  }
  // Interleave by shuffling deterministically (not grouped by engine) so network conditions are shared
  // across engines the same way benchmarks/latency_probe.js does, rather than one engine getting an
  // unlucky run of congestion.
  for (let i = jobs.length - 1; i > 0; i--) {
    const j = Math.floor(((i * 2654435761) % 2147483647) / 2147483647 * (i + 1));
    [jobs[i], jobs[j]] = [jobs[j], jobs[i]];
  }

  const results = [];
  let done = 0;
  for (const job of jobs) {
    try {
      let out;
      if (job.engine === "piper") { piperWs = await ensureOpen(piperWs, authorize, "piper"); out = await wsSynthOn(piperWs, job.text, job.voice); }
      else if (job.engine === "kokoro") { kokoroWs = await ensureOpen(kokoroWs, authorize, "kokoro"); out = await wsSynthOn(kokoroWs, job.text, job.voice); }
      else out = await elevenSynth(job.text, EL_MODELS[job.engine]);
      const dir = path.join(AUDIO_DIR, job.engine);
      fs.mkdirSync(dir, { recursive: true });
      const wavPath = path.join(dir, `${job.lang}-${job.id}.wav`);
      fs.writeFileSync(wavPath, pcmToWav(out.pcm, out.sr));
      results.push({ engine: job.engine, lang: job.lang, id: job.id, text: job.text, first_byte_ms: Math.round(out.firstMs), audio_bytes: out.pcm.length, wav: path.relative(HERE, wavPath) });
      done++;
      console.log(`[${done}/${jobs.length}] ${job.engine} ${job.lang}/${job.id} first_byte=${Math.round(out.firstMs)}ms`);
    } catch (e) {
      results.push({ engine: job.engine, lang: job.lang, id: job.id, text: job.text, error: e.message });
      console.log(`[${done + 1}/${jobs.length}] ${job.engine} ${job.lang}/${job.id} ERROR ${e.message}`);
      done++;
    }
    // The connection itself stays open the whole run (that's the point - warm, reused); nothing to
    // refresh per request. Session tokens only gate the initial WebSocket handshake, not messages sent
    // after it's open, so a 60s token TTL doesn't affect an already-open connection.
  }

  fs.writeFileSync(path.join(RESULTS_DIR, "latency.json"), JSON.stringify(results, null, 2));
  console.log(`wrote ${path.join(RESULTS_DIR, "latency.json")}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
