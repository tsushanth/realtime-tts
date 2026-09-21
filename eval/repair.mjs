#!/usr/bin/env node
// Reruns only the jobs that errored in the last synth.mjs run (e.g. a dropped connection) and merges
// the fresh results into eval/results/latency.json in place, instead of repaying for engines (notably
// ElevenLabs, which costs real money) that already succeeded.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RESULTS = path.join(HERE, "results", "latency.json");
const results = JSON.parse(fs.readFileSync(RESULTS, "utf8"));
const AUDIO_DIR = path.join(HERE, "audio");

const GATEWAY_KEY = process.env.TTS_GATEWAY_API_KEY;
const EL_KEY = process.env.ELEVENLABS_API_KEY;

const PIPER_VOICE = { en: "custom:en-us-john", es: "custom:es-es-davefx", de: "custom:de-de-mls-f", fr: "custom:fr-fr-mls-f" };
const KOKORO_VOICE = "af_heart";
const EL_VOICE = process.env.ELEVENLABS_VOICE_ID || "JBFqnCBsd6RMkjVDRZzb";
const EL_MODELS = { "elevenlabs-flash": "eleven_flash_v2_5", "elevenlabs-multilingual": "eleven_multilingual_v2" };

async function authorize(engine) {
  const r = await fetch("https://api.readaloudai.org/tts/authorize", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ key: GATEWAY_KEY, engine }),
  });
  if (!r.ok) throw new Error(`authorize ${engine} failed: ${r.status} ${await r.text()}`);
  return r.json();
}
function openPersistent(url, token) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`${url}?token=${token}`);
    ws.binaryType = "arraybuffer";
    const t = setTimeout(() => reject(new Error("ws open timeout")), 15000);
    ws.onopen = () => { clearTimeout(t); resolve(ws); };
    ws.onerror = () => { clearTimeout(t); reject(new Error("ws open error")); };
  });
}
async function ensureOpen(ws, engine) {
  if (ws && ws.readyState === WebSocket.OPEN) return ws;
  const auth = await authorize(engine);
  return openPersistent(auth.url, auth.token);
}
function wsSynthOn(ws, text, voice) {
  return new Promise((resolve, reject) => {
    const chunks = []; let t0, firstMs;
    const timer = setTimeout(() => { cleanup(); reject(new Error("timeout")); }, 30000);
    const onMessage = async (e) => {
      try {
        if (typeof e.data !== "string") {
          const ab = e.data instanceof Blob ? await e.data.arrayBuffer() : e.data;
          firstMs ??= performance.now() - t0; chunks.push(Buffer.from(ab)); return;
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
    method: "POST", headers: { "xi-api-key": EL_KEY, "content-type": "application/json" },
    body: JSON.stringify({ text, model_id: model }),
  });
  if (!r.ok) throw new Error(`elevenlabs ${model} failed: ${r.status} ${await r.text()}`);
  const reader = r.body.getReader(); const chunks = []; let firstMs;
  for (;;) { const { done, value } = await reader.read(); if (done) break; firstMs ??= performance.now() - t0; chunks.push(Buffer.from(value)); }
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
  let piperWs, kokoroWs;
  const RETARGET = (process.env.RETARGET_IDS || "").split(",").filter(Boolean);
  const failed = RETARGET.length
    ? results.filter((r) => RETARGET.includes(`${r.engine}|${r.lang}|${r.id}`))
    : results.filter((r) => r.error);
  process.stdout.write(`repairing ${failed.length} jobs\n`);
  if (RETARGET.length) {
    // Warm each engine's connection with a throwaway request first so the FIRST real measurement
    // isn't penalized by reconnect/cold-voice-load overhead the way the earlier repair run was.
    for (const engine of new Set(failed.map((f) => f.engine))) {
      if (engine === "piper") { piperWs = await ensureOpen(piperWs, "piper"); await wsSynthOn(piperWs, "warm up.", "custom:en-us-john"); }
      else if (engine === "kokoro") { kokoroWs = await ensureOpen(kokoroWs, "kokoro"); await wsSynthOn(kokoroWs, "warm up.", KOKORO_VOICE); }
    }
    process.stdout.write("warm-up done\n");
  }
  for (const job of failed) {
    try {
      let out;
      if (job.engine === "piper") { piperWs = await ensureOpen(piperWs, "piper"); out = await wsSynthOn(piperWs, job.text, PIPER_VOICE[job.lang]); }
      else if (job.engine === "kokoro") { kokoroWs = await ensureOpen(kokoroWs, "kokoro"); out = await wsSynthOn(kokoroWs, job.text, KOKORO_VOICE); }
      else out = await elevenSynth(job.text, EL_MODELS[job.engine]);
      const dir = path.join(AUDIO_DIR, job.engine);
      fs.mkdirSync(dir, { recursive: true });
      const wavPath = path.join(dir, `${job.lang}-${job.id}.wav`);
      fs.writeFileSync(wavPath, pcmToWav(out.pcm, out.sr));
      delete job.error;
      job.first_byte_ms = Math.round(out.firstMs); job.audio_bytes = out.pcm.length; job.wav = path.relative(HERE, wavPath);
      process.stdout.write(`OK ${job.engine} ${job.lang}/${job.id} first_byte=${job.first_byte_ms}ms\n`);
    } catch (e) {
      job.error = e.message;
      process.stdout.write(`STILL FAILING ${job.engine} ${job.lang}/${job.id}: ${e.message}\n`);
    }
  }
  fs.writeFileSync(RESULTS, JSON.stringify(results, null, 2));
  const stillFailing = results.filter((r) => r.error).length;
  console.log(`done. ${stillFailing} still failing out of ${results.length} total.`);
}
main().catch((e) => { console.error(e); process.exit(1); });
