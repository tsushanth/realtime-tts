const WebSocket = require('ws');
const KEY = process.env.TTS_GATEWAY_API_KEY;
const TARGETS = [
  { name: 'piper-on-fly(4 threads,private)', url: 'ws://piper-tts-sjc.internal:8080/tts', fresh: 5, warm: 8 },
  { name: 'piper-on-modal(direct)', url: 'wss://t-sushanth--realtime-tts-worker-piper-web.modal.run/tts', fresh: 5, warm: 8 },
  { name: 'kokoro-via-cloudflare(prod)', url: 'wss://tts.readaloudai.org/tts', fresh: 4, warm: 6 },
];
const TEXTS = ['Thanks for calling, how can I help you today?', 'Your extension is six six three five.',
  'Is there anything else I can help you with today?', 'I understand your frustration, let me see what I can do.',
  'Your order should arrive within three to five business days.', 'Let me pull up your account details right now.'];
const now = () => Number(process.hrtime.bigint() / 1000n) / 1000; // ms
function open(url) {
  return new Promise((res, rej) => {
    const t0 = now();
    const ws = new WebSocket(url, { headers: { Authorization: `Bearer ${KEY}` }, handshakeTimeout: 60000 });
    ws.once('open', () => res({ ws, connectMs: now() - t0, t0 }));
    ws.once('error', rej);
  });
}
function synth(ws, text) {
  return new Promise((res, rej) => {
    const t0 = now(); let first = null; let bytes = 0;
    const to = setTimeout(() => rej(new Error('timeout')), 60000);
    ws.on('message', function h(data, isBinary) {
      if (isBinary) { if (first === null) first = now() - t0; bytes += data.length; return; }
      const j = JSON.parse(data.toString());
      if (j.type === 'done' || j.type === 'cancelled' || j.type === 'error') {
        ws.off('message', h); clearTimeout(to);
        j.type === 'error' ? rej(new Error(j.message)) : res({ firstAudioMs: first, totalMs: now() - t0, audioSeconds: bytes / 48000 });
      }
    });
    ws.send(JSON.stringify({ type: 'synthesize', text, voice: 'af_heart', speed: 1.0 }));
  });
}
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const round = (x) => Math.round(x);
const https = require('https');
const EKEY = process.env.ELEVENLABS_API_KEY;
const VOICE = process.env.ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb';
function elevenReq(agent, text, model) {
  return new Promise((res, rej) => {
    const t0 = now(); const body = JSON.stringify({ text, model_id: model });
    const req = https.request({ host: 'api.elevenlabs.io', path: `/v1/text-to-speech/${VOICE}/stream?output_format=pcm_24000`, method: 'POST', agent,
      headers: { 'xi-api-key': EKEY, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, (r) => {
      if (r.statusCode !== 200) { let b = ''; r.on('data', (d) => b += d); r.on('end', () => rej(new Error('status ' + r.statusCode + ' ' + b.slice(0, 120)))); return; }
      let first = null, bytes = 0;
      r.on('data', (d) => { if (first === null) first = now() - t0; bytes += d.length; });
      r.on('end', () => res({ firstAudioMs: first, totalMs: now() - t0, audioSeconds: bytes / 48000 }));
    });
    req.on('error', rej); req.write(body); req.end();
  });
}
async function measureEleven(model, freshN, warmN) {
  const r = { fresh: [], warm: [] };
  { const p0 = now(); await elevenReq(new https.Agent({ keepAlive: false }), TEXTS[0], model); r.primeMs = round(now() - p0); }
  for (let i = 0; i < freshN; i++) { const a = new https.Agent({ keepAlive: false }); const x = await elevenReq(a, TEXTS[i % TEXTS.length], model); r.fresh.push({ connectMs: 0, connectToFirstAudioMs: round(x.firstAudioMs) }); }
  const a = new https.Agent({ keepAlive: true }); await elevenReq(a, TEXTS[0], model);
  const w = []; for (let i = 0; i < warmN; i++) w.push((await elevenReq(a, TEXTS[i % TEXTS.length], model)).firstAudioMs);
  r.warm = { firstAudioMs: w.map(round), p50: round(pct(w, 0.5)), max: round(Math.max(...w)) }; r.rttMs = { p50: 0, max: 0 };
  a.destroy(); return r;
}

(async () => {
  const out = {};
  for (let rd = 1; rd <= 2; rd++) {
   for (const m of ['eleven_multilingual_v2', 'eleven_flash_v2_5']) {
    try { out[`elevenlabs ${m} #r${rd}`] = await measureEleven(m, 5, 6); } catch (e) { out[`elevenlabs ${m} #r${rd}`] = { error: String(e.message || e).slice(0, 200) }; }
   }
   for (const t of TARGETS) {
    const r = { fresh: [], warm: [] };
    try {
      { const p0 = now(); const { ws } = await open(t.url); await synth(ws, TEXTS[0]); r.primeMs = round(now() - p0); ws.close(); }
      for (let i = 0; i < t.fresh; i++) {
        const { ws, connectMs, t0 } = await open(t.url);
        const s = await synth(ws, TEXTS[i % TEXTS.length]);
        r.fresh.push({ connectMs: round(connectMs), connectToFirstAudioMs: round(connectMs + s.firstAudioMs) });
        ws.close();
      }
      const { ws, connectMs } = await open(t.url);
      await synth(ws, TEXTS[0]); // discard: first request on this connection
      const rt = [];
      for (let i = 0; i < 5; i++) { const t0 = now(); await new Promise((res) => { ws.once('message', res); ws.send('not json'); }); rt.push(now() - t0); }
      r.rttMs = { p50: round(pct(rt, 0.5)), max: round(Math.max(...rt)) };
      const w = [];
      for (let i = 0; i < t.warm; i++) w.push((await synth(ws, TEXTS[i % TEXTS.length])).firstAudioMs);
      r.warm = { firstAudioMs: w.map(round), p50: round(pct(w, 0.5)), max: round(Math.max(...w)) };
      ws.close();
    } catch (e) { r.error = String(e.message || e).slice(0, 200); }
    out[`${t.name} #r${rd}`] = r;
  }
  }
  console.log(JSON.stringify(out, null, 1));
  process.exit(0);
})();
