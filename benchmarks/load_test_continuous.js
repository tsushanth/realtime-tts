const WebSocket = require('ws');
const KEY = process.env.TTS_GATEWAY_API_KEY, URL_ = 'ws://piper-tts-sjc.internal:8080/tts';
const TEXTS = ['Thanks for calling, I can help you with that. Let me pull up your account details right now.',
  'Your order should arrive within three to five business days, and I will send a confirmation email shortly.',
  'I understand your frustration, let me see what I can do to make this right.',
  'Is there anything else I can help you with today?', 'Your extension is six six three five.'];
const now = () => Number(process.hrtime.bigint() / 1000n) / 1000;
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : null; };
function worker(ms, wid, log) {
  return new Promise((resolve) => {
    const ws = new WebSocket(URL_, { headers: { Authorization: `Bearer ${KEY}` } });
    const end = now() + ms; let i = wid;
    ws.on('open', next);
    ws.on('error', () => resolve());
    let t0, first, bytes;
    function next() {
      if (now() > end) { ws.close(); return resolve(); }
      t0 = now(); first = null; bytes = 0;
      ws.send(JSON.stringify({ type: 'synthesize', text: TEXTS[i++ % TEXTS.length], speed: 1.0 }));
    }
    ws.on('message', (d, bin) => {
      if (bin) { if (first === null) first = now() - t0; bytes += d.length; return; }
      const j = JSON.parse(d.toString());
      if (j.type === 'done') { const a = bytes / 48000; log.push({ at: t0, ttfa: first, audio: a, total: now() - t0 }); setTimeout(next, a * 1000 + 1500); }
    });
  });
}
async function run(n, ms) {
  const log = []; const t0 = now();
  await Promise.all(Array.from({ length: n }, (_, k) => worker(ms, k, log)));
  return { log, wall: (now() - t0) / 1000 };
}
(async () => {
  const out = {};
  await new Promise((r) => setTimeout(r, 90000)); // let any burst balance recover first
  { const { log } = await run(1, 8000); out['after 90s idle, 1 caller'] = { ttfa_p50: Math.round(pct(log.map((l) => l.ttfa), .5)), n: log.length }; }
  for (const n of [2, 4, 6, 8, 12]) {
    const { log, wall } = await run(n, 45000);
    const t00 = Math.min(...log.map((l) => l.at)); const half = log.filter((l) => l.at - t00 > 22000).map((l) => l.ttfa);
    const tt = log.map((l) => l.ttfa);
    out[`${n} realistic calls, 45s`] = { requests: log.length, ttfa_p50: Math.round(pct(tt, .5)), ttfa_p95: Math.round(pct(tt, .95)), second_half_p50: Math.round(pct(half, .5)) };
  }
  console.log(JSON.stringify(out, null, 1)); process.exit(0);
})();
