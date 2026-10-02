// Local end-to-end smoke for the Telnyx path. No provider keys, no real Telnyx, no real database:
//   - a tiny fake PostgREST/storage server stands in for Supabase (one number routed with carrier=telnyx,
//     one tenant with a 1 s intro jingle, which gives the session real audio to write back without TTS/LLM)
//   - the real server.js runs on a spare port with TELNYX_ENABLED=1
//   - a fake "Telnyx" posts the TeXML webhook, opens the websocket URL the TeXML returned, sends `start`
//     plus 3 s of synthetic PCMU speech-like audio, and checks the session wrote paced 160-byte frames back.
// Run: node scripts/telnyxSmoke.mjs
import http from 'node:http';
import { spawn } from 'node:child_process';
import { WebSocket } from 'ws';

const FAKE_PORT = 18931;
const APP_PORT = 18932;
const KEY = 'smoke-webhook-key';

// G.711 mu-law encoder (same algorithm as twilioAdapter.js) just to synthesize caller audio.
function mulaw(s) {
  let sign = 0; if (s < 0) { s = -s; sign = 0x80; } if (s > 32635) s = 32635; s += 0x84;
  let e = 7; for (let m = 0x4000; (s & m) === 0 && e > 0; m >>= 1) e--;
  return ~(sign | (e << 4) | ((s >> (e + 3)) & 0x0f)) & 0xff;
}
const JINGLE = Buffer.from(Array.from({ length: 8000 }, (_, i) => mulaw(Math.round(8000 * Math.sin(i / 6)))));

const rows = {
  calldesk_phone_numbers: [{ tenant_id: 'tenant-smoke', inbound_agent_version_id: 'v1', carrier: 'telnyx' }],
  calldesk_agent_versions: [{ voice_engine: 'poc', tts_backend: null, flow_id: 'f1', agent_id: 'a1' }],
  calldesk_conversation_flows: [{ nodes: [{ id: 'n1', type: 'greeting', prompt: 'Say hello.', edges: [] }], global_settings: { startNodeId: 'n1' } }],
  calldesk_tenants: [{ settings: {} }],
  tenant_call_audio_assets: [{ id: 'asset1', asset_type: 'jingle', name: 'intro', description: '', mulaw8k_storage_path: 'smoke/j.mulaw', enabled: true }],
};
const fake = http.createServer((req, res) => {
  if (req.url.startsWith('/storage/')) { res.writeHead(200, { 'content-type': 'application/octet-stream' }); return res.end(JINGLE); }
  const m = req.url.match(/\/rest\/v1\/([a-z_]+)/);
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify(req.method === 'GET' ? rows[m?.[1]] || [] : []));
});
await new Promise((r) => fake.listen(FAKE_PORT, '127.0.0.1', r));

const logs = [];
const srv = spawn('node', ['server.js'], {
  env: {
    PATH: process.env.PATH, PORT: String(APP_PORT), SUPABASE_URL: `http://127.0.0.1:${FAKE_PORT}`, SUPABASE_SERVICE_ROLE_KEY: 'fake',
    TELNYX_ENABLED: '1', TELNYX_WEBHOOK_SECRET: KEY, TELNYX_PUBLIC_HOST: `127.0.0.1:${APP_PORT}`,
  },
});
srv.stdout.on('data', (d) => logs.push(String(d)));
srv.stderr.on('data', (d) => logs.push(String(d)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
const check = (name, ok, extra = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? '  ' + extra : ''}`); };

try {
  for (let i = 0; i < 50; i++) { try { await fetch(`http://127.0.0.1:${APP_PORT}/`); break; } catch { await sleep(100); } }

  const bad = await fetch(`http://127.0.0.1:${APP_PORT}/telnyx/voice`, { method: 'POST', body: 'CallSid=S1&To=%2B15550001' });
  check('webhook without key rejected', bad.status === 403);

  const res = await fetch(`http://127.0.0.1:${APP_PORT}/telnyx/voice?key=${KEY}`, {
    method: 'POST', headers: { 'content-type': 'application/x-www-form-urlencoded' }, body: 'CallSid=SMOKE-1&To=%2B15550001&From=%2B15550002',
  });
  const texml = await res.text();
  const url = texml.match(/url="([^"]+)"/)?.[1].replace(/&amp;/g, '&');
  check('webhook returns bidirectional <Stream> TeXML', res.status === 200 && !!url && texml.includes('bidirectionalMode="rtp"'), url);

  const unsigned = await new Promise((resolve) => {
    const w = new WebSocket(`ws://127.0.0.1:${APP_PORT}/telnyx-stream?cs=SMOKE-1&exp=9999999999&sig=bad`);
    w.on('unexpected-response', (_q, r) => resolve(r.statusCode)); w.on('open', () => resolve('open')); w.on('error', () => resolve('error'));
  });
  check('websocket with a bad signature rejected (401)', unsigned === 401, String(unsigned));

  const ws = new WebSocket(url.replace('wss://', 'ws://'));
  const out = []; let t0 = 0;
  ws.on('message', (d) => { const m = JSON.parse(d.toString()); if (m.event === 'media') out.push({ t: Date.now(), len: Buffer.from(m.media.payload, 'base64').length }); });
  await new Promise((r) => ws.on('open', r));
  ws.send(JSON.stringify({ event: 'connected', version: '1.0.0' }));
  ws.send(JSON.stringify({ event: 'start', sequence_number: '1', stream_id: 'smoke-stream', start: { call_control_id: 'v2:smoke', call_session_id: 's', from: '+15550002', to: '+15550001', media_format: { encoding: 'PCMU', sample_rate: 8000, channels: 1 } } }));
  // 3 s of synthetic speech-like PCMU: 150 x 20 ms frames, amplitude-modulated 220 Hz carrier, sent in real time.
  for (let f = 0; f < 150; f++) {
    const b = Buffer.alloc(160);
    for (let i = 0; i < 160; i++) { const n = f * 160 + i; b[i] = mulaw(Math.round(6000 * Math.sin(n * 2 * Math.PI * 220 / 8000) * (0.5 + 0.5 * Math.sin(n / 800)))); }
    ws.send(JSON.stringify({ event: 'media', sequence_number: String(f + 2), media: { track: 'inbound', chunk: String(f + 1), timestamp: String(f * 20), payload: b.toString('base64') }, stream_id: 'smoke-stream' }));
    await sleep(20);
  }
  ws.send(JSON.stringify({ event: 'dtmf', stream_id: 'smoke-stream', dtmf: { digit: '5' } }));
  await sleep(500);
  ws.send(JSON.stringify({ event: 'stop', stream_id: 'smoke-stream' }));
  await sleep(300);
  ws.close();

  const gaps = out.slice(1).map((o, i) => o.t - out[i].t);
  const avg = gaps.reduce((a, b) => a + b, 0) / (gaps.length || 1);
  check('session wrote audio back', out.length >= 40, `${out.length} frames`);
  check('every outbound frame is 160 bytes', out.length > 0 && out.every((o) => o.len === 160));
  check('outbound pacing near 20 ms', avg > 15 && avg < 30, `avg ${avg.toFixed(1)} ms`);
  const log = logs.join('');
  check('server logged telnyx stream start + jingle', log.includes('[telnyx] stream started: smoke-stream (callSid: SMOKE-1') && log.includes('intro jingle'));
  check('session processed caller dtmf/stop without crashing', !/Unhandled|TypeError/.test(log) && log.includes('[telnyx] stream stopped'));
} finally {
  srv.kill();
  fake.close();
}
console.log(failed ? `\n${failed} check(s) FAILED` : '\nall checks passed');
if (failed) console.log(logs.join('').slice(-3000));
process.exit(failed ? 1 : 0);
