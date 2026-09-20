import test from 'node:test';
import assert from 'node:assert/strict';
import { ReadAloud, AuthError, QuotaError, CapacityError, VoiceError, ApiError, pcmToWav } from '../index.js';

const json = (status, body, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } });

class FakeWS {
  static script = []; static sent = []; static closed = 0;
  readyState = 0;
  constructor(url) {
    this.url = url;
    queueMicrotask(async () => {
      this.readyState = 1; this.onopen?.();
      await new Promise((r) => setTimeout(r, 5));
      for (const step of FakeWS.script) {
        if (this.readyState !== 1) return;
        if (step.close) { this.readyState = 3; return this.onclose?.({ code: step.close }); }
        this.onmessage?.({ data: step });
        await new Promise((r) => setTimeout(r, 2));
      }
    });
  }
  send(d) { FakeWS.sent.push(JSON.parse(d)); }
  close() { FakeWS.closed++; this.readyState = 3; }
}
const bin = (n) => new Uint8Array([n, n, n, n]).buffer;
const reset = (script) => { FakeWS.script = script; FakeWS.sent = []; FakeWS.closed = 0; };
const mk = (fetchImpl, key = 'k') => new ReadAloud({ apiKey: key, fetch: fetchImpl, WebSocket: FakeWS });
const okAuth = (extra = {}) => async () => json(200, { token: 't', url: 'wss://x/ws', ...extra });

test('stream yields chunks and finishes', async () => {
  reset(['{"type":"chunk_meta"}', bin(1), '{"type":"chunk_meta"}', bin(2), '{"type":"done"}']);
  const out = [];
  for await (const c of mk(okAuth()).stream('hi')) out.push(c);
  assert.deepEqual(out.map((c) => c[0]), [1, 2]);
  assert.deepEqual(FakeWS.sent[0], { type: 'synthesize', text: 'hi', voice: 'default', speed: 1, format: 'pcm_24000' });
  assert.equal(FakeWS.closed, 1);
});

test('break sends stop', async () => {
  reset(['{}', bin(1), bin(2), bin(3), '{"type":"done"}']);
  for await (const _ of mk(okAuth()).stream('hi')) break;
  assert.ok(FakeWS.sent.some((m) => m.type === 'stop'));
});

test('abort signal cancels', async () => {
  reset(['{}', bin(1), bin(2), bin(3), bin(4), '{"type":"done"}']);
  const ac = new AbortController();
  let n = 0;
  await assert.rejects(async () => {
    for await (const _ of mk(okAuth()).stream('hi', { signal: ac.signal })) { n++; ac.abort(); }
  }, { name: 'AbortError' });
  assert.equal(n, 1);
  assert.ok(FakeWS.sent.some((m) => m.type === 'stop'));
});

test('error mapping: capacity, voice, close 1013', async () => {
  reset(['{"type":"error","message":"at capacity, retry shortly"}']);
  await assert.rejects(() => mk(okAuth()).convert('x'), CapacityError);
  reset(['{"type":"error","message":"unknown voice"}']);
  await assert.rejects(() => mk(okAuth()).convert('x', { voice: 'custom:z' }), VoiceError);
  reset([{ close: 1013 }]);
  await assert.rejects(() => mk(okAuth()).convert('x'), CapacityError);
  reset(['{"type":"error","message":"boom"}']);
  await assert.rejects(() => mk(okAuth()).convert('x'), (e) => e instanceof ApiError && !(e instanceof VoiceError));
});

test('authorize errors', async () => {
  await assert.rejects(() => mk(async () => json(401, { error: 'invalid key' }), 'bad').convert('x'), AuthError);
  await assert.rejects(() => mk(async () => json(402, { error: 'exhausted' })).convert('x'), QuotaError);
});

test('convert falls back to websocket without http_url', async () => {
  reset([bin(1), bin(2), '{"type":"done"}']);
  const out = await mk(okAuth()).convert('x');
  assert.equal(out.length, 8);
});

test('convert uses http_url when present', async () => {
  const calls = [];
  const f = async (url, init) => {
    calls.push([url, init]);
    if (url.endsWith('/tts/authorize')) return json(200, { token: 't', url: 'wss://x', http_url: 'https://x/v1/tts/stream' });
    return new Response(new Uint8Array([1, 2, 3, 4]), { status: 200 });
  };
  const out = await mk(f).convert('hello', { format: 'mulaw_8000' });
  assert.deepEqual([...out], [1, 2, 3, 4]);
  assert.equal(calls[1][1].headers.authorization, 'Bearer t');
  assert.equal(JSON.parse(calls[1][1].body).format, 'mulaw_8000');
});

test('http 503 -> CapacityError with retryAfter', async () => {
  const f = async (url) => url.endsWith('/tts/authorize')
    ? json(200, { token: 't', url: 'wss://x', http_url: 'https://x/h' })
    : json(503, { error: 'busy' }, { 'retry-after': '3' });
  await assert.rejects(() => mk(f).convert('x'), (e) => e instanceof CapacityError && e.retryAfter === 3);
});

test('pcmToWav header', () => {
  const w = pcmToWav(new Uint8Array(20), 24000);
  assert.equal(w.length, 64);
  assert.equal(new TextDecoder().decode(w.slice(0, 4)), 'RIFF');
  assert.equal(new DataView(w.buffer).getUint32(24, true), 24000);
  assert.equal(new DataView(w.buffer).getUint32(40, true), 20);
});
