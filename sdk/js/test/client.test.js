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

const httpFetch = (status = 200, parts = [[1, 2], [3, 4]], seen = []) => async (url, init) => {
  seen.push([url, init]);
  if (url.endsWith('/tts/authorize')) return json(200, { token: 't', url: 'wss://x', http_url: 'https://x/v1/tts/stream' });
  if (status !== 200) return json(status, { error: 'nope' }, status === 503 ? { 'retry-after': '2' } : {});
  return new Response(new ReadableStream({
    start(c) { for (const p of parts) c.enqueue(new Uint8Array(p)); c.close(); },
  }), { status });
};

test('streamHttp yields chunked bytes with bearer token, custom voice and format', async () => {
  const seen = [];
  const out = [];
  for await (const c of mk(httpFetch(200, undefined, seen)).streamHttp('hi', { voice: 'custom:abc', format: 'alaw_8000', speed: 1.5 })) out.push(...c);
  assert.deepEqual(out, [1, 2, 3, 4]);
  assert.equal(seen[1][0], 'https://x/v1/tts/stream');
  assert.equal(seen[1][1].headers.authorization, 'Bearer t');
  assert.deepEqual(JSON.parse(seen[1][1].body), { text: 'hi', voice: 'custom:abc', speed: 1.5, format: 'alaw_8000' });
});

test('streamHttp: no http_url, 503 and 401 errors; early break ok', async () => {
  await assert.rejects(async () => { for await (const _ of mk(okAuth()).streamHttp('x')); }, ApiError);
  await assert.rejects(async () => { for await (const _ of mk(httpFetch(503)).streamHttp('x')); },
    (e) => e instanceof CapacityError && e.retryAfter === 2);
  await assert.rejects(async () => { for await (const _ of mk(httpFetch(401)).streamHttp('x')); }, AuthError);
  for await (const _ of mk(httpFetch()).streamHttp('x')) break;
});

test('all audio formats and custom voice reach the websocket request', async () => {
  for (const format of ['pcm_24000', 'pcm_8000', 'mulaw_8000', 'alaw_8000', 'mp3_24000_64', 'mp3_24000_128', 'opus_24000']) {
    reset(['{"type":"done"}']);
    await mk(okAuth()).convert('x', { format, voice: 'custom:q1' });
    assert.equal(FakeWS.sent[0].format, format);
    assert.equal(FakeWS.sent[0].voice, 'custom:q1');
  }
});

test('http 404 unknown voice -> VoiceError', async () => {
  await assert.rejects(async () => { for await (const _ of mk(httpFetch(404)).streamHttp('x', { voice: 'custom:z' })); },
    (e) => e instanceof ApiError && !(e instanceof VoiceError));  // body "nope": not a voice error
  const f = async (url) => url.endsWith('/tts/authorize')
    ? json(200, { token: 't', url: 'wss://x', http_url: 'https://x/h' }) : json(404, { error: 'unknown voice' });
  await assert.rejects(() => mk(f).convert('x', { voice: 'custom:z' }), VoiceError);
});

// ---- POST /v1/text-to-speech (mp3/opus) against a real local HTTP server ----
import http from 'node:http';

async function withGateway(handler, fn) {
  const seen = [];
  const srv = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const body = JSON.parse(Buffer.concat(chunks).toString() || '{}');
      seen.push({ url: req.url, method: req.method, auth: req.headers.authorization, body });
      handler(req, res, body);
    });
  });
  await new Promise((r) => srv.listen(0, '127.0.0.1', r));
  try { await fn(`http://127.0.0.1:${srv.address().port}`, seen); }
  finally { srv.closeAllConnections?.(); await new Promise((r) => srv.close(r)); }
}
const reply = (res, status, body, headers = {}) => {
  res.writeHead(status, { 'content-type': 'application/json', ...headers });
  res.end(JSON.stringify(body));
};
const mkReal = (apiBase, extra = {}) => new ReadAloud({ apiKey: 'sk', apiBase, ...extra });

test('textToSpeechBytes: mp3 success, request shape', async () => {
  const mp3 = Buffer.from([0xff, 0xfb, 0x90, 0x00, 1, 2, 3]);
  await withGateway((req, res) => { res.writeHead(200, { 'content-type': 'audio/mpeg' }); res.end(mp3); }, async (base, seen) => {
    const out = await mkReal(base, { engine: 'kokoro' }).textToSpeechBytes('hello', { voice: 'af_heart', speed: 1.2, format: 'mp3_24000_64' });
    assert.deepEqual([...out], [...mp3]);
    assert.equal(seen[0].url, '/v1/text-to-speech');
    assert.equal(seen[0].method, 'POST');
    assert.equal(seen[0].auth, 'Bearer sk');
    assert.deepEqual(seen[0].body, { text: 'hello', voice: 'af_heart', speed: 1.2, format: 'mp3_24000_64', engine: 'kokoro' });
  });
});

test('textToSpeech: defaults to mp3_24000_128 and streams chunks', async () => {
  await withGateway((req, res) => {
    res.writeHead(200, { 'content-type': 'audio/mpeg' });
    res.write(Buffer.from([1, 2]));
    setTimeout(() => { res.write(Buffer.from([3, 4])); setTimeout(() => res.end(Buffer.from([5])), 20); }, 20);
  }, async (base, seen) => {
    const got = [];
    for await (const c of mkReal(base).textToSpeech('hi', { engine: 'piper' })) got.push(...c);
    assert.deepEqual(got, [1, 2, 3, 4, 5]);
    assert.equal(seen[0].body.format, 'mp3_24000_128');
    assert.equal(seen[0].body.engine, 'piper');
  });
});

test('textToSpeech: break cancels the request', async () => {
  let closed;
  const done = new Promise((r) => { closed = r; });
  await withGateway((req, res) => {
    res.writeHead(200, { 'content-type': 'audio/mpeg' });
    res.on('close', closed);
    const t = setInterval(() => res.write(Buffer.alloc(64)), 5);
    res.on('close', () => clearInterval(t));
  }, async (base) => {
    for await (const _ of mkReal(base).textToSpeech('hi')) break;
    await Promise.race([done, new Promise((_, rej) => setTimeout(() => rej(new Error('server never saw close')), 2000))]);
  });
});

test('textToSpeech: HTTP error mapping', async () => {
  const cases = [
    [401, { error: 'invalid or missing API key' }, AuthError],
    [402, { error: 'Free tier exhausted' }, QuotaError],
    [400, { error: 'text must be a non-empty string' }, ApiError],
    [400, { error: 'unknown voice' }, VoiceError],
    [413, { error: 'text too long (max 5000 chars)' }, ApiError],
    [501, { error: 'kokoro engine not available' }, ApiError],
    [502, { error: 'TTS worker unavailable, please retry' }, ApiError],
    [503, { error: 'encoder at capacity, please retry' }, CapacityError],
  ];
  for (const [status, body, Cls] of cases) {
    await withGateway((req, res) => reply(res, status, body), async (base) => {
      for (const call of [(c) => c.textToSpeechBytes('x'), async (c) => { for await (const _ of c.textToSpeech('x')); }]) {
        await assert.rejects(() => call(mkReal(base)), (e) => {
          assert.ok(e instanceof Cls, `${status}: expected ${Cls.name}, got ${e.constructor.name}`);
          assert.equal(e.status, status);
          assert.equal(e.message, body.error);
          if (Cls === ApiError) assert.equal(e.constructor, ApiError);
          return true;
        });
      }
    });
  }
});

test('textToSpeech: 503 honors Retry-After; absent header -> undefined', async () => {
  await withGateway((req, res) => reply(res, 503, { error: 'encoder at capacity' }, { 'retry-after': '2' }), async (base) => {
    await assert.rejects(() => mkReal(base).textToSpeechBytes('x'), (e) => e instanceof CapacityError && e.retryAfter === 2);
  });
  await withGateway((req, res) => reply(res, 503, { error: 'busy' }), async (base) => {
    await assert.rejects(() => mkReal(base).textToSpeechBytes('x'), (e) => e instanceof CapacityError && e.retryAfter === undefined);
  });
});
