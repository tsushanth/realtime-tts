// Piper TTS backend: client protocol against a local mock Piper WebSocket server, CallSession._speakPiper behaviour
// (ordering, barge-in, fallback to ElevenLabs on error / connect timeout / overflow), per-version selection, cost, other backends.
import { describe, it, expect, beforeEach, afterEach, afterAll, vi } from 'vitest';
import net from 'node:net';
import WebSocket, { WebSocketServer } from 'ws';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';
process.env.ELEVENLABS_API_KEY = 'test-eleven-key';
process.env.ELEVENLABS_VOICE_ID = 'DEFAULTELEVENVOICE0001';
process.env.PIPER_TTS_TOKEN = 'test-piper-token';

const { CallSession, buildTenantContextMessage } = await import('../server.js');
const { TwilioCallAdapter } = await import('../twilioAdapter.js');
const { resolveInboundCall } = await import('../tenantLookup.js');
const { validateVersionVoice } = await import('../voiceSelection.js');
const { PiperPool, piperSynthesize, piperConfig, piperPool } = await import('../piperTts.js');
const { validatePiperVoice, PIPER_DEFAULT_VOICE } = await import('../piperVoices.js');
const { CallCostTracker } = await import('../costTracker.js');

// ---- mock Piper server -------------------------------------------------------------------------------------------
let wss, url, behavior, requests, connections, authHeaders;
function startMock() {
  return new Promise((resolve) => {
    wss = new WebSocketServer({ port: 0 }, () => {
      url = `ws://127.0.0.1:${wss.address().port}/tts`;
      resolve();
    });
    wss.on('connection', (ws, req) => {
      connections++;
      authHeaders.push(req.headers.authorization);
      ws.on('message', (raw) => {
        const msg = JSON.parse(raw.toString());
        if (msg.type === 'synthesize') { requests.push(msg); behavior(ws, msg); }
        else if (msg.type === 'stop') { requests.push(msg); ws.send(JSON.stringify({ type: 'cancelled' })); }
      });
    });
  });
}
const audioFor = (text) => Buffer.from(`AUDIO:${text}`);
const speakOk = (ws, msg) => {
  ws.send(JSON.stringify({ type: 'chunk_meta', text: '', gen_ms: 5, audio_s: 1 }));
  ws.send(audioFor(msg.text));
  ws.send(JSON.stringify({ type: 'done' }));
};

beforeEach(async () => {
  connections = 0; requests = []; authHeaders = []; behavior = speakOk;
  await startMock();
  process.env.PIPER_TTS_URL = url;
  process.env.PIPER_MAX_STREAMS = '8';
  process.env.PIPER_CONNECT_TIMEOUT_MS = '1500';
  process.env.PIPER_FIRST_AUDIO_TIMEOUT_MS = '4000';
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(async () => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  for (const k of ['PIPER_TTS_URL', 'PIPER_MAX_STREAMS', 'PIPER_CONNECT_TIMEOUT_MS', 'PIPER_FIRST_AUDIO_TIMEOUT_MS', 'PIPER_COST_PER_1K_CHARS']) delete process.env[k];
  for (const c of wss.clients) c.terminate();
  await new Promise((r) => wss.close(r));
});
afterAll(() => piperPool.closeSession('x'));

// ---- a session-shaped stub that runs the real prototype methods ---------------------------------------------------
let seq = 0;
function makeSession({ twilio = true } = {}) {
  const sent = [];
  const clientWs = twilio
    ? Object.assign(Object.create(TwilioCallAdapter.prototype), { readyState: WebSocket.OPEN, send: (d, o) => sent.push({ data: Buffer.from(d), opts: o }) })
    : { readyState: WebSocket.OPEN, send: (d) => sent.push({ data: Buffer.from(d) }) };
  const s = {
    id: `piper-test-${++seq}`, callSid: 'CAtest', ttsBackend: 'piper', piperVoiceId: PIPER_DEFAULT_VOICE,
    elevenVoiceId: 'DEFAULTELEVENVOICE0001', elevenStability: 0.5, ttsModel: null, expressiveDelivery: false,
    activeTurn: 1, turnState: { id: 1, pendingTts: 0, startedSpeaking: false }, clientWs, sent,
    _latency: null, _maybeRetireTurn: vi.fn(),
  };
  for (const m of ['_speakPiper', '_speakHttpTts', '_elevenLabsFetch', '_markTtsFirstByte']) s[m] = CallSession.prototype[m];
  return s;
}
const speak = async (s, text, turn = 1) => {
  s.turnState.pendingTts++;
  s._speakPiper(text, turn, Date.now());
  // wait for the sentence queue (the _sendChain tail is this sentence's slot)
  await s._sendChain;
};
const stubEleven = (body = Buffer.from('ELEVEN')) => {
  const f = vi.fn(async () => ({ ok: true, body: (async function* () { yield body; })() }));
  vi.stubGlobal('fetch', f);
  return f;
};
const payloads = (s) => s.sent.map((x) => x.data.toString());
const fallbackWarned = () => console.warn.mock.calls.some((c) => String(c[0]).includes('[tts] piper failed, falling back'));

describe('Piper client protocol (mock server)', () => {
  it('sends a synthesize request with bearer auth, voice and format, and returns frames in order', async () => {
    const pool = new PiperPool();
    const got = [];
    behavior = (ws, msg) => {
      ws.send(JSON.stringify({ type: 'chunk_meta' })); ws.send(Buffer.from('one'));
      ws.send(JSON.stringify({ type: 'chunk_meta' })); ws.send(Buffer.from('two'));
      ws.send(JSON.stringify({ type: 'done' }));
    };
    const r = await piperSynthesize({ pool, sessionKey: 'a', text: 'Hi there.', voice: 'custom:en-us-ljspeech', format: 'mulaw_8000', onChunk: (b) => got.push(b.toString()) });
    expect(got).toEqual(['one', 'two']);
    expect(r.frames).toBe(2);
    expect(r.firstAudioMs).toBeGreaterThanOrEqual(0);
    expect(requests[0]).toEqual({ type: 'synthesize', text: 'Hi there.', voice: 'custom:en-us-ljspeech', speed: 1.0, format: 'mulaw_8000' });
    expect(authHeaders[0]).toBe('Bearer test-piper-token');
    pool.closeSession('a');
  });

  it('reuses the warm socket for the next request of the same session', async () => {
    const pool = new PiperPool();
    const run = (t) => piperSynthesize({ pool, sessionKey: 'a', text: t, voice: 'default', format: 'pcm_24000', onChunk: () => {} });
    const r1 = await run('one'); const r2 = await run('two');
    expect(connections).toBe(1);
    expect(r1.reused).toBe(false); expect(r2.reused).toBe(true);
    pool.closeSession('a');
  });

  it('prewarm opens an idle socket that the first request then reuses', async () => {
    const pool = new PiperPool();
    await pool.prewarm('a');
    expect(connections).toBe(1);
    const r = await piperSynthesize({ pool, sessionKey: 'a', text: 'x', voice: 'default', format: 'pcm_24000', onChunk: () => {} });
    expect(r.reused).toBe(true);
    pool.closeSession('a');
  });

  it('server error message rejects with a server PiperError and the socket is not reused', async () => {
    const pool = new PiperPool();
    behavior = (ws) => ws.send(JSON.stringify({ type: 'error', message: 'at capacity, retry shortly' }));
    await expect(piperSynthesize({ pool, sessionKey: 'a', text: 'x', voice: 'default', format: 'pcm_24000', onChunk: () => {} })).rejects.toMatchObject({ code: 'server' });
    expect(pool.open).toBe(0);
  });

  it('times out when no audio arrives', async () => {
    const pool = new PiperPool();
    behavior = () => {};
    const cfg = { ...piperConfig(), firstAudioTimeoutMs: 80 };
    await expect(piperSynthesize({ pool, sessionKey: 'a', text: 'x', voice: 'default', format: 'pcm_24000', onChunk: () => {}, cfg })).rejects.toMatchObject({ code: 'timeout' });
    expect(pool.busy).toBe(0);
  });

  it('enforces the stream cap (overflow) and frees the slot afterwards', async () => {
    process.env.PIPER_MAX_STREAMS = '1';
    const pool = new PiperPool();
    let hold;
    behavior = (ws, msg) => { hold = () => speakOk(ws, msg); };
    const first = piperSynthesize({ pool, sessionKey: 'a', text: 'x', voice: 'default', format: 'pcm_24000', onChunk: () => {} });
    await vi.waitFor(() => expect(requests.length).toBe(1));
    await expect(piperSynthesize({ pool, sessionKey: 'b', text: 'y', voice: 'default', format: 'pcm_24000', onChunk: () => {} })).rejects.toMatchObject({ code: 'overflow' });
    hold(); await first;
    expect(pool.busy).toBe(0);
    pool.closeSession('a');
  });

  it('a warm idle socket of another session is evicted to make room under the cap', async () => {
    process.env.PIPER_MAX_STREAMS = '1';
    const pool = new PiperPool();
    await pool.prewarm('a');
    expect(pool.idleCount).toBe(1);
    await piperSynthesize({ pool, sessionKey: 'b', text: 'y', voice: 'default', format: 'pcm_24000', onChunk: () => {} });
    expect(pool.open).toBe(1);
    pool.closeSession('b'); pool.closeSession('a');
  });
});

describe('CallSession._speakPiper', () => {
  it('happy path: sentences go out in call order as mulaw 8k, one warm socket, [latency] reports ttsBackend piper', async () => {
    const fetchSpy = stubEleven();
    const s = makeSession();
    s._latency = { turnStart: Date.now() - 500, llmFirstToken: Date.now() - 300 };
    s._speakPiper('First sentence.', 1, Date.now());
    s._speakPiper('Second sentence.', 1, Date.now());
    await s._sendChain;
    expect(payloads(s)).toEqual(['AUDIO:First sentence.', 'AUDIO:Second sentence.']);
    expect(s.sent.every((x) => x.opts?.format === 'mulaw8k')).toBe(true);
    expect(requests.map((r) => r.format)).toEqual(['mulaw_8000', 'mulaw_8000']);
    expect(requests[0].voice).toBe('custom:en-us-ljspeech');
    expect(connections).toBe(1);
    expect(fetchSpy).not.toHaveBeenCalled();
    const line = console.log.mock.calls.map((c) => String(c[0])).find((l) => l.startsWith('[latency]'));
    expect(line).toBeTruthy();
    const j = JSON.parse(line.slice(line.indexOf('{')));
    expect(j).toMatchObject({ source: 'piper', ttsBackend: 'piper' });
    expect(typeof j.ttsLegMs).toBe('number'); expect(typeof j.responseMs).toBe('number'); expect(typeof j.llmTtfbMs).toBe('number');
    piperPool.closeSession(s.id);
  });

  it('non-telephony client gets pcm_24000 and pcm16 framing', async () => {
    stubEleven();
    const s = makeSession({ twilio: false });
    behavior = (ws) => { ws.send(Buffer.alloc(12, 1)); ws.send(JSON.stringify({ type: 'done' })); };
    await speak(s, 'Hello.');
    expect(requests[0].format).toBe('pcm_24000');
    expect(s.sent[0].data.length).toBe(12);
    piperPool.closeSession(s.id);
  });

  it('barge-in: aborting sends stop, forwards no audio and releases the turn slot', async () => {
    stubEleven();
    const s = makeSession();
    behavior = () => {}; // never answers
    s.turnState.pendingTts++;
    s._speakPiper('Long sentence.', 1, Date.now());
    await vi.waitFor(() => expect(requests.some((r) => r.type === 'synthesize')).toBe(true));
    s.activeTurn = 0;
    for (const c of s._httpTtsAborts) c.abort();
    await s._sendChain;
    await vi.waitFor(() => expect(requests.some((r) => r.type === 'stop')).toBe(true));
    expect(s.sent).toEqual([]);
    expect(s.turnState.pendingTts).toBe(0);
    expect(fallbackWarned()).toBe(false); // an abort is not a Piper failure
    expect(piperPool.busy).toBe(0);
  });

  it('falls back to ElevenLabs when Piper reports an error', async () => {
    const fetchSpy = stubEleven(Buffer.from('ELEVEN-AUDIO'));
    const s = makeSession();
    behavior = (ws) => ws.send(JSON.stringify({ type: 'error', message: 'boom' }));
    s._latency = { turnStart: Date.now() - 500, llmFirstToken: Date.now() - 300 };
    await speak(s, 'Hello there.');
    expect(fallbackWarned()).toBe(true);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [u, init] = fetchSpy.mock.calls[0];
    expect(String(u)).toContain('/v1/text-to-speech/DEFAULTELEVENVOICE0001/stream?output_format=ulaw_8000');
    expect(JSON.parse(init.body).text).toBe('Hello there.');
    expect(payloads(s)).toEqual(['ELEVEN-AUDIO']);
    const line = console.log.mock.calls.map((c) => String(c[0])).find((l) => l.startsWith('[latency]'));
    expect(JSON.parse(line.slice(line.indexOf('{')))).toMatchObject({ source: 'piper-fallback-elevenlabs', ttsBackend: 'piper' });
    piperPool.closeSession(s.id);
  });

  it('falls back to ElevenLabs on connect timeout', async () => {
    // a TCP listener that never completes the WebSocket handshake
    const blackhole = net.createServer((sock) => sock.on('error', () => {}));
    await new Promise((r) => blackhole.listen(0, '127.0.0.1', r));
    process.env.PIPER_TTS_URL = `ws://127.0.0.1:${blackhole.address().port}/tts`;
    process.env.PIPER_CONNECT_TIMEOUT_MS = '100';
    const fetchSpy = stubEleven();
    const s = makeSession();
    await speak(s, 'Hello.');
    blackhole.close();
    expect(fallbackWarned()).toBe(true);
    expect(console.warn.mock.calls.some((c) => String(c[0]).includes('connect'))).toBe(true);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(payloads(s)).toEqual(['ELEVEN']);
  });

  it('falls back to ElevenLabs on overflow (cap reached by another call)', async () => {
    process.env.PIPER_MAX_STREAMS = '1';
    const fetchSpy = stubEleven();
    let release;
    behavior = (ws, msg) => { release = () => speakOk(ws, msg); };
    const a = makeSession(); const b = makeSession();
    a.turnState.pendingTts++;
    a._speakPiper('Call A.', 1, Date.now());
    await vi.waitFor(() => expect(requests.length).toBe(1));
    await speak(b, 'Call B.');
    expect(console.warn.mock.calls.some((c) => String(c[0]).includes('overflow'))).toBe(true);
    expect(payloads(b)).toEqual(['ELEVEN']);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    release(); await a._sendChain;
    expect(payloads(a)).toEqual(['AUDIO:Call A.']);
    piperPool.closeSession(a.id); piperPool.closeSession(b.id);
  });

  it('does not repeat a sentence when Piper fails after audio already went out', async () => {
    const fetchSpy = stubEleven();
    const s = makeSession();
    behavior = (ws, msg) => { ws.send(audioFor(msg.text)); ws.send(JSON.stringify({ type: 'error', message: 'late' })); };
    await speak(s, 'Partial.');
    expect(payloads(s)).toEqual(['AUDIO:Partial.']);
    expect(fetchSpy).not.toHaveBeenCalled();
    piperPool.closeSession(s.id);
  });

  it('later sentences still use Piper after one fell back', async () => {
    stubEleven();
    const s = makeSession();
    let n = 0;
    behavior = (ws, msg) => (n++ === 0 ? ws.send(JSON.stringify({ type: 'error', message: 'once' })) : speakOk(ws, msg));
    await speak(s, 'One.');
    await speak(s, 'Two.');
    expect(payloads(s)).toEqual(['ELEVEN', 'AUDIO:Two.']);
    piperPool.closeSession(s.id);
  });
});

describe('per-version selection', () => {
  const ROW = {
    tenant_id: 'tenant-1', inbound_agent_version_id: 'version-1', outbound_agent_version_id: 'version-1',
    voice_engine: 'poc', tts_backend: 'piper', flow_id: 'flow-1', agent_id: 'agent-1', voice_id: 'custom:en-us-john',
    nodes: [{ id: 'n1', type: 'greeting', prompt: 'Say hello.', edges: [] }], global_settings: { startNodeId: 'n1' },
    stripe_customer_id: null, provider: null, api_key: null, event_type_id: null, settings: {},
  };

  it('tenantLookup reads tts_backend=piper and voice_id; the context message carries both', async () => {
    global.fetch = vi.fn(async () => ({ ok: true, status: 200, json: async () => [ROW], text: async () => '' }));
    const r = await resolveInboundCall('+15550001111');
    expect(r.ttsBackend).toBe('piper');
    expect(r.voiceId).toBe('custom:en-us-john');
    const msg = buildTenantContextMessage(r);
    expect(msg.ttsBackend).toBe('piper'); expect(msg.voiceId).toBe('custom:en-us-john');
  });

  const fakeWs = () => ({ readyState: WebSocket.OPEN, send() {}, close() {} });
  const open = (extra) => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network disabled in test'))));
    const s = new CallSession(fakeWs());
    s.onClientMessage(JSON.stringify({ type: 'context', systemPrompt: 'x', ...extra }), false);
    return s;
  };

  it('context ttsBackend=piper selects the backend; valid version voice applied, default is the licence-cleared voice', () => {
    const a = open({ ttsBackend: 'piper', voiceId: 'custom:en-us-john' });
    expect(a.ttsBackend).toBe('piper'); expect(a.cost.ttsBackend).toBe('piper'); expect(a.piperVoiceId).toBe('custom:en-us-john');
    const b = open({ ttsBackend: 'piper' });
    expect(b.piperVoiceId).toBe('custom:en-us-ljspeech');
    const c = open({ ttsBackend: 'piper', voiceId: 'custom:nope' });
    expect(c.ttsBackend).toBe('piper'); expect(c.piperVoiceId).toBe('custom:en-us-ljspeech');
    for (const x of [a, b, c]) { piperPool.closeSession(x.id); x.close(); }
  });

  it('existing backends are unchanged by a piper-shaped voice id', () => {
    const s = open({ ttsBackend: 'elevenlabs', voiceId: 'custom:en-us-john' });
    expect(s.ttsBackend).toBe('elevenlabs');
    expect(s.elevenVoiceId).toBe('DEFAULTELEVENVOICE0001');
    s.close();
  });

  it('voice validation: known ids only', () => {
    for (const id of ['default', 'custom:en-us-john', 'custom:en-us-ljspeech', ' custom:en-us-ljspeech ']) expect(validateVersionVoice('piper', id).ok).toBe(true);
    for (const id of ['af_heart', 'custom:other', '', null, 5, 'EXAVITQu4vr4xnSDxMaL']) expect(validateVersionVoice('piper', id).ok).toBe(false);
    expect(validatePiperVoice('custom:en-us-ljspeech').voiceId).toBe(PIPER_DEFAULT_VOICE);
    expect(validateVersionVoice('kokoro', 'custom:en-us-john').ok).toBe(false);
  });
});

describe('cost tracking', () => {
  it('piper is priced from PIPER_COST_PER_1K_CHARS, default 0', () => {
    const t = new CallCostTracker({ ttsBackend: 'piper' });
    t.addTtsChars(2000);
    expect(t.breakdown().tts).toBe(0);
    process.env.PIPER_COST_PER_1K_CHARS = '0.5';
    expect(t.breakdown().tts).toBeCloseTo(1.0, 10);
    expect(t.breakdown().ttsBackend).toBe('piper');
  });
});
