// Telnyx call recording: start request, soft failure, webhook auth/idempotency/call-log write, retention, proxy, Twilio unchanged.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';
process.env.TELNYX_ENABLED = '1';
process.env.TELNYX_WEBHOOK_SECRET = 'hook-secret';
process.env.TELNYX_API_KEY = 'KEYfake';
process.env.TEST_CALL_SECRET = 'admin-secret';
process.env.TWILIO_ACCOUNT_SID = 'ACfake';
process.env.TWILIO_AUTH_TOKEN = 'fake-token';

const { app, enforceRecordingRetention } = await import('../server.js');
const rec = await import('../telnyxRecording.js');

const ENV = { TELNYX_ENABLED: '1', TELNYX_API_KEY: 'KEYfake', TELNYX_WEBHOOK_SECRET: 'hook-secret' };
const okRes = (extra = {}) => ({ ok: true, status: 200, json: async () => ({}), text: async () => '', ...extra });
const noSleep = async () => {};

describe('startTelnyxRecording', () => {
  it('POSTs the documented TeXML request: dual channel, no beep, completed callback, bearer auth', async () => {
    const f = vi.fn(async () => okRes());
    const r = await rec.startTelnyxRecording({ callSid: 'CS1', accountSid: 'ACC1', host: 'h.example', env: ENV, fetchImpl: f, sleep: noSleep });
    expect(r).toEqual({ started: true, attempts: 1 });
    const [url, init] = f.mock.calls[0];
    expect(url).toBe('https://api.telnyx.com/v2/texml/Accounts/ACC1/Calls/CS1/Recordings.json');
    expect(init.method).toBe('POST');
    expect(init.headers.Authorization).toBe('Bearer KEYfake');
    const b = Object.fromEntries(init.body);
    expect(b).toMatchObject({ RecordingChannels: 'dual', RecordingTrack: 'both', PlayBeep: 'false', RecordingStatusCallbackEvent: 'completed', RecordingStatusCallbackMethod: 'POST' });
    expect(b.RecordingStatusCallback).toBe('https://h.example/telnyx/recording-status?key=hook-secret');
  });

  it('flag off or missing prerequisites: no API call', async () => {
    for (const [env, reason] of [[{ ...ENV, TELNYX_ENABLED: undefined }, 'telnyx_disabled'], [{ ...ENV, TELNYX_API_KEY: '' }, 'no_api_key'], [{ ...ENV, TELNYX_WEBHOOK_SECRET: '' }, 'webhook_auth_not_configured']]) {
      const f = vi.fn();
      const r = await rec.startTelnyxRecording({ callSid: 'CS1', accountSid: 'A', host: 'h', env, fetchImpl: f, sleep: noSleep });
      expect(r).toMatchObject({ started: false, reason });
      expect(f).not.toHaveBeenCalled();
    }
    const f = vi.fn();
    expect(await rec.startTelnyxRecording({ callSid: 'CS1', accountSid: null, host: 'h', env: ENV, fetchImpl: f })).toMatchObject({ reason: 'no_call_or_account' });
    expect(f).not.toHaveBeenCalled();
  });

  it('4xx: soft failure, no retry, never throws, key not in the result', async () => {
    const f = vi.fn(async () => ({ ok: false, status: 401, text: async () => 'no' }));
    const r = await rec.startTelnyxRecording({ callSid: 'CS1', accountSid: 'A', host: 'h', env: ENV, fetchImpl: f, sleep: noSleep });
    expect(r).toMatchObject({ started: false, reason: 'http_401', status: 401 });
    expect(f).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(r)).not.toContain('KEYfake');
  });

  it('5xx then success: retries', async () => {
    const f = vi.fn().mockResolvedValueOnce({ ok: false, status: 503 }).mockResolvedValueOnce(okRes());
    const sleep = vi.fn(async () => {});
    expect(await rec.startTelnyxRecording({ callSid: 'C', accountSid: 'A', host: 'h', env: ENV, fetchImpl: f, sleep })).toEqual({ started: true, attempts: 2 });
    expect(sleep).toHaveBeenCalledTimes(1);
  });

  it('persistent 5xx and network errors give up after 3 attempts; timeouts are not retried', async () => {
    const f5 = vi.fn(async () => ({ ok: false, status: 500 }));
    expect(await rec.startTelnyxRecording({ callSid: 'C', accountSid: 'A', host: 'h', env: ENV, fetchImpl: f5, sleep: noSleep })).toMatchObject({ started: false, reason: 'http_500' });
    expect(f5).toHaveBeenCalledTimes(3);
    const fn = vi.fn(async () => { throw new Error('ECONNRESET'); });
    expect(await rec.startTelnyxRecording({ callSid: 'C', accountSid: 'A', host: 'h', env: ENV, fetchImpl: fn, sleep: noSleep })).toMatchObject({ reason: 'network_error' });
    expect(fn).toHaveBeenCalledTimes(3);
    const ft = vi.fn(async () => { const e = new Error('t'); e.name = 'TimeoutError'; throw e; });
    expect(await rec.startTelnyxRecording({ callSid: 'C', accountSid: 'A', host: 'h', env: ENV, fetchImpl: ft, sleep: noSleep })).toMatchObject({ reason: 'timeout' });
    expect(ft).toHaveBeenCalledTimes(1);
  });

  it('account id: env override wins, else the webhook AccountSid', () => {
    expect(rec.resolveTelnyxAccountSid('W', { TELNYX_ACCOUNT_SID: 'E' })).toBe('E');
    expect(rec.resolveTelnyxAccountSid('W', {})).toBe('W');
    expect(rec.resolveTelnyxAccountSid(null, {})).toBeNull();
  });
});

describe('/telnyx/recording-status webhook', () => {
  const realFetch = globalThis.fetch;
  let patches;
  beforeEach(() => {
    rec._resetSeenForTests();
    patches = [];
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    globalThis.fetch = vi.fn(async (url, init) => {
      if (String(url).includes('calldesk_call_logs') && init?.method === 'PATCH') patches.push({ url: String(url), body: JSON.parse(init.body) });
      return okRes();
    });
  });
  afterEach(() => { globalThis.fetch = realFetch; vi.restoreAllMocks(); });
  const post = (q = '?key=hook-secret') => request(app).post(`/telnyx/recording-status${q}`).set('content-type', 'application/x-www-form-urlencoded');
  const payload = (o = {}) => new URLSearchParams({ CallSid: 'CS9', RecordingSid: 'a1b2c3d4-0000-1111-2222-333344445555', RecordingStatus: 'completed', RecordingUrl: 'https://s3.example/x.mp3', ...o }).toString();

  it('rejects a missing or wrong key and writes nothing', async () => {
    expect((await post('').send(payload())).status).toBe(403);
    expect((await post('?key=nope').send(payload())).status).toBe(403);
    expect(patches).toHaveLength(0);
  });

  it('writes the same columns as Twilio (recording_url + recording_sid) on the call log row', async () => {
    const r = await post().send(payload());
    expect(r.status).toBe(200);
    expect(patches).toHaveLength(1);
    expect(patches[0].url).toContain('retell_call_id=eq.CS9');
    expect(patches[0].body).toEqual({ recording_url: 'https://api.telnyx.com/v2/recordings/a1b2c3d4-0000-1111-2222-333344445555', recording_sid: 'telnyx:a1b2c3d4-0000-1111-2222-333344445555' });
  });

  it('is idempotent: a redelivery is acknowledged without a second write', async () => {
    expect((await post().send(payload())).status).toBe(200);
    expect((await post().send(payload())).status).toBe(200);
    expect(patches).toHaveLength(1);
  });

  it('ignores non-completed events and missing ids; accepts a JSON body', async () => {
    await post().send(payload({ RecordingStatus: 'in-progress' }));
    await post().send(payload({ RecordingSid: '' }));
    expect(patches).toHaveLength(0);
    const r = await request(app).post('/telnyx/recording-status?key=hook-secret').set('content-type', 'application/json')
      .send({ CallSid: 'CSJ', RecordingSid: 'deadbeef-aaaa-bbbb-cccc-000011112222', RecordingStatus: 'completed' });
    expect(r.status).toBe(200);
    expect(patches).toHaveLength(1);
  });
});

describe('retention delete', () => {
  it('DELETEs via the Telnyx API with the bearer key; 404 counts as gone; other errors do not', async () => {
    const f = vi.fn(async () => ({ ok: true, status: 200 }));
    expect(await rec.deleteTelnyxRecording('telnyx:abc-12345678', { env: ENV, fetchImpl: f })).toBe(true);
    expect(f.mock.calls[0][0]).toBe('https://api.telnyx.com/v2/recordings/abc-12345678');
    expect(f.mock.calls[0][1]).toMatchObject({ method: 'DELETE', headers: { Authorization: 'Bearer KEYfake' } });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(await rec.deleteTelnyxRecording('telnyx:x', { env: ENV, fetchImpl: async () => ({ ok: false, status: 404 }) })).toBe(true);
    expect(await rec.deleteTelnyxRecording('telnyx:x', { env: ENV, fetchImpl: async () => ({ ok: false, status: 500 }) })).toBe(false);
    expect(await rec.deleteTelnyxRecording('telnyx:x', { env: ENV, fetchImpl: async () => { throw new Error('boom'); } })).toBe(false);
    const never = vi.fn();
    expect(await rec.deleteTelnyxRecording('RE0123', { env: ENV, fetchImpl: never })).toBe(false); // a Twilio sid is never sent to Telnyx
    expect(await rec.deleteTelnyxRecording('telnyx:x', { env: { ...ENV, TELNYX_API_KEY: '' }, fetchImpl: never })).toBe(false);
    expect(never).not.toHaveBeenCalled();
  });
});

describe('/recording-audio proxy', () => {
  const realFetch = globalThis.fetch;
  afterEach(() => { globalThis.fetch = realFetch; vi.restoreAllMocks(); });
  const get = (url, auth = 'Bearer admin-secret') => request(app).get('/recording-audio').query({ url }).set('authorization', auth);
  const ID = 'a1b2c3d4-0000-1111-2222-333344445555';

  it('streams a Telnyx recording: fresh download url, key only sent to api.telnyx.com', async () => {
    const calls = [];
    globalThis.fetch = vi.fn(async (url, init) => {
      calls.push({ url: String(url), auth: init?.headers?.Authorization });
      if (String(url).startsWith('https://api.telnyx.com/')) return okRes({ json: async () => ({ data: { download_urls: { mp3: 'https://bucket.example/r.mp3?sig=1' } } }) });
      return { ok: true, status: 200, headers: new Headers({ 'content-type': 'audio/mpeg' }), body: (async function* () { yield Buffer.from('AUDIO'); })() };
    });
    const r = await get(`https://api.telnyx.com/v2/recordings/${ID}`);
    expect(r.status).toBe(200);
    expect(r.headers['content-type']).toContain('audio/mpeg');
    expect(Buffer.from(r.body).toString()).toBe('AUDIO');
    expect(calls[0]).toEqual({ url: `https://api.telnyx.com/v2/recordings/${ID}`, auth: 'Bearer KEYfake' });
    expect(calls[1].url).toBe('https://bucket.example/r.mp3?sig=1');
    expect(calls[1].auth).toBeUndefined();
  });

  it('requires the admin secret; rejects non-recording telnyx urls; Telnyx errors map to 404/502', async () => {
    globalThis.fetch = vi.fn();
    expect((await get(`https://api.telnyx.com/v2/recordings/${ID}`, 'Bearer bad')).status).toBe(401);
    expect((await get('https://api.telnyx.com/v2/messages/x')).status).toBe(400);
    expect((await get(`https://api.telnyx.com/v2/recordings/${ID}/../x`)).status).toBe(400);
    expect(globalThis.fetch).not.toHaveBeenCalled();
    vi.spyOn(console, 'error').mockImplementation(() => {});
    globalThis.fetch = vi.fn(async () => ({ ok: false, status: 404 }));
    expect((await get(`https://api.telnyx.com/v2/recordings/${ID}`)).status).toBe(404);
    globalThis.fetch = vi.fn(async () => ({ ok: false, status: 500 }));
    expect((await get(`https://api.telnyx.com/v2/recordings/${ID}`)).status).toBe(502);
  });

  it('Twilio recording urls still go to Twilio with Basic auth (unchanged)', async () => {
    const seen = [];
    globalThis.fetch = vi.fn(async (url, init) => { seen.push([String(url), init?.headers?.Authorization]); return { ok: true, status: 200, headers: new Headers(), body: (async function* () { yield Buffer.from('T'); })() }; });
    const r = await get('https://api.twilio.com/2010-04-01/Accounts/ACfake/Recordings/RE1.mp3');
    expect(r.status).toBe(200);
    expect(seen[0][0]).toBe('https://api.twilio.com/2010-04-01/Accounts/ACfake/Recordings/RE1.mp3');
    expect(seen[0][1]).toMatch(/^Basic /);
  });
});

describe('sid helpers', () => {
  it('round-trips and distinguishes Twilio sids', () => {
    expect(rec.isTelnyxRecordingSid('telnyx:abc')).toBe(true);
    expect(rec.isTelnyxRecordingSid('RE' + '0'.repeat(32))).toBe(false);
    expect(rec.telnyxRecordingId('telnyx:abc')).toBe('abc');
    expect(rec.parseTelnyxRecordingUrl(rec.telnyxRecordingUrl('abc-12345678'))).toBe('abc-12345678');
    expect(rec.parseTelnyxRecordingUrl('https://evil.example/v2/recordings/abc-12345678')).toBeNull();
  });
});

describe('retention sweep (server wiring, mocked fetch only)', () => {
  const realFetch = globalThis.fetch;
  afterEach(() => { globalThis.fetch = realFetch; vi.restoreAllMocks(); });
  it('routes Telnyx rows to the Telnyx DELETE and Twilio rows to Twilio, clearing both references', async () => {
    const old = new Date(Date.now() - 40 * 86400_000).toISOString();
    const calls = [];
    globalThis.fetch = vi.fn(async (url, init) => {
      const u = String(url); const m = init?.method || 'GET';
      calls.push(`${m} ${u}`);
      if (u.includes('/calldesk_tenants')) return okRes({ json: async () => [{ id: 't1', settings: { recording_retention_days: 30 } }] });
      if (u.includes('/calldesk_call_logs') && m === 'GET') return okRes({ json: async () => [
        { id: 'L1', tenant_id: 't1', recording_sid: 'telnyx:aaaa-bbbb-cccc', created_at: old },
        { id: 'L2', tenant_id: 't1', recording_sid: 'RE123', created_at: old },
      ] });
      return okRes();
    });
    await enforceRecordingRetention();
    expect(calls).toContain('DELETE https://api.telnyx.com/v2/recordings/aaaa-bbbb-cccc');
    expect(calls).toContain('DELETE https://api.twilio.com/2010-04-01/Accounts/ACfake/Recordings/RE123.json');
    expect(calls.filter((c) => c.startsWith('PATCH') && c.includes('id=eq.L1'))).toHaveLength(1);
    expect(calls.filter((c) => c.startsWith('PATCH') && c.includes('id=eq.L2'))).toHaveLength(1);
  });
  it('a failed Telnyx delete keeps the reference', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const old = new Date(Date.now() - 40 * 86400_000).toISOString();
    const calls = [];
    globalThis.fetch = vi.fn(async (url, init) => {
      const u = String(url); const m = init?.method || 'GET';
      calls.push(`${m} ${u}`);
      if (u.includes('/calldesk_tenants')) return okRes({ json: async () => [{ id: 't1', settings: { recording_retention_days: 30 } }] });
      if (u.includes('/calldesk_call_logs') && m === 'GET') return okRes({ json: async () => [{ id: 'L1', tenant_id: 't1', recording_sid: 'telnyx:aaaa-bbbb-cccc', created_at: old }] });
      if (m === 'DELETE') return { ok: false, status: 500 };
      return okRes();
    });
    await enforceRecordingRetention();
    expect(calls.some((c) => c.startsWith('PATCH'))).toBe(false);
  });
});
