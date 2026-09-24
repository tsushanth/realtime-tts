import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';
process.env.TEST_CALL_SECRET = 'test-secret';
process.env.TWILIO_ACCOUNT_SID = 'AC' + '1'.repeat(32);
process.env.TWILIO_AUTH_TOKEN = 'twilio-token';
process.env.DEMO_FROM_NUMBER = '+15550000009';

const { app, pendingCallContext, sampleCalleeOverrides, sampleCallSids } = await import('../server.js');

const callee = { systemPrompt: 'You are the after-hours line for a fictional brokerage.', greeting: 'Thanks for calling, how can I help?' };
const TO = '+15550000001';
const FROM = '+15550000009';
const SID = (c) => 'CA' + c.repeat(32);
const AUTH = { Authorization: 'Bearer test-secret' };
const voice = (sid, to, from) => request(app).post('/twilio/voice').type('form').send({ CallSid: sid, To: to, From: from });

// Stub of everything server.js fetches: Supabase (rate limiter / tenant lookup) and Twilio.
let twilioCalls;
function stubFetch() {
  twilioCalls = [];
  vi.stubGlobal('fetch', vi.fn(async (url, init) => {
    const u = String(url);
    const json = (b, ok = true, status = 200) => ({ ok, status, json: async () => b, text: async () => JSON.stringify(b) });
    if (u.includes('calldesk_try_acquire_token')) return json(true);
    if (u.endsWith('/Calls.json')) { twilioCalls.push(String(init?.body)); return json({ sid: SID('d'), status: 'queued' }); }
    if (u.includes('/Recordings.json')) return json({ recordings: [{ sid: 'RE' + '2'.repeat(32), status: 'completed', channels: 2, duration: '70' }] });
    return json([]);
  }));
}

describe('sampleCallee', () => {
  beforeEach(() => {
    sampleCalleeOverrides.clear(); pendingCallContext.clear(); sampleCallSids.clear();
    process.env.SAMPLE_CALLEE_NUMBERS = TO;
    stubFetch();
  });
  afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); delete process.env.SAMPLE_CALLEE_NUMBERS; });

  const place = (body) => request(app).post('/place-test-call').set(AUTH).send(body);

  it('advertises the capability without revealing the allowlist', async () => {
    const res = await request(app).get('/sample-callee-capability');
    expect(res.body).toEqual({ sampleCallee: 1 });
  });

  it('rejects sampleCallee without shopper:true, and a malformed sampleCallee', async () => {
    expect((await place({ toNumber: TO, routeAs: '+15550000002', sampleCallee: callee })).status).toBe(400);
    expect((await place({ toNumber: TO, shopper: true, sampleCallee: { systemPrompt: 'x', greeting: '' } })).status).toBe(400);
  });

  it('rejects a non-E.164 toNumber (400) before dialing', async () => {
    for (const bad of ['5550000001', '+0555000000', '+1 555 000 0001', 'tel:+15550000001']) {
      const res = await place({ toNumber: bad, shopper: true, sampleCallee: callee });
      expect(res.status).toBe(400);
    }
    expect(twilioCalls).toEqual([]);
  });

  it('refuses (403) a number not in SAMPLE_CALLEE_NUMBERS, and refuses everything when unset/empty', async () => {
    expect((await place({ toNumber: '+15559999999', shopper: true, sampleCallee: callee })).status).toBe(403);
    delete process.env.SAMPLE_CALLEE_NUMBERS;
    expect((await place({ toNumber: TO, shopper: true, sampleCallee: callee })).status).toBe(403);
    process.env.SAMPLE_CALLEE_NUMBERS = ' , ';
    expect((await place({ toNumber: TO, shopper: true, sampleCallee: callee })).status).toBe(403);
    expect(twilioCalls).toEqual([]);
    expect(sampleCalleeOverrides.size).toBe(0);
  });

  it('places an allowlisted call: registers override + sid, sets TimeLimit', async () => {
    const res = await place({ toNumber: TO, shopper: true, record: true, sampleCallee: callee });
    expect(res.status).toBe(200);
    expect(res.body.sampleCallee).toBe(true);
    expect(twilioCalls[0]).toContain('TimeLimit=150');
    expect(sampleCalleeOverrides.get(TO)?.fromNumber).toBe(FROM);
    expect(sampleCallSids.has(SID('d'))).toBe(true);
  });

  it('/twilio/voice consumes a matching override exactly once (second hit falls through)', async () => {
    await place({ toNumber: TO, shopper: true, sampleCallee: callee });
    await voice(SID('a'), TO, FROM);
    expect(pendingCallContext.get(SID('a'))?.isSampleCallee).toBe(true);
    expect(sampleCalleeOverrides.size).toBe(0);
    await voice(SID('b'), TO, FROM);
    expect(pendingCallContext.get(SID('b'))?.isSampleCallee).toBeUndefined();
  });

  it('a non-matching From does not consume the override (real caller keeps the tenant path)', async () => {
    await place({ toNumber: TO, shopper: true, sampleCallee: callee });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await voice(SID('b'), TO, '+15551111111');
    expect(pendingCallContext.get(SID('b'))?.isSampleCallee).toBeUndefined();
    expect(sampleCalleeOverrides.size).toBe(1);
    expect(warn.mock.calls.flat().join(' ')).toMatch(/non-matching From/);
    warn.mockRestore();
    await voice(SID('a'), TO, FROM); // the genuine leg still gets it
    expect(pendingCallContext.get(SID('a'))?.isSampleCallee).toBe(true);
  });

  it('the override expires after 30s', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));
    await place({ toNumber: TO, shopper: true, sampleCallee: callee });
    vi.setSystemTime(new Date('2026-01-01T00:00:31Z'));
    await voice(SID('a'), TO, FROM);
    expect(pendingCallContext.get(SID('a'))?.isSampleCallee).toBeUndefined();
    expect(sampleCalleeOverrides.size).toBe(0);
  });

  it('/call-recording: 401 without secret, 404 for a sid we did not place, 200 for one we did', async () => {
    expect((await request(app).get('/call-recording/' + SID('d'))).status).toBe(401);
    expect((await request(app).get('/call-recording/' + SID('e')).set(AUTH)).status).toBe(404);
    await place({ toNumber: TO, shopper: true, sampleCallee: callee });
    const ok = await request(app).get('/call-recording/' + SID('d')).set(AUTH);
    expect(ok.status).toBe(200);
    expect(ok.body.status).toBe('completed');
    expect(ok.body.url).toMatch(/^https:\/\/api\.twilio\.com\/.*\.mp3$/);
    expect((await request(app).get('/call-recording/bad').set(AUTH)).status).toBe(400);
  });

  it('the sid set is bounded', async () => {
    for (let i = 0; i < 60; i++) {
      const hex = i.toString(16).padStart(32, '0');
      sampleCallSids.set('CA' + hex, Date.now() + 1e6);
    }
    await place({ toNumber: TO, shopper: true, sampleCallee: callee });
    expect(sampleCallSids.size).toBeLessThanOrEqual(50);
    expect(sampleCallSids.has(SID('d'))).toBe(true);
  });
});
