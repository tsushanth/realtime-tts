import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';
process.env.TEST_CALL_SECRET = 'test-secret';
process.env.TWILIO_ACCOUNT_SID = 'AC' + '1'.repeat(32);
process.env.TWILIO_AUTH_TOKEN = 'twilio-token';
process.env.DEMO_FROM_NUMBER = '+15550000009';

const { app, pendingCallContext, sampleCalleeOverrides, sampleCallSids, sampleAudioEvents, samplePlacedSidByTo } = await import('../server.js');

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
    if (/\/Calls\/CA[0-9a-f]{32}\.json$/.test(u)) return json({ status: 'completed', duration: '148' });
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
    expect(res.body).toEqual({ sampleCallee: 1, callAudio: 1 });
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
    expect(twilioCalls[0]).toContain('TimeLimit=210'); // raised from 150: a sample with a jingle + a booking was getting cut off before the confirmation
    expect(sampleCalleeOverrides.get(TO)?.fromNumber).toBe(FROM);
    expect(sampleCallSids.has(SID('d'))).toBe(true);
  });

  it('remembers which shopper-leg sid was placed for the callee number, so audio events can be filed under it', async () => {
    samplePlacedSidByTo.clear();
    await place({ toNumber: TO, shopper: true, sampleCallee: callee });
    expect(samplePlacedSidByTo.get(TO)).toBe(SID('d'));
  });

  it('tells the demo-agent leg which number it is answering (sampleTo), so it can find that placement', async () => {
    await place({ toNumber: TO, shopper: true, sampleCallee: callee });
    await voice(SID('a'), TO, FROM);
    expect(pendingCallContext.get(SID('a'))?.sampleTo).toBe(TO);
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

  describe('callAudio (jingle + sound effects on a sample call)', () => {
    const b64 = (n, v = 1) => Buffer.alloc(n, v).toString('base64');
    const audio = () => ({ jingle: { name: 'intro', audio: b64(32000, 0xaa) }, effects: [{ name: 'booking_confirmed_chime', description: 'After the booking is confirmed', audio: b64(16000, 0xbb) }] });

    it('accepts it, confirms it back to the caller, and keeps it on the override', async () => {
      const res = await place({ toNumber: TO, shopper: true, sampleCallee: { ...callee, callAudio: audio() } });
      expect(res.status).toBe(200);
      expect(res.body.callAudio).toBe(true);
      expect(sampleCalleeOverrides.get(TO)?.callAudio.effects[0].name).toBe('booking_confirmed_chime');
    });
    it('a sample with no callAudio reports callAudio:false, so a generator can tell the difference', async () => {
      const res = await place({ toNumber: TO, shopper: true, sampleCallee: callee });
      expect(res.status).toBe(200);
      expect(res.body.callAudio).toBe(false);
    });
    it('accepts a full-size payload (12s jingle) that exceeds express.json\'s 100kb default', async () => {
      const res = await place({ toNumber: TO, shopper: true, sampleCallee: { ...callee, callAudio: { jingle: { name: 'intro', audio: b64(96000) } } } });
      expect(res.status).toBe(200);
      expect(res.body.callAudio).toBe(true);
    });
    it('FAILS CLOSED (400, no call placed) when callAudio is present but unusable, instead of silently producing an audio-less sample', async () => {
      for (const bad of [{ jingle: { name: 'x', audio: '' } }, { effects: [{ name: 'bad name', description: 'd', audio: b64(10) }] }, 'nope', {}]) {
        const res = await place({ toNumber: TO, shopper: true, sampleCallee: { ...callee, callAudio: bad } });
        expect(res.status).toBe(400);
      }
      expect(twilioCalls).toEqual([]);
      expect(sampleCalleeOverrides.size).toBe(0);
    });
    it('hands the callAudio to the call context when /twilio/voice consumes the override', async () => {
      await place({ toNumber: TO, shopper: true, sampleCallee: { ...callee, callAudio: audio() } });
      await voice(SID('a'), TO, FROM);
      const ctx = pendingCallContext.get(SID('a'));
      expect(ctx.isSampleCallee).toBe(true);
      expect(ctx.callAudio.jingle.name).toBe('intro');
    });
  });

  describe('/call-status audioEvents (lets a generator verify what actually played, without scraping logs)', () => {
    beforeEach(() => sampleAudioEvents.clear());
    it('requires the secret like the rest of the sample endpoints', async () => {
      expect((await request(app).get('/call-status/' + SID('d'))).status).toBe(401);
    });
    it('returns the recorded jingle/effect events for that call, with the call status', async () => {
      sampleAudioEvents.set(SID('d'), [{ kind: 'jingle', name: 'intro', atMs: 5 }, { kind: 'effect', name: 'booking_chime', atMs: 91000 }]);
      const res = await request(app).get('/call-status/' + SID('d')).set(AUTH);
      expect(res.status).toBe(200);
      expect(res.body.status).toBe('completed');
      expect(res.body.duration).toBe(148);
      expect(res.body.audioEvents).toEqual([{ kind: 'jingle', name: 'intro', atMs: 5 }, { kind: 'effect', name: 'booking_chime', atMs: 91000 }]);
    });
    it('a call with no events reports an empty list (so "nothing played" is distinguishable from "field missing")', async () => {
      const res = await request(app).get('/call-status/' + SID('e')).set(AUTH);
      expect(res.body.audioEvents).toEqual([]);
    });
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
