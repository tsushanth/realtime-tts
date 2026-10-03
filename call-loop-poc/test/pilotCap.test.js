// Pilot cap enforcement (docs/pilot-cap-enforcement.md): flag read with the existing tenant row fetch, fail open, blocked call speaks then ends
// on both carriers, per-call duration limit for pilots, call log outcome. Fakes for fetch / the carrier adapter; no network.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';
import { EventEmitter } from 'node:events';
import WebSocket from 'ws';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';
process.env.TELNYX_ENABLED = '1';
process.env.TELNYX_WEBHOOK_SECRET = 'hook-secret';
process.env.PILOT_LOOKUP_TIMEOUT_MS = '60';

const { app, pendingCallContext, contextWrittenCallSids, wirePhoneAdapter, PILOT_BLOCKED_MESSAGE, CallSession } = await import('../server.js');
const { resolveInboundCall, _resetPilotLookupState, PILOT_MAX_CALL_SEC } = await import('../tenantLookup.js');

const BASE = {
  tenant_id: 'tenant-1', inbound_agent_version_id: 'v1', outbound_agent_version_id: 'v1', voice_engine: 'poc', tts_backend: 'elevenlabs',
  tts_model: null, voice_id: 'voice-x', flow_id: 'f1', agent_id: 'a1', nodes: [{ id: 'n1', type: 'greeting', prompt: 'hi', edges: [] }],
  global_settings: { startNodeId: 'n1' }, stripe_customer_id: null, provider: null, api_key: null, event_type_id: null, settings: {},
};

// Every table returns BASE except calldesk_tenants, whose row/behaviour the test controls. `tenantMode` decides how the pilot-column query behaves.
let urls;
function stubFetch({ tenant = {}, pilotCols = 'ok', globalSettings } = {}) {
  urls = [];
  globalThis.fetch = vi.fn(async (url, init) => {
    const u = String(url);
    urls.push(u);
    const body = (rows) => ({ ok: true, status: 200, json: async () => rows, text: async () => '' });
    if (u.includes('/calldesk_call_logs')) return body([{ id: 'log-1' }]);
    if (u.includes('/calldesk_tenants')) {
      const asksPilot = u.includes('pilot_blocked');
      if (asksPilot && pilotCols === 'missing') return { ok: false, status: 400, text: async () => 'column does not exist', json: async () => ({}) };
      if (asksPilot && pilotCols === 'error') throw new Error('network down');
      if (asksPilot && pilotCols === 'hang') {
        return new Promise((_, reject) => init?.signal?.addEventListener('abort', () => reject(new DOMException('timed out', 'TimeoutError'))));
      }
      if (asksPilot && pilotCols === 'no-embed' && u.includes('calldesk_pilots(')) return { ok: false, status: 400, text: async () => 'no relationship', json: async () => ({}) };
      return body([asksPilot ? { settings: {}, ...tenant } : { settings: {} }]);
    }
    if (u.includes('/calldesk_conversation_flows')) return body([{ nodes: BASE.nodes, global_settings: globalSettings || BASE.global_settings }]);
    return body([BASE]);
  });
}
const tenantFetches = () => urls.filter((u) => u.includes('/calldesk_tenants'));

const realFetch = globalThis.fetch;
beforeEach(() => {
  _resetPilotLookupState();
  pendingCallContext.clear();
  contextWrittenCallSids.clear();
  for (const m of ['log', 'warn', 'error']) vi.spyOn(console, m).mockImplementation(() => {});
});
afterEach(() => { globalThis.fetch = realFetch; vi.restoreAllMocks(); vi.useRealTimers(); });

describe('lookup: pilot flag', () => {
  it('blocked pilot: returns a minimal blocked object with the reason, no flow work', async () => {
    stubFetch({ tenant: { pilot_blocked: true, pilot_blocked_reason: 'cap' } });
    const r = await resolveInboundCall('+1555');
    expect(r).toMatchObject({ tenantId: 'tenant-1', pilotBlocked: true, pilotBlockedReason: 'cap', ttsBackend: 'elevenlabs', voiceId: 'voice-x' });
    expect(r.flow.nodes).toEqual([]);
    expect(console.log).toHaveBeenCalledWith('[pilot] blocked tenant=tenant-1 reason=cap');
  });
  it('blocked without a reason logs reason=unknown', async () => {
    stubFetch({ tenant: { pilot_blocked: true } });
    expect((await resolveInboundCall('+1555')).pilotBlockedReason).toBe('unknown');
  });
  it('piggybacks on the existing tenant fetch: still exactly one calldesk_tenants query, selecting settings too', async () => {
    stubFetch({ tenant: { pilot_blocked: false } });
    await resolveInboundCall('+1555');
    expect(tenantFetches()).toHaveLength(1);
    expect(tenantFetches()[0]).toContain('select=settings,pilot_blocked');
  });
  it('flag false / not a pilot: resolved object has no pilot keys and the flow is untouched', async () => {
    stubFetch({ tenant: { pilot_blocked: false } });
    const r = await resolveInboundCall('+1555');
    expect('pilotBlocked' in r).toBe(false);
    expect('isPilot' in r).toBe(false);
    expect(r.flow.globalSettings).toEqual({ startNodeId: 'n1' });
  });
  it('only a literal true blocks (truthy strings, 1, null do not)', async () => {
    for (const v of ['true', 1, null, 'false']) {
      _resetPilotLookupState();
      stubFetch({ tenant: { pilot_blocked: v } });
      expect('pilotBlocked' in (await resolveInboundCall('+1555'))).toBe(false);
    }
  });
  it('column missing (pre-migration): call served normally, retried without the columns, then not asked again for a while', async () => {
    stubFetch({ pilotCols: 'missing' });
    const r = await resolveInboundCall('+1555');
    expect(r.tenantId).toBe('tenant-1');
    expect('pilotBlocked' in r).toBe(false);
    expect(tenantFetches().some((u) => u.endsWith('select=settings'))).toBe(true);
    urls.length = 0;
    await resolveInboundCall('+1555');
    expect(tenantFetches()).toHaveLength(1); // backoff: only the plain query this time
    expect(tenantFetches()[0]).not.toContain('pilot_blocked');
  });
  it('pilots relationship not visible: falls back to the flag-only query and still blocks', async () => {
    stubFetch({ pilotCols: 'no-embed', tenant: { pilot_blocked: true, pilot_blocked_reason: 'expired' } });
    expect((await resolveInboundCall('+1555')).pilotBlocked).toBe(true);
  });
  it('FAIL OPEN on query error: call served, flag ignored', async () => {
    stubFetch({ pilotCols: 'error', tenant: { pilot_blocked: true } });
    const r = await resolveInboundCall('+1555');
    expect(r.tenantId).toBe('tenant-1');
    expect('pilotBlocked' in r).toBe(false);
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('[pilot] lookup failed tenant=tenant-1'));
  });
  it('FAIL OPEN on timeout: bounded by the short timeout, call served', async () => {
    stubFetch({ pilotCols: 'hang', tenant: { pilot_blocked: true } });
    const t0 = Date.now();
    const r = await resolveInboundCall('+1555');
    expect(Date.now() - t0).toBeLessThan(1500);
    expect(r.tenantId).toBe('tenant-1');
    expect('pilotBlocked' in r).toBe(false);
  });
  it('outbound direction is never blocked', async () => {
    stubFetch({ tenant: { pilot_blocked: true, pilot_blocked_reason: 'cap' } });
    const r = await resolveInboundCall('+1555', 'outbound');
    expect('pilotBlocked' in r).toBe(false);
  });
});

describe('lookup: per-call duration limit for pilots', () => {
  it('active pilot with no limit gets the pilot cap', async () => {
    stubFetch({ tenant: { pilot_blocked: false, calldesk_pilots: { status: 'active' } } });
    const r = await resolveInboundCall('+1555');
    expect(r.isPilot).toBe(true);
    expect(r.flow.globalSettings.maxCallDurationSec).toBe(PILOT_MAX_CALL_SEC);
    expect(PILOT_MAX_CALL_SEC).toBe(600);
  });
  it('keeps a shorter existing limit, shortens a longer one', async () => {
    stubFetch({ tenant: { calldesk_pilots: [{ status: 'active' }] }, globalSettings: { startNodeId: 'n1', maxCallDurationSec: 300 } });
    expect((await resolveInboundCall('+1555')).flow.globalSettings.maxCallDurationSec).toBe(300);
    _resetPilotLookupState();
    stubFetch({ tenant: { calldesk_pilots: { status: 'capped' } }, globalSettings: { startNodeId: 'n1', maxCallDurationSec: 3600 } });
    expect((await resolveInboundCall('+1555')).flow.globalSettings.maxCallDurationSec).toBe(600);
  });
  it('converted pilot, no pilot row, or unreadable status: no limit added', async () => {
    for (const rel of [{ status: 'converted' }, null, {}, undefined]) {
      _resetPilotLookupState();
      stubFetch({ tenant: { calldesk_pilots: rel } });
      const r = await resolveInboundCall('+1555');
      expect('isPilot' in r).toBe(false);
      expect('maxCallDurationSec' in r.flow.globalSettings).toBe(false);
    }
  });
});

describe('carrier webhooks carry the blocked context', () => {
  it('Twilio /twilio/voice', async () => {
    stubFetch({ tenant: { pilot_blocked: true, pilot_blocked_reason: 'stopped' } });
    const res = await request(app).post('/twilio/voice').type('form').send({ CallSid: 'CAtw1', To: '+1555', From: '+1999' });
    expect(res.status).toBe(200);
    expect(res.text).toContain('<Stream');
    const ctx = pendingCallContext.get('CAtw1');
    expect(ctx).toMatchObject({ pilotBlocked: true, pilotBlockedReason: 'stopped', fromNumber: '+1999', tenantNumber: '+1555', direction: 'inbound' });
  });
  it('Telnyx /telnyx/voice', async () => {
    stubFetch({ tenant: { pilot_blocked: true, pilot_blocked_reason: 'cap' } });
    const orig = globalThis.fetch;
    globalThis.fetch = vi.fn(async (url, init) => {
      const r = await orig(url, init);
      if (String(url).includes('/calldesk_phone_numbers')) return { ...r, json: async () => [{ ...BASE, carrier: 'telnyx' }] };
      return r;
    });
    const res = await request(app).post('/telnyx/voice?key=hook-secret').type('form').send({ CallSid: 'CStx1', To: '+1555', From: '+1999' });
    expect(res.status).toBe(200);
    expect(res.text).toContain('<Stream');
    expect(pendingCallContext.get('CStx1')).toMatchObject({ pilotBlocked: true, carrier: 'telnyx', tenantId: 'tenant-1' });
  });
});

// A fake carrier adapter: same surface wirePhoneAdapter/CallSession use.
function fakeAdapter(carrier) {
  const a = new EventEmitter();
  a.readyState = WebSocket.OPEN;
  a.sent = [];
  a.closed = false;
  if (carrier) a.carrier = carrier;
  a.send = (d) => a.sent.push(d);
  a.close = () => { a.closed = true; a.readyState = WebSocket.CLOSED; };
  a.clearQueue = () => {};
  a.isSpeaking = () => false;
  return a;
}

describe.each([[undefined, 'twilio'], ['telnyx', 'telnyx']])('blocked call on %s path', (carrier, label) => {
  let spoken;
  let session;
  beforeEach(() => {
    spoken = [];
    session = null;
    vi.spyOn(CallSession.prototype, '_speak').mockImplementation(function (text, turnId) { session = this; spoken.push({ text, turnId }); });
    vi.spyOn(CallSession.prototype, '_runPostCallAnalysis').mockResolvedValue(null);
  });
  const ctx = (sid) => ({ tenantId: 'tenant-1', pilotBlocked: true, pilotBlockedReason: 'cap', flow: { nodes: [], globalSettings: {} }, fromNumber: '+1999', tenantNumber: '+1555', ttsBackend: 'elevenlabs', voiceId: 'voice-x', ...(carrier ? { carrier } : {}), sid });
  const start = async (sid) => {
    const adapter = fakeAdapter(carrier);
    wirePhoneAdapter(adapter, label);
    pendingCallContext.set(sid, ctx(sid));
    adapter.emit('start', sid);
    await new Promise((r) => setTimeout(r, 10));
    return adapter;
  };
  const posts = () => globalThis.fetch.mock.calls.filter(([u, i]) => String(u).includes('/calldesk_call_logs') && i?.method === 'POST');
  const patches = () => globalThis.fetch.mock.calls.filter(([u, i]) => String(u).includes('/calldesk_call_logs') && i?.method === 'PATCH');

  it('speaks the polite message through the TTS path (no flow/LLM), logs abandoned+blocked, hangs up once it finishes', async () => {
    stubFetch();
    const adapter = await start(`CS-${label}`);
    expect(console.log).toHaveBeenCalledWith(expect.stringContaining('[pilot] blocked tenant=tenant-1 reason=cap'));
    expect(spoken.map((x) => x.text)).toEqual([PILOT_BLOCKED_MESSAGE]);
    expect(PILOT_BLOCKED_MESSAGE).not.toMatch(/pilot|cap|minute|trial/i);
    expect(session.tenantId).toBe('tenant-1'); // context applied (the backend itself falls back to kokoro here: no key in tests)
    expect(adapter.closed).toBe(false); // still speaking
    expect(posts()).toHaveLength(1);
    expect(JSON.parse(posts()[0][1].body)).toMatchObject({ tenant_id: 'tenant-1', retell_call_id: `CS-${label}`, outcome: 'abandoned', direction: 'inbound', analysis: { blocked: 'pilot' } });
    expect(globalThis.fetch.mock.calls.some(([u]) => /anthropic|deepgram|openai/i.test(String(u)))).toBe(false);

    // message finishes playing -> hang up
    session.turnState.pendingTts = 0;
    session._maybeRetireTurn(spoken[0].turnId);
    expect(adapter.closed).toBe(true);
    await new Promise((r) => setTimeout(r, 10));
    // finalize only adds the duration: no outcome/analysis overwrite, no post-call analysis, no webhooks
    expect(patches()).toHaveLength(1);
    expect(Object.keys(JSON.parse(patches()[0][1].body))).toEqual(['duration_seconds']);
    expect(CallSession.prototype._runPostCallAnalysis).not.toHaveBeenCalled();
    expect(urls.some((u) => u.includes('/calldesk_webhooks'))).toBe(false);
  });

  it('hangs up by itself if the message never finishes (TTS outage), and leaves no timer behind', async () => {
    vi.useFakeTimers();
    stubFetch();
    const adapter = fakeAdapter(carrier);
    wirePhoneAdapter(adapter, label);
    pendingCallContext.set(`CSx-${label}`, ctx('x'));
    adapter.emit('start', `CSx-${label}`);
    await vi.advanceTimersByTimeAsync(5);
    expect(adapter.closed).toBe(false);
    await vi.advanceTimersByTimeAsync(20_001);
    expect(adapter.closed).toBe(true);
    expect(console.log).toHaveBeenCalledWith(expect.stringContaining('message did not finish'));
  });

  it('a caller who hangs up during the message clears the safety timer (no leak)', async () => {
    vi.useFakeTimers();
    stubFetch();
    const adapter = fakeAdapter(carrier);
    wirePhoneAdapter(adapter, label);
    pendingCallContext.set(`CSy-${label}`, ctx('y'));
    adapter.emit('start', `CSy-${label}`);
    await vi.advanceTimersByTimeAsync(5);
    const before = vi.getTimerCount();
    adapter.emit('close');
    expect(vi.getTimerCount()).toBeLessThan(before);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(console.log).not.toHaveBeenCalledWith(expect.stringContaining('message did not finish'));
  });

  it('non-blocked call on the same wiring is unchanged: normal context, normal log row, no blocked handling', async () => {
    stubFetch();
    const adapter = fakeAdapter(carrier);
    wirePhoneAdapter(adapter, label);
    pendingCallContext.set(`CSn-${label}`, { tenantId: 'tenant-1', flow: { nodes: [{ id: 'n1', type: 'greeting', prompt: 'hi', edges: [] }], startNodeId: 'n1', globalSettings: {} }, fromNumber: '+1999', tenantNumber: '+1555', ...(carrier ? { carrier } : {}) });
    vi.spyOn(CallSession.prototype, '_runNodeTurn').mockImplementation(() => {});
    adapter.emit('start', `CSn-${label}`);
    await new Promise((r) => setTimeout(r, 10));
    expect(spoken).toHaveLength(0);
    expect(session).toBeNull();
    expect(console.log).not.toHaveBeenCalledWith(expect.stringContaining('[pilot]'));
    expect(JSON.parse(posts()[0][1].body)).toMatchObject({ outcome: 'answered' });
    expect('analysis' in JSON.parse(posts()[0][1].body)).toBe(false);
    adapter.emit('close');
  });
});

describe('per-call duration limit timer (existing maxCallDurationSec mechanism, set to the pilot cap)', () => {
  it('fires a graceful end at the limit and is cleared on a normal hangup (no leaked timer)', async () => {
    vi.useFakeTimers();
    stubFetch();
    const adapter = fakeAdapter();
    wirePhoneAdapter(adapter, 'twilio');
    pendingCallContext.set('CSd1', { tenantId: 'tenant-1', flow: { nodes: [{ id: 'n1', type: 'greeting', prompt: 'hi', edges: [] }], startNodeId: 'n1', globalSettings: { maxCallDurationSec: PILOT_MAX_CALL_SEC } } });
    const run = vi.spyOn(CallSession.prototype, '_runNodeTurn').mockImplementation(() => {});
    void run;
    const end = vi.spyOn(CallSession.prototype, '_endCallGracefully');
    adapter.emit('start', 'CSd1');
    await vi.advanceTimersByTimeAsync(PILOT_MAX_CALL_SEC * 1000 - 1000);
    expect(end).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1100);
    expect(end).toHaveBeenCalledWith(expect.stringContaining('maxCallDurationSec 600s'));
    expect(adapter.closed).toBe(true);
  });
  it('normal hangup before the limit cancels the timer', async () => {
    vi.useFakeTimers();
    stubFetch();
    const adapter = fakeAdapter();
    wirePhoneAdapter(adapter, 'twilio');
    pendingCallContext.set('CSd2', { tenantId: 'tenant-1', flow: { nodes: [{ id: 'n1', type: 'greeting', prompt: 'hi', edges: [] }], startNodeId: 'n1', globalSettings: { maxCallDurationSec: PILOT_MAX_CALL_SEC } } });
    vi.spyOn(CallSession.prototype, '_runNodeTurn').mockImplementation(() => {});
    const end = vi.spyOn(CallSession.prototype, '_endCallGracefully');
    adapter.emit('start', 'CSd2');
    await vi.advanceTimersByTimeAsync(1000);
    const withTimer = vi.getTimerCount();
    adapter.emit('close');
    await vi.advanceTimersByTimeAsync(0);
    expect(vi.getTimerCount()).toBeLessThan(withTimer);
    await vi.advanceTimersByTimeAsync(PILOT_MAX_CALL_SEC * 1000);
    expect(end).not.toHaveBeenCalled();
  });
});
