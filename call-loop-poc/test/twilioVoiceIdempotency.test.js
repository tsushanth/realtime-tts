// Real regression test for today's "two greetings back to back" bug (commit eb64070).
//
// Root cause: /twilio/voice's real-call branch called resolveInboundCall() (2 sequential + 4
// parallel Supabase HTTP round trips) and wrote a pendingCallContext entry with NO protection
// against Twilio retrying the same CallSid — a slow response, a network blip, or a Fly cold start
// can all trigger a real retry. Each independent invocation used to run its own lookup and its own
// context write; a second Media Stream connection could then consume a second, later-written entry
// and greet again.
//
// This test hits the real Express route (via supertest against the exported `app`) with a mocked
// `fetch` standing in for Supabase, and asserts on the real module-level state
// (`pendingCallContext`, `contextWrittenCallSids`) rather than guessing at behavior from the
// outside — proving both halves of the fix: concurrent requests coalesce onto one lookup, and a
// late (sequential) retry after full completion is a no-op rather than a second context write.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import request from 'supertest';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';

const { app, pendingCallContext, contextWrittenCallSids } = await import('../server.js');

// One generic row shape covering every table resolveInboundCall/tenantLookup.js queries — pg()
// just indexes array[0].<field>, so a superset of fields across all real queries is harmless.
const FAKE_ROW = {
  tenant_id: 'tenant-1',
  inbound_agent_version_id: 'version-1',
  outbound_agent_version_id: 'version-1',
  voice_engine: 'poc',
  tts_backend: null,
  flow_id: 'flow-1',
  agent_id: 'agent-1',
  nodes: [{ id: 'n1', type: 'greeting', prompt: 'Say hello.', edges: [] }],
  global_settings: { startNodeId: 'n1' },
  stripe_customer_id: null,
  provider: null,
  api_key: null,
  event_type_id: null,
  settings: {},
};

// A small real delay per call matters here, not just for realism: with a near-instant mock, the
// first request's whole resolveInboundCall() chain can finish before the second request's HTTP
// bytes even arrive at the server (real supertest/HTTP-layer overhead vs. an instant mock), so the
// two requests never actually overlap — the coalescing path never gets exercised and only the
// separate "late retry" guard fires. Real slow-DB conditions were the actual root cause found
// today (2 sequential + 4 parallel Supabase round trips, no timeout) — this delay is standing in
// for that, not padding.
function mockFetch(delayMs = 15) {
  let callCount = 0;
  const impl = vi.fn(async () => {
    callCount++;
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    return { ok: true, json: async () => [FAKE_ROW] };
  });
  impl.callCount = () => callCount;
  return impl;
}

describe('/twilio/voice real-call idempotency (today\'s regression)', () => {
  beforeEach(() => {
    pendingCallContext.clear();
    contextWrittenCallSids.clear();
  });

  it('a single request resolves normally and writes context exactly once', async () => {
    global.fetch = mockFetch();
    const callSid = 'CAsingle000000000000000000000000';
    const res = await request(app)
      .post('/twilio/voice')
      .type('form')
      .send({ CallSid: callSid, To: '+15550001111', From: '+15559998888' });

    expect(res.status).toBe(200);
    expect(res.text).toContain('<Stream');
    expect(pendingCallContext.has(callSid)).toBe(true);
    expect(contextWrittenCallSids.has(callSid)).toBe(true);
  });

  it('two concurrent requests for the same CallSid coalesce onto one lookup, not two', async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock;
    const callSid = 'CAconcurrent0000000000000000000';
    const body = { CallSid: callSid, To: '+15550001111', From: '+15559998888' };

    // Fired together, mirroring a real Twilio retry landing while the first request is still
    // resolving — the exact race that used to cause a second, independent resolveInboundCall().
    const [res1, res2] = await Promise.all([
      request(app).post('/twilio/voice').type('form').send(body),
      request(app).post('/twilio/voice').type('form').send(body),
    ]);

    expect(res1.status).toBe(200);
    expect(res2.status).toBe(200);
    // Exactly one full resolveInboundCall() should have run: 2 sequential + 4 parallel real
    // queries = 6 fetch calls. If the two requests had raced independently (the bug), this would
    // be 12. This is the direct, black-box proof the coalescing fix works.
    expect(fetchMock.callCount()).toBe(6);
    expect(contextWrittenCallSids.has(callSid)).toBe(true);
  });

  it('a late retry after the first request already fully completed is a no-op, not a second write', async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock;
    const callSid = 'CAlateretry00000000000000000000';
    const body = { CallSid: callSid, To: '+15550001111', From: '+15559998888' };

    const first = await request(app).post('/twilio/voice').type('form').send(body);
    expect(first.status).toBe(200);
    expect(contextWrittenCallSids.has(callSid)).toBe(true);
    const callsAfterFirst = fetchMock.callCount();

    // Simulate the Media Stream having already consumed the first context (the real flow: the
    // stream's 'start' handler deletes its pendingCallContext entry once it hands off to the
    // session) — this is exactly the state a late retry would actually see.
    pendingCallContext.delete(callSid);

    const second = await request(app).post('/twilio/voice').type('form').send(body);
    expect(second.status).toBe(200);
    expect(second.text).toContain('<Stream'); // still gets valid TwiML, just no duplicate context
    // The late retry still re-runs resolveInboundCall (contextWrittenCallSids only guards the
    // WRITE, not the lookup itself — see the fix's own comment on why that's the simpler, safer
    // choice) but must NOT write pendingCallContext again.
    expect(fetchMock.callCount()).toBeGreaterThan(callsAfterFirst);
    expect(pendingCallContext.has(callSid)).toBe(false); // the real assertion: no duplicate write
  });

  it('different CallSids never coalesce with each other', async () => {
    const fetchMock = mockFetch();
    global.fetch = fetchMock;
    const [res1, res2] = await Promise.all([
      request(app).post('/twilio/voice').type('form').send({ CallSid: 'CAone00000000000000000000000000', To: '+15550001111', From: '+15559998888' }),
      request(app).post('/twilio/voice').type('form').send({ CallSid: 'CAtwo00000000000000000000000000', To: '+15550001111', From: '+15559998888' }),
    ]);
    expect(res1.status).toBe(200);
    expect(res2.status).toBe(200);
    expect(pendingCallContext.has('CAone00000000000000000000000000')).toBe(true);
    expect(pendingCallContext.has('CAtwo00000000000000000000000000')).toBe(true);
    // Two independent calls, two independent lookups: 12, not 6.
    expect(fetchMock.callCount()).toBe(12);
  });
});
