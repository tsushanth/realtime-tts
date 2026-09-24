import { describe, it, expect, beforeEach, vi } from 'vitest';
import request from 'supertest';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';
process.env.TEST_CALL_SECRET = 'test-secret';

const { app, pendingCallContext, sampleCalleeOverrides } = await import('../server.js');

const callee = { systemPrompt: 'You are the after-hours line for a fictional brokerage.', greeting: 'Thanks for calling, how can I help?' };

describe('sampleCallee', () => {
  it('advertises the capability', async () => {
    const res = await request(app).get('/sample-callee-capability');
    expect(res.body).toEqual({ sampleCallee: 1 });
  });

  beforeEach(() => { sampleCalleeOverrides.clear(); pendingCallContext.clear(); });

  it('rejects sampleCallee without shopper:true', async () => {
    const res = await request(app).post('/place-test-call').set('Authorization', 'Bearer test-secret')
      .send({ toNumber: '+15550000001', routeAs: '+15550000002', sampleCallee: callee });
    expect(res.status).toBe(400);
  });

  it('rejects a malformed sampleCallee', async () => {
    const res = await request(app).post('/place-test-call').set('Authorization', 'Bearer test-secret')
      .send({ toNumber: '+15550000001', shopper: true, sampleCallee: { systemPrompt: 'x', greeting: '' } });
    expect(res.status).toBe(400);
  });

  it('/twilio/voice consumes a matching override exactly once', async () => {
    sampleCalleeOverrides.set('+15550000001', { ...callee, fromNumber: '+15550000009', expiresAt: Date.now() + 60_000 });
    await request(app).post('/twilio/voice').type('form').send({ CallSid: 'CA' + 'a'.repeat(32), To: '+15550000001', From: '+15550000009' });
    expect(pendingCallContext.get('CA' + 'a'.repeat(32))?.isSampleCallee).toBe(true);
    expect(sampleCalleeOverrides.size).toBe(0);
  });

  it('ignores an override when From does not match', async () => {
    sampleCalleeOverrides.set('+15550000001', { ...callee, fromNumber: '+15550000009', expiresAt: Date.now() + 60_000 });
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, json: async () => [], text: async () => '[]' })));
    await request(app).post('/twilio/voice').type('form').send({ CallSid: 'CA' + 'b'.repeat(32), To: '+15550000001', From: '+15551111111' });
    expect(pendingCallContext.get('CA' + 'b'.repeat(32))?.isSampleCallee).toBeUndefined();
    expect(sampleCalleeOverrides.size).toBe(1);
    vi.unstubAllGlobals();
  });
});
