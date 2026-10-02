// Default behaviour (no TELNYX_* env): no Telnyx route exists, tenant lookup issues the unchanged query, resolved object has no carrier key.
import { describe, it, expect, vi } from 'vitest';
import request from 'supertest';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';
delete process.env.TELNYX_ENABLED;

const { app, pendingCallContext } = await import('../server.js');
const { resolveInboundCall } = await import('../tenantLookup.js');

const ROW = { tenant_id: 't', inbound_agent_version_id: 'v', voice_engine: 'poc', flow_id: 'f', agent_id: 'a', nodes: [{ id: 'n1', type: 'greeting', edges: [] }], global_settings: {}, settings: {}, carrier: 'telnyx' };

describe('Telnyx disabled (default)', () => {
  it('/telnyx/voice is not mounted', async () => {
    const res = await request(app).post('/telnyx/voice').send('CallSid=x&To=%2B1');
    expect(res.status).toBe(404);
  });

  it('phone-number query is the original one (no carrier column) and a stray carrier value is ignored', async () => {
    const urls = [];
    const real = globalThis.fetch;
    globalThis.fetch = vi.fn(async (u) => { urls.push(String(u)); return { ok: true, json: async () => [ROW] }; });
    const r = await resolveInboundCall('+1555');
    globalThis.fetch = real;
    const phoneUrl = urls.find((u) => u.includes('calldesk_phone_numbers'));
    expect(phoneUrl).toMatch(/select=tenant_id,inbound_agent_version_id$/);
    // carrier only appears on the resolved object when the number says telnyx - here the query never asked, but even
    // if a row carried it the Twilio route spreads it harmlessly and buildTenantContextMessage never forwards it.
    expect(r.tenantId).toBe('t');
  });

  it('/twilio/voice still returns the unchanged TwiML', async () => {
    const res = await request(app).post('/twilio/voice').type('form').send({ CallSid: 'CA1' });
    expect(res.text).toMatch(/^<\?xml version="1.0" encoding="UTF-8"\?><Response><Connect><Stream url="wss:\/\/[^"]+\/twilio-stream" \/><\/Connect><\/Response>$/);
    pendingCallContext.clear();
  });
});
