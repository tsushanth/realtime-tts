// Routing + auth for the opt-in Telnyx path: TELNYX_ENABLED=1, number carrier=telnyx, webhook auth, signed stream URL.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';
import crypto from 'node:crypto';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';
process.env.TELNYX_ENABLED = '1';
process.env.TELNYX_WEBHOOK_SECRET = 'hook-secret';
const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519');
const PUB_B64 = publicKey.export({ format: 'der', type: 'spki' }).subarray(-32).toString('base64');

const { app, pendingCallContext, contextWrittenCallSids } = await import('../server.js');
const { verifyTelnyxSignature, signStreamParams, verifyStreamParams, buildTelnyxStreamTexml } = await import('../telnyx.js');

const ROW = {
  tenant_id: 'tenant-1', inbound_agent_version_id: 'v1', outbound_agent_version_id: 'v1', voice_engine: 'poc', tts_backend: null,
  flow_id: 'f1', agent_id: 'a1', nodes: [{ id: 'n1', type: 'greeting', prompt: 'hi', edges: [] }], global_settings: { startNodeId: 'n1' },
  stripe_customer_id: null, provider: null, api_key: null, event_type_id: null, settings: {},
};
const urls = [];
function mockFetch({ carrier, carrierColumnExists = true }) {
  globalThis.fetch = vi.fn(async (url) => {
    urls.push(String(url));
    const u = String(url);
    if (u.includes('calldesk_phone_numbers') && u.includes('carrier') && !carrierColumnExists) return { ok: false, status: 400, text: async () => 'no column', json: async () => ({}) };
    const row = u.includes('calldesk_phone_numbers') && carrierColumnExists && carrier !== undefined ? { ...ROW, carrier } : ROW;
    return { ok: true, json: async () => [row] };
  });
}
const form = (o) => new URLSearchParams(o).toString();
const post = (q = '?key=hook-secret') => request(app).post(`/telnyx/voice${q}`).set('content-type', 'application/x-www-form-urlencoded');

describe('/telnyx/voice routing', () => {
  const realFetch = globalThis.fetch;
  beforeEach(() => { pendingCallContext.clear(); contextWrittenCallSids.clear(); urls.length = 0; vi.spyOn(console, 'warn').mockImplementation(() => {}); });
  afterEach(() => { globalThis.fetch = realFetch; vi.restoreAllMocks(); });

  it('number with carrier=telnyx: stores context and returns signed bidirectional <Stream> TeXML', async () => {
    mockFetch({ carrier: 'telnyx' });
    const res = await post().send(form({ CallSid: 'CS1', To: '+15551230000', From: '+15559990000' }));
    expect(res.status).toBe(200);
    expect(res.text).toMatch(/<Connect><Stream url="wss:\/\/[^"/]+\/telnyx-stream\?cs=CS1&amp;exp=\d+&amp;sig=[0-9a-f]{64}" bidirectionalMode="rtp" bidirectionalCodec="PCMU">/);
    expect(res.text).toContain('<Parameter name="callSid" value="CS1"/>');
    const ctx = pendingCallContext.get('CS1');
    expect(ctx.carrier).toBe('telnyx');
    expect(ctx.tenantId).toBe('tenant-1');
    expect(ctx.fromNumber).toBe('+15559990000');
    expect(ctx.tenantNumber).toBe('+15551230000');
    expect(urls.some((u) => u.includes('calldesk_phone_numbers') && u.includes(',carrier'))).toBe(true);
  });

  it('a retried webhook does not write a second context', async () => {
    mockFetch({ carrier: 'telnyx' });
    await post().send(form({ CallSid: 'CS2', To: '+1555', From: '+1666' }));
    pendingCallContext.delete('CS2');
    await post().send(form({ CallSid: 'CS2', To: '+1555', From: '+1666' }));
    expect(pendingCallContext.has('CS2')).toBe(false);
  });

  it('number whose carrier is twilio (default / column absent) is refused: Hangup, no context', async () => {
    for (const carrier of [undefined, 'twilio', null]) {
      mockFetch({ carrier });
      const res = await post().send(form({ CallSid: 'CS3', To: '+1555', From: '+1666' }));
      expect(res.text).toContain('<Hangup/>');
      expect(res.text).not.toContain('<Stream');
      expect(pendingCallContext.has('CS3')).toBe(false);
    }
  });

  it('carrier column missing (pre-migration): query is retried without it and the number routes as twilio', async () => {
    mockFetch({ carrier: 'telnyx', carrierColumnExists: false });
    const res = await post().send(form({ CallSid: 'CS4', To: '+1555', From: '+1666' }));
    expect(res.text).toContain('<Hangup/>');
    expect(urls.filter((u) => u.includes('calldesk_phone_numbers')).length).toBe(2);
    expect(urls.filter((u) => u.includes('calldesk_phone_numbers')).at(-1)).not.toContain('carrier');
  });

  it('missing/incorrect webhook key is rejected with 403 before any lookup', async () => {
    mockFetch({ carrier: 'telnyx' });
    expect((await post('').send(form({ CallSid: 'X', To: '+1' }))).status).toBe(403);
    expect((await post('?key=wrong').send(form({ CallSid: 'X', To: '+1' }))).status).toBe(403);
    expect(urls).toHaveLength(0);
  });

  it('accepts a JSON body too', async () => {
    mockFetch({ carrier: 'telnyx' });
    const res = await request(app).post('/telnyx/voice?key=hook-secret').send({ CallSid: 'CS5', To: '+1555', From: '+1666' });
    expect(res.text).toContain('<Stream');
    expect(pendingCallContext.has('CS5')).toBe(true);
  });

  it('missing CallSid/To -> Hangup', async () => {
    mockFetch({ carrier: 'telnyx' });
    expect((await post().send(form({ From: '+1' }))).text).toContain('<Hangup/>');
  });
});

describe('Ed25519 webhook signature', () => {
  const body = Buffer.from('CallSid=CS9&To=%2B1555');
  const now = 1_800_000_000;
  const sign = (ts, b = body) => crypto.sign(null, Buffer.concat([Buffer.from(`${ts}|`), b]), privateKey).toString('base64');

  it('accepts a valid signature over `${timestamp}|${rawBody}`', () => {
    expect(verifyTelnyxSignature({ rawBody: body, signature: sign(now), timestamp: String(now), publicKeyB64: PUB_B64, nowSec: now })).toBe(true);
  });
  it('rejects tampered body, wrong key, stale timestamp, garbage', () => {
    const ok = { rawBody: body, signature: sign(now), timestamp: String(now), publicKeyB64: PUB_B64, nowSec: now };
    expect(verifyTelnyxSignature({ ...ok, rawBody: Buffer.from('tampered') })).toBe(false);
    const other = crypto.generateKeyPairSync('ed25519').publicKey.export({ format: 'der', type: 'spki' }).subarray(-32).toString('base64');
    expect(verifyTelnyxSignature({ ...ok, publicKeyB64: other })).toBe(false);
    expect(verifyTelnyxSignature({ ...ok, nowSec: now + 301 })).toBe(false);
    expect(verifyTelnyxSignature({ ...ok, signature: 'AAAA' })).toBe(false);
    expect(verifyTelnyxSignature({ ...ok, publicKeyB64: 'short' })).toBe(false);
    expect(verifyTelnyxSignature({})).toBe(false);
  });
});

describe('signed stream URL', () => {
  const env = { TELNYX_STREAM_SECRET: 's3cret' };
  const sp = (o) => new URLSearchParams(o);
  it('round-trips and rejects tampering/expiry', () => {
    const p = signStreamParams('CSX', env, 1000);
    expect(verifyStreamParams(sp(p), env, 1001)).toBe('CSX');
    expect(verifyStreamParams(sp({ ...p, cs: 'OTHER' }), env, 1001)).toBeNull();
    expect(verifyStreamParams(sp(p), env, 1000 + 121)).toBeNull();
    expect(verifyStreamParams(sp({ ...p, sig: '0'.repeat(64) }), env, 1001)).toBeNull();
    expect(verifyStreamParams(sp({}), env, 1001)).toBeNull();
    expect(verifyStreamParams(sp(p), { TELNYX_STREAM_SECRET: 'different' }, 1001)).toBeNull();
  });
  it('TeXML escapes attribute values', () => {
    const t = buildTelnyxStreamTexml({ host: 'h.example', callSid: 'a"b<c', env });
    expect(t).not.toContain('a"b<c');
    expect(t).toContain('a&quot;b&lt;c');
  });
});
