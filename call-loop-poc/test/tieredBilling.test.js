// Per-agent tiered billing, engine side: tier read, context message, call log attribution. (Billing itself is done by the web app's cron from call logs.)
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';

const { buildTenantContextMessage } = await import('../server.js');
const { resolveInboundCall, insertCallLog, updateCallLogByCallSid, normalizeTier } = await import('../tenantLookup.js');

const ROW = {
  tenant_id: 'tenant-1', inbound_agent_version_id: 'version-1', outbound_agent_version_id: 'version-1',
  voice_engine: 'poc', tts_backend: 'elevenlabs', flow_id: 'flow-1', agent_id: 'agent-1',
  nodes: [{ id: 'n1', type: 'greeting', prompt: 'Say hello.', edges: [] }], global_settings: { startNodeId: 'n1' },
  stripe_customer_id: null, provider: null, api_key: null, event_type_id: null, settings: {},
};

beforeEach(() => { vi.spyOn(console, 'error').mockImplementation(() => {}); vi.spyOn(console, 'warn').mockImplementation(() => {}); vi.spyOn(console, 'log').mockImplementation(() => {}); });
afterEach(() => { vi.restoreAllMocks(); });

describe('tenantLookup tier', () => {
  function stub(rows, failWhen = () => false) {
    const urls = [];
    global.fetch = vi.fn(async (url) => {
      urls.push(String(url));
      if (failWhen(String(url))) return { ok: false, status: 400, text: async () => 'no column', json: async () => ({}) };
      return { ok: true, status: 200, json: async () => rows, text: async () => '' };
    });
    return urls;
  }

  it('reads a valid tier and queries the column', async () => {
    const urls = stub([{ ...ROW, tier: 'pro' }]);
    const r = await resolveInboundCall('+15550001111');
    expect(r.tier).toBe('pro');
    expect(urls.some((u) => u.includes('calldesk_agent_versions') && u.includes('tts_model,tier'))).toBe(true);
  });
  it.each([['gold'], [''], [null], [42], ['PRO ']])('invalid/absent value %j: handled', async (v) => {
    stub([{ ...ROW, tier: v }]);
    const r = await resolveInboundCall('+15550001111');
    expect(r.tier).toBe(v === 'PRO ' ? 'pro' : undefined);
    if (v !== 'PRO ') expect('tier' in r).toBe(false);
  });
  it('column missing: call still resolves, tier absent, model choice still read', async () => {
    const urls = stub([{ ...ROW, llm_model: 'm1' }], (u) => u.includes('calldesk_agent_versions') && u.includes(',tier'));
    const r = await resolveInboundCall('+15550001111');
    expect(r).not.toBeNull();
    expect(urls.filter((u) => u.includes('calldesk_agent_versions'))).toHaveLength(2);
    expect(r.tier).toBeUndefined();
    expect(r.llmModel).toBe('m1');
  });
  it('neither tier nor model columns exist: still routes with the base columns', async () => {
    const urls = stub([ROW], (u) => u.includes('calldesk_agent_versions') && (u.includes('tier') || u.includes('llm_model')));
    const r = await resolveInboundCall('+15550001111');
    expect(r).not.toBeNull();
    expect(urls.filter((u) => u.includes('calldesk_agent_versions'))).toHaveLength(3);
    expect(r.tier).toBeUndefined();
  });
  it('normalizeTier', () => {
    expect(normalizeTier('lite')).toBe('lite');
    expect(normalizeTier('enterprise')).toBeUndefined();
    expect(normalizeTier(undefined)).toBeUndefined();
  });
});

describe('buildTenantContextMessage tier', () => {
  it('is identical to today when there is no tier', () => {
    const base = { flow: { nodes: [] }, tenantId: 't' };
    expect(buildTenantContextMessage(base)).toEqual({ type: 'context', flow: { nodes: [] }, tenantId: 't' });
    expect('tier' in buildTenantContextMessage(base)).toBe(false);
  });
  it('carries tier when set', () => {
    expect(buildTenantContextMessage({ flow: {}, tier: 'standard' }).tier).toBe('standard');
  });
});

describe('call log tier write', () => {
  function stub(failTier) {
    const bodies = [];
    global.fetch = vi.fn(async (url, opts) => {
      const body = JSON.parse(opts.body);
      bodies.push(body);
      if (failTier && 'tier' in body) return { ok: false, status: 400, text: async () => 'no column', json: async () => ({}) };
      return { ok: true, status: 200, text: async () => '', json: async () => [{ id: 'log-1' }] };
    });
    return bodies;
  }
  it('insert with tier column present', async () => {
    const b = stub(false);
    expect(await insertCallLog({ a: 1, tier: 'pro' })).toBe('log-1');
    expect(b).toEqual([{ a: 1, tier: 'pro' }]);
  });
  it('insert retries without tier when the column is missing', async () => {
    const b = stub(true);
    expect(await insertCallLog({ a: 1, tier: 'pro' })).toBe('log-1');
    expect(b).toEqual([{ a: 1, tier: 'pro' }, { a: 1 }]);
  });
  it('insert without tier: single request, as before', async () => {
    const b = stub(true);
    await insertCallLog({ a: 1 });
    expect(b).toHaveLength(1);
  });
  it('update retries without tier when the column is missing', async () => {
    const b = stub(true);
    await updateCallLogByCallSid('CA1', { duration_seconds: 5, tier: 'lite' });
    expect(b).toEqual([{ duration_seconds: 5, tier: 'lite' }, { duration_seconds: 5 }]);
  });
  it('update without tier: single request', async () => {
    const b = stub(false);
    await updateCallLogByCallSid('CA1', { duration_seconds: 5 });
    expect(b).toHaveLength(1);
  });
});
