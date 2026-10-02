// A published agent version can name its language model (llm_model) and voice model (tts_model). The engine reads them when a call
// arrives and puts them in the call's context message; a version without them behaves exactly as before.
import { describe, it, expect, beforeEach, vi } from 'vitest';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';

const { buildTenantContextMessage } = await import('../server.js');
const { resolveInboundCall } = await import('../tenantLookup.js');

const ROW = {
  tenant_id: 'tenant-1', inbound_agent_version_id: 'version-1', outbound_agent_version_id: 'version-1',
  voice_engine: 'poc', tts_backend: 'elevenlabs', flow_id: 'flow-1', agent_id: 'agent-1',
  nodes: [{ id: 'n1', type: 'greeting', prompt: 'Say hello.', edges: [] }], global_settings: { startNodeId: 'n1' },
  stripe_customer_id: null, provider: null, api_key: null, event_type_id: null, settings: {},
};

// fetch stub: every table returns `rows`; `failWhen(url)` makes a request fail with HTTP 400 (like selecting a column that does not exist).
function stubFetch(rows, failWhen = () => false) {
  const urls = [];
  global.fetch = vi.fn(async (url) => {
    urls.push(String(url));
    if (failWhen(String(url))) return { ok: false, status: 400, text: async () => 'column does not exist', json: async () => ({}) };
    return { ok: true, status: 200, json: async () => rows, text: async () => '' };
  });
  return urls;
}

describe('tenantLookup: version model choice', () => {
  beforeEach(() => { vi.spyOn(console, 'error').mockImplementation(() => {}); vi.spyOn(console, 'warn').mockImplementation(() => {}); });

  it('returns llmModel and ttsModel when the version has them, and asks for the columns', async () => {
    const urls = stubFetch([{ ...ROW, llm_model: 'gpt-6-luna', tts_model: 'eleven_flash_v2_5' }]);
    const r = await resolveInboundCall('+15550001111', 'inbound');
    expect(r.llmModel).toBe('gpt-6-luna');
    expect(r.ttsModel).toBe('eleven_flash_v2_5');
    expect(urls.some((u) => u.includes('calldesk_agent_versions') && u.includes('llm_model') && u.includes('tts_model'))).toBe(true);
  });

  it('a version with no model choice resolves exactly as before (both undefined)', async () => {
    stubFetch([{ ...ROW, llm_model: null, tts_model: null }]);
    const r = await resolveInboundCall('+15550001111', 'inbound');
    expect(r.llmModel).toBeUndefined();
    expect(r.ttsModel).toBeUndefined();
    expect(r.ttsBackend).toBe('elevenlabs');
    expect(r.flow.nodes).toHaveLength(1);
  });

  it('still connects the call when the new columns do not exist yet (query fails, retries without them)', async () => {
    const urls = stubFetch([ROW], (u) => u.includes('calldesk_agent_versions') && u.includes('llm_model'));
    const r = await resolveInboundCall('+15550001111', 'inbound');
    expect(r).not.toBeNull();
    expect(r.flow.nodes).toHaveLength(1);
    expect(r.llmModel).toBeUndefined();
    const versionQueries = urls.filter((u) => u.includes('calldesk_agent_versions'));
    // the tier column is requested first, then the model columns, then the base columns (see tieredBilling.test.js)
    expect(versionQueries).toHaveLength(3);
    expect(versionQueries[2]).not.toContain('llm_model');
  });
});

describe('buildTenantContextMessage', () => {
  const base = { flow: { nodes: [], startNodeId: 'n1' }, tenantId: 't1' };
  it('puts the model in the context only when the version chose one', () => {
    expect(buildTenantContextMessage({ ...base, llmModel: 'gpt-6-luna' })).toMatchObject({ type: 'context', model: 'gpt-6-luna', tenantId: 't1' });
    expect('model' in buildTenantContextMessage(base)).toBe(false);
    expect('ttsModel' in buildTenantContextMessage(base)).toBe(false);
  });
  it('carries the voice model with the voice backend', () => {
    expect(buildTenantContextMessage({ ...base, ttsBackend: 'elevenlabs', ttsModel: 'eleven_v4_turbo' })).toMatchObject({ ttsBackend: 'elevenlabs', ttsModel: 'eleven_v4_turbo' });
  });
  it('keeps every field it carried before', () => {
    const m = buildTenantContextMessage({ ...base, stripeCustomerId: 'cus_1', fromNumber: '+1555', tenantNumber: '+1666', calendar: { provider: 'cal' }, callAudio: { jingle: null, effects: [] } });
    expect(m).toMatchObject({ stripeCustomerId: 'cus_1', phoneNumber: '+1555', tenantNumber: '+1666', calendar: { provider: 'cal' }, callAudio: { jingle: null, effects: [] } });
  });
});
