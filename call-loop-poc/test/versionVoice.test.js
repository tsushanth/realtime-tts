// The agent version's saved voice (calldesk_agent_versions.voice_id): lookup, context message, validation, and how a session applies it.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import WebSocket from 'ws';
import { resolveLanguage } from '../languages.js';

process.env.NODE_ENV = 'test';
process.env.SUPABASE_URL = 'https://fake-project.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'fake-service-role-key';
// Keys/config so every backend is "ready" for the context message (module constants are read at import time).
process.env.ELEVENLABS_API_KEY = 'test-eleven-key';
process.env.ELEVENLABS_VOICE_ID = 'DEFAULTELEVENVOICE0001';
process.env.CARTESIA_API_KEY = 'test-cartesia-key';
process.env.CARTESIA_VOICE_ID = '11111111-1111-4111-8111-111111111111';
process.env.MINIMAX_API_KEY = 'test-minimax-key';
process.env.MINIMAX_GROUP_ID = 'test-group';
process.env.MINIMAX_VOICE_ID = 'English_Default_Voice';
process.env.TTS_VOICE = 'af_heart';

const { CallSession, buildTenantContextMessage } = await import('../server.js');
const { resolveInboundCall } = await import('../tenantLookup.js');
const { validateVersionVoice } = await import('../voiceSelection.js');

const EL_ID = 'EXAVITQu4vr4xnSDxMaL'; // 20 alphanumerics, an ElevenLabs-shaped id
const CARTESIA_ID = '0bfbea6c-2f8f-4f86-b411-aa2316561e36';
const MINIMAX_ID = 'English_Graceful_Lady';

const ROW = {
  tenant_id: 'tenant-1', inbound_agent_version_id: 'version-1', outbound_agent_version_id: 'version-1',
  voice_engine: 'poc', tts_backend: 'elevenlabs', flow_id: 'flow-1', agent_id: 'agent-1',
  nodes: [{ id: 'n1', type: 'greeting', prompt: 'Say hello.', edges: [] }], global_settings: { startNodeId: 'n1' },
  stripe_customer_id: null, provider: null, api_key: null, event_type_id: null, settings: {},
};

beforeEach(() => { vi.spyOn(console, 'error').mockImplementation(() => {}); vi.spyOn(console, 'warn').mockImplementation(() => {}); vi.spyOn(console, 'log').mockImplementation(() => {}); });
afterEach(() => { vi.restoreAllMocks(); });

function stubFetch(rows, failWhen = () => false) {
  const urls = [];
  global.fetch = vi.fn(async (url) => {
    urls.push(String(url));
    if (failWhen(String(url))) return { ok: false, status: 400, text: async () => 'column does not exist', json: async () => ({}) };
    return { ok: true, status: 200, json: async () => rows, text: async () => '' };
  });
  return urls;
}
const versionQueries = (urls) => urls.filter((u) => u.includes('calldesk_agent_versions'));

describe('tenantLookup: voice_id', () => {
  it('reads voice_id in the first version query and exposes voiceId', async () => {
    const urls = stubFetch([{ ...ROW, voice_id: ` ${EL_ID} ` }]);
    const r = await resolveInboundCall('+15550001111');
    expect(r.voiceId).toBe(EL_ID);
    expect(versionQueries(urls)).toHaveLength(1);
    expect(versionQueries(urls)[0]).toContain('voice_id');
  });

  it.each([[null], [''], ['   '], [42], [undefined]])('absent/blank voice_id %j: no voiceId key at all', async (v) => {
    stubFetch([{ ...ROW, voice_id: v }]);
    const r = await resolveInboundCall('+15550001111');
    expect('voiceId' in r).toBe(false);
  });

  it('column missing: retries without it, keeps the model and tier columns, call still resolves', async () => {
    const urls = stubFetch([{ ...ROW, llm_model: 'm1', tier: 'pro' }], (u) => u.includes('calldesk_agent_versions') && u.includes('voice_id'));
    const r = await resolveInboundCall('+15550001111');
    expect(r).not.toBeNull();
    expect('voiceId' in r).toBe(false);
    expect(r.llmModel).toBe('m1');
    expect(r.tier).toBe('pro');
    const q = versionQueries(urls);
    expect(q).toHaveLength(4);
    expect(q[3]).toContain('tier');
    expect(q[3]).not.toContain('voice_id');
  });

  it('tier column missing but voice_id present: the voice is kept', async () => {
    const urls = stubFetch([{ ...ROW, voice_id: EL_ID }], (u) => u.includes('calldesk_agent_versions') && u.includes(',tier'));
    const r = await resolveInboundCall('+15550001111');
    expect(r.voiceId).toBe(EL_ID);
    expect(versionQueries(urls)).toHaveLength(2);
  });
});

describe('buildTenantContextMessage: voiceId', () => {
  const base = { flow: { nodes: [], startNodeId: 'n1', globalSettings: {} }, ttsBackend: 'elevenlabs' };
  it('unchanged when unset', () => {
    expect(buildTenantContextMessage(base)).toEqual({ type: 'context', flow: base.flow, ttsBackend: 'elevenlabs' });
    expect('voiceId' in buildTenantContextMessage({ ...base, voiceId: undefined })).toBe(false);
  });
  it('carries voiceId when set', () => {
    expect(buildTenantContextMessage({ ...base, voiceId: EL_ID }).voiceId).toBe(EL_ID);
  });
});

describe('validateVersionVoice', () => {
  it.each([
    ['elevenlabs', EL_ID, true], ['elevenlabs', '11labs-Adrian', false], ['elevenlabs', 'short', false], ['elevenlabs', 'has space in it 1234', false],
    ['elevenlabs', `${EL_ID}${EL_ID}`, false], ['elevenlabs', CARTESIA_ID, false],
    ['cartesia', CARTESIA_ID, true], ['cartesia', EL_ID, false], ['cartesia', 'cartesia-Cleo', false],
    ['minimax', MINIMAX_ID, true], ['minimax', 'minimax-Cimo', false], ['minimax', 'bad voice!', false], ['minimax', 'ab', false],
    ['kokoro', 'af_heart', true], ['kokoro', 'am_adam', true], ['kokoro', 'bf_emma', false], ['kokoro', 'nope', false],
    ['fish', 'anything12345678901', false], ['piper', 'en_US-lessac-medium', false],
    ['elevenlabs', '', false], ['elevenlabs', '   ', false], ['elevenlabs', null, false], ['elevenlabs', 7, false],
  ])('%s %j -> %s', (backend, value, ok) => {
    expect(validateVersionVoice(backend, value).ok).toBe(ok);
  });
  it('trims and returns the normalized id', () => {
    expect(validateVersionVoice('kokoro', '  af_sky ')).toEqual({ ok: true, voiceId: 'af_sky' });
  });
});

describe('session: precedence per backend', () => {
  const fakeWs = () => ({ readyState: WebSocket.OPEN, sent: [], send(d, o) { this.sent.push({ d, o }); }, close() {} });
  function open(extra) {
    const s = new CallSession(fakeWs());
    s.onClientMessage(JSON.stringify({ type: 'context', systemPrompt: 'You are a test receptionist.', ...extra }), false);
    return s;
  }
  const sessions = [];
  const mk = (extra) => { const s = open(extra); sessions.push(s); return s; };
  afterEach(() => { while (sessions.length) sessions.pop().close(); });

  it('no voiceId: every voice stays on the env default (unchanged behaviour)', () => {
    const s = mk({ ttsBackend: 'elevenlabs' });
    expect(s.elevenVoiceId).toBe('DEFAULTELEVENVOICE0001');
    expect(s.cartesiaVoiceId).toBe(process.env.CARTESIA_VOICE_ID);
    expect(s.minimaxVoiceId).toBe('English_Default_Voice');
    expect(s.voice).toBe('af_heart');
    expect(s._versionVoice).toBeNull();
  });

  it('elevenlabs: version voice replaces the default', () => {
    const s = mk({ ttsBackend: 'elevenlabs', voiceId: EL_ID });
    expect(s.elevenVoiceId).toBe(EL_ID);
    expect(s.cartesiaVoiceId).toBe(process.env.CARTESIA_VOICE_ID);
  });
  it('cartesia: version voice replaces the default', () => {
    const s = mk({ ttsBackend: 'cartesia', voiceId: CARTESIA_ID });
    expect(s.cartesiaVoiceId).toBe(CARTESIA_ID);
    expect(s.elevenVoiceId).toBe('DEFAULTELEVENVOICE0001');
  });
  it('minimax: version voice replaces the default', () => {
    const s = mk({ ttsBackend: 'minimax', voiceId: MINIMAX_ID });
    expect(s.minimaxVoiceId).toBe(MINIMAX_ID);
  });
  it('kokoro: version voice sets the voice name', () => {
    const s = mk({ ttsBackend: 'kokoro', voiceId: 'am_adam' });
    expect(s.voice).toBe('am_adam');
  });
  it('no ttsBackend in the context: validated against the engine default backend', () => {
    // TTS_BACKEND defaults to kokoro in the test env
    const s = mk({ voiceId: 'af_nova' });
    expect(s.ttsBackend).toBe('kokoro');
    expect(s.voice).toBe('af_nova');
  });

  it('explicit per-call voice (context `voice`) beats the version voice on kokoro', () => {
    const s = mk({ ttsBackend: 'kokoro', voice: 'af_bella', voiceId: 'am_adam' });
    expect(s.voice).toBe('af_bella');
  });
  it('explicit sample/shopper voice beats the version voice on elevenlabs', () => {
    const s = mk({ ttsBackend: 'elevenlabs', voiceId: EL_ID, samplePlayback: true, voice: 'brian' });
    expect(s.elevenVoiceId).toBe('nPczCjzI2devNBz1zQrb');
  });

  it.each([
    ['elevenlabs', '11labs-Adrian'], ['elevenlabs', 'x'], ['cartesia', EL_ID], ['minimax', 'minimax-Cimo'], ['kokoro', 'not_a_voice'], ['kokoro', '../etc'],
  ])('invalid saved voice for %s (%j): default kept, warning logged, no throw', (backend, bad) => {
    const s = mk({ ttsBackend: backend, voiceId: bad });
    expect(s.elevenVoiceId).toBe('DEFAULTELEVENVOICE0001');
    expect(s.cartesiaVoiceId).toBe(process.env.CARTESIA_VOICE_ID);
    expect(s.minimaxVoiceId).toBe('English_Default_Voice');
    expect(s.voice).toBe('af_heart');
    expect(s._versionVoice).toBeNull();
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('version voice_id ignored'));
  });

  it('non-string voiceId (object/number): ignored with a warning', () => {
    const s = mk({ ttsBackend: 'elevenlabs', voiceId: { a: 1 } });
    expect(s.elevenVoiceId).toBe('DEFAULTELEVENVOICE0001');
    const s2 = mk({ ttsBackend: 'elevenlabs', voiceId: 12 });
    expect(s2.elevenVoiceId).toBe('DEFAULTELEVENVOICE0001');
  });

  it('voice saved for another backend than the one in use (test override) is not applied', () => {
    // simulates the context after /test-tts-override: backend swapped, voice from the version's own backend
    const s = new CallSession(fakeWs());
    sessions.push(s);
    s.ttsBackend = 'cartesia';
    s._applyVersionVoice(EL_ID, 'elevenlabs', false);
    expect(s.cartesiaVoiceId).toBe(process.env.CARTESIA_VOICE_ID);
    expect(s._versionVoice).toBeNull();
  });
});

describe('session: language interplay', () => {
  const fakeWs = () => ({ readyState: WebSocket.OPEN, sent: [], send() {}, close() {} });
  const sessions = [];
  afterEach(() => { while (sessions.length) sessions.pop().close(); });
  function mk(backend, voiceId) {
    const s = new CallSession(fakeWs());
    sessions.push(s);
    s.onClientMessage(JSON.stringify({ type: 'context', systemPrompt: 'x', ttsBackend: backend, voiceId }), false);
    return s;
  }
  beforeEach(() => { vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network disabled in test')))); });
  afterEach(() => vi.unstubAllGlobals());

  it('elevenlabs + Spanish: the language only supplies the generic default voice, so the version voice stays', () => {
    const s = mk('elevenlabs', EL_ID);
    s._applyLanguage(resolveLanguage('es'));
    expect(s.ttsBackend).toBe('elevenlabs');
    expect(s.elevenVoiceId).toBe(EL_ID);
  });
  it('elevenlabs without a version voice + Spanish: language voice applies exactly as before', () => {
    const s = mk('elevenlabs', undefined);
    const lang = resolveLanguage('es');
    s._applyLanguage(lang);
    expect(s.elevenVoiceId).toBe(lang.tts.elevenVoiceId);
  });
  it('elevenlabs + Hungarian (cartesia-only language): backend moves, the language pinned cartesia voice applies', () => {
    const s = mk('elevenlabs', EL_ID);
    const lang = resolveLanguage('hu');
    s._applyLanguage(lang);
    expect(s.ttsBackend).toBe('cartesia');
    expect(s.cartesiaVoiceId).toBe(lang.tts.voiceId);
  });
  it('cartesia version voice + Hungarian: the language-pinned voice wins (current behaviour kept)', () => {
    const s = mk('cartesia', CARTESIA_ID);
    const lang = resolveLanguage('hu');
    s._applyLanguage(lang);
    expect(s.cartesiaVoiceId).toBe(lang.tts.voiceId);
  });
  it('kokoro version voice + Spanish: backend moves to elevenlabs and uses the language voice, kokoro name untouched', () => {
    const s = mk('kokoro', 'am_adam');
    const lang = resolveLanguage('es');
    s._applyLanguage(lang);
    expect(s.ttsBackend).toBe('elevenlabs');
    expect(s.elevenVoiceId).toBe(lang.tts.elevenVoiceId);
    expect(s.voice).toBe('am_adam');
  });
  it('the pre-language snapshot (restored on a mid-call switch back to English) holds the version voice, not the env default', () => {
    const s = mk('elevenlabs', EL_ID);
    s._applyLanguage(resolveLanguage('es'));
    expect(s._preLangState.elevenVoiceId).toBe(EL_ID);
    const c = mk('cartesia', CARTESIA_ID);
    expect(c._preLangState.cartesiaVoiceId).toBe(CARTESIA_ID);
  });
});
