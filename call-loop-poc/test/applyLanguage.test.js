// Tests CallSession.prototype._applyLanguage (server.js) via
// `CallSession.prototype._applyLanguage.call(stub, lang)` on a minimal stub — chosen over
// extracting a pure function because _applyLanguage reads several module-level constants
// (ELEVENLABS_API_KEY, CARTESIA_API_KEY, CARTESIA_VOICE_ID, FISH_AUDIO_API_KEY) that are computed
// once from process.env at import time, plus calls the module-private prewarmLangFillers() as a
// side effect. Extracting the decision logic cleanly would mean threading those constants through
// as parameters, which risks diverging from the real method; calling the real prototype method on
// a stub exercises the exact shipped code, unmodified.
//
// Because those constants are frozen at import time, each key-presence scenario below imports
// server.js fresh (a distinct `?scenario=` query string busts Node/Vitest's module cache) with
// process.env set beforehand for that scenario.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { resolveLanguage } from '../languages.js';

const ENV_KEYS = ['ELEVENLABS_API_KEY', 'CARTESIA_API_KEY', 'CARTESIA_VOICE_ID', 'FISH_AUDIO_API_KEY'];
const savedEnv = {};

beforeEach(() => {
  for (const k of ENV_KEYS) savedEnv[k] = process.env[k];
});

afterEach(() => {
  for (const k of ENV_KEYS) {
    if (savedEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedEnv[k];
  }
  vi.unstubAllGlobals();
});

// Minimal stub carrying only the fields _applyLanguage actually reads or writes (verified against
// the real implementation, server.js's CallSession constructor + _applyLanguage body):
// ttsBackend, cost.ttsBackend, elevenVoiceId, ttsModel, cartesiaVoiceId, fishReferenceId,
// backchannelWords.
function makeStub(ttsBackend) {
  return {
    ttsBackend,
    cost: { ttsBackend },
    elevenVoiceId: 'stub-eleven-voice',
    ttsModel: null,
    cartesiaVoiceId: 'stub-cartesia-voice',
    fishReferenceId: null,
    backchannelWords: null,
  };
}

describe('_applyLanguage routing (today\'s regression scenarios)', () => {
  it('elevenlabs default + cartesia-required lang (hu) + cartesia keys present => switches to cartesia', async () => {
    delete process.env.ELEVENLABS_API_KEY; // keep prewarmLangFillers a no-op (elevenlabs key gates it)
    process.env.CARTESIA_API_KEY = 'test-cartesia-key';
    process.env.CARTESIA_VOICE_ID = 'test-cartesia-voice';
    const { CallSession } = await import('../server.js?scenario=cartesia-keys-present');

    const stub = makeStub('elevenlabs');
    const lang = resolveLanguage('hu');
    expect(lang.tts.backend).toBe('cartesia'); // sanity on the real languages.js data

    CallSession.prototype._applyLanguage.call(stub, lang);

    expect(stub.ttsBackend).toBe('cartesia');
    expect(stub.cost.ttsBackend).toBe('cartesia');
    expect(stub.cartesiaVoiceId).toBe(lang.tts.voiceId);
    expect(stub.backchannelWords).toBe(lang.say.backchannel);
  });

  it('elevenlabs default + fish-required lang (ca) + fish key present => switches to fish', async () => {
    delete process.env.ELEVENLABS_API_KEY;
    process.env.FISH_AUDIO_API_KEY = 'test-fish-key';
    const { CallSession } = await import('../server.js?scenario=fish-key-present');

    const stub = makeStub('elevenlabs');
    const lang = resolveLanguage('ca');
    expect(lang.tts.backend).toBe('fish');

    CallSession.prototype._applyLanguage.call(stub, lang);

    expect(stub.ttsBackend).toBe('fish');
    expect(stub.cost.ttsBackend).toBe('fish');
    expect(stub.fishReferenceId).toBe(lang.tts.referenceId);
    expect(stub.backchannelWords).toBe(lang.say.backchannel);
  });

  it('kokoro default + elevenlabs-required lang (es) + ELEVENLABS_API_KEY set => switches to elevenlabs', async () => {
    process.env.ELEVENLABS_API_KEY = 'test-eleven-key';
    delete process.env.CARTESIA_API_KEY;
    delete process.env.FISH_AUDIO_API_KEY;
    const { CallSession } = await import('../server.js?scenario=eleven-key-present-kokoro');
    // Avoid a real network call from prewarmLangFillers, which fires once ttsBackend becomes
    // 'elevenlabs' and ELEVENLABS_API_KEY is set — stub fetch so any such call fails fast locally.
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network disabled in test'))));

    const stub = makeStub('kokoro');
    const lang = resolveLanguage('es');
    expect(lang.tts.backend).toBe('elevenlabs');

    CallSession.prototype._applyLanguage.call(stub, lang);

    expect(stub.ttsBackend).toBe('elevenlabs');
    expect(stub.cost.ttsBackend).toBe('elevenlabs');
    expect(stub.elevenVoiceId).toBe(lang.tts.elevenVoiceId);
    expect(stub.backchannelWords).toBe(lang.say.backchannel);
  });

  it('minimax default + elevenlabs-required lang (es) + ELEVENLABS_API_KEY set => switches to elevenlabs', async () => {
    process.env.ELEVENLABS_API_KEY = 'test-eleven-key';
    const { CallSession } = await import('../server.js?scenario=eleven-key-present-minimax');
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network disabled in test'))));

    const stub = makeStub('minimax');
    const lang = resolveLanguage('es');

    CallSession.prototype._applyLanguage.call(stub, lang);

    expect(stub.ttsBackend).toBe('elevenlabs');
    expect(stub.cost.ttsBackend).toBe('elevenlabs');
  });

  it('kokoro/minimax + elevenlabs-required lang but ELEVENLABS_API_KEY unset => stays on original backend (today\'s exact shipped bug guard)', async () => {
    delete process.env.ELEVENLABS_API_KEY;
    const { CallSession } = await import('../server.js?scenario=eleven-key-absent');

    const stub = makeStub('kokoro');
    const lang = resolveLanguage('es');

    CallSession.prototype._applyLanguage.call(stub, lang);

    // No ElevenLabs key -> the redirect must NOT fire; this is exactly the guard whose flaw
    // caused the real 404 that shipped today, so it must stay provably intact.
    expect(stub.ttsBackend).toBe('kokoro');
    expect(stub.cost.ttsBackend).toBe('kokoro');
  });

  it('elevenlabs default + cartesia-required lang but cartesia keys unset => stays on elevenlabs', async () => {
    delete process.env.ELEVENLABS_API_KEY;
    delete process.env.CARTESIA_API_KEY;
    delete process.env.CARTESIA_VOICE_ID;
    const { CallSession } = await import('../server.js?scenario=cartesia-keys-absent');

    const stub = makeStub('elevenlabs');
    const lang = resolveLanguage('hu');

    CallSession.prototype._applyLanguage.call(stub, lang);

    expect(stub.ttsBackend).toBe('elevenlabs');
    expect(stub.cost.ttsBackend).toBe('elevenlabs');
    // Falls through to the elevenlabs branch at the bottom of _applyLanguage since ttsBackend is
    // still 'elevenlabs'.
    expect(stub.elevenVoiceId).toBe(lang.tts.elevenVoiceId);
  });
});
