import { describe, it, expect } from 'vitest';
import { resolveLanguage, detectSpokenLanguage, LANGS } from '../languages.js';

describe('resolveLanguage', () => {
  it('returns null for English / unset / empty', () => {
    expect(resolveLanguage('en')).toBeNull();
    expect(resolveLanguage('en-US')).toBeNull();
    expect(resolveLanguage('EN')).toBeNull();
    expect(resolveLanguage(undefined)).toBeNull();
    expect(resolveLanguage(null)).toBeNull();
    expect(resolveLanguage('')).toBeNull();
    expect(resolveLanguage('   ')).toBeNull();
  });

  it('returns null for an unsupported code', () => {
    expect(resolveLanguage('xx')).toBeNull();
    expect(resolveLanguage('klingon')).toBeNull();
  });

  it('resolves an ElevenLabs-backed code (es) with the default ElevenLabs backend', () => {
    const lang = resolveLanguage('es');
    expect(lang).not.toBeNull();
    expect(lang.code).toBe('es');
    expect(lang.tts.backend).toBe('elevenlabs');
    expect(typeof lang.tts.elevenVoiceId).toBe('string');
    expect(lang.tts.elevenVoiceId.length).toBeGreaterThan(0);
  });

  it('resolves a Cartesia-backed code (hu)', () => {
    const lang = resolveLanguage('hu');
    expect(lang).not.toBeNull();
    expect(lang.code).toBe('hu');
    expect(lang.tts.backend).toBe('cartesia');
    expect(typeof lang.tts.voiceId).toBe('string');
    expect(lang.tts.voiceId.length).toBeGreaterThan(0);
  });

  it('resolves a Fish-backed code (ca)', () => {
    const lang = resolveLanguage('ca');
    expect(lang).not.toBeNull();
    expect(lang.code).toBe('ca');
    expect(lang.tts.backend).toBe('fish');
    expect(typeof lang.tts.referenceId).toBe('string');
    expect(lang.tts.referenceId.length).toBeGreaterThan(0);
  });

  it('resolves via ALIASES (pt -> pt-BR)', () => {
    const lang = resolveLanguage('pt');
    expect(lang).not.toBeNull();
    expect(lang.code).toBe('pt-BR');
  });

  it('resolves a region-tagged code (es-MX) via the base-language fallback', () => {
    // resolveLanguage: when the exact code and ALIASES both miss, it strips off the
    // region subtag (split on -/_ , lowercased) and checks LANGS for that base code.
    const lang = resolveLanguage('es-MX');
    expect(lang).not.toBeNull();
    expect(lang.code).toBe('es');
  });
});

describe('LANGS entries (loop over all, no hand-written per-language cases)', () => {
  const codes = Object.keys(LANGS);

  it('has more than one language defined (sanity)', () => {
    expect(codes.length).toBeGreaterThan(30);
  });

  for (const code of codes) {
    describe(`LANGS['${code}']`, () => {
      const entry = LANGS[code];

      it('has a non-empty name', () => {
        expect(typeof entry.name).toBe('string');
        expect(entry.name.length).toBeGreaterThan(0);
      });

      it('has a valid dg config', () => {
        expect(['flux', 'nova3']).toContain(entry.dg.kind);
        if (entry.dg.kind === 'flux') {
          expect(typeof entry.dg.hint).toBe('string');
          expect(entry.dg.hint.length).toBeGreaterThan(0);
          expect(entry.dg.code).toBeUndefined();
        } else {
          expect(typeof entry.dg.code).toBe('string');
          expect(entry.dg.code.length).toBeGreaterThan(0);
          expect(entry.dg.hint).toBeUndefined();
        }
      });

      it('has a say object with all 5 required phrase fields non-empty', () => {
        const { say } = entry;
        expect(typeof say.warmup).toBe('string');
        expect(say.warmup.length).toBeGreaterThan(0);
        expect(typeof say.calendar).toBe('string');
        expect(say.calendar.length).toBeGreaterThan(0);
        expect(typeof say.goodbye).toBe('string');
        expect(say.goodbye.length).toBeGreaterThan(0);
        expect(typeof say.transfer).toBe('string');
        expect(say.transfer.length).toBeGreaterThan(0);
        expect(Array.isArray(say.backchannel)).toBe(true);
        expect(say.backchannel.length).toBeGreaterThan(0);
        for (const phrase of say.backchannel) {
          expect(typeof phrase).toBe('string');
          expect(phrase.length).toBeGreaterThan(0);
        }
      });

      it('has real RegExp phoneAskRe/closingRe that do not throw against plain text', () => {
        expect(entry.phoneAskRe).toBeInstanceOf(RegExp);
        expect(entry.closingRe).toBeInstanceOf(RegExp);
        expect(() => entry.phoneAskRe.test('just a plain sentence with no special content')).not.toThrow();
        expect(() => entry.closingRe.test('just a plain sentence with no special content')).not.toThrow();
      });
    });
  }
});

describe('detectSpokenLanguage', () => {
  it('detects a language from its real DETECT_MARKERS text', () => {
    // Spanish markers include 'gracias' and the ñ/á/é/í/ó/ú/ü/¿/¡ character class.
    expect(detectSpokenLanguage('Muchas gracias, hasta luego', ['en', 'es'])).toBe('es');
  });

  it('detects English from its own markers', () => {
    expect(detectSpokenLanguage('yes please, thanks a lot', ['en', 'es'])).toBe('en');
  });

  it('returns null for no match', () => {
    expect(detectSpokenLanguage('xyz qwerty zzz', ['en', 'es'])).toBeNull();
  });

  it('returns null for invalid input (non-string text, empty/missing allowedCodes)', () => {
    expect(detectSpokenLanguage(123, ['en'])).toBeNull();
    expect(detectSpokenLanguage('gracias', [])).toBeNull();
    expect(detectSpokenLanguage('gracias', undefined)).toBeNull();
    expect(detectSpokenLanguage('hi', ['en'])).toBeNull(); // too short (<4 chars)
  });

  it('respects allowedCodes restriction — a language not in allowedCodes is never returned', () => {
    // Text has clear Spanish markers, but 'es' isn't in allowedCodes, so it must not be returned.
    // No other allowed code's markers should hit this text, so the result is null.
    expect(detectSpokenLanguage('Muchas gracias, hasta luego', ['en', 'fr'])).toBeNull();
  });
});
