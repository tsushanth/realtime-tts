// Jingle / sound-effect playback (see docs/superpowers/specs/2026-09-29-call-audio-jingle-sfx-design.md).
// Covers the pure pieces: kill switch, context parsing, tool construction, and dispatch — the
// server.js wiring (jingle-on-connect, tool handler) is covered separately in callAudioSession.test.js.

import { describe, it, expect } from 'vitest';
import {
  isCallAudioEnabled,
  parseCallAudioContext,
  buildPlaySoundEffectTool,
  pickSoundEffect,
  encodeCallAudioForContext,
  SOUND_EFFECT_TOOL_NAME,
} from '../callAudio.js';

const b64 = (bytes) => Buffer.from(bytes).toString('base64');

const RAW = {
  jingle: { name: 'intro', audio: b64([1, 2, 3, 4]) },
  effects: [
    { name: 'booking_confirmed_chime', description: 'Play right after an appointment is successfully booked.', audio: b64([9, 9]) },
    { name: 'error_buzz', description: 'Play when a request cannot be fulfilled.', audio: b64([7]) },
  ],
};

describe('isCallAudioEnabled (global kill switch)', () => {
  it('defaults to enabled when unset', () => {
    expect(isCallAudioEnabled({})).toBe(true);
  });
  it('is disabled by "false" / "0" (case-insensitive), enabled by anything else', () => {
    expect(isCallAudioEnabled({ CALL_AUDIO_ASSETS_ENABLED: 'false' })).toBe(false);
    expect(isCallAudioEnabled({ CALL_AUDIO_ASSETS_ENABLED: 'FALSE' })).toBe(false);
    expect(isCallAudioEnabled({ CALL_AUDIO_ASSETS_ENABLED: '0' })).toBe(false);
    expect(isCallAudioEnabled({ CALL_AUDIO_ASSETS_ENABLED: 'true' })).toBe(true);
    expect(isCallAudioEnabled({ CALL_AUDIO_ASSETS_ENABLED: '1' })).toBe(true);
  });
});

describe('parseCallAudioContext', () => {
  it('decodes jingle and effects into Buffers', () => {
    const ca = parseCallAudioContext(RAW, { CALL_AUDIO_ASSETS_ENABLED: 'true' });
    expect(Buffer.isBuffer(ca.jingle)).toBe(true);
    expect([...ca.jingle]).toEqual([1, 2, 3, 4]);
    expect(ca.effects.size).toBe(2);
    expect([...ca.effects.get('booking_confirmed_chime').audio]).toEqual([9, 9]);
    expect(ca.effects.get('error_buzz').description).toMatch(/cannot be fulfilled/);
  });
  it('returns null when the global flag is off, regardless of configured assets', () => {
    expect(parseCallAudioContext(RAW, { CALL_AUDIO_ASSETS_ENABLED: 'false' })).toBeNull();
  });
  it('returns null for missing / empty / malformed input (tenant with no assets = zero behavior change)', () => {
    expect(parseCallAudioContext(undefined, {})).toBeNull();
    expect(parseCallAudioContext(null, {})).toBeNull();
    expect(parseCallAudioContext('nope', {})).toBeNull();
    expect(parseCallAudioContext({ jingle: null, effects: [] }, {})).toBeNull();
  });
  it('drops effect entries with a bad name, empty audio, or non-string fields, keeping the valid ones', () => {
    const ca = parseCallAudioContext(
      {
        jingle: null,
        effects: [
          { name: 'ok', description: 'fine', audio: b64([1]) },
          { name: '', description: 'no name', audio: b64([1]) },
          { name: 'has space', description: 'bad slug', audio: b64([1]) },
          { name: 'empty_audio', description: 'x', audio: '' },
          { name: 'no_audio', description: 'x' },
          null,
        ],
      },
      {}
    );
    expect([...ca.effects.keys()]).toEqual(['ok']);
    expect(ca.jingle).toBeNull();
  });
  it('keeps a jingle-only config (no effects)', () => {
    const ca = parseCallAudioContext({ jingle: RAW.jingle, effects: [] }, {});
    expect(ca.jingle).not.toBeNull();
    expect(ca.effects.size).toBe(0);
  });
});

describe('buildPlaySoundEffectTool', () => {
  it('is null when there are no effects (jingle-only or no config)', () => {
    expect(buildPlaySoundEffectTool(null)).toBeNull();
    expect(buildPlaySoundEffectTool(parseCallAudioContext({ jingle: RAW.jingle, effects: [] }, {}))).toBeNull();
  });
  it('has one enum value per effect and surfaces each description to the model', () => {
    const tool = buildPlaySoundEffectTool(parseCallAudioContext(RAW, {}));
    expect(tool.name).toBe(SOUND_EFFECT_TOOL_NAME);
    expect(tool.name).toBe('play_sound_effect');
    expect(tool.input_schema.properties.name.enum).toEqual(['booking_confirmed_chime', 'error_buzz']);
    expect(tool.input_schema.required).toEqual(['name']);
    expect(tool.input_schema.properties.name.description).toContain('booking_confirmed_chime: Play right after an appointment is successfully booked.');
    expect(tool.input_schema.properties.name.description).toContain('error_buzz: Play when a request cannot be fulfilled.');
  });
  it('tells the model it only ADDS a sound and must still speak', () => {
    const tool = buildPlaySoundEffectTool(parseCallAudioContext(RAW, {}));
    expect(tool.description).toMatch(/still (say|speak)/i);
  });
});

describe('pickSoundEffect', () => {
  const ca = parseCallAudioContext(RAW, {});
  it('returns the cached buffer for a known name', () => {
    expect([...pickSoundEffect(ca, 'error_buzz')]).toEqual([7]);
  });
  it('returns null for an unknown name (model hallucinated an enum value) or no config', () => {
    expect(pickSoundEffect(ca, 'nonexistent')).toBeNull();
    expect(pickSoundEffect(null, 'error_buzz')).toBeNull();
  });
});

describe('encodeCallAudioForContext (tenantLookup -> context message)', () => {
  it('splits DB rows into one jingle + effects, base64-encoding audio, skipping disabled/blank rows', () => {
    const out = encodeCallAudioForContext([
      { asset_type: 'jingle', name: 'intro', description: '', enabled: true, audio: Buffer.from([1, 2]) },
      { asset_type: 'sound_effect', name: 'chime', description: 'd', enabled: true, audio: Buffer.from([3]) },
      { asset_type: 'sound_effect', name: 'off', description: 'd', enabled: false, audio: Buffer.from([4]) },
      { asset_type: 'sound_effect', name: 'blank', description: 'd', enabled: true, audio: Buffer.alloc(0) },
      { asset_type: 'sound_effect', name: 'missing', description: 'd', enabled: true, audio: null },
    ]);
    expect(out.jingle).toEqual({ name: 'intro', audio: Buffer.from([1, 2]).toString('base64') });
    expect(out.effects).toEqual([{ name: 'chime', description: 'd', audio: Buffer.from([3]).toString('base64') }]);
  });
  it('enforces at most one jingle (first enabled wins) — the spec\'s app-layer constraint, defensively', () => {
    const out = encodeCallAudioForContext([
      { asset_type: 'jingle', name: 'a', enabled: true, audio: Buffer.from([1]) },
      { asset_type: 'jingle', name: 'b', enabled: true, audio: Buffer.from([2]) },
    ]);
    expect(out.jingle.name).toBe('a');
  });
  it('returns undefined when nothing is usable, so the context message stays unchanged for a tenant with no assets', () => {
    expect(encodeCallAudioForContext([])).toBeUndefined();
    expect(encodeCallAudioForContext(null)).toBeUndefined();
    expect(encodeCallAudioForContext([{ asset_type: 'sound_effect', name: 'x', enabled: false, audio: Buffer.from([1]) }])).toBeUndefined();
  });
});
