// Jingle / sound-effect playback for live calls — see
// docs/superpowers/specs/2026-09-29-call-audio-jingle-sfx-design.md.
//
// Assets are generated and converted to mu-law@8kHz OFFLINE (dashboard-triggered), never during a
// call. This module only handles the runtime half: the global kill switch, turning the per-tenant
// asset payload into playable buffers, and building/dispatching the `play_sound_effect` tool. It's
// deliberately free of any server.js state so it can be unit-tested in isolation.

export const SOUND_EFFECT_TOOL_NAME = 'play_sound_effect';

// Effect names double as tool enum values, so they must be simple slugs.
const NAME_RE = /^[A-Za-z0-9_-]{1,64}$/;

// Emergency, org-wide off switch layered ON TOP of per-tenant asset config: default on, only an
// explicit "false"/"0" disables it. When off, no jingle plays and the tool is never offered,
// regardless of what assets exist.
export function isCallAudioEnabled(env = process.env) {
  const v = String(env.CALL_AUDIO_ASSETS_ENABLED ?? 'true').trim().toLowerCase();
  return v !== 'false' && v !== '0';
}

// Narrower switch for sound effects only, layered under the global one. Effects put a tool in front of
// the LLM on every turn (so they can change what it says); the jingle never touches the LLM. That
// difference is why an operator may want to disable effects without silencing every tenant's jingle.
// The global switch still wins: if it's off, effects are off too.
export function isSoundEffectsEnabled(env = process.env) {
  if (!isCallAudioEnabled(env)) return false;
  const v = String(env.CALL_AUDIO_EFFECTS_ENABLED ?? 'true').trim().toLowerCase();
  return v !== 'false' && v !== '0';
}

function decodeAudio(b64) {
  if (typeof b64 !== 'string' || b64.length === 0) return null;
  const buf = Buffer.from(b64, 'base64');
  return buf.length > 0 ? buf : null;
}

// Context-message payload ({ jingle?: {name, audio}, effects?: [{name, description, audio}] },
// audio = base64 mu-law@8kHz) -> { jingle: Buffer|null, effects: Map<name, {description, audio}> },
// or null when the feature is off / nothing usable is configured (so callers can treat "no
// config" and "disabled" identically: zero behavior change).
export function parseCallAudioContext(raw, env = process.env) {
  if (!isCallAudioEnabled(env)) return null;
  if (!raw || typeof raw !== 'object') return null;

  const jingle = raw.jingle && typeof raw.jingle === 'object' ? decodeAudio(raw.jingle.audio) : null;

  const effects = new Map();
  if (Array.isArray(raw.effects) && isSoundEffectsEnabled(env)) {
    for (const e of raw.effects) {
      if (!e || typeof e !== 'object' || typeof e.name !== 'string' || !NAME_RE.test(e.name)) continue;
      const audio = decodeAudio(e.audio);
      if (!audio) continue;
      effects.set(e.name, { description: typeof e.description === 'string' ? e.description : '', audio });
    }
  }

  if (!jingle && effects.size === 0) return null;
  return { jingle, effects };
}

// One enum value per effect name. A JSON-schema enum can't carry per-value descriptions, so each
// asset's description is listed alongside its name in the parameter description — that's what the
// model reads to decide when to use it, same as it does for transition_flow's edge conditions.
export function buildPlaySoundEffectTool(callAudio) {
  if (!callAudio || callAudio.effects.size === 0) return null;
  const names = [...callAudio.effects.keys()];
  const lines = names.map((n) => `- ${n}: ${callAudio.effects.get(n).description || '(no description)'}`);
  return {
    name: SOUND_EFFECT_TOOL_NAME,
    description:
      'Plays a short sound effect to the caller. Use it ONLY at the moment one of the listed situations actually ' +
      'happens, never just to fill time. It only ADDS a sound — you must still say your spoken reply in the same turn.',
    input_schema: {
      type: 'object',
      properties: {
        name: { type: 'string', enum: names, description: `Which sound to play:\n${lines.join('\n')}` },
      },
      required: ['name'],
    },
  };
}

export function pickSoundEffect(callAudio, name) {
  return callAudio?.effects.get(name)?.audio ?? null;
}

// tenantLookup rows (audio already downloaded to a Buffer) -> the JSON-safe context payload
// parseCallAudioContext consumes. Buffers can't ride through the JSON context message directly
// (they'd serialize as a byte-array of numbers), hence base64.
// Returns undefined when nothing is usable so the context message is unchanged for a tenant with
// no assets.
export function encodeCallAudioForContext(rows) {
  if (!Array.isArray(rows)) return undefined;
  let jingle = null;
  const effects = [];
  for (const r of rows) {
    if (!r || r.enabled === false || !r.audio || r.audio.length === 0) continue;
    if (r.asset_type === 'jingle') {
      if (!jingle) jingle = { name: r.name, audio: Buffer.from(r.audio).toString('base64') };
    } else if (r.asset_type === 'sound_effect') {
      effects.push({ name: r.name, description: r.description || '', audio: Buffer.from(r.audio).toString('base64') });
    }
  }
  if (!jingle && effects.length === 0) return undefined;
  return { jingle, effects };
}
