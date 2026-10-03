// Validation of the voice id an agent version carries (calldesk_agent_versions.voice_id) before a call uses it.
//
// The builder stores voice_id as free text, and the same column holds Retell voice ids ("11labs-Adrian", "minimax-Cimo", a tenant
// voice uuid) for Retell-engine versions. A saved value that is not a real voice for this engine's backend would make every
// utterance of the call fail at the provider, so anything that does not look right is rejected and the caller keeps the default.
// Never throws.

import { validatePiperVoice } from './piperVoices.js';

// Kokoro voices the realtime-tts worker can load. The worker hard-codes KPipeline(lang_code='a'), so only the American English
// set (af_* / am_*) works; any other name raises inside the worker. Update this list when the worker's voice set changes.
export const KOKORO_VOICES = new Set([
  'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
  'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa',
]);

// Retell catalog ids are "<provider>-<Name>": never a valid voice id for a direct provider call.
const RETELL_STYLE = /^(11labs|elevenlabs|cartesia|openai|minimax|deepgram|playht|fish|retell|platform|custom)[-_]/i;
const ELEVENLABS_ID = /^[A-Za-z0-9]{15,32}$/; // ElevenLabs voice ids are 20 alphanumeric characters
const CARTESIA_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i; // Cartesia voice ids are uuids
const MINIMAX_ID = /^[A-Za-z0-9_-]{3,64}$/; // e.g. English_Graceful_Lady, or a cloned voice id

// Returns { ok: true, voiceId } with the normalized id, or { ok: false, reason }.
export function validateVersionVoice(backend, value) {
  if (typeof value !== 'string') return { ok: false, reason: 'not a string' };
  const id = value.trim();
  if (!id) return { ok: false, reason: 'empty' };
  if (id.length > 64) return { ok: false, reason: 'too long' };
  if (backend === 'kokoro') {
    return KOKORO_VOICES.has(id) ? { ok: true, voiceId: id } : { ok: false, reason: 'not a known kokoro voice' };
  }
  if (backend === 'elevenlabs' || backend === 'cartesia' || backend === 'minimax') {
    if (RETELL_STYLE.test(id)) return { ok: false, reason: 'looks like a Retell catalog voice id' };
    const re = backend === 'elevenlabs' ? ELEVENLABS_ID : backend === 'cartesia' ? CARTESIA_ID : MINIMAX_ID;
    return re.test(id) ? { ok: true, voiceId: id } : { ok: false, reason: `not a valid ${backend} voice id` };
  }
  if (backend === 'piper') return validatePiperVoice(id);
  return { ok: false, reason: `per-version voice is not supported for backend ${backend}` };
}
