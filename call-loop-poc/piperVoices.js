// Voice ids the owned Piper service (Fly app piper-tts-sjc, worker-piper-fly/server.py) accepts, and the validation for an agent
// version's voice_id when its backend is 'piper'. Self-contained so voiceSelection.js (and any later voice-id work) can reuse it.
//
// The service resolves `voice` as: anything not prefixed `custom:` -> the service's built-in `default` voice; `custom:<id>` -> a
// published voice stored as <id>/model.onnx. Only ids known to be published are accepted here, so a typo can never reach the service.
//
// Licence: internal-docs cost-lab/tts-gates/LICENCE.md clears only `custom:en-us-ljspeech` (public-domain data, trained from scratch)
// as the safest voice for a paid tier. `custom:en-us-john` is clean on paper but not counsel-cleared. `default` is NOT to be sold
// (Lessac research-only lineage plus Polly-generated training audio). `default` is still accepted for internal testing.
export const PIPER_DEFAULT_VOICE = 'custom:en-us-ljspeech';
export const PIPER_VOICES = new Set(['default', 'custom:en-us-john', 'custom:en-us-ljspeech']);

// Returns { ok: true, voiceId } or { ok: false, reason }. Never throws.
export function validatePiperVoice(value) {
  if (typeof value !== 'string') return { ok: false, reason: 'not a string' };
  const id = value.trim();
  if (!id) return { ok: false, reason: 'empty' };
  return PIPER_VOICES.has(id) ? { ok: true, voiceId: id } : { ok: false, reason: 'not a known piper voice' };
}
