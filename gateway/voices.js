/**
 * Static catalog of built-in Kokoro voices (worker/models/voices-v1.0.bin).
 * These are always available and do not require training or payment to use.
 * Voice IDs match the Kokoro naming exactly so synthesize calls pass through
 * unchanged to the worker.
 */

/**
 * @typedef {Object} Voice
 * @property {string} voice_id
 * @property {string} name
 * @property {string} category - "premade" (ElevenLabs parity)
 * @property {string} language
 * @property {string} gender
 * @property {string} [accent]
 * @property {string} [description]
 */

/** @type {Voice[]} */
const VOICES = [
  // American English — Female
  { voice_id: "af_heart", name: "Heart", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_bella", name: "Bella", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_nicole", name: "Nicole", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_sky", name: "Sky", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_river", name: "River", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_alloy", name: "Alloy", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_aoede", name: "Aoede", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_jessica", name: "Jessica", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_kore", name: "Kore", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_nova", name: "Nova", category: "premade", language: "en", gender: "female", accent: "American" },
  { voice_id: "af_sarah", name: "Sarah", category: "premade", language: "en", gender: "female", accent: "American" },
  // American English — Male
  { voice_id: "am_adam", name: "Adam", category: "premade", language: "en", gender: "male", accent: "American" },
  { voice_id: "am_echo", name: "Echo", category: "premade", language: "en", gender: "male", accent: "American" },
  { voice_id: "am_eric", name: "Eric", category: "premade", language: "en", gender: "male", accent: "American" },
  { voice_id: "am_fenrir", name: "Fenrir", category: "premade", language: "en", gender: "male", accent: "American" },
  { voice_id: "am_liam", name: "Liam", category: "premade", language: "en", gender: "male", accent: "American" },
  { voice_id: "am_michael", name: "Michael", category: "premade", language: "en", gender: "male", accent: "American" },
  { voice_id: "am_onyx", name: "Onyx", category: "premade", language: "en", gender: "male", accent: "American" },
  { voice_id: "am_puck", name: "Puck", category: "premade", language: "en", gender: "male", accent: "American" },
  { voice_id: "am_santa", name: "Santa", category: "premade", language: "en", gender: "male", accent: "American" },
  // British English — Female
  { voice_id: "bf_alice", name: "Alice", category: "premade", language: "en", gender: "female", accent: "British" },
  { voice_id: "bf_emma", name: "Emma", category: "premade", language: "en", gender: "female", accent: "British" },
  { voice_id: "bf_isabella", name: "Isabella", category: "premade", language: "en", gender: "female", accent: "British" },
  { voice_id: "bf_lily", name: "Lily", category: "premade", language: "en", gender: "female", accent: "British" },
  // British English — Male
  { voice_id: "bm_daniel", name: "Daniel", category: "premade", language: "en", gender: "male", accent: "British" },
  { voice_id: "bm_fable", name: "Fable", category: "premade", language: "en", gender: "male", accent: "British" },
  { voice_id: "bm_george", name: "George", category: "premade", language: "en", gender: "male", accent: "British" },
  { voice_id: "bm_lewis", name: "Lewis", category: "premade", language: "en", gender: "male", accent: "British" },
  // French
  { voice_id: "ff_siwis", name: "Siwis", category: "premade", language: "fr", gender: "female", accent: "French" },
  // Greek
  { voice_id: "hf_alpha", name: "Alpha", category: "premade", language: "el", gender: "female", accent: "Greek" },
  { voice_id: "hf_beta", name: "Beta", category: "premade", language: "el", gender: "female", accent: "Greek" },
  { voice_id: "hm_omega", name: "Omega", category: "premade", language: "el", gender: "male", accent: "Greek" },
  { voice_id: "hm_psi", name: "Psi", category: "premade", language: "el", gender: "male", accent: "Greek" },
  // Italian
  { voice_id: "if_sara", name: "Sara", category: "premade", language: "it", gender: "female", accent: "Italian" },
  { voice_id: "im_nicola", name: "Nicola", category: "premade", language: "it", gender: "male", accent: "Italian" },
  // Japanese
  { voice_id: "jf_alpha", name: "Alpha", category: "premade", language: "ja", gender: "female", accent: "Japanese" },
  { voice_id: "jf_gongitsune", name: "Gongitsune", category: "premade", language: "ja", gender: "female", accent: "Japanese" },
  { voice_id: "jf_nezumi", name: "Nezumi", category: "premade", language: "ja", gender: "female", accent: "Japanese" },
  { voice_id: "jf_tebukuro", name: "Tebukuro", category: "premade", language: "ja", gender: "female", accent: "Japanese" },
  { voice_id: "jm_kumo", name: "Kumo", category: "premade", language: "ja", gender: "male", accent: "Japanese" },
  // Portuguese
  { voice_id: "pf_dora", name: "Dora", category: "premade", language: "pt", gender: "female", accent: "Portuguese" },
  { voice_id: "pm_alex", name: "Alex", category: "premade", language: "pt", gender: "male", accent: "Portuguese" },
  { voice_id: "pm_santa", name: "Santa", category: "premade", language: "pt", gender: "male", accent: "Portuguese" },
  // Chinese
  { voice_id: "zf_xiaobei", name: "Xiaobei", category: "premade", language: "zh", gender: "female", accent: "Mandarin" },
  { voice_id: "zf_xiaoni", name: "Xiaoni", category: "premade", language: "zh", gender: "female", accent: "Mandarin" },
  { voice_id: "zf_xiaoxiao", name: "Xiaoxiao", category: "premade", language: "zh", gender: "female", accent: "Mandarin" },
  { voice_id: "zf_xiaoyi", name: "Xiaoyi", category: "premade", language: "zh", gender: "female", accent: "Mandarin" },
  { voice_id: "zm_yunjian", name: "Yunjian", category: "premade", language: "zh", gender: "male", accent: "Mandarin" },
  { voice_id: "zm_yunxi", name: "Yunxi", category: "premade", language: "zh", gender: "male", accent: "Mandarin" },
  { voice_id: "zm_yunxia", name: "Yunxia", category: "premade", language: "zh", gender: "male", accent: "Mandarin" },
  { voice_id: "zm_yunyang", name: "Yunyang", category: "premade", language: "zh", gender: "male", accent: "Mandarin" },
  // Other English variants
  { voice_id: "ef_dora", name: "Dora", category: "premade", language: "en", gender: "female" },
  { voice_id: "em_alex", name: "Alex", category: "premade", language: "en", gender: "male" },
  { voice_id: "em_santa", name: "Santa", category: "premade", language: "en", gender: "male" },
];

const BY_ID = new Map(VOICES.map((v) => [v.voice_id, v]));
const VOICE_ID_RE = /^v-[0-9a-f]{10}$/;

export function isBuiltInVoice(voiceId) {
  return BY_ID.has(voiceId);
}

export function isCustomVoice(voiceId) {
  return VOICE_ID_RE.test(voiceId);
}

export function getBuiltInVoice(voiceId) {
  return BY_ID.get(voiceId) || null;
}

export function listBuiltInVoices() {
  return VOICES.slice();
}

/** Fast lookup without allocation (used in hot paths like /v1/text-to-speech validate). */
export function voiceExists(voiceId) {
  return BY_ID.has(voiceId) || VOICE_ID_RE.test(voiceId);
}
