// Retell-style low-latency call loop POC, built on top of this repo's own
// realtime-tts gateway (see ../README.md for that leg's protocol/latency
// numbers). New pieces added here: streaming STT (Deepgram Flux — native
// semantic turn detection, not acoustic-silence VAD), streaming LLM
// (Anthropic), sentence-boundary chunking to pipeline LLM->TTS, and
// barge-in (kills in-flight TTS the moment the caller starts talking again).
//
// Browser <-WS-> this server <-WS-> Deepgram Flux (STT + turn detection)
//                            <-HTTP streaming-> Anthropic (LLM)
//                            <-WS-> realtime-tts gateway (TTS)
//
// Run: DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... npm start
import 'dotenv/config';
import express from 'express';
import http from 'node:http';
import { WebSocketServer, WebSocket } from 'ws';
import { randomUUID } from 'node:crypto';
import Anthropic from '@anthropic-ai/sdk';
import { SentenceChunker } from './sentenceChunker.js';
import { TwilioCallAdapter } from './twilioAdapter.js';
import { CallCostTracker } from './costTracker.js';
import { reportCallUsage } from './stripeMeter.js';
import { resolveInboundCall, fetchKnowledgeItems } from './tenantLookup.js';

const PORT = process.env.PORT || 8090;
// The Deepgram+Claude+TTS pipeline — the only engine this repo runs now.
// (A speech-to-speech alternative on OpenAI's Realtime API used to live
// here behind VOICE_ENGINE=s2s; removed 2026-08-28 — no flow/billing
// support, and the OpenAI key was pulled after disabling that billing
// account, so it could no longer run anyway. Not worth maintaining a dead
// code path.)
const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const TTS_GATEWAY_WS_URL = process.env.TTS_GATEWAY_WS_URL || 'ws://127.0.0.1:8080/tts';
const TTS_GATEWAY_API_KEY = process.env.TTS_GATEWAY_API_KEY;
const TTS_VOICE = process.env.TTS_VOICE || 'af_heart';
// 'kokoro' (default) is our own realtime-tts gateway (CPU or GPU backend,
// unaffected by this flag). 'elevenlabs', 'cartesia', and 'minimax' are
// alternate backends for the cascaded engine's TTS leg — each calls that
// provider's HTTP API directly with its own key (real per-character/per-
// request cost) and buffers the full response before handing it to the
// resample pipeline (see _speakElevenLabs for why: resampling is a stateful
// decimation that can't restart per-chunk). This is only the process-wide
// default — a per-session {"type":"context"} message with a `ttsBackend`
// field overrides it per call (see CallSession.onClientMessage), so a
// multi-tenant caller can pick a different backend per tenant without a
// restart.
const VALID_TTS_BACKENDS = ['kokoro', 'elevenlabs', 'cartesia', 'minimax'];
const TTS_BACKEND = VALID_TTS_BACKENDS.includes(process.env.TTS_BACKEND) ? process.env.TTS_BACKEND : 'kokoro';
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'JBFqnCBsd6RMkjVDRZzb';
// eleven_flash_v2_5 (the previous hardcoded default) is ElevenLabs' fastest
// AND lowest-quality tier — ~75ms model latency, explicitly traded against
// naturalness. Real complaint on a live call, side-by-side against Retell's
// own ElevenLabs integration: "night and day" worse. Retell almost
// certainly isn't using the bargain tier. eleven_multilingual_v2 is
// ElevenLabs' stable, established high-quality model (their newer eleven_v3
// is not used here — it's newer/experimental and may need a different API
// shape than this streaming endpoint, not worth the risk in the same
// change as fixing an active quality complaint). The latency cost is real
// but small relative to our actual per-turn budget, which is dominated by
// the LLM leg (700-900ms) — worth it for a direct quality complaint.
const ELEVENLABS_MODEL = process.env.ELEVENLABS_MODEL || 'eleven_multilingual_v2';
if (TTS_BACKEND === 'elevenlabs' && !ELEVENLABS_API_KEY) {
  console.warn('[call-loop] TTS_BACKEND=elevenlabs but ELEVENLABS_API_KEY not set — TTS will fail');
}
// Cartesia's TTS-bytes endpoint (see docs.cartesia.ai/api-reference/tts/bytes)
// — a single POST returning raw audio bytes, same shape as ElevenLabs's
// stream endpoint. voice.id is a UUID from Cartesia's voice library, not a
// memorable name, so there's no baked-in default — it must be set.
const CARTESIA_API_KEY = process.env.CARTESIA_API_KEY;
const CARTESIA_VOICE_ID = process.env.CARTESIA_VOICE_ID;
const CARTESIA_MODEL = process.env.CARTESIA_MODEL || 'sonic-3.6';
// Per-call ttsModel override allowlist (see CallSession.onClientMessage and
// /test-tts-override) — lets the mystery-shopper latency harness A/B TTS
// model/provider tiers against a live tenant's real agent without touching
// its DB config or the process-wide default, which would affect every real
// caller for the duration of the test.
const VALID_TTS_MODELS = {
  elevenlabs: new Set(['eleven_multilingual_v2', 'eleven_turbo_v2_5', 'eleven_flash_v2_5']),
  cartesia: new Set(['sonic-3.6', 'sonic-2']),
};
if (TTS_BACKEND === 'cartesia' && (!CARTESIA_API_KEY || !CARTESIA_VOICE_ID)) {
  console.warn('[call-loop] TTS_BACKEND=cartesia but CARTESIA_API_KEY/CARTESIA_VOICE_ID not set — TTS will fail');
}
// MiniMax's T2A v2 endpoint (see platform.minimax.io/docs/api-reference/
// speech-t2a-http) — needs both an API key and a GroupId (account/org id,
// distinct from the key itself) passed as a query param.
const MINIMAX_API_KEY = process.env.MINIMAX_API_KEY;
const MINIMAX_GROUP_ID = process.env.MINIMAX_GROUP_ID;
const MINIMAX_VOICE_ID = process.env.MINIMAX_VOICE_ID || 'English_Graceful_Lady';
const MINIMAX_MODEL = process.env.MINIMAX_MODEL || 'speech-2.8-hd';
if (TTS_BACKEND === 'minimax' && (!MINIMAX_API_KEY || !MINIMAX_GROUP_ID)) {
  console.warn('[call-loop] TTS_BACKEND=minimax but MINIMAX_API_KEY/MINIMAX_GROUP_ID not set — TTS will fail');
}
// Shared readiness check — used both at process startup (above) and when a
// per-session context message requests a backend (CallSession.onClientMessage),
// so a tenant can't silently end up with dead TTS just because a key was
// never configured on this Fly app.
function ttsBackendMissingKey(backend) {
  if (backend === 'elevenlabs') return !ELEVENLABS_API_KEY;
  if (backend === 'cartesia') return !CARTESIA_API_KEY || !CARTESIA_VOICE_ID;
  if (backend === 'minimax') return !MINIMAX_API_KEY || !MINIMAX_GROUP_ID;
  return false; // kokoro needs no key
}

// Backchanneling — the same technique Retell exposes as enable_backchannel /
// backchannel_frequency / backchannel_words: the moment the caller finishes
// talking, if the LLM hasn't produced its first token within
// BACKCHANNEL_DELAY_MS, speak a short filler ("Mm-hmm.", "Got it.", ...) so
// the call never sits in genuine dead air while Claude is still thinking.
// This doesn't reduce real LLM/TTS latency at all (see DECISIONS.md — that
// was investigated separately and the remaining gap isn't fixable on our
// side) — it masks the *perceived* latency, same as Retell does.
//
// Configurable process-wide via env vars, and per-session via the same
// {"type":"context"} message ttsBackend/voice already use (see
// CallSession.onClientMessage) — mirrors Retell's three knobs exactly:
//   BACKCHANNEL_ENABLED    -> enable_backchannel
//   BACKCHANNEL_FREQUENCY  -> backchannel_frequency (0-1: chance a slow
//                             turn actually gets a filler, so it doesn't
//                             fire mechanically on every single slow turn)
//   BACKCHANNEL_WORDS      -> backchannel_words (comma-separated)
// BACKCHANNEL_DELAY_MS isn't one of Retell's named params (their docs don't
// expose the threshold) but needs to be tunable too, so it gets the same
// env-var + per-session treatment.
// Was opt-in, not opt-out, at a 400ms threshold: that fired on ~80% of
// turns on every backend, including fast ones like ElevenLabs, which was
// never the intent — the intent was masking genuine latency (e.g. kokoro's
// cold start), not a filler word before most replies. Re-enabled by default
// now that mystery-shopper's real [latency] data gives an actual number to
// set the threshold from: ordinary turns land at 500-900ms llmTtfbMs, the
// closing/goodbye turn (see the isNodeEntry note above) regularly spikes to
// 3-4.4s. 1500ms sits well clear of the former and well under the latter,
// so this should now fire almost only on the turns that actually need
// masking. A tenant/call can still override via the {"type":"context"}
// message's backchannelEnabled/backchannelDelayMs fields.
const BACKCHANNEL_ENABLED_DEFAULT = process.env.BACKCHANNEL_ENABLED !== 'false';
const BACKCHANNEL_FREQUENCY_DEFAULT = Number(process.env.BACKCHANNEL_FREQUENCY ?? 0.8);
const BACKCHANNEL_DELAY_MS_DEFAULT = Number(process.env.BACKCHANNEL_DELAY_MS ?? 1500);
const BACKCHANNEL_WORDS_DEFAULT = (process.env.BACKCHANNEL_WORDS || 'Mm-hmm.,Got it.,One sec.,Sure thing.')
  .split(',')
  .map((w) => w.trim())
  .filter(Boolean);

// Kokoro-specific cold-start mask. Reproduced twice on real calls: the
// Modal-hosted TTS gateway can take longer to cold-start than the time
// until the flow's opening line needs to speak, even with the eager
// _ensureTtsSocket() call in the constructor — that reduces the window but
// doesn't guarantee it closes before the greeting is ready. Real fix:
// speak a short warmup line through a backend with no cold-start problem
// (elevenlabs — an HTTP call, not a persistent gateway connection) the
// moment a kokoro session needs to speak before its own socket is open, so
// the caller hears *something* instead of dead air while it finishes
// connecting. This is deliberately not the same mechanism as
// backchanneling above (which masks per-turn LLM latency mid-conversation)
// — this masks a one-time connection-setup delay at the start of a call.
const KOKORO_WARMUP_PHRASE = process.env.KOKORO_WARMUP_PHRASE || 'One moment while I get set up.';

// backend::voice::text -> Buffer (PCM16LE mono 24kHz). Populated at startup
// for the HTTP TTS backends that have a single, global, env-configured voice
// (elevenlabs/cartesia/minimax) so playing a filler costs zero network
// latency — synthesizing one live would be just as slow as the real
// response it's meant to hide. Not pre-warmed for kokoro, whose voice
// varies per session/tenant rather than being one fixed value; a kokoro
// session with no cached filler for its voice just skips backchanneling
// (see CallSession._maybeSpeakBackchannel) rather than synthesizing live.
const fillerCache = new Map();

async function fetchElevenLabsPcmOnce(text) {
  const res = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}/stream?output_format=pcm_24000`,
    {
      method: 'POST',
      headers: { 'xi-api-key': ELEVENLABS_API_KEY, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        model_id: ELEVENLABS_MODEL,
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    }
  );
  if (!res.ok || !res.body) throw new Error(`ElevenLabs prewarm failed: ${res.status}`);
  const parts = [];
  for await (const chunk of res.body) parts.push(Buffer.from(chunk));
  return Buffer.concat(parts);
}

async function fetchCartesiaPcmOnce(text) {
  const res = await fetch('https://api.cartesia.ai/tts/bytes', {
    method: 'POST',
    headers: {
      'Cartesia-Version': '2026-08-14',
      Authorization: `Bearer ${CARTESIA_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_id: CARTESIA_MODEL,
      transcript: text,
      voice: { id: CARTESIA_VOICE_ID },
      output_format: { container: 'raw', encoding: 'pcm_s16le', sample_rate: 24000 },
    }),
  });
  if (!res.ok || !res.body) throw new Error(`Cartesia prewarm failed: ${res.status}`);
  const parts = [];
  for await (const chunk of res.body) parts.push(Buffer.from(chunk));
  return Buffer.concat(parts);
}

async function fetchMinimaxPcmOnce(text) {
  const res = await fetch(`https://api-uw.minimax.io/v1/t2a_v2?GroupId=${encodeURIComponent(MINIMAX_GROUP_ID)}`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${MINIMAX_API_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MINIMAX_MODEL,
      text,
      stream: false,
      output_format: 'hex',
      voice_setting: { voice_id: MINIMAX_VOICE_ID, speed: 1.0, vol: 1.0, pitch: 0 },
      audio_setting: { sample_rate: 24000, format: 'pcm', channel: 1 },
    }),
  });
  if (!res.ok) throw new Error(`MiniMax prewarm failed: ${res.status}`);
  const body = await res.json();
  if (body.base_resp?.status_code !== 0 || !body.data?.audio) {
    throw new Error(`MiniMax prewarm synthesis error: ${body.base_resp?.status_msg || 'no audio'}`);
  }
  return Buffer.from(body.data.audio, 'hex');
}

async function prewarmFillerCache() {
  const jobs = [];
  for (const text of BACKCHANNEL_WORDS_DEFAULT) {
    if (ELEVENLABS_API_KEY) {
      jobs.push(
        fetchElevenLabsPcmOnce(text)
          .then((buf) => fillerCache.set(`elevenlabs::${ELEVENLABS_VOICE_ID}::${text}`, buf))
          .catch((err) => console.warn(`[call-loop] filler prewarm (elevenlabs, "${text}") failed:`, err.message))
      );
    }
    if (CARTESIA_API_KEY && CARTESIA_VOICE_ID) {
      jobs.push(
        fetchCartesiaPcmOnce(text)
          .then((buf) => fillerCache.set(`cartesia::${CARTESIA_VOICE_ID}::${text}`, buf))
          .catch((err) => console.warn(`[call-loop] filler prewarm (cartesia, "${text}") failed:`, err.message))
      );
    }
    if (MINIMAX_API_KEY && MINIMAX_GROUP_ID) {
      jobs.push(
        fetchMinimaxPcmOnce(text)
          .then((buf) => fillerCache.set(`minimax::${MINIMAX_VOICE_ID}::${text}`, buf))
          .catch((err) => console.warn(`[call-loop] filler prewarm (minimax, "${text}") failed:`, err.message))
      );
    }
  }
  // Kokoro cold-start warmup line — always via elevenlabs regardless of
  // which backend the session ends up using, since the whole point is to
  // speak while kokoro's own gateway is still connecting. Keyed under a
  // 'warmup::' namespace, distinct from the 'elevenlabs::' backchannel
  // entries above, even though both happen to use the same provider here.
  if (ELEVENLABS_API_KEY) {
    jobs.push(
      fetchElevenLabsPcmOnce(KOKORO_WARMUP_PHRASE)
        .then((buf) => fillerCache.set(`warmup::elevenlabs::${ELEVENLABS_VOICE_ID}::${KOKORO_WARMUP_PHRASE}`, buf))
        .catch((err) => console.warn('[call-loop] kokoro warmup line prewarm failed:', err.message))
    );
  }
  await Promise.all(jobs);
  console.log(`[call-loop] filler cache warmed: ${fillerCache.size} clip(s)`);
}
// Haiku over Sonnet for the voice path specifically — a phone reply doesn't
// need Sonnet's depth of reasoning, and LLM TTFB was the single biggest
// latency line item measured in real calls (1.3-1.8s), bigger than
// everything else in the pipeline combined. Process-wide default, overridable
// per call via the context message's `model` field (see CallSession.
// onClientMessage) so a quality-sensitive tenant can opt into a stronger
// model without changing what every other call pays/suffers in latency.
const LLM_MODEL = process.env.LLM_MODEL || 'claude-haiku-4-5-20251001';
const VALID_LLM_MODELS = new Set(['claude-haiku-4-5-20251001', 'claude-sonnet-4-6']);
// Only needed for a flow's 'transfer' node type on a real (Twilio) phone
// call — redirects the live call via Twilio's REST API. Not needed for
// browser calls (there's nothing to redirect) or flows with no transfer node.
const TWILIO_ACCOUNT_SID = process.env.TWILIO_ACCOUNT_SID;
const TWILIO_AUTH_TOKEN = process.env.TWILIO_AUTH_TOKEN;
// Needed to build absolute URLs (Pay's action callback, the reconnect
// TwiML's Stream url) from code paths with no live request to read a host
// from (_executePayment runs from _maybeRetireTurn, not an Express handler).
const PUBLIC_HOST = process.env.PUBLIC_HOST || 'call-loop-poc.fly.dev';
// Shared secret guarding the GET /active-calls admin endpoint (see below).
// Same shape as the gateway's ADMIN_SECRET (Bearer token, 401 when unset or
// mismatched) — this only exposes read-only live-call state, no control.
const ACTIVE_CALLS_SECRET = process.env.ACTIVE_CALLS_SECRET;
// Guards POST /place-test-call — see that route for why it's a distinct
// secret from ACTIVE_CALLS_SECRET rather than reusing it: that one only
// ever reads state, this one places real, billable outbound calls.
const TEST_CALL_SECRET = process.env.TEST_CALL_SECRET;
const SYSTEM_PROMPT =
  'You are a concise, friendly voice assistant on a phone call. Keep replies to 1-2 short ' +
  'sentences unless asked for more detail. Never use markdown, bullet points, or emoji — ' +
  'this is spoken audio.';

// The "mystery shopper" persona (see MYSTERY_SHOPPER_DECISIONS.md) — a
// customer with a fixed goal, deliberately generic so the same call works
// against any business's booking flow (ours or a competitor's) without
// knowing its exact question order in advance.
const SHOPPER_SYSTEM_PROMPT =
  'You are calling a business on the phone to book an appointment. You are NOT an assistant ' +
  'or an AI in this conversation — you are playing the role of a real customer named Alex ' +
  'Morgan. Your goal: book an appointment for tomorrow afternoon. Wait for the business to ' +
  'speak first and answer their questions naturally, one at a time, in whatever order they ' +
  'ask — do not volunteer your name or the appointment time before they ask for it. If asked ' +
  'to confirm something, confirm it. Once the booking is confirmed, thank them briefly ONE ' +
  'time and say a single goodbye — do not keep exchanging further pleasantries, goodbyes, or ' +
  '"you too, thanks" back-and-forth after that, even if the other party keeps talking; the ' +
  'call will be ended automatically after your goodbye. Keep replies short ' +
  'and conversational, like a real phone call. Never break character, never mention you are ' +
  'an AI, a test, or a language model, even if asked directly — just answer as Alex would.';
const SHOPPER_MAX_DURATION_MS = 3 * 60 * 1000;
// How many times _maybeRetireTurn will nudge a silent/stalled extraction
// node before giving up and forcing a hard-coded spoken fallback instead —
// see the deadlock-nudge fix there. Real call reproduced 2 consecutive
// silent turns on the same node (nudge itself went silent too), so 1 nudge
// wasn't enough; this is deliberately small so a model that's genuinely
// stuck doesn't nudge indefinitely before falling back.
const MAX_NUDGE_ATTEMPTS = 2;
// How long to wait after the last keypad digit before treating a DTMF
// entry as complete (see CallSession._onDtmfDigit) — long enough that a
// caller dialing a multi-digit code at a normal pace doesn't get cut off
// between digits, short enough that a single menu-selection digit ("press
// 1 for sales") doesn't leave a caller waiting a full second-plus for a
// response with no feedback that their press registered.
const DTMF_DEBOUNCE_MS = 1200;
// Shared by both the shopper's own reply (server.js's assistant-turn
// handling) and the check on what it just heard from the other party (see
// _onUserTurnComplete below) — see MYSTERY_SHOPPER_PATTERNS.md pattern 2
// for why watching only one side wasn't enough: dead-air mis-transcription
// after a real goodbye produces garbage that never matches this pattern,
// so a detector that only counts the SHOPPER's own replies can stall
// forever once the OTHER party has already said an equally clear goodbye.
// Cycle 13 finding: the original pattern's `thanks?...$` branch matched
// ANY reply ending in ordinary "thanks!" — which happens constantly in
// normal mid-conversation acknowledgment ("Perfect, thanks!"), not just
// real goodbyes. That falsely incremented the shopper's closing counter
// early in a call, and a later unrelated business turn then tripped the
// either-party hangup check, killing a call mid-booking before it ever
// reached a real conclusion. Dropped the bare "thanks" branch entirely —
// only real, close-to-complete farewell phrases count now.
const CLOSING_SHAPED_RE = /\b(goodbye|take care|have a (great|good|wonderful) day)\b|\bbye\b/i;

// Flux decides "they're done talking" from the words themselves, not just
// silence — eot_threshold is the confidence bar for a real EndOfTurn, higher
// = waits for more certainty before handing off to the LLM. StartOfTurn
// (below) is what drives barge-in and fires independently of eot_threshold.
// Two variants, not one: a browser call sends PCM16@16kHz mic audio, but a
// real Twilio call's native wire format is mu-law@8kHz. Flux accepts mulaw
// directly (encoding=mulaw&sample_rate=8000) — sending that straight through
// avoids exactly the same naive nearest-neighbor upsample-then-hope problem
// already fixed on the TTS output side (see twilioAdapter.js/_speakElevenLabs
// etc.): decoding real phone audio to PCM16 and resampling it before Deepgram
// even sees it can only lose information, never add it back. `numerals=true`
// converts spoken numbers to digits, which matters for a booking flow parsing
// dates/times out of the transcript.
//
// Latency/quality knobs (defaults set from the latency review — the
// mystery-shopper timing analysis added in the same change now measures
// real response-onset on every call, so these can be tuned against actual
// numbers rather than by hand):
// - eot_threshold lowered 0.7 -> 0.6: the semantic-certainty bar before the
//   LLM is allowed to start. 0.7 was conservative — it gates response onset
//   behind a high-confidence end-of-Thought that typically costs a few
//   hundred ms of the perceived turn-taking gap. 0.6 is still above the
//   acoustic-turn floor, so it rarely cuts a speaker off mid-word; the guard
//   against premature handoff is eot_timeout_ms below.
// - keyterm=<list>: Deepgram boosts recognition of the supplied phrases (e.g.
//   a tenant's known caller names or product terms). Unset = no boosting.
//   (smart_format was tried here too, but Flux rejects it outright --
//   INVALID_QUERY_PARAMETER, "Unknown query parameters: smart_format" --
//   it's a nova-model-only param. Every call's Deepgram socket was failing
//   its WS handshake with a 400 until this was caught: no STT connected on
//   any call, so no transcript, no turns, and no [latency] line ever fired.)
const DEEPGRAM_EOT_THRESHOLD = process.env.DEEPGRAM_EOT_THRESHOLD ?? '0.6';
const DEEPGRAM_EOT_TIMEOUT_MS = process.env.DEEPGRAM_EOT_TIMEOUT_MS ?? '5000';
const DEEPGRAM_KEYWORDS = process.env.DEEPGRAM_KEYWORDS?.trim() || null;
const deepgramParams =
  `model=flux-general-en&eot_threshold=${DEEPGRAM_EOT_THRESHOLD}&eot_timeout_ms=${DEEPGRAM_EOT_TIMEOUT_MS}` +
  `&numerals=true${DEEPGRAM_KEYWORDS ? `&keyterm=${encodeURIComponent(DEEPGRAM_KEYWORDS)}` : ''}`;
const DEEPGRAM_WS_URL_BROWSER = `wss://api.deepgram.com/v2/listen?${deepgramParams}&encoding=linear16&sample_rate=16000`;
const DEEPGRAM_WS_URL_TWILIO = `wss://api.deepgram.com/v2/listen?${deepgramParams}&encoding=mulaw&sample_rate=8000`;

if (!DEEPGRAM_API_KEY) console.warn('[call-loop] DEEPGRAM_API_KEY not set — STT will fail');
if (!ANTHROPIC_API_KEY) console.warn('[call-loop] ANTHROPIC_API_KEY not set — LLM will fail');

const anthropic = ANTHROPIC_API_KEY ? new Anthropic({ apiKey: ANTHROPIC_API_KEY }) : null;

const app = express();
app.use(express.static('public'));

app.use(express.urlencoded({ extended: false })); // Twilio POSTs form-encoded fields (To, From, CallSid)

// Per-call context resolved here (before the WS even connects) and handed
// off to the Media Stream once it starts — keyed by CallSid, since that's
// the only identifier both this webhook and the later 'start' event share.
// Swept on a timer so a call that gets a TwiML response but never actually
// opens the stream (e.g. caller hangs up mid-ring) doesn't leak forever.
const pendingCallContext = new Map();

// Session-resume state, keyed by CallSid — the infrastructure piece a
// mid-call TwiML detour needs (e.g. redirecting out to Twilio's <Pay> verb
// for PCI-compliant payment capture, then reconnecting our own
// <Connect><Stream> once it's done — see the payment-node design; that
// node type isn't built yet, this is the reusable piece underneath it).
// Redirecting a live call to different TwiML tears down the CURRENT Media
// Stream connection (a 'stop' event fires, same as a real hangup) and, once
// the detour's own TwiML finishes, opens a BRAND NEW one — a new
// CallSession would normally start that fresh, with none of the
// in-progress conversation. Stashing the outgoing session's live state
// here (in-process object references, not serialized — this never leaves
// memory, so there's no JSON-roundtrip cost or shape to maintain) lets the
// new session rehydrate instead of starting over.
//
// Not swept on a fixed timer like pendingCallContext — checked lazily on
// read (mirrors testTtsOverrides' expiresAt pattern) since a resume is
// rare and the entries are small; RESUME_TTL_MS below is generous enough
// to cover a real payment flow without being effectively unbounded.
const pendingResumeSessions = new Map();
const RESUME_TTL_MS = 10 * 60 * 1000; // 10 min — Pay can involve retries (bad card, etc.), give it real room

// Test-only, in-memory, TEST_CALL_SECRET-gated: lets the mystery-shopper
// latency harness force a real tenant's INBOUND agent (resolved from
// Supabase in /twilio/voice below) onto a different ttsBackend/ttsModel for
// a short window, without writing to that tenant's DB row -- which would
// change the experience for any real caller hitting the number during the
// test, not just the synthetic shopper call. Keyed by the dialed number;
// swept lazily on read (expired entries are just ignored, not deleted
// proactively -- this map is expected to have at most a couple of entries
// at once, size isn't a concern).
const testTtsOverrides = new Map();

// Registry of calls currently in progress — one entry per live CallSession,
// keyed by the session's own id. This is *live call state* (who's on a call
// right now, on what flow node), NOT an audio stream: the flow position it
// exposes is the same currentNodeId/nodeType the session already pushes to its
// own client via 'flow_state' events (collectedData is deliberately left out —
// see activeCallSnapshot), just also readable out-of-band by the dashboard
// through GET /active-calls. Entries are added in the CallSession constructor
// and removed in close(), so it always reflects open sessions.
const activeSessions = new Map();
setInterval(() => {
  const cutoff = Date.now() - 60_000;
  for (const [callSid, entry] of pendingCallContext) {
    if (entry.createdAt < cutoff) pendingCallContext.delete(callSid);
  }
}, 30_000).unref();

// Twilio POSTs here when the call is answered — the TwiML tells it to open
// a Media Streams WebSocket back to us at /twilio-stream, on this same host
// (works automatically whether that's localhost through a tunnel or a real
// deploy, since it's derived from the request itself, not hardcoded). Also
// resolves which tenant/flow owns the dialed number (see tenantLookup.js) —
// unrouted numbers (unregistered, or a 'retell'-engine version) fall back to
// this process's static single-tenant config exactly as before.
app.post('/twilio/voice', async (req, res) => {
  const callSid = req.body.CallSid;
  // Routing normally keys off the dialed number (req.body.To) — correct for
  // a real inbound call, but wrong for an outbound call WE place to a real
  // phone (e.g. a live demo call): there, To is the callee's own number, not
  // any tenant's routed number. ?routeAs=<number> lets an outbound call
  // explicitly say which tenant's flow it should run, without touching the
  // real inbound-routing path at all.
  // Mystery-shopper mode: an outbound call WHERE WE PLAY THE CUSTOMER, dialed
  // at an arbitrary target number (our own business line, or a competitor's
  // like Retell's agent number) — not a tenant lookup at all, so it bypasses
  // resolveInboundCall entirely. See MYSTERY_SHOPPER_DECISIONS.md decision 3
  // for why this reuses /twilio/voice + the normal CallSession machinery
  // instead of a separate service.
  if (req.query.mode === 'shopper' && callSid) {
    pendingCallContext.set(callSid, { isShopper: true, createdAt: Date.now() });
  } else {
    const toNumber = req.query.routeAs || req.body.To;
    if (callSid) {
      const resolved = await resolveInboundCall(toNumber).catch((err) => {
        console.error('[call-loop] tenant lookup failed', err);
        return null;
      });
      // Caller's number (From), carried through for the live-call registry's
      // display — the dialed tenant number (To) is the same for every call, the
      // caller's isn't.
      if (resolved) {
        // Test-only override (see testTtsOverrides above) — wins over the
        // tenant's real DB config, but only for the window set via
        // /test-tts-override, and only for this one call.
        const override = testTtsOverrides.get(toNumber);
        const ttsOverride = override && override.expiresAt > Date.now() ? override : null;
        pendingCallContext.set(callSid, {
          ...resolved,
          ...(ttsOverride ? { ttsBackend: ttsOverride.ttsBackend, ttsModel: ttsOverride.ttsModel } : {}),
          fromNumber: req.body.From || null,
          createdAt: Date.now(),
        });
      }
    }
  }
  const twiml =
    `<?xml version="1.0" encoding="UTF-8"?>` +
    `<Response><Connect><Stream url="wss://${req.headers.host}/twilio-stream" /></Connect></Response>`;
  res.type('text/xml').send(twiml);
});

// <Pay>'s action callback — Twilio POSTs here once a payment session ends
// (success, failure, or the caller pressed * to cancel). Writes ONLY the
// outcome into the paused session's stashed collectedData — Result, last
// four digits, card brand — never PaymentCardNumber or any other raw
// cardholder field, even though Twilio's webhook payload can include one
// for certain transaction types (see CreatePayments' own Input field note:
// digits are redacted from Twilio's LOGS, which is a different guarantee
// than "never appears in this webhook body" — the responsibility not to
// let it into OUR history/logs/DB is on this handler, not Twilio).
// Reconnects the call into a fresh <Connect><Stream> either way; the
// payment node's own outgoing edges (flow-author-defined, e.g. "payment
// succeeded" / "payment failed or was canceled") take it from there once
// _tryResumeFromCallSid rehydrates and _runNodeTurn re-evaluates.
app.post('/twilio/pay-result', (req, res) => {
  const callSid = req.query.callSid || req.body.CallSid;
  const stashed = callSid ? pendingResumeSessions.get(callSid) : null;
  if (stashed) {
    stashed.collectedData.payment_status = req.body.Result || 'unknown';
    if (req.body.PaymentCardNumber) {
      // Last-resort guard: this field is meant to already be masked
      // ("XXXXXXXXXXXX1234") per Twilio's redaction guarantee, but never
      // trust that blindly — only ever keep a trailing run of <=4 digits,
      // regardless of what actually arrives here.
      const digitsOnly = String(req.body.PaymentCardNumber).replace(/\D/g, '');
      stashed.collectedData.payment_last4 = digitsOnly.slice(-4);
    }
    if (req.body.PaymentConfirmationCode) stashed.collectedData.payment_confirmation = req.body.PaymentConfirmationCode;
    console.log(`[call-loop] pay-result for ${callSid}: ${stashed.collectedData.payment_status}`);
  } else {
    console.warn(`[call-loop] pay-result for ${callSid} — no stashed session found (expired or never redirected from here)`);
  }
  const twiml = `<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="wss://${req.headers.host}/twilio-stream" /></Connect></Response>`;
  res.type('text/xml').send(twiml);
});

// Read-only view of calls in progress right now, for the dashboard's live
// call state view. Guarded by a shared secret, mirroring the gateway's
// requireAdmin (Bearer token; 401 when the secret is unset or doesn't match)
// — see gateway/server.js. This returns presence + current flow node only,
// NOT audio or transcripts. Optional ?tenantId= filters to one tenant so a
// per-tenant caller never sees other tenants' calls.
app.get('/active-calls', (req, res) => {
  const auth = req.headers['authorization'] || '';
  if (!ACTIVE_CALLS_SECRET || auth !== `Bearer ${ACTIVE_CALLS_SECRET}`) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  const tenantId = typeof req.query.tenantId === 'string' ? req.query.tenantId : null;
  let calls = [...activeSessions.values()].map((s) => s.activeCallSnapshot());
  if (tenantId) calls = calls.filter((c) => c.tenantId === tenantId);
  res.json({ calls, count: calls.length, serverTime: Date.now() });
});

// Places a real outbound PSTN call from this app's own Twilio number to an
// arbitrary destination, routed through a specific tenant's flow via
// ?routeAs= on the /twilio/voice webhook (see that route's own comment for
// why routeAs exists — To on a real outbound call is the callee's number,
// not any tenant's routed number). This is the in-house engine's equivalent
// of calldesktech's Retell-based demo-call trigger; nothing wired it up
// before now because every prior end-to-end test drove the browser/text-
// debug WS path directly rather than a real phone call. Same admin-secret
// guard as /active-calls (Bearer token; 401 when unset or mismatched) —
// this places real, billable calls, not just reads state.
app.post('/place-test-call', express.json(), async (req, res) => {
  const auth = req.headers['authorization'] || '';
  if (!TEST_CALL_SECRET || auth !== `Bearer ${TEST_CALL_SECRET}`) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  const { toNumber, routeAs, record, shopper } = req.body || {};
  // Shopper mode (see MYSTERY_SHOPPER_DECISIONS.md): we're calling OUT to
  // play the customer, so there's no tenant to route as — toNumber is
  // whatever business we're dialing (our own number, or a competitor's).
  if (!toNumber || (!routeAs && !shopper)) {
    return res.status(400).json({ error: 'toNumber is required, plus either routeAs or shopper:true' });
  }
  if (!TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) {
    return res.status(500).json({ error: 'TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not configured' });
  }

  try {
    const auth64 = Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64');
    // No FROM number configured explicitly — ask Twilio which number(s) this
    // account actually owns and use the first, rather than guessing one.
    const numbersRes = await fetch(
      `https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/IncomingPhoneNumbers.json?PageSize=1`,
      { headers: { Authorization: `Basic ${auth64}` } }
    );
    const numbersBody = await numbersRes.json();
    const fromNumber = numbersBody.incoming_phone_numbers?.[0]?.phone_number;
    if (!fromNumber) {
      return res.status(500).json({ error: 'No Twilio phone number found on this account', detail: numbersBody });
    }

    const voiceUrl = shopper
      ? `https://${req.headers.host}/twilio/voice?mode=shopper`
      : `https://${req.headers.host}/twilio/voice?routeAs=${encodeURIComponent(routeAs)}`;
    const params = new URLSearchParams({ To: toNumber, From: fromNumber, Url: voiceUrl });
    // Opt-in only — a normal test call shouldn't silently start recording.
    // Twilio's own dual-channel recording (caller + callee on separate
    // tracks) gives a real downloadable wav for the ASR eval harnesses,
    // which need actual audio, not just a live transcript.
    if (record) {
      params.set('Record', 'true');
      params.set('RecordingChannels', 'dual');
    }
    const callRes = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Calls.json`, {
      method: 'POST',
      headers: { Authorization: `Basic ${auth64}`, 'Content-Type': 'application/x-www-form-urlencoded' },
      body: params,
    });
    const callBody = await callRes.json();
    if (!callRes.ok) {
      return res.status(callRes.status).json({ error: 'Twilio call creation failed', detail: callBody });
    }
    console.log(`[call-loop] test call placed: ${fromNumber} -> ${toNumber} (routeAs=${routeAs}), sid=${callBody.sid}`);
    res.json({ sid: callBody.sid, from: fromNumber, to: toNumber, status: callBody.status });
  } catch (err) {
    console.error('[call-loop] place-test-call failed', err);
    res.status(500).json({ error: err.message });
  }
});

// Sets/clears a short-lived per-number TTS override for A/B latency testing
// against a real tenant's live agent (see testTtsOverrides above for why
// this exists instead of just editing the tenant's DB row). Same admin-
// secret guard as /place-test-call. Body: { number, ttsBackend, ttsModel,
// ttsExpiresInMs? } to set (defaults to a 5-minute TTL so a forgotten
// override can't linger and silently affect real callers); { number } alone
// (no ttsBackend) clears it early.
app.post('/test-tts-override', express.json(), (req, res) => {
  const auth = req.headers['authorization'] || '';
  if (!TEST_CALL_SECRET || auth !== `Bearer ${TEST_CALL_SECRET}`) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  const { number, ttsBackend, ttsModel, ttsExpiresInMs } = req.body || {};
  if (!number) return res.status(400).json({ error: 'number is required' });
  if (!ttsBackend) {
    testTtsOverrides.delete(number);
    return res.json({ cleared: number });
  }
  if (!VALID_TTS_BACKENDS.includes(ttsBackend)) {
    return res.status(400).json({ error: `invalid ttsBackend: ${ttsBackend}` });
  }
  if (ttsModel && !VALID_TTS_MODELS[ttsBackend]?.has(ttsModel)) {
    return res.status(400).json({ error: `invalid ttsModel ${ttsModel} for backend ${ttsBackend}` });
  }
  const expiresAt = Date.now() + Math.min(Number(ttsExpiresInMs) || 5 * 60_000, 30 * 60_000);
  testTtsOverrides.set(number, { ttsBackend, ttsModel: ttsModel || null, expiresAt });
  console.log(`[call-loop] test TTS override set for ${number}: ${ttsBackend}${ttsModel ? `/${ttsModel}` : ''}, expires ${new Date(expiresAt).toISOString()}`);
  res.json({ number, ttsBackend, ttsModel: ttsModel || null, expiresAt });
});

const server = http.createServer(app);
// Two WebSocketServer instances each bound with {server, path} don't reliably
// coexist on one shared http.Server in this ws version — the second path
// registered gets a 400 on every handshake instead of falling through to it.
// ws's own docs' fix for multiple paths on one server: noServer mode on both,
// with manual routing on the server's single 'upgrade' event instead.
const wss = new WebSocketServer({ noServer: true });
const twilioWss = new WebSocketServer({ noServer: true });

server.on('upgrade', (req, socket, head) => {
  const { pathname } = new URL(req.url, `http://${req.headers.host}`);
  if (pathname === '/call') {
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
  } else if (pathname === '/twilio-stream') {
    twilioWss.handleUpgrade(req, socket, head, (ws) => twilioWss.emit('connection', ws, req));
  } else {
    socket.destroy();
  }
});

wss.on('connection', (clientWs) => {
  console.log('[call-loop] client connected');
  const session = new CallSession(clientWs);
  clientWs.on('message', (data, isBinary) => session.onClientMessage(data, isBinary));
  clientWs.on('close', () => session.close());
  clientWs.on('error', (err) => console.error('[call-loop] client ws error', err));
});

twilioWss.on('connection', (twilioWs) => {
  console.log('[call-loop] Twilio call connected');
  // The adapter presents the exact same interface a browser WebSocket would
  // — the session class has no idea this is a phone call, not a browser tab.
  const adapter = new TwilioCallAdapter(twilioWs);
  const session = new CallSession(adapter);
  adapter.on('message', (data, isBinary) => session.onClientMessage(data, isBinary));
  adapter.on('close', () => session.close());
  adapter.on('error', (err) => console.error('[call-loop] twilio adapter error', err));
  adapter.on('dtmf', (digit) => session._onDtmfDigit(digit));
  // Per-tenant routing resolved back in /twilio/voice (see tenantLookup.js)
  // — handed to the session the same way a browser client does it, via a
  // synthesized 'context' message, rather than duplicating that parsing
  // logic here.
  adapter.on('start', (callSid) => {
    // Tags every "turn N user/assistant" log line below with the real
    // Twilio CallSid, so a concurrent-run script (mystery-shopper-run.sh)
    // can grep logs for one specific call instead of hand-disentangling
    // multiple sessions' turn counters that collide when several calls run
    // at once (real gap found running the mystery-shopper framework, see
    // MYSTERY_SHOPPER_DECISIONS.md).
    session.callSid = callSid;

    // A reconnect after a mid-call detour (see _redirectForDetour) has no
    // pendingCallContext entry — that context was already established on
    // this call's FIRST segment, and the detour's own TwiML (e.g. <Pay>)
    // is what reconnects us, not another /twilio/voice hit. Check this
    // BEFORE the pendingCallContext lookup below, which would otherwise
    // just silently no-op on a resumed call (no entry to find).
    if (session._tryResumeFromCallSid(callSid)) {
      // No fresh caller utterance to react to (the caller was just talking
      // to Twilio's own Pay prompts, not us) — same gap _runNodeTurn
      // already bridges for any other auto-advanced node entry.
      session._runNodeTurn(session.currentNodeId);
      return;
    }

    const resolved = pendingCallContext.get(callSid);
    if (!resolved) return;
    pendingCallContext.delete(callSid);
    if (resolved.isShopper) {
      // No flow, no greeting — a flow-less CallSession already stays silent
      // until it hears something (see MYSTERY_SHOPPER_DECISIONS.md decision
      // 2), which is exactly right for a customer who calls IN and waits for
      // the business to greet first, rather than speaking first.
      session.isShopper = true;
      session.onClientMessage(JSON.stringify({
        type: 'context',
        systemPrompt: SHOPPER_SYSTEM_PROMPT,
        ttsBackend: 'elevenlabs',
      }), false);
      // Safety net (see decision 5): a flow-less session never hangs up on
      // its own — normally fine, since the real business side ends the call
      // — but if something on either end gets stuck, this stops a live PSTN
      // call (real per-minute cost on both Twilio and whatever it's dialing)
      // from running forever.
      setTimeout(() => {
        console.log(`[call-loop] shopper call ${callSid} hit max duration, hanging up`);
        session.close();
      }, SHOPPER_MAX_DURATION_MS);
      return;
    }
    session.onClientMessage(JSON.stringify({
      type: 'context',
      flow: resolved.flow,
      ...(resolved.ttsBackend ? { ttsBackend: resolved.ttsBackend } : {}),
      ...(resolved.ttsModel ? { ttsModel: resolved.ttsModel } : {}),
      ...(resolved.stripeCustomerId ? { stripeCustomerId: resolved.stripeCustomerId } : {}),
      ...(resolved.tenantId ? { tenantId: resolved.tenantId } : {}),
      ...(resolved.fromNumber ? { phoneNumber: resolved.fromNumber } : {}),
    }), false);
  });
});

class CallSession {
  constructor(clientWs) {
    this.clientWs = clientWs;
    this.history = [];
    this.turnSeq = 0;
    this.activeTurn = 0; // turn id currently allowed to speak; 0 once nothing is in flight
    this.turnState = null; // { id, llmDone, pendingTts } — tracks when activeTurn can go back to 0
    this.ttsWs = null;
    this.dgConnection = null;
    // Per-call context — overridable by a {"type":"context"} message sent
    // right after connect (see onClientMessage), so a caller (e.g. a
    // multi-tenant product embedding this) can hand each session its own
    // business prompt/voice instead of every call getting the same
    // hardcoded assistant. Falls back to the module defaults untouched.
    this.systemPrompt = SYSTEM_PROMPT;
    this.voice = TTS_VOICE;
    this.greeting = null;
    this.ttsBackend = TTS_BACKEND;
    this.ttsModel = null; // per-call override via context `ttsModel` — see onClientMessage
    this.llmModel = LLM_MODEL; // per-call override via context `model` — see onClientMessage
    this._latency = null; // per-turn latency instrumentation — see _markTtsFirstByte
    this.backchannelEnabled = BACKCHANNEL_ENABLED_DEFAULT;
    this.backchannelFrequency = BACKCHANNEL_FREQUENCY_DEFAULT;
    this.backchannelDelayMs = BACKCHANNEL_DELAY_MS_DEFAULT;
    this.backchannelWords = BACKCHANNEL_WORDS_DEFAULT;
    // Conversation flow (optional) — a real node-based state machine, set via
    // the {"type":"context"} message's `flow` field. When absent, the session
    // behaves exactly as before (this.systemPrompt used verbatim every turn).
    // See _buildNodeSystemPrompt/_buildTransitionTool/_applyTransition for how
    // a node's own instructions + edges become a per-turn prompt + a real
    // tool call the LLM uses to advance state, instead of flattening the
    // whole graph into one static prompt upfront.
    this.flow = null;
    this.flowNodesById = null;
    this.currentNodeId = null;
    this.collectedData = {};
    // Set via the {"type":"context"} message's `stripeCustomerId` field —
    // present only when this call belongs to a real billed tenant (browser
    // demo calls and flow-MCP test calls have none, and simply don't get
    // metered). See stripeMeter.js.
    this.stripeCustomerId = null;
    // Mystery-shopper flag (see MYSTERY_SHOPPER_DECISIONS.md) — enables the
    // closing-loop-detection hangup below, since a flow-less session
    // otherwise never ends a call on its own.
    this.isShopper = false;
    this.callSid = null; // set once the Twilio 'start' event arrives (browser calls never get one)
    this._shopperClosingCount = 0;
    this._closing = false;
    // Set by _stashForResume() right before a mid-call TwiML detour (e.g.
    // Pay) redirects the live call — the 'stop' event that follows is this
    // session's transport actually closing, but NOT the call ending, so
    // close() must skip billing finalization and the live-call-registry
    // removal it normally does on a real hangup. See pendingResumeSessions.
    this._pausedForResume = false;
    this._nudgeAttempts = new Map(); // nodeId -> count, see _maybeRetireTurn's deadlock-nudge fix
    this._queuedUserText = null; // see _onUserTurnComplete's in-flight-turn guard
    this._dtmfBuffer = ''; // see _onDtmfDigit — buffered keypad digits not yet flushed into a turn
    this._dtmfTimer = null;
    // Live-monitoring metadata — which tenant owns this call and the phone
    // number involved, both set from the {"type":"context"} message (see
    // onClientMessage). Null for anonymous browser demo calls, which carry no
    // tenant; a real Twilio call gets both from the routing lookup. Only used
    // to populate the /active-calls registry — nothing in the call path reads
    // them.
    this.tenantId = null;
    this.phoneNumber = null;
    this.cost = new CallCostTracker({ ttsBackend: TTS_BACKEND, voiceEngine: 'cascaded' });
    this._callStartedAt = Date.now();
    // Stable id for the active-calls registry (also handy in logs). Registered
    // here so the call shows up the instant the session opens, even before any
    // context/flow arrives; close() removes it.
    this.id = randomUUID();
    activeSessions.set(this.id, this);
    this._connectDeepgram();
    // Real bug, reproduced twice: _ensureTtsSocket() used to open the Modal-
    // hosted TTS gateway lazily, on the FIRST _speak() call. Modal cold
    // starts can take 10+ seconds, and if that exceeds how long until the
    // next turn arrives, the superseded-turn check in _speak's dispatch()
    // silently drops the first turn's audio entirely — genuine dead air,
    // not just slow. Opening the socket here instead means the cold start
    // happens concurrently with STT setup and the caller's first utterance,
    // so by the time there's actually something to say, the socket is
    // already warm (or at least much further along). Cheap to do
    // unconditionally even for non-kokoro calls — an unused idle WS gets
    // closed normally in close(), same as today.
    this._ensureTtsSocket();
  }

  // Read-only snapshot for the /active-calls endpoint. Deliberately excludes
  // conversation content — this is presence + current flow position, not a
  // transcript or audio feed.
  activeCallSnapshot() {
    return {
      id: this.id,
      tenantId: this.tenantId,
      phoneNumber: this.phoneNumber,
      startedAt: this._callStartedAt,
      currentNodeId: this.currentNodeId,
      nodeType: this.currentNodeId ? this.flowNodesById?.get(this.currentNodeId)?.type ?? null : null,
    };
  }

  send(obj) {
    if (this.clientWs.readyState === WebSocket.OPEN) this.clientWs.send(JSON.stringify(obj));
  }

  _connectDeepgram() {
    if (!DEEPGRAM_API_KEY) return;
    const isTwilio = this.clientWs instanceof TwilioCallAdapter;
    const url = isTwilio ? DEEPGRAM_WS_URL_TWILIO : DEEPGRAM_WS_URL_BROWSER;
    const dg = new WebSocket(url, { headers: { Authorization: `Token ${DEEPGRAM_API_KEY}` } });
    this.dgConnection = dg;

    dg.on('open', () => console.log('[call-loop] deepgram (flux) connected'));

    dg.on('message', (data) => {
      let msg;
      try {
        msg = JSON.parse(data.toString());
      } catch {
        return;
      }
      if (msg.type !== 'TurnInfo') return;

      if (msg.event === 'StartOfTurn') {
        // Real complaint from a real call: the assistant kept getting cut
        // off — including right near the end of a response — by things
        // that were never actual interruptions (a breath, a stray "mm",
        // background noise). StartOfTurn is Deepgram's raw acoustic
        // voice-onset signal, no transcript yet, and firing _bargeIn()
        // straight off it means ANY detected sound while the assistant is
        // talking kills its audio. Don't commit to a barge-in yet — wait
        // for Deepgram to actually transcribe real words for this turn
        // (below) before cutting the assistant off. A sound that never
        // produces a transcript (because it wasn't real speech) now never
        // barges in at all, instead of always eventually doing so.
        this._pendingBargeIn = true;
      } else if (msg.event === 'Update') {
        const text = msg.transcript?.trim();
        if (text) {
          this.send({ type: 'transcript', text, isFinal: false });
          if (this._pendingBargeIn) {
            this._pendingBargeIn = false;
            this._bargeIn();
          }
        }
      } else if (msg.event === 'EndOfTurn') {
        // High-confidence, semantically-aware turn end (eot_threshold) — this
        // is what nova-2 + acoustic VAD couldn't do: it waits for a complete
        // thought, not just a gap in the audio.
        const text = msg.transcript?.trim();
        if (text) this._onUserTurnComplete(text);
      }
    });

    dg.on('error', (err) => console.error('[call-loop] deepgram error', err));
    dg.on('close', () => console.log('[call-loop] deepgram closed'));
  }

  onClientMessage(data, isBinary) {
    if (isBinary) {
      // Raw PCM16 mono 16kHz audio frame from the browser mic.
      if (this.dgConnection?.readyState === WebSocket.OPEN) {
        this.dgConnection.send(data);
      }
      return;
    }
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === 'hangup') this.close();
    // Text-input debug path — lets a turn be driven without a mic/Deepgram
    // (e.g. scripting a flow test), bypassing STT entirely. Real calls never
    // send this; it's additive and doesn't change the audio path at all.
    if (msg.type === 'user_text' && typeof msg.text === 'string' && msg.text.trim()) {
      this._onUserTurnComplete(msg.text.trim());
    }
    if (msg.type === 'context') {
      // Must arrive before the first user turn to take effect — it's read
      // fresh at the start of each turn, not snapshotted at connect time.
      if (typeof msg.systemPrompt === 'string' && msg.systemPrompt.trim()) {
        this.systemPrompt = msg.systemPrompt;
      }
      if (typeof msg.voice === 'string' && msg.voice.trim()) {
        this.voice = msg.voice;
      }
      if (typeof msg.stripeCustomerId === 'string' && msg.stripeCustomerId.trim()) {
        this.stripeCustomerId = msg.stripeCustomerId.trim();
      }
      // Live-monitoring metadata (see the registry / GET /active-calls) — a
      // real routed call passes both; anonymous browser demos pass neither.
      if (typeof msg.tenantId === 'string' && msg.tenantId.trim()) {
        this.tenantId = msg.tenantId.trim();
      }
      if (typeof msg.phoneNumber === 'string' && msg.phoneNumber.trim()) {
        this.phoneNumber = msg.phoneNumber.trim();
      }
      if (VALID_TTS_BACKENDS.includes(msg.ttsBackend)) {
        if (msg.ttsBackend !== 'kokoro' && ttsBackendMissingKey(msg.ttsBackend)) {
          console.warn(`[call-loop] context requested ttsBackend=${msg.ttsBackend} but its API key/config isn't set — falling back to kokoro`);
        } else {
          this.ttsBackend = msg.ttsBackend;
          this.cost.ttsBackend = msg.ttsBackend;
        }
      }
      // Test-only knob (see /test-tts-override): which model/tier within
      // this.ttsBackend to use, validated per-backend so a typo can't reach
      // the provider's API as an arbitrary string. Unset = that backend's
      // process-wide default (ELEVENLABS_MODEL/CARTESIA_MODEL).
      if (typeof msg.ttsModel === 'string' && VALID_TTS_MODELS[this.ttsBackend]?.has(msg.ttsModel.trim())) {
        this.ttsModel = msg.ttsModel.trim();
      }
      // Same knob as Retell's LLM choice, one model per call: a tenant that
      // needs a stronger/cheaper model than the process default picks it here
      // instead of requiring an app-wide restart.
      if (typeof msg.model === 'string' && VALID_LLM_MODELS.has(msg.model.trim())) {
        this.llmModel = msg.model.trim();
      }
      // Same three knobs Retell exposes as enable_backchannel/
      // backchannel_frequency/backchannel_words, plus our own tunable delay
      // threshold — see BACKCHANNEL_*_DEFAULT above for what each controls.
      if (typeof msg.backchannelEnabled === 'boolean') {
        this.backchannelEnabled = msg.backchannelEnabled;
      }
      if (typeof msg.backchannelFrequency === 'number' && msg.backchannelFrequency >= 0 && msg.backchannelFrequency <= 1) {
        this.backchannelFrequency = msg.backchannelFrequency;
      }
      if (typeof msg.backchannelDelayMs === 'number' && msg.backchannelDelayMs > 0) {
        this.backchannelDelayMs = msg.backchannelDelayMs;
      }
      if (Array.isArray(msg.backchannelWords) && msg.backchannelWords.every((w) => typeof w === 'string' && w.trim())) {
        this.backchannelWords = msg.backchannelWords.map((w) => w.trim());
      }
      if (Array.isArray(msg.flow?.nodes) && msg.flow.nodes.length > 0) {
        this.flow = msg.flow;
        this.flowNodesById = new Map(this.flow.nodes.map((n) => [n.id, n]));
        this.currentNodeId = msg.flow.startNodeId || this.flow.nodes[0].id;
        console.log(`[call-loop] flow set — ${this.flow.nodes.length} nodes, starting at "${this.currentNodeId}"`);
        this.send({ type: 'flow_state', currentNodeId: this.currentNodeId, nodeType: this.flowNodesById.get(this.currentNodeId)?.type, collectedData: this.collectedData });
      }
      if (typeof msg.greeting === 'string' && msg.greeting.trim()) {
        this.greeting = msg.greeting;
        this.history.push({ role: 'assistant', content: msg.greeting });
        // Speak it as a real turn (not a special-cased id) so barge-in works
        // on the greeting exactly like it does on every other response.
        const turnId = ++this.turnSeq;
        this.activeTurn = turnId;
        this.turnState = { id: turnId, llmDone: true, pendingTts: 0, startedSpeaking: false, saidNothing: false };
        this._speak(msg.greeting, turnId, Date.now());
      } else if (this.flow) {
        // No explicit greeting text — let the flow's own start node (usually
        // a 'greeting'-type node) generate the opening line itself, same as
        // every other node's turn.
        this._runNodeTurn(this.currentNodeId);
      }
      console.log(`[call-loop] context set — prompt: ${this.systemPrompt.length} chars, voice: ${this.voice}, ttsBackend: ${this.ttsBackend}, model: ${this.llmModel}`);
    }
  }

  // Real Twilio keypad input (see twilioAdapter.js's 'dtmf' event) — a
  // caller navigating an IVR-style menu ("press 1 for sales") or entering a
  // multi-digit code. Deliberately NOT a new flow-node type: a flow's edges
  // already get evaluated by the model against whatever the caller's last
  // turn said, so feeding a digit sequence in as a synthetic user turn
  // ("[Caller pressed 1 on the keypad]") lets a flow author write an edge
  // condition like "caller pressed 1" using the exact same mechanism as any
  // spoken condition — no schema change, no separate menu-node concept to
  // keep in sync with the real one.
  //
  // Buffered, not fired per-keypress: entering "1234#" for a 4-digit code
  // would otherwise trigger four separate conversational turns (one per
  // digit) before the caller finishes, which is wrong — a real IVR either
  // waits for a terminator key or a short pause. '#' flushes immediately
  // (the universal "I'm done entering" key); otherwise a short debounce
  // covers the common single-digit menu-selection case (a lone "1" with no
  // more coming) without needing the caller to also press '#' for that.
  _onDtmfDigit(digit) {
    if (this._closing) return;
    if (digit === '#') {
      this._flushDtmfBuffer();
      return;
    }
    if (digit === '*') {
      // No real meaning defined yet (Retell's own IVR templates use it for
      // "start over" in some flows) — drop rather than buffer garbage into
      // what the model sees as caller input.
      return;
    }
    this._dtmfBuffer += digit;
    if (this._dtmfTimer) clearTimeout(this._dtmfTimer);
    this._dtmfTimer = setTimeout(() => this._flushDtmfBuffer(), DTMF_DEBOUNCE_MS);
  }

  _flushDtmfBuffer() {
    if (this._dtmfTimer) {
      clearTimeout(this._dtmfTimer);
      this._dtmfTimer = null;
    }
    if (!this._dtmfBuffer) return;
    const digits = this._dtmfBuffer;
    this._dtmfBuffer = '';
    console.log(`[call-loop] [call ${this.callSid || this.id}] dtmf: "${digits}"`);
    this._onUserTurnComplete(`[Caller pressed ${digits} on the keypad]`);
  }

  async _onUserTurnComplete(userText) {
    // Cycle-2 mystery-shopper finding (see MYSTERY_SHOPPER_DECISIONS.md):
    // close() only disconnects once the audio queue drains, but nothing
    // stopped a NEW turn from being generated (and re-filling that queue)
    // in the meantime if the other party kept talking — so if they never
    // stopped, the real disconnect never happened either. Once a hangup
    // has been decided, stop generating new turns outright, regardless of
    // what's still coming in from the other side.
    if (this._closing) {
      console.log('[call-loop] turn ignored — session is closing');
      return;
    }
    // Pattern-2 fix (MYSTERY_SHOPPER_PATTERNS.md): once the shopper has said
    // its OWN goodbye at least once, treat the other party's very next
    // reply as the second (and stronger) signal to hang up if it's ALSO
    // closing-shaped — don't wait for a second closing-shaped reply from
    // the shopper itself, which a dead-air mis-transcription can prevent
    // from ever arriving. This is checked before the turn is even
    // generated, so the shopper never has a chance to respond to what
    // might already be silence/noise on the line.
    if (this.isShopper && this._shopperClosingCount >= 1 && CLOSING_SHAPED_RE.test(userText.trim())) {
      console.log(`[call-loop] shopper: other party also closing-shaped (count=${this._shopperClosingCount}, matched text: "${userText.trim()}"), hanging up immediately`);
      this._closing = true;
      this.close();
      return;
    }
    // Real bug, found via mystery-shopper + code review: this function used
    // to blindly overwrite this.activeTurn/this.turnState for every final
    // transcript, with no check for whether the PREVIOUS turn's LLM call
    // was still in flight. _generateTurn only speaks/logs/records a turn's
    // result if this.activeTurn still matches its own turnId when the LLM
    // stream finishes (see the check there) — so a second utterance arriving
    // a second or two after the first (completely normal in a real
    // conversation, e.g. "Yes." then "That's correct." in quick succession)
    // would silently blow away whatever the first turn's LLM call was about
    // to say, with NO log line, no fallback, nothing. On a live call this
    // ate turns that happened to be the actual booking confirmation, making
    // the call end with no acknowledgment at all.
    //
    // This is NOT the same thing as a real barge-in (see _bargeIn, driven by
    // Deepgram's StartOfTurn+real-words while the assistant is actively
    // speaking) — a real barge-in explicitly resets activeTurn to 0 first,
    // so by the time its EndOfTurn arrives here there's nothing in flight to
    // race against. This guard only catches the case _bargeIn was never
    // meant to cover: the assistant hasn't said a single word of its reply
    // yet (turnState.startedSpeaking is still false), so there's nothing to
    // legitimately interrupt — queue the new utterance and let
    // _maybeRetireTurn replay it the moment the in-flight turn finishes,
    // instead of discarding it.
    if (this.activeTurn !== 0 && this.turnState?.id === this.activeTurn && !this.turnState.startedSpeaking) {
      this._queuedUserText = this._queuedUserText ? `${this._queuedUserText} ${userText}` : userText;
      console.log(`[call-loop] turn ${this.activeTurn} hasn't spoken yet — queuing instead of preempting: "${userText}"`);
      return;
    }
    const turnId = ++this.turnSeq;
    this.activeTurn = turnId;
    this.turnState = { id: turnId, llmDone: false, pendingTts: 0, startedSpeaking: false, saidNothing: false };
    const turnStartedAt = Date.now();
    // Seed the per-turn latency trace: turnStart is the server-side moment
    // Deepgram's EndOfTurn arrived (i.e. end of the caller's speech), so the
    // downstream timestamps in _markTtsFirstByte measure the true response
    // latency. Cleared after its TTS first byte is logged. Only a real user
    // turn seeds a trace — a flow auto-advance (nudge/_nodeTurn) has no
    // caller utterance to measure a response to.
    this._latency = { turnStart: turnStartedAt };
    console.log(`[call-loop] [call ${this.callSid || this.id}] turn ${turnId} user: "${userText}"`);

    this.history.push({ role: 'user', content: userText });
    this.send({ type: 'user_turn', turnId, text: userText });

    await this._generateTurn(turnId, turnStartedAt);
  }

  // Generates an assistant turn for the *current flow node* with no new
  // caller utterance — used for the flow's opening line, and for node types
  // (function/knowledge_base/goodbye/transfer) that act as soon as the flow
  // enters them rather than waiting on the caller to say something first.
  async _runNodeTurn(nodeId) {
    this.currentNodeId = nodeId;
    const turnId = ++this.turnSeq;
    this.activeTurn = turnId;
    this.turnState = { id: turnId, llmDone: false, pendingTts: 0, startedSpeaking: false, saidNothing: false };
    // The model needs a fresh `user` turn at the end of history to actually
    // have something to respond to. The call's very opening has nothing at
    // all yet (isCallOpening). A LATER auto-advance — the flow moved to
    // this node as a side effect of the previous turn's transition_flow
    // call, with no new caller utterance since — has the exact same gap:
    // history still ends on the model's OWN prior assistant reply. This
    // was the real root cause of a bug reproduced on every real call that
    // reached the goodbye node: Haiku consistently produced empty text for
    // it (nothing fresh to react to), silently caught by the fallback
    // below — which had its own bug, speaking the node's raw instruction
    // text ("Thank the caller for calling and say a warm goodbye.")
    // verbatim to the actual caller instead of a real goodbye. Both are
    // fixed now: this bridges the gap so the model has something to
    // respond to (fixing the empty-text root cause for auto-advanced
    // nodes generally, not just goodbye), and the fallback text below no
    // longer vocalizes developer instructions either way.
    const isCallOpening = this.history.length === 0;
    const lastMsg = this.history[this.history.length - 1];
    if (isCallOpening) {
      this.history.push({ role: 'user', content: '[Call connected — begin the flow.]' });
    } else if (!lastMsg || lastMsg.role !== 'user') {
      this.history.push({
        role: 'user',
        content: `[System note: the flow has moved to the "${nodeId}" step. Give your opening line for this step now.]`,
      });
    }
    await this._generateTurn(turnId, Date.now(), { isNodeEntry: true, suppressTransitionTool: isCallOpening, isCallOpening });
  }

  // Shared by both a real caller turn and a flow auto-advance turn — the
  // only difference is whether a user message was already pushed to
  // this.history before calling in. When this.flow is set, the system
  // prompt/tools are scoped to just the current node (see
  // _buildNodeSystemPrompt/_buildTransitionTool) instead of the flat
  // this.systemPrompt every other turn uses, and the LLM signals when to
  // advance by calling the transition_flow tool rather than us guessing from
  // the model's prose.
  async _generateTurn(turnId, turnStartedAt, { isNodeEntry = false, suppressTransitionTool = false, isCallOpening = false } = {}) {
    if (!anthropic) {
      this.send({ type: 'error', message: 'ANTHROPIC_API_KEY not configured' });
      if (this.turnState?.id === turnId) this.turnState.llmDone = true;
      return;
    }

    const node = this.flow ? this.flowNodesById.get(this.currentNodeId) : null;
    if (this.turnState?.id === turnId) {
      this.turnState.nodeType = node?.type || null;
      this.turnState.nodeParams = node?.params || null;
    }

    // A function node's side effect happens before it says anything, and
    // only once — on the turn that actually enters the node (via
    // _runNodeTurn). Without the isNodeEntry guard, every later turn where
    // the conversation just happens to still be sitting in this node (e.g.
    // the caller asking a follow-up before transitioning away) would re-run
    // the webhook, which is wrong — it's a one-time side effect of arriving
    // at the node, not a per-turn one.
    if (node?.type === 'function' && isNodeEntry) {
      await this._executeFunctionNode(node);
    }
    if (node?.type === 'knowledge_base' && isNodeEntry) {
      await this._executeKnowledgeBaseNode(node);
    }

    const systemPrompt = node ? this._buildNodeSystemPrompt(node, isNodeEntry) : this.systemPrompt;
    // The call's very opening turn has no real caller utterance to justify
    // any edge yet — only the synthetic "[Call connected]" seed message —
    // so the transition tool is withheld for that one turn specifically.
    // Without this, Haiku would sometimes call it anyway (observed jumping
    // straight from greeting to the next node before the caller had said a
    // word), reading "begin the flow" too literally. Every other node-entry
    // turn (function/knowledge_base/goodbye/transfer auto-advance, or the
    // greeting reached via a real transition) keeps the tool as normal.
    const tools = [];
    if (node && node.edges.length > 0 && !suppressTransitionTool) {
      tools.push(this._buildTransitionTool(node));
    }
    // Real bug found via mystery-shopper testing: this.collectedData was
    // only ever populated as a side effect of transition_flow, meaning a
    // field captured several turns before the node's transition condition
    // was met existed ONLY in conversation history, not in any tracked
    // state — so the "Already collected this call" context the prompt
    // shows is stale/empty for anything not yet transitioned on. Under a
    // confusing turn (e.g. dead-air noise right after a goodbye), the model
    // relying purely on re-reading history "forgot" a name it had already
    // been given three turns earlier and re-asked for it multiple times.
    // A lightweight tool that persists a field the MOMENT it's captured,
    // independent of transitioning, closes that gap.
    if (node && node.extract) {
      tools.push({
        name: 'record_field',
        description:
          'Call this immediately whenever the caller provides one of this step\'s fields, even ' +
          'if you are not ready to transition yet. Safe to call multiple times.',
        input_schema: {
          type: 'object',
          properties: { field: { type: 'string', enum: Object.keys(node.extract) }, value: { type: 'string' } },
          required: ['field', 'value'],
        },
      });
    }

    let firstTokenAt = null;
    let assistantText = '';
    const chunker = new SentenceChunker((sentence) => {
      if (this.activeTurn !== turnId) {
        console.log(`[call-loop] turn ${turnId} sentence chunk dropped — activeTurn is now ${this.activeTurn}: "${sentence}"`);
        return;
      }
      this._speak(sentence, turnId, turnStartedAt);
    });

    // Backchanneling. Used to be gated on !isNodeEntry ("no caller utterance
    // to be acknowledging" for a flow auto-advance turn like the goodbye
    // node) — but mystery-shopper's real [latency] data shows the closing/
    // goodbye turn is consistently the single slowest turn in a call
    // (llmTtfbMs regularly 3-4.4s there vs 500-900ms on ordinary turns,
    // every call checked), and that's an isNodeEntry turn, so it could never
    // get a filler regardless of this flag — dead air on exactly the turn
    // most likely to need masking. A filler doesn't have to be replying to
    // anything to be worth playing; it just has to fill the gap. Cleared the
    // moment the LLM's first token actually arrives (below) or the turn
    // ends (finally, below) so a stray filler never fires after the real
    // response.
    //
    // Real call finding (2026-09-16): removing the isNodeEntry gate also
    // let a filler fire on the call's very FIRST turn — a caller picking up
    // the phone and hearing "Got it." before the agent has said a single
    // word, let alone before they've said anything themselves. There is
    // nothing to acknowledge yet; unlike goodbye (which follows a real
    // conversation), the call-opening turn has no context a filler could
    // plausibly be reacting to. Explicitly excluding just this one turn,
    // not reintroducing the old blanket isNodeEntry gate the goodbye fix
    // was for.
    let backchannelTimer = null;
    if (this.backchannelEnabled && !isCallOpening) {
      backchannelTimer = setTimeout(() => {
        backchannelTimer = null;
        if (Math.random() < this.backchannelFrequency) this._maybeSpeakBackchannel(turnId);
      }, this.backchannelDelayMs);
    }

    try {
      const stream = anthropic.messages.stream({
        model: this.llmModel,
        system: systemPrompt,
        max_tokens: 300,
        messages: this.history,
        ...(tools.length > 0 ? { tools } : {}),
      });

      stream.on('text', (delta) => {
        if (this.activeTurn !== turnId) return;
        if (!firstTokenAt) {
          firstTokenAt = Date.now();
          if (backchannelTimer) {
            clearTimeout(backchannelTimer);
            backchannelTimer = null;
          }
          if (this._latency?.turnStart) this._latency.llmFirstToken = firstTokenAt;
          console.log(`[call-loop] turn ${turnId} LLM TTFB: ${firstTokenAt - turnStartedAt}ms`);
        }
        assistantText += delta;
        chunker.push(delta);
      });

      const final = await stream.finalMessage();
      if (final.usage) this.cost.addLlmUsage(this.llmModel, final.usage.input_tokens, final.usage.output_tokens);
      // Real bug found via mystery-shopper: this used to live inside the
      // `if (this.activeTurn === turnId)` block below, alongside the
      // speaking/logging logic that's correctly gated on it (a superseded
      // turn shouldn't get to speak or log after the fact). But a recorded
      // field is real data the model already genuinely extracted from what
      // the caller said BEFORE any barge-in happened — interrupting the
      // agent's spoken acknowledgment of a fact doesn't make that fact
      // untrue. Gating this behind the same check as speaking meant a
      // barge-in (or anything else that moved activeTurn on) silently
      // discarded already-correct data: caught on a live call where the
      // caller gave their name, got barged over mid-reply, and the agent
      // asked for the name again two turns later because it was never
      // persisted. stream.finalMessage() already waited for the complete
      // response regardless of activeTurn, so this only needs to run once,
      // unconditionally, right here.
      for (const block of final.content) {
        if (block.type === 'tool_use' && block.name === 'record_field' && block.input?.field) {
          this.collectedData[block.input.field] = block.input.value;
          console.log(`[call-loop] recorded field "${block.input.field}" = "${block.input.value}"`);
        }
      }
      // Cycles 10-11 (mystery-shopper) finding: every "produced no speech"
      // fallback fired exactly when the model also called record_field in
      // the same turn — 100% correlation across two separate real calls,
      // even after rewording the prompt to explicitly say recording a
      // field must never substitute for a spoken reply. That the prompt
      // fix had zero effect points at a code-level cause instead: the
      // streaming 'text' event (which is all assistantText was ever built
      // from) may not fire for a text block that arrives after a tool_use
      // block in the same message. Cross-checking the final message
      // directly, once streaming is done, is a cheap way to confirm and
      // recover from that rather than falling back to a canned line when
      // real text was there all along.
      if (!assistantText) {
        const textBlock = final.content.find((b) => b.type === 'text' && b.text);
        if (textBlock) {
          console.warn(`[call-loop] turn ${turnId}: streamed text was empty but final.content had real text — using it instead of the generic fallback`);
          assistantText = textBlock.text;
          this._speak(assistantText, turnId, turnStartedAt);
        }
      }
      if (this.activeTurn === turnId) {
        // Safety net for a terminal node (goodbye/transfer): dead air there
        // is a much worse failure than anywhere else in the flow — it's the
        // caller's very last impression, or happens right as they're being
        // handed off. This used to fall back to node.prompt — the node's
        // own DEVELOPER-FACING instruction text (e.g. "Thank the caller for
        // calling and say a warm goodbye.") — spoken to the actual caller
        // verbatim, read literally as dialogue, immediately before hanging
        // up on them. Real bug, caught on a live call. node.prompt is an
        // instruction *for the model*, never customer-facing copy; a
        // generic but real spoken line is always safe here, unlike
        // vocalizing whatever a flow author happened to write as the
        // node's prompt.
        if (!assistantText && (node?.type === 'goodbye' || node?.type === 'transfer')) {
          console.warn(`[call-loop] node "${node.id}" (${node.type}) produced no speech — falling back to a generic line`);
          assistantText = node.type === 'goodbye'
            ? 'Thank you so much for calling. Have a great day!'
            : "I'm connecting you now — one moment please.";
          this._speak(assistantText, turnId, turnStartedAt);
        } else if (!assistantText && !final.content.some((b) => b.type === 'tool_use')) {
          // Cycle 9 finding: a non-terminal node occasionally produced
          // neither text nor a tool call, leaving dead silence with
          // nothing forcing the call forward — this fallback covers that.
          //
          // Cycle 12 finding (important refinement): this fallback was ALSO
          // firing whenever record_field was called with no accompanying
          // text — confirmed real Claude Haiku behavior (final.content
          // genuinely has no text block, not a missed-streaming-event bug;
          // see the check added above this block). In that case the field
          // WAS captured correctly — nothing was actually unclear — so
          // saying "Sorry, could you say that again?" is not just
          // unnecessary, it's actively wrong and makes a working system
          // sound broken (exactly what the judge flagged: "the system
          // committed to a booking on a repeat of an utterance it had just
          // declared unrecognizable"). Only use this apologetic fallback
          // when NO tool call happened either — genuine silence — not when
          // a tool call succeeded silently.
          console.warn(`[call-loop] node "${node?.id}" (${node?.type}) produced no speech and no tool call — falling back to a generic clarifying line`);
          assistantText = "Sorry, could you say that again?";
          this._speak(assistantText, turnId, turnStartedAt);
        } else if (!assistantText) {
          console.log(`[call-loop] node "${node?.id}" (${node?.type}) called a tool with no spoken text — letting it pass silently rather than falsely claiming something was unclear`);
          // Real bug found via mystery-shopper: this silent pass-through is
          // correct on its own (see the cycle-12 note above — saying "sorry,
          // could you repeat that" when data WAS captured is worse than
          // saying nothing), but the existing deadlock-nudge below only
          // fires once EVERY field for the node is captured. A turn that
          // captures only SOME fields (e.g. the caller gave a name and a
          // vague "tomorrow afternoon", the model records name but the
          // vague time isn't extractable as a field) still ends this turn
          // with nothing said and nothing asked — and if the caller has
          // nothing more to volunteer unprompted, both sides wait forever
          // with no captured-fields threshold ever met to trigger the
          // existing nudge. Mark it here so _maybeRetireTurn can nudge on
          // ANY silent turn, not just a fully-captured one.
          if (this.turnState?.id === turnId) this.turnState.saidNothing = true;
        }
        chunker.flush();
        if (assistantText) {
          this.history.push({ role: 'assistant', content: assistantText });
          // Only the CALLER's side was ever logged server-side (see "turn N
          // user:" below in _onUserTurnComplete) — meaning there was no way
          // to pull a full transcript for a past call to compare against
          // anything, e.g. Retell's own stored transcripts. Log our own
          // side too, same format, so `flyctl logs` has both halves of the
          // conversation.
          console.log(`[call-loop] [call ${this.callSid || this.id}] turn ${turnId} assistant: "${assistantText}"`);
        }
        // Real bug, caught on the first mystery-shopper run (see
        // MYSTERY_SHOPPER_DECISIONS.md decision 6): a flow-less shopper
        // session never hangs up on its own, and the business side often
        // says goodbye without proactively hanging up either (normal phone
        // etiquette — waiting for the caller to hang up). With neither side
        // ending the call, both looped exchanging "bye"/"take care" for
        // over a minute until the repetitive context caused the LLM to
        // break character and hallucinate a bizarre "let's do another
        // roleplay scenario" meta-conversation — on BOTH the Retell call
        // and our own, identically, confirming this is a shopper-design
        // bug, not something either backend did wrong. Fix: once the
        // shopper itself has said something closing-shaped twice, hang up
        // proactively instead of waiting on the other side.
        if (this.isShopper && assistantText) {
          const isClosing = CLOSING_SHAPED_RE.test(assistantText.trim());
          if (isClosing) {
            this._shopperClosingCount = (this._shopperClosingCount || 0) + 1;
            if (this._shopperClosingCount >= 2) {
              console.log('[call-loop] shopper: second closing-shaped reply, hanging up proactively');
              this._closing = true;
              setTimeout(() => this.close(), 2000);
            }
          }
        }
        // TTS-backend-independent observability — chunk_meta only exists on
        // the kokoro path, so a client (browser UI, or a headless test
        // harness like the flow MCP server) that wants "what did the
        // assistant actually say" needs a signal that doesn't depend on
        // which TTS backend this call happens to be using.
        this.send({ type: 'assistant_turn', turnId, text: assistantText, nodeId: this.currentNodeId });

        const toolUse = final.content.find((b) => b.type === 'tool_use' && b.name === 'transition_flow');
        if (toolUse && this.turnState?.id === turnId) {
          this.turnState.transition = toolUse.input;
        }
      }
    } catch (err) {
      console.error('[call-loop] LLM error', err);
      this.send({ type: 'error', message: 'LLM request failed' });
    } finally {
      if (backchannelTimer) clearTimeout(backchannelTimer);
      if (this.turnState?.id === turnId) {
        this.turnState.llmDone = true;
        this._maybeRetireTurn(turnId);
      }
    }
  }

  // Which cached filler clip (if any) applies for this session's current
  // backend+voice — see fillerCache/prewarmFillerCache above for how it's
  // populated. Backends with a single global env-configured voice
  // (elevenlabs/cartesia/minimax) look themselves up directly; kokoro's
  // voice varies per session so it isn't prewarmed, meaning a kokoro call
  // simply has no cached filler to play (see _maybeSpeakBackchannel).
  _fillerCacheKey(text) {
    const voice =
      this.ttsBackend === 'elevenlabs' ? ELEVENLABS_VOICE_ID :
      this.ttsBackend === 'cartesia' ? CARTESIA_VOICE_ID :
      this.ttsBackend === 'minimax' ? MINIMAX_VOICE_ID :
      this.voice;
    return `${this.ttsBackend}::${voice}::${text}`;
  }

  _maybeSpeakBackchannel(turnId) {
    if (this.activeTurn !== turnId) return; // barge-in or turn already resolved
    const word = this.backchannelWords[Math.floor(Math.random() * this.backchannelWords.length)];
    const buf = fillerCache.get(this._fillerCacheKey(word));
    // No cached clip for this backend/voice — skip rather than synthesize
    // live, which would be just as slow as the real response it's meant to
    // hide (see fillerCache's comment).
    if (!buf) return;
    console.log(`[call-loop] turn ${turnId} backchannel: "${word}"`);
    this._speakCached(buf, turnId).catch((err) => console.error('[call-loop] backchannel send failed', err));
  }

  // Sends a pre-synthesized filler clip immediately — no network call, so
  // this is the one "speak" path with zero added latency. Still
  // participates in the same pendingTts/turnState bookkeeping every other
  // TTS path uses (see _speak/_speakHttpTts) so barge-in and
  // _maybeRetireTurn stay correct even when a filler is the only thing a
  // superseded turn ever said.
  async _speakCached(buffer, turnId) {
    if (this.activeTurn !== turnId) return;
    if (this.turnState?.id === turnId) this.turnState.pendingTts++;
    // Same ordering chain _speakHttpTts uses — a cached clip (backchannel
    // filler, kokoro warmup line) sent straight to clientWs.send() with no
    // regard for _sendChain could jump ahead of or interleave with a
    // sentence still being flushed for another turn. Only a live risk now
    // that backchanneling is enabled by default again; claim a ticket the
    // same way so this can never race a real sentence's audio.
    const prior = this._sendChain || Promise.resolve();
    let releaseNext;
    this._sendChain = new Promise((resolve) => {
      releaseNext = resolve;
    });
    await prior;
    try {
      if (this.activeTurn !== turnId) return;
      if (this.clientWs.readyState !== WebSocket.OPEN) return;
      this.clientWs.send(buffer, { binary: true });
    } finally {
      releaseNext();
      if (this.turnState?.id === turnId) {
        this.turnState.pendingTts = Math.max(0, this.turnState.pendingTts - 1);
        this._maybeRetireTurn(turnId);
      }
    }
  }

  // Builds this turn's system prompt from just the current flow node's own
  // instructions — not the whole flow flattened into one wall of text — so
  // the model only ever has to reason about "what am I doing right now,"
  // with collected-so-far data and the node's real edges (via the
  // transition_flow tool) as its only additional context.
  // `isNodeEntry` is true only for the single turn that actually arrives at
  // this node (see _runNodeTurn) — every later turn spent still in the same
  // node (the caller said something that didn't trigger a transition yet)
  // passes false. This distinction matters: reproduced on a real call, a
  // node's own instructions (e.g. the greeting node's "Greet the caller")
  // read as a standing per-turn instruction rather than a one-time opening
  // action, so a caller saying something ambiguous ("Hello?") while still
  // in that node got a SECOND greeting-flavored reply right on top of the
  // first — not corrupted audio, just the model doing exactly what its
  // system prompt said to do, again. Once isNodeEntry is false, the prompt
  // explicitly says the opening line has already been given and not to
  // repeat it, leaning on conversation history (which already has it)
  // instead of restating the step instructions as if they were still owed.
  _buildNodeSystemPrompt(node, isNodeEntry) {
    const gs = this.flow?.globalSettings || {};
    // Real call finding (2026-09-16, live test call): the prompt never told
    // the model what day it actually is, so a relative date like "the day
    // after" or "tomorrow" could only be echoed back verbatim, never
    // resolved to a real calendar date — "Perfect, I've got you down for
    // the day after tomorrow" instead of "that's Thursday the 18th".
    // Anchoring today's date here fixes that for every node, not just
    // booking ones, since any step could reasonably need it.
    const todayStr = new Date().toLocaleDateString('en-US', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
      timeZone: gs.timezone || 'America/Los_Angeles', // no per-tenant timezone field exists yet; defaults to the server's own region
    });
    let prompt =
      `You are a concise, friendly voice assistant on a phone call, currently in the ` +
      `"${node.id}" step of a structured conversation flow.\n\n` +
      `Today's date is ${todayStr}. When the caller gives a relative date ("tomorrow", ` +
      `"the day after", "next Tuesday"), resolve it to a specific calendar date yourself and ` +
      `say the actual date out loud (e.g. "that's Thursday the 18th") — never just repeat the ` +
      `caller's relative phrasing back as if it were a booked date.\n\n`;
    if (isNodeEntry) {
      prompt += `Step instructions: ${node.prompt}\n`;
    } else {
      prompt +=
        `Step instructions (for context — you already acted on these; don't repeat your ` +
        `opening line for this step): ${node.prompt}\n` +
        `You've already given this step's opening line earlier in the conversation. Respond ` +
        `naturally to what the caller just said — don't greet them again or restate your ` +
        `opening line.\n`;
    }
    if (node.extract) {
      const fields = Object.keys(node.extract);
      const hasTimeField = fields.some((f) => /time|date|when/i.test(f));
      // Missing-fields tracking (cycle 18, MYSTERY_SHOPPER_PATTERNS.md):
      // the prior wording asked the model to infer "don't re-ask anything
      // already volunteered" from conversation history, with the actual
      // captured state (collectedData) dumped separately, after the
      // numbered steps, as a bare JSON fact with no directive tying it
      // back to step 1. Reproduced live, twice in one day: the model
      // re-asked for a field it had already correctly recorded via
      // record_field two turns earlier. Computing the missing set in code
      // and stating it as an explicit, closed list — instead of asking the
      // model to infer it — removes the inference step entirely rather
      // than adding yet another prose instruction on top (see the cycle
      // 7-9 instruction-overload note this block already exists to avoid).
      const missingFields = fields.filter((f) => !this.collectedData[f]);
      // Consolidated from what was 4 separately-appended paragraphs (see
      // MYSTERY_SHOPPER_PATTERNS.md's cycles 7-9 note on instruction
      // overload: each fix landed as one more paragraph stacked onto this
      // node's prompt, and by cycle 9 that volume itself was a plausible
      // cause of inconsistent tool-calling — e.g. record_field getting
      // called for one field but not another given in the same breath).
      // Rewritten as one ordered checklist instead of accreted paragraphs,
      // and a new final step (confirm the full summary before
      // transitioning) added directly from the cycle-9 judge's own
      // suggestion: closes without ever confirming a booking looked
      // unfinished even when the data was actually correct.
      prompt += `For this step, in order:\n`;
      if (missingFields.length > 0) {
        prompt += `1. You still need: ${missingFields.join(', ')}. Ask for ALL of these together in ONE question, in the SAME turn — don't ask one at a time. Do not ask about anything not in this list — it's already been captured (see "Already collected" below).\n`;
      } else {
        prompt += `1. Every field for this step is already captured (see "Already collected" below) — do not ask for any of them again. Move straight to confirming/summarizing.\n`;
      }
      prompt += `2. The moment the caller gives you a field, call record_field for it — but ALWAYS also say something out loud to the caller in that same turn. Calling record_field is a silent background action, never a substitute for actually replying — never let a turn consist of only a tool call with nothing spoken.\n`;
      if (hasTimeField) {
        prompt += `3. If a date/time answer is vague ("afternoon", "next week"), propose ONE concrete slot inside their range and get a yes before treating it as captured.\n`;
      }
      // Real call finding (2026-09-16): this step used to say "read back
      // what you captured before relying on it" — soft enough that a live
      // call skipped straight from record_field to a confident summary
      // ("Perfect, Vishan! I've got you down for...") on a name Deepgram
      // had actually mis-transcribed, with no yes/no question in between.
      // record_field only means the model heard SOMETHING, not that it's
      // correct — reworded as a hard requirement that's explicit about
      // what "confirmed" means, instead of leaving the model to treat its
      // own capture as sufficient.
      prompt += `${hasTimeField ? '4' : '3'}. Names and numbers are easy to mishear. Before treating any field as final, you MUST ask the caller a direct yes/no question repeating back exactly what you captured (e.g. "Got it, Alex, for 3pm — did I get that right?"). Calling record_field is NOT confirmation — it only means you heard something. Wait for the caller to actually say yes (or correct you) before moving on.\n`;
      prompt += `${hasTimeField ? '5' : '4'}. Only after the caller has explicitly confirmed every field this way, say ONE summary sentence with all of them ("So that's Alex Morgan at 2pm tomorrow.") and THEN call transition_flow in the same turn — don't transition silently, without a prior yes/no confirmation, or without ever stating the final summary.\n`;
    }
    if (Object.keys(this.collectedData).length > 0) {
      prompt += `Already collected this call (do NOT ask for these again): ${JSON.stringify(this.collectedData)}\n`;
    }
    if (node.edges.length > 0) {
      // Real call finding (2026-09-16): transitioning into an extraction
      // step (e.g. booking) doesn't itself trigger a proactive opening
      // question — only 'function'/'knowledge_base'/'goodbye'/'transfer'
      // nodes auto-speak on entry (see _applyTransition's AUTO_ADVANCE_TYPES).
      // A live call hit this directly: the caller said "I wanted to make an
      // appointment", the model replied with a bare acknowledgment ("Great!
      // I can help you book an appointment with us.") and silently called
      // transition_flow with nothing else asked — leaving the caller to
      // carry the conversation until they guessed to ask about availability.
      // Deliberately not adding extraction to auto-advance types to fix
      // this: some flows already have the CURRENT node ask the next step's
      // question itself before transitioning (seen working correctly in
      // earlier mystery-shopper runs), and an unconditional auto-turn on
      // every entry would double-ask in those cases — the same bug already
      // fixed today, in the opposite direction. Instead, make the
      // requirement explicit at the point of transitioning: never hand off
      // silently.
      prompt +=
        `\nWhen this step's goal has been met, call the transition_flow tool to move to the ` +
        `next step. If it hasn't been met yet, keep talking and don't call the tool. If you call ` +
        `transition_flow, your spoken reply in that SAME turn must move the conversation forward ` +
        `— never a bare acknowledgment with nothing else (e.g. never just "Great, I can help you ` +
        `book an appointment" with no question and no info). The caller should never be left ` +
        `wondering what to say next: if the next step needs information from the caller, ask for ` +
        `it yourself, in this same reply — don't just acknowledge and wait, since nothing else ` +
        `will proactively ask on your behalf.\n`;
    }
    // Real pattern found across mystery-shopper runs: a non-goodbye node
    // would sometimes phrase its own line as if the call were already
    // ending ("thanks for calling... have a great day"), and then the
    // ACTUAL goodbye node fired afterward and said a full second closing —
    // sounding like two goodbyes back to back even though no code path
    // literally spoke twice in one turn. Closing language belongs only to
    // the goodbye node itself.
    if (node.type !== 'goodbye') {
      prompt +=
        `Do not use closing/goodbye language in this step (e.g. "thanks for calling", "have a ` +
        `great day") — that belongs only to the call's actual final goodbye, which is a later ` +
        `step, not this one.\n`;
      // Related but distinct bug seen after the above fix landed: a
      // confirmation node re-asked "Does that work for you?" immediately
      // after the caller had already explicitly said yes to the same
      // thing — a wasted turn from re-confirming something not actually in
      // question anymore.
      prompt +=
        `If the caller has already clearly said yes/confirmed something, don't ask them to ` +
        `confirm it again — move on.\n`;
    } else {
      // The other half of this bug: even once a call correctly reaches the
      // real goodbye node, that node's OWN single turn was itself
      // internally redundant ("Thank you, Alex. Thanks so much for
      // calling, Alex.") — thanking twice and repeating the name twice in
      // one breath, which read as a script glitch rather than a person
      // speaking. This is a single-turn discipline problem, separate from
      // the cross-node issue fixed above.
      prompt +=
        `This is the final goodbye. Say it as ONE short, natural closing sentence: mention the ` +
        `caller's name at most once, thank them at most once, and say goodbye once. Do not stack ` +
        `multiple thank-yous or repeat their name within this line.\n`;
    }
    prompt += gs.allowInterruptions === false
      ? 'Complete your sentences before listening.\n'
      : 'Allow the caller to interrupt you.\n';
    prompt +=
      'Keep replies to 1-2 short sentences unless asked for more detail. Never use markdown, ' +
      'bullet points, or emoji — this is spoken audio. Always say at least one sentence out ' +
      'loud on every turn, even if you are also calling a tool — never respond with nothing.';
    return prompt;
  }

  // The current node's real edges become the tool's actual enum of valid
  // targets (and their natural-language conditions become the tool's
  // description) — the model picks one instead of us parsing free text to
  // guess where the conversation should go next.
  _buildTransitionTool(node) {
    const properties = {
      next_node_id: {
        type: 'string',
        enum: node.edges.map((e) => e.target),
        description: node.edges.map((e) => `${e.target}: ${e.condition}`).join('; '),
      },
    };
    if (node.extract) {
      properties.extracted = {
        type: 'object',
        properties: Object.fromEntries(Object.keys(node.extract).map((k) => [k, { type: 'string' }])),
        description: 'Fields the caller has actually provided during this step so far.',
      };
    }
    return {
      name: 'transition_flow',
      description: "Call this once this step's goal has been met and it's time to move to the next step in the flow.",
      input_schema: { type: 'object', properties, required: ['next_node_id'] },
    };
  }

  // Applied once the turn that produced this transition has fully finished
  // speaking (see _maybeRetireTurn) — not the instant the tool call arrives —
  // so the caller always hears the current node's full response before the
  // flow moves on.
  _applyTransition({ next_node_id, extracted }) {
    if (extracted && typeof extracted === 'object') {
      Object.assign(this.collectedData, extracted);
    }
    const nextNode = this.flowNodesById.get(next_node_id);
    if (!nextNode) {
      console.warn(`[call-loop] flow transition to unknown node "${next_node_id}" — ignoring`);
      return;
    }
    // Billable event = actually completing the node's job (leaving it via a
    // real transition), not just visiting it — matches flowBuilder.ts's
    // fixed node ids ('booking'/'take_message') for the wizard-built flows.
    // A custom flow that reuses these ids for something else would also bill
    // here; that's an acceptable trade-off for not needing a separate
    // "billable" flag on the node schema yet.
    if (this.currentNodeId === 'booking') this.cost.addBillableEvent('booking');
    else if (this.currentNodeId === 'take_message') this.cost.addBillableEvent('message');
    console.log(`[call-loop] flow transition -> "${next_node_id}" (${nextNode.type})`);
    this.send({ type: 'flow_state', currentNodeId: next_node_id, nodeType: nextNode.type, collectedData: this.collectedData });
    // These node types act as soon as the flow enters them (run a webhook,
    // look something up, say goodbye and hang up, transfer the call) rather
    // than waiting for the caller to speak first.
    const AUTO_ADVANCE_TYPES = new Set(['function', 'knowledge_base', 'goodbye', 'transfer', 'payment']);
    if (AUTO_ADVANCE_TYPES.has(nextNode.type)) {
      this._runNodeTurn(next_node_id);
    } else {
      this.currentNodeId = next_node_id;
    }
  }

  // Runs a flow 'function' node's configured webhook and folds the result
  // into history as a system-style note before the node's own turn is
  // generated, so the model's dialogue and next transition are actually
  // informed by what the webhook returned instead of the model guessing.
  // Was previously a disclosed gap: a knowledge_base node ran as a plain
  // prompt with nothing behind it — no real Supabase lookup. node.params.
  // knowledgeBaseId is stamped on by tenantLookup.js (resolved from the
  // agent's actual calldesk_knowledge_bases row, since the node itself has
  // no column for it); folded into history the same way a function node's
  // webhook result is, so the model answers from real content instead of
  // guessing.
  async _executeKnowledgeBaseNode(node) {
    const knowledgeBaseId = node.params?.knowledgeBaseId;
    if (!knowledgeBaseId) {
      console.warn(`[call-loop] knowledge_base node "${node.id}" has no knowledgeBaseId — running with no KB content`);
      return;
    }
    const items = await fetchKnowledgeItems(knowledgeBaseId);
    if (items.length === 0) {
      console.warn(`[call-loop] knowledge_base node "${node.id}" — KB ${knowledgeBaseId} has no items`);
      return;
    }
    const qa = items.map((it, i) => `${i + 1}. Q: ${it.question}\n   A: ${it.answer}`).join('\n');
    this.history.push({
      role: 'user',
      content: `[System note: knowledge base content for this step — answer using these facts when relevant, otherwise say you're not sure and offer to have someone follow up:\n${qa}]`,
    });
  }

  async _executeFunctionNode(node) {
    const url = node.params?.webhookUrl;
    if (!url) {
      console.warn(`[call-loop] function node "${node.id}" has no params.webhookUrl — skipping call`);
      return;
    }
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ function: node.function, collectedData: this.collectedData }),
        signal: AbortSignal.timeout(8000),
      });
      const result = await res.json().catch(() => ({}));
      console.log(`[call-loop] function node "${node.id}" (${node.function}) -> HTTP ${res.status}`);
      this.history.push({
        role: 'user',
        content: `[System note: function "${node.function}" returned ${JSON.stringify(result)}]`,
      });
    } catch (err) {
      console.error(`[call-loop] function node "${node.id}" webhook failed`, err);
      this.history.push({
        role: 'user',
        content: `[System note: function "${node.function}" failed — let the caller know something went wrong and offer to have someone follow up]`,
      });
    }
  }

  // Redirects the live call via Twilio's REST API — only possible on a real
  // phone call (TwilioCallAdapter exposes .callSid; a browser call has
  // nothing to redirect) and only with Twilio credentials configured.
  // Otherwise falls back to hanging up rather than silently doing nothing.
  async _executeTransfer(params) {
    const to = params?.transferTo;
    const callSid = this.clientWs?.callSid;
    if (!to || !callSid || !TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) {
      console.warn(
        `[call-loop] transfer requested but cannot complete it (to=${to || 'unset'}, ` +
        `callSid=${callSid ? 'present' : 'missing — not a Twilio call?'}, ` +
        `twilioCredsConfigured=${Boolean(TWILIO_ACCOUNT_SID && TWILIO_AUTH_TOKEN)}) — hanging up instead`
      );
      this.close();
      return;
    }
    try {
      const twiml = `<?xml version="1.0" encoding="UTF-8"?><Response><Dial>${to}</Dial></Response>`;
      const auth = Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64');
      const res = await fetch(
        `https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Calls/${callSid}.json`,
        {
          method: 'POST',
          headers: { Authorization: `Basic ${auth}`, 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ Twiml: twiml }).toString(),
        }
      );
      console.log(`[call-loop] transfer -> ${to} (Twilio responded HTTP ${res.status})`);
      if (res.ok) this.cost.addBillableEvent('transfer');
      // Twilio's own <Dial> now owns the call — our media-stream WS leg will
      // get a 'stop' event and close() normally once that dial ends.
    } catch (err) {
      console.error('[call-loop] transfer failed', err);
      this.close();
    }
  }

  // Redirects the live call to different TwiML the same way _executeTransfer
  // does, but for a detour we expect the call to come BACK from (e.g. Pay),
  // not hand off permanently. Stashes this session's in-progress state
  // first so the reconnected stream can rehydrate it — see
  // pendingResumeSessions and _tryResumeFromCallSid.
  //
  // Deliberately generic (redirectTwiml is caller-supplied, not hardcoded
  // to Pay) — this is the reusable primitive; the payment-node type that
  // would actually call this with a <Pay> TwiML string isn't built yet
  // (see the 2026-09-16 Twilio <Pay> design notes) pending confirming the
  // exact redirect/reconnect and recording-suppression behavior directly
  // against Twilio, not assumed from secondary sources.
  async _redirectForDetour(redirectTwiml) {
    const callSid = this.clientWs?.callSid;
    if (!callSid || !TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) {
      console.warn(
        `[call-loop] detour redirect requested but cannot complete it (callSid=${callSid ? 'present' : 'missing'}, ` +
        `twilioCredsConfigured=${Boolean(TWILIO_ACCOUNT_SID && TWILIO_AUTH_TOKEN)})`
      );
      return false;
    }
    this._stashForResume(callSid);
    try {
      const auth = Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64');
      const res = await fetch(
        `https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Calls/${callSid}.json`,
        {
          method: 'POST',
          headers: { Authorization: `Basic ${auth}`, 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ Twiml: redirectTwiml }).toString(),
        }
      );
      console.log(`[call-loop] detour redirect for ${callSid} -> Twilio responded HTTP ${res.status}`);
      if (!res.ok) {
        // The redirect itself failed — nothing to resume into, so don't
        // leave a stashed entry that'll just sit until RESUME_TTL_MS expires.
        pendingResumeSessions.delete(callSid);
      }
      return res.ok;
      // Our media-stream WS leg gets a 'stop' event now, same as a real
      // hangup — close() checks _pausedForResume (set by _stashForResume)
      // to skip billing/registry finalization for this non-final close.
    } catch (err) {
      console.error('[call-loop] detour redirect failed', err);
      pendingResumeSessions.delete(callSid);
      return false;
    }
  }

  // Real finding (2026-09-17, MYSTERY_SHOPPER-adjacent research via the
  // Twilio MCP): tried triggering Payments purely via REST
  // (POST .../Calls/{CallSid}/Payments.json) against a call already
  // connected via <Connect><Stream>, hoping to skip a TwiML detour
  // entirely — got error 21220 ("call not in the expected state") even
  // against a genuinely in-progress call. The REST Payments resource
  // appears to control/update a Pay session already running because the
  // call's TwiML is actively executing <Pay>, not to cold-start one
  // independent of TwiML. So this goes through _redirectForDetour after
  // all, same as a real <Pay> verb would need.
  //
  // chargeAmount omitted/0 tokenizes only (see CreatePayments' own
  // description) — a real charge needs params.amount set explicitly by
  // whoever authors this flow node, not a default.
  async _executePayment(params) {
    const callSid = this.clientWs?.callSid;
    if (!callSid) {
      console.warn('[call-loop] payment node requires a Twilio call (no callSid) — skipping');
      return;
    }
    const amount = params?.amount ? Number(params.amount) : 0;
    const connector = params?.paymentConnector || 'Default';
    const description = params?.description || '';
    const payTwiml =
      `<?xml version="1.0" encoding="UTF-8"?><Response><Pay ` +
      `chargeAmount="${amount}" paymentConnector="${connector}" ` +
      `${description ? `description="${description}" ` : ''}` +
      `action="https://${PUBLIC_HOST}/twilio/pay-result?callSid=${callSid}" /></Response>`;
    const ok = await this._redirectForDetour(payTwiml);
    if (!ok) {
      console.error(`[call-loop] payment redirect failed for ${callSid} — hanging up rather than leaving the call stuck`);
      this.close();
    }
  }

  // Captures everything a resumed session needs to continue the same
  // conversation rather than start fresh — stored as live object
  // references (this Map never leaves process memory) so there's no
  // serialization shape to keep in sync with CallSession's own fields.
  _stashForResume(callSid) {
    this._pausedForResume = true;
    pendingResumeSessions.set(callSid, {
      history: this.history,
      collectedData: this.collectedData,
      currentNodeId: this.currentNodeId,
      flow: this.flow,
      flowNodesById: this.flowNodesById,
      cost: this.cost, // same tracker instance carries forward — billed once, on the real final close()
      tenantId: this.tenantId,
      phoneNumber: this.phoneNumber,
      stripeCustomerId: this.stripeCustomerId,
      ttsBackend: this.ttsBackend,
      ttsModel: this.ttsModel,
      llmModel: this.llmModel,
      voice: this.voice,
      systemPrompt: this.systemPrompt,
      backchannelEnabled: this.backchannelEnabled,
      backchannelFrequency: this.backchannelFrequency,
      backchannelDelayMs: this.backchannelDelayMs,
      backchannelWords: this.backchannelWords,
      turnSeq: this.turnSeq,
      stashedAt: Date.now(),
    });
  }

  // Called from the Twilio 'start' handler before falling back to the
  // normal fresh-session path — returns true and rehydrates `this` if a
  // resume is pending for this CallSid, false otherwise (an ordinary new
  // call, or a resume that expired before Twilio reconnected — e.g. the
  // caller hung up mid-Pay and never came back).
  _tryResumeFromCallSid(callSid) {
    const stashed = pendingResumeSessions.get(callSid);
    if (!stashed) return false;
    pendingResumeSessions.delete(callSid);
    if (Date.now() - stashed.stashedAt > RESUME_TTL_MS) {
      console.warn(`[call-loop] resume for ${callSid} expired (stashed ${Date.now() - stashed.stashedAt}ms ago) — starting fresh instead`);
      return false;
    }
    Object.assign(this, {
      history: stashed.history,
      collectedData: stashed.collectedData,
      currentNodeId: stashed.currentNodeId,
      flow: stashed.flow,
      flowNodesById: stashed.flowNodesById,
      cost: stashed.cost,
      tenantId: stashed.tenantId,
      phoneNumber: stashed.phoneNumber,
      stripeCustomerId: stashed.stripeCustomerId,
      ttsBackend: stashed.ttsBackend,
      ttsModel: stashed.ttsModel,
      llmModel: stashed.llmModel,
      voice: stashed.voice,
      systemPrompt: stashed.systemPrompt,
      backchannelEnabled: stashed.backchannelEnabled,
      backchannelFrequency: stashed.backchannelFrequency,
      backchannelDelayMs: stashed.backchannelDelayMs,
      backchannelWords: stashed.backchannelWords,
      turnSeq: stashed.turnSeq,
    });
    console.log(`[call-loop] resumed session for ${callSid} at node "${this.currentNodeId}" (${this.history.length} history entries carried over)`);
    return true;
  }

  // A turn only stops being "active" (eligible for barge-in) once the LLM
  // has finished generating AND every TTS chunk it dispatched has finished
  // playing/erroring/being cancelled — otherwise a stray VAD blip after the
  // conversation has gone quiet reads as a spurious barge-in on nothing.
  _maybeRetireTurn(turnId) {
    if (this.turnState?.id !== turnId) return;
    if (this.turnState.llmDone && this.turnState.pendingTts <= 0 && this.activeTurn === turnId) {
      this.activeTurn = 0;
      // Flow side effects run only after the caller has fully heard this
      // node's response — a transition, hangup, or transfer landing mid-TTS
      // would cut the assistant off on its own words.
      if (this.turnState.nodeType === 'goodbye') {
        console.log('[call-loop] flow reached goodbye node — hanging up');
        this._closing = true;
        this.close();
        return;
      }
      if (this.turnState.nodeType === 'transfer') {
        this._executeTransfer(this.turnState.nodeParams);
        return;
      }
      if (this.turnState.nodeType === 'payment') {
        this._executePayment(this.turnState.nodeParams);
        return;
      }
      if (this.turnState.transition) {
        this._applyTransition(this.turnState.transition);
        return;
      }
      // See _onUserTurnComplete's in-flight-turn guard: the caller may have
      // said something new while this turn's LLM call was still running.
      // Answer it now, before considering an artificial nudge below — real
      // caller input always beats a synthetic one. (Known gap: this only
      // covers the plain fallthrough case, i.e. this turn was ordinary
      // back-and-forth with no transition/goodbye/transfer — the rarer case
      // of queued text arriving exactly around a flow transition isn't
      // handled here yet and needs its own follow-up.)
      if (this._queuedUserText) {
        const queued = this._queuedUserText;
        this._queuedUserText = null;
        this._onUserTurnComplete(queued);
        return;
      }
      // Diagnostic-driven fix (found via mystery-shopper cycle 14, raw
      // logs): the model recorded every required field via record_field —
      // satisfying the node's transition condition — but called neither
      // transition_flow nor said anything else in that turn. With no new
      // caller utterance coming (the caller had nothing left to add,
      // correctly), NOTHING was left to prompt the model to continue —
      // both sides silently waited on each other forever, only ending via
      // the 3-minute safety cap. This is a real deadlock, not something
      // more prompt wording can fix reliably on its own — nudge the model
      // to continue immediately instead of waiting on the caller.
      const node = this.flowNodesById?.get(this.currentNodeId);
      if (node?.extract && node.edges.length > 0) {
        const fields = Object.keys(node.extract);
        const allCaptured = fields.every((f) => this.collectedData[f]);
        // See the "saidNothing" note above (_generateTurn's silent-tool-call
        // branch): a turn that captures only SOME fields and says nothing
        // deadlocks exactly like the all-captured case below, just without
        // ever meeting allCaptured — the caller has nothing left to
        // volunteer unprompted, so nobody ever nudges the model to ask for
        // what's still missing. Nudge on either condition.
        const saidNothing = this.turnState.saidNothing;
        // Real bug found running this exact fix live: using allCaptured as
        // an independent trigger (alongside saidNothing) meant that once
        // every field was captured, EVERY future turn retirement on this
        // node re-entered this branch — including completely normal turns
        // where the model spoke just fine and simply hadn't transitioned
        // yet. Combined with the bounded-attempts fallback below, that
        // turned into a runaway loop: the canned fallback line fired on
        // nearly every turn for the rest of the call, drowning out the
        // model's real (perfectly fine) replies, because allCaptured never
        // stops being true once it's true. saidNothing is per-turn and
        // always resets to false on a new turnState, so gating on it alone
        // — using allCaptured only to pick which message to say, not as an
        // independent trigger — means this can only ever fire on a turn
        // that genuinely just said nothing, never on an ordinary one.
        if (saidNothing) {
          // Real bug found via mystery-shopper: a ONE-SHOT "once per node"
          // guard (a bare nodeId flag) meant that if the model went silent
          // AGAIN on the nudge turn itself — genuinely reproduced on a live
          // call: nudged, model recorded the remaining field but still said
          // nothing — there was no nudge left to fire a second time, and the
          // call deadlocked for real, only ending 2+ minutes later via the
          // shopper's hard safety-timeout, not any actual conversational
          // resolution. Bounded retries fix the immediate deadlock; the
          // final fallback guarantees SOMETHING is always said rather than
          // silently giving up after the cap.
          const attempts = this._nudgeAttempts.get(this.currentNodeId) || 0;
          if (attempts < MAX_NUDGE_ATTEMPTS) {
            this._nudgeAttempts.set(this.currentNodeId, attempts + 1);
            console.log(
              `[call-loop] node "${node.id}" ${allCaptured ? 'has all fields captured but didn\'t transition' : 'ended a silent turn with fields still missing'} — nudging (attempt ${attempts + 1}/${MAX_NUDGE_ATTEMPTS})`
            );
            this.history.push({
              role: 'user',
              content: allCaptured
                ? `[System note: every required field for this step has been captured. Say your one-sentence summary now and call transition_flow.]`
                : `[System note: you just recorded a field but said nothing out loud. Continue now — acknowledge what you captured and ask for whatever is still missing, in one spoken sentence.]`,
            });
            const nudgeTurnId = ++this.turnSeq;
            this.activeTurn = nudgeTurnId;
            this.turnState = { id: nudgeTurnId, llmDone: false, pendingTts: 0, startedSpeaking: false, saidNothing: false };
            this._generateTurn(nudgeTurnId, Date.now(), { isNodeEntry: false });
          } else {
            console.warn(`[call-loop] node "${node.id}" still silent after ${MAX_NUDGE_ATTEMPTS} nudges — forcing a spoken fallback instead of deadlocking`);
            // Real bug found via mystery-shopper: allCaptured only means
            // every field HAS a value, not that the caller actually
            // confirmed a concrete one — a vague "tomorrow afternoon"
            // satisfies it just as well as an agreed "2pm tomorrow". The
            // old "I've got you down" phrasing asserted a confirmed booking
            // that, per the transcript, was never actually agreed to — the
            // judge correctly called this a fabricated confirmation. Read
            // the values back as a question instead of asserting them as
            // fact, whether or not every field technically has a value.
            const missing = fields.filter((f) => !this.collectedData[f]);
            const fallback = missing.length > 0
              ? `Sorry, I want to make sure I get this right — could you tell me ${missing.join(' and ')} one more time?`
              : `Sorry, let me just double check — ${fields.map((f) => `${f.replace(/_/g, ' ')}: ${this.collectedData[f]}`).join(', ')}. Is that all correct?`;
            const fallbackTurnId = ++this.turnSeq;
            this.activeTurn = fallbackTurnId;
            this.turnState = { id: fallbackTurnId, llmDone: true, pendingTts: 0, startedSpeaking: false, saidNothing: false };
            this.history.push({ role: 'assistant', content: fallback });
            this._speak(fallback, fallbackTurnId, Date.now());
          }
        }
      }
    }
  }

  _ensureTtsSocket() {
    if (this.ttsWs && this.ttsWs.readyState === WebSocket.OPEN) return this.ttsWs;
    // Only the cloud gateway (realtime-tts-gateway.fly.dev) requires this —
    // a local dev worker (ws://127.0.0.1:8080/tts) has no auth at all, so
    // this is a no-op there when the env var is unset.
    const ws = new WebSocket(
      TTS_GATEWAY_WS_URL,
      TTS_GATEWAY_API_KEY ? { headers: { Authorization: `Bearer ${TTS_GATEWAY_API_KEY}` } } : undefined
    );
    ws.binaryType = 'arraybuffer';
    ws.on('open', () => console.log('[call-loop] tts gateway connected'));
    ws.on('error', (err) => console.error('[call-loop] tts gateway error', err));
    ws.on('close', () => {
      console.log('[call-loop] tts gateway closed');
      if (this.ttsWs === ws) this.ttsWs = null;
    });
    ws.on('message', (data, isBinary) => this._onTtsMessage(data, isBinary));
    this.ttsWs = ws;
    return ws;
  }

  _speak(text, turnId, turnStartedAt) {
    this.cost.addTtsChars(text.length);
    // Reserve this turn's "still speaking" slot immediately, synchronously —
    // not inside a possibly-deferred dispatch. On a session's first turn,
    // the TTS connection has to open/request fresh, and if the counter only
    // incremented once that finished, a fast LLM response could see
    // pendingTts still at 0 in between, conclude the turn was already fully
    // done, and reset activeTurn — causing dispatch to then silently drop
    // audio that was actually still in flight.
    if (this.turnState?.id === turnId) this.turnState.pendingTts++;

    if (this.ttsBackend === 'elevenlabs') {
      this._speakElevenLabs(text, turnId, turnStartedAt);
      return;
    }
    if (this.ttsBackend === 'cartesia') {
      this._speakCartesia(text, turnId, turnStartedAt);
      return;
    }
    if (this.ttsBackend === 'minimax') {
      this._speakMinimax(text, turnId, turnStartedAt);
      return;
    }

    const ws = this._ensureTtsSocket();
    // Cold-start mask — see KOKORO_WARMUP_PHRASE's comment above. Only
    // fires once per session (_warmupPlayed), and only if the gateway
    // genuinely isn't ready yet — a warm gateway (the common case once
    // Modal's instance has been serving traffic for a while) never
    // triggers this at all.
    if (!this._warmupPlayed && ws.readyState !== WebSocket.OPEN) {
      this._warmupPlayed = true;
      const warmupBuf = fillerCache.get(`warmup::elevenlabs::${ELEVENLABS_VOICE_ID}::${KOKORO_WARMUP_PHRASE}`);
      if (warmupBuf) {
        console.log(`[call-loop] turn ${turnId}: kokoro gateway not ready yet, playing warmup line`);
        this._speakCached(warmupBuf, turnId).catch((err) => console.error('[call-loop] warmup send failed', err));
      }
    }
    const dispatch = () => {
      if (this.activeTurn !== turnId) {
        console.log(`[call-loop] turn ${turnId} dropped before dispatch — activeTurn is now ${this.activeTurn} (superseded while waiting for TTS socket)`);
        if (this.turnState?.id === turnId) this.turnState.pendingTts = Math.max(0, this.turnState.pendingTts - 1);
        return;
      }
      this._pendingTurnStart = this._pendingTurnStart || turnStartedAt;
      ws.send(JSON.stringify({ type: 'synthesize', text, voice: this.voice, speed: 1.0 }));
    };
    if (ws.readyState === WebSocket.OPEN) dispatch();
    else ws.once('open', dispatch);
  }

  // Shared plumbing for every HTTP-based TTS backend (elevenlabs, cartesia,
  // minimax): manage this turn's AbortController and forward audio down the
  // same wire format our own TTS gateway emits (PCM16LE mono 24kHz — see
  // _onTtsMessage's binary branch, which does the same forwarding job for
  // the gateway) AS IT ARRIVES rather than buffering the whole response
  // first.
  //
  // This used to buffer everything before sending a single frame, on the
  // theory that downstream resampling (twilioAdapter.js's resampleInt16,
  // 24kHz -> 8kHz for real phone calls) is a stateful decimation that can't
  // restart per chunk. That reasoning doesn't hold for this specific ratio:
  // 24000/8000 is an exact integer (3), so "take every 3rd sample" has no
  // fractional phase to carry between chunks — each chunk resamples
  // correctly on its own as long as it's trimmed to a whole number of
  // 16-bit samples (handled below via `carry`), which twilioAdapter's own
  // frame-pacing queue already supports being fed incrementally (see
  // _sendMediaFrames/_startPacing there). Measured cost of the old
  // buffer-everything approach: 150-950ms of pure added latency per turn
  // (see the [timing] log breakdown this file's git history has — the
  // provider's own connect+TTFB was consistently ~100-400ms, matching
  // Retell's own reported ElevenLabs TTS-leg latency of ~150-170ms; the
  // rest was us waiting to finish downloading the full response).
  //
  // `fetchPcm` does the provider-specific request and calls `onChunk(buf)`
  // for each piece of audio as it arrives (checking `turnId` against
  // `this.activeTurn` itself wherever it can bail out early on barge-in).
  // `format` is 'pcm16' (default — PCM16LE @ 24kHz, what the browser client
  // and the kokoro gateway both speak) or 'mulaw8k' (mu-law @ 8kHz,
  // requested directly from the provider for a real Twilio call — see
  // _speakElevenLabs/_speakCartesia/_speakMinimax). mu-law is 1 byte per
  // sample, so unlike pcm16 there's no multi-byte alignment to preserve
  // across chunk boundaries — every byte is independently valid, and
  // TwilioCallAdapter forwards it straight through with no resampling at
  // all (see its send()). pcm16 still needs the 6-byte (3-sample) carry
  // alignment: a chunk boundary landing mid-decimation-group would shift
  // twilioAdapter's naive resampleInt16 phase for everything after it.
  // Sentence-level ordering (see SentenceChunker's callback in _runLlmTurn):
  // one _speakHttpTts call per sentence, fired as each completes during LLM
  // streaming, NOT awaited by its caller — so with a fast-enough TTS
  // provider, sentence 2's fetch can genuinely finish and start writing to
  // clientWs before sentence 1's does. That never showed up against
  // ElevenLabs (its ~1-1.4s leg time meant sentence N's stream reliably
  // finished long before sentence N+1 was even chunked), but it's a real,
  // audible bug against Cartesia (~150-380ms) — heard on a real call as
  // overlapping "hi there"s and swallowed words. Fetches stay concurrent
  // (kept unbuffered/streamed within a sentence, so first-byte latency for
  // the turn's opening sentence is unaffected), but writes to clientWs are
  // serialized through this chain so sentence N+1's audio can never reach
  // the client before sentence N's has fully gone out.
  async _speakHttpTts(label, fetchPcm, text, turnId, turnStartedAt, format = 'pcm16') {
    if (!this._httpTtsAborts) this._httpTtsAborts = new Set();
    const controller = new AbortController();
    this._httpTtsAborts.add(controller);

    // Claim this sentence's place in the turn's send order SYNCHRONOUSLY,
    // before the fetch below even starts — ordering must reflect call order
    // (the order SentenceChunker detected sentences in), not fetch-
    // completion order. Claiming this after the fetch (an earlier version
    // of this fix did exactly that) let a short, fast-to-synthesize later
    // sentence jump ahead of a longer earlier one whenever it happened to
    // finish first — heard on a real call as "How can I help you today?"
    // playing before "Hi there, thanks for calling CallDeskTech."
    const prior = this._sendChain || Promise.resolve();
    let releaseNext;
    this._sendChain = new Promise((resolve) => {
      releaseNext = resolve;
    });

    let carry = Buffer.alloc(0);
    let firstByteAt = null;
    const chunks = [];

    const onChunk = (chunk) => {
      if (this.activeTurn !== turnId) return; // barge-in — stop forwarding
      let buf = chunk;
      if (format === 'pcm16') {
        buf = carry.length > 0 ? Buffer.concat([carry, chunk]) : chunk;
        const remainder = buf.length % 6;
        if (remainder !== 0) {
          carry = Buffer.from(buf.subarray(buf.length - remainder));
          buf = buf.subarray(0, buf.length - remainder);
        } else {
          carry = Buffer.alloc(0);
        }
      }
      if (buf.length === 0) return;
      chunks.push(buf);
    };

    try {
      await fetchPcm(text, controller.signal, turnId, onChunk);
    } catch (err) {
      if (err.name !== 'AbortError') console.error(`[call-loop] ${label} stream error`, err);
    }

    // Fetch may have finished well before an earlier sentence's flush did —
    // wait for it (already complete at this point, so this is just
    // queueing, no extra network wait) before this sentence's audio goes
    // out, so playback order always matches call order.
    await prior;
    try {
      for (const buf of chunks) {
        if (this.activeTurn !== turnId) break; // barge-in — stop forwarding
        if (this.clientWs.readyState !== WebSocket.OPEN) break;
        if (firstByteAt === null) {
          firstByteAt = Date.now();
          this._markTtsFirstByte(turnId, label, firstByteAt);
          console.log(`[call-loop] TTS TTFB: ${firstByteAt - turnStartedAt}ms (turn latency end-to-end, ${label})`);
        }
        // See _onUserTurnComplete's in-flight-turn guard: real audio has now
        // gone out for this turn, so a new caller utterance from here on is
        // a genuine (would-be) barge-in, not a race against a turn that
        // hasn't said anything yet.
        if (this.turnState?.id === turnId) this.turnState.startedSpeaking = true;
        this.clientWs.send(buf, { binary: true, format: format === 'mulaw8k' ? 'mulaw8k' : undefined });
      }
    } finally {
      releaseNext();
      this._httpTtsAborts.delete(controller);
      if (this.turnState?.id === turnId) {
        this.turnState.pendingTts = Math.max(0, this.turnState.pendingTts - 1);
        this._maybeRetireTurn(turnId);
      }
    }
  }

  // ElevenLabs streaming. Requests ulaw_8000 directly for a real Twilio
  // call instead of our own pcm_24000 + naive resample — see
  // TwilioCallAdapter.send()'s 'mulaw8k' branch for why: nearest-neighbor
  // decimation with no anti-aliasing filter audibly corrupts real speech,
  // and this sidesteps it entirely by letting ElevenLabs's own (properly
  // filtered) resampler produce 8kHz mu-law directly. Forwards each chunk
  // to onChunk as it arrives instead of buffering the full response.
  _speakElevenLabs(text, turnId, turnStartedAt) {
    const isTwilio = this.clientWs instanceof TwilioCallAdapter;
    const format = isTwilio ? 'mulaw8k' : 'pcm16';
    const outputFormat = isTwilio ? 'ulaw_8000' : 'pcm_24000';
    this._speakHttpTts('elevenlabs', async (text, signal, turnId, onChunk) => {
      const res = await fetch(
        `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}/stream?output_format=${outputFormat}`,
        {
          method: 'POST',
          headers: { 'xi-api-key': ELEVENLABS_API_KEY, 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            model_id: this.ttsModel || ELEVENLABS_MODEL,
            voice_settings: { stability: 0.5, similarity_boost: 0.75 },
          }),
          signal,
        }
      );
      if (!res.ok || !res.body) {
        console.error(`[call-loop] ElevenLabs request failed: ${res.status}`);
        return;
      }
      for await (const chunk of res.body) {
        if (this.activeTurn !== turnId) break; // barge-in mid-stream
        onChunk(Buffer.from(chunk));
      }
    }, text, turnId, turnStartedAt, format);
  }

  // Cartesia's TTS-bytes endpoint — same shape as ElevenLabs's stream
  // endpoint (single POST, raw audio bytes back), just a different body
  // schema. See docs.cartesia.ai/api-reference/tts/bytes. Requests
  // pcm_mulaw @ 8000 directly for a real Twilio call — same reasoning as
  // ElevenLabs above.
  _speakCartesia(text, turnId, turnStartedAt) {
    const isTwilio = this.clientWs instanceof TwilioCallAdapter;
    const format = isTwilio ? 'mulaw8k' : 'pcm16';
    const outputFormat = isTwilio
      ? { container: 'raw', encoding: 'pcm_mulaw', sample_rate: 8000 }
      : { container: 'raw', encoding: 'pcm_s16le', sample_rate: 24000 };
    this._speakHttpTts('cartesia', async (text, signal, turnId, onChunk) => {
      const res = await fetch('https://api.cartesia.ai/tts/bytes', {
        method: 'POST',
        headers: {
          'Cartesia-Version': '2026-08-14',
          Authorization: `Bearer ${CARTESIA_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: this.ttsModel || CARTESIA_MODEL,
          transcript: text,
          voice: { id: CARTESIA_VOICE_ID },
          output_format: outputFormat,
        }),
        signal,
      });
      if (!res.ok || !res.body) {
        console.error(`[call-loop] Cartesia request failed: ${res.status}`);
        return;
      }
      for await (const chunk of res.body) {
        if (this.activeTurn !== turnId) break; // barge-in mid-stream
        onChunk(Buffer.from(chunk));
      }
    }, text, turnId, turnStartedAt, format);
  }

  // MiniMax's T2A v2 endpoint — non-streaming (stream: false): it returns
  // one JSON object with the full utterance's audio hex-encoded in
  // data.audio, not a raw byte stream, so there's no per-chunk barge-in
  // check to make (nothing to check partway through — the request either
  // completes or it doesn't). See platform.minimax.io/docs/api-reference/
  // speech-t2a-http. Requests pcmu_raw (G.711 mu-law, fixed 8kHz) directly
  // for a real Twilio call — same reasoning as ElevenLabs/Cartesia above.
  _speakMinimax(text, turnId, turnStartedAt) {
    const isTwilio = this.clientWs instanceof TwilioCallAdapter;
    const format = isTwilio ? 'mulaw8k' : 'pcm16';
    const audioSetting = isTwilio
      ? { sample_rate: 8000, format: 'pcmu_raw', channel: 1 }
      : { sample_rate: 24000, format: 'pcm', channel: 1 };
    this._speakHttpTts('minimax', async (text, signal, turnId, onChunk) => {
      const res = await fetch(`https://api-uw.minimax.io/v1/t2a_v2?GroupId=${encodeURIComponent(MINIMAX_GROUP_ID)}`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${MINIMAX_API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: MINIMAX_MODEL,
          text,
          stream: false,
          output_format: 'hex',
          voice_setting: { voice_id: MINIMAX_VOICE_ID, speed: 1.0, vol: 1.0, pitch: 0 },
          audio_setting: audioSetting,
        }),
        signal,
      });
      if (!res.ok) {
        console.error(`[call-loop] MiniMax request failed: ${res.status}`);
        return;
      }
      const body = await res.json();
      if (body.base_resp?.status_code !== 0 || !body.data?.audio) {
        console.error(`[call-loop] MiniMax synthesis error: ${body.base_resp?.status_msg || 'no audio in response'}`);
        return;
      }
      onChunk(Buffer.from(body.data.audio, 'hex'));
    }, text, turnId, turnStartedAt, format);
  }

  // One structured [latency] line per real user turn (see the _latency seed
  // in _onUserTurnComplete) — end-to-end response time plus its LLM/TTS
  // split, tagged with the CallSid so the mystery-shopper pipeline can grep
  // one session's numbers out of interleaved logs. Emitted on the TTS first
  // byte (per-provider: chunk_meta for the kokoro gateway, first audio chunk
  // for the HTTP backends), once per turn, and only for turns that actually
  // had a caller utterance to respond to.
  _markTtsFirstByte(turnId, source, at) {
    if (!this._latency?.turnStart || this._latency.ttsFirstByte !== undefined) return;
    this._latency.ttsFirstByte = at;
    const llmMs = this._latency.llmFirstToken ? this._latency.llmFirstToken - this._latency.turnStart : null;
    const ttsMs = at - this._latency.turnStart;
    const llmToTtsMs = this._latency.llmFirstToken ? at - this._latency.llmFirstToken : null;
    console.log(
      `[latency] [call ${this.callSid || this.id}] turn ${turnId} ${JSON.stringify({
        source,
        ttsBackend: this.ttsBackend,
        llmTtfbMs: llmMs,
        ttsLegMs: llmToTtsMs,
        responseMs: ttsMs,
      })}`
    );
    this._latency = null;
  }

  _onTtsMessage(data, isBinary) {
    if (isBinary) {
      // PCM16LE mono 24kHz chunk immediately following a chunk_meta message.
      if (this.clientWs.readyState === WebSocket.OPEN) {
        if (this.turnState) this.turnState.startedSpeaking = true; // see _onUserTurnComplete
        this.clientWs.send(data, { binary: true });
      }
      return;
    }
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.type === 'chunk_meta' && this._pendingTurnStart) {
      this._markTtsFirstByte(this.turnState?.id ?? 0, 'kokoro', Date.now());
      console.log(`[call-loop] TTS TTFB: ${Date.now() - this._pendingTurnStart}ms (turn latency end-to-end)`);
      this._pendingTurnStart = null;
    }
    if (msg.type === 'done' || msg.type === 'cancelled' || msg.type === 'error') {
      if (this.turnState) {
        this.turnState.pendingTts = Math.max(0, this.turnState.pendingTts - 1);
        this._maybeRetireTurn(this.turnState.id);
      }
    }
    // Forward control messages (chunk_meta/done/error) so the browser can
    // sequence playback and know when the assistant has finished speaking.
    this.send({ type: 'tts_event', event: msg });
  }

  _bargeIn() {
    if (this.activeTurn === 0) return;
    console.log(`[call-loop] barge-in — cancelling turn ${this.activeTurn}`);
    this.activeTurn = 0; // no active turn is allowed to speak until the next final transcript
    if (this._httpTtsAborts) {
      for (const controller of this._httpTtsAborts) controller.abort();
      this._httpTtsAborts.clear();
    }
    if (this.ttsWs?.readyState === WebSocket.OPEN) {
      this.ttsWs.send(JSON.stringify({ type: 'stop' }));
    }
    if (typeof this.clientWs.clearQueue === 'function') this.clientWs.clearQueue();
    this.send({ type: 'barge_in' });
  }

  close() {
    if (this._closed) return;
    this._closed = true;
    this.dgConnection?.close();
    this.ttsWs?.close();
    // A flow's goodbye node calls this proactively to end the call — unlike
    // every other close() caller (browser tab closed, Twilio's own 'stop'
    // event), which is already reacting to the transport having closed
    // itself. Without this, a real phone call would just sit connected but
    // silent forever once the flow decided it was over — STT/TTS torn down,
    // but the actual call never hung up. Safe to call even when close() was
    // itself triggered BY the transport closing — both TwilioCallAdapter and
    // a plain ws.WebSocket treat a second close() as a no-op.
    this.clientWs.close?.();

    // A detour redirect (see _redirectForDetour/_stashForResume) closes
    // THIS session's transport but the call itself isn't over — a new
    // CallSession will pick it back up via _tryResumeFromCallSid once
    // Twilio reconnects. Billing finalization and the live-call registry
    // removal below both mean "the call is done" — neither is true yet,
    // so skip both here and let the eventual real close() (paused=false)
    // do them once, covering the accumulated cost across every segment.
    if (this._pausedForResume) {
      console.log('[call-loop] session paused for resume (detour in progress) — not finalizing billing/registry yet');
      return;
    }

    activeSessions.delete(this.id);
    console.log('[call-loop] client disconnected');
    // Deepgram bills for the whole connected duration, not per-turn audio —
    // total call wall-clock is the right proxy, not summed turn lengths.
    const voiceSeconds = (Date.now() - this._callStartedAt) / 1000;
    this.cost.addSttSeconds(voiceSeconds);
    this.cost.logSummary();
    // Fire-and-forget — close() must not block hangup on a Stripe round
    // trip, and a metering failure shouldn't surface as a call failure.
    reportCallUsage(this.stripeCustomerId, {
      voiceSeconds,
      bookingEvents: this.cost.bookingEvents,
      transferEvents: this.cost.transferEvents,
      messageEvents: this.cost.messageEvents,
    }).catch((err) => console.error('[call-loop] usage reporting failed', err));
  }
}

server.listen(PORT, () => {
  console.log(`[call-loop] listening on http://localhost:${PORT}`);
  console.log(`[call-loop] TTS gateway: ${TTS_GATEWAY_WS_URL}`);
  // Always prewarm regardless of the process-wide default — a per-session
  // context override can enable backchanneling even when it's off globally,
  // and the prewarm cost (a handful of short TTS calls, once, at startup)
  // is trivial either way.
  prewarmFillerCache();
});
