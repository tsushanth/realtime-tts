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
// Opt-in, not opt-out: this was firing on ~80% of turns on every backend,
// including fast ones like ElevenLabs, which was never the intent — the
// intent was masking genuine latency (e.g. kokoro's cold start), not a
// filler word before most replies. A tenant/call can still turn it on via
// the {"type":"context"} message's backchannelEnabled field.
const BACKCHANNEL_ENABLED_DEFAULT = process.env.BACKCHANNEL_ENABLED === 'true';
const BACKCHANNEL_FREQUENCY_DEFAULT = Number(process.env.BACKCHANNEL_FREQUENCY ?? 0.8);
const BACKCHANNEL_DELAY_MS_DEFAULT = Number(process.env.BACKCHANNEL_DELAY_MS ?? 400);
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
// everything else in the pipeline combined.
const LLM_MODEL = process.env.LLM_MODEL || 'claude-haiku-4-5-20251001';
// Only needed for a flow's 'transfer' node type on a real (Twilio) phone
// call — redirects the live call via Twilio's REST API. Not needed for
// browser calls (there's nothing to redirect) or flows with no transfer node.
const TWILIO_ACCOUNT_SID = process.env.TWILIO_ACCOUNT_SID;
const TWILIO_AUTH_TOKEN = process.env.TWILIO_AUTH_TOKEN;
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
const DEEPGRAM_WS_URL_BROWSER =
  'wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000' +
  '&eot_threshold=0.7&eot_timeout_ms=5000&numerals=true';
const DEEPGRAM_WS_URL_TWILIO =
  'wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=mulaw&sample_rate=8000' +
  '&eot_threshold=0.7&eot_timeout_ms=5000&numerals=true';

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
      if (resolved) pendingCallContext.set(callSid, { ...resolved, fromNumber: req.body.From || null, createdAt: Date.now() });
    }
  }
  const twiml =
    `<?xml version="1.0" encoding="UTF-8"?>` +
    `<Response><Connect><Stream url="wss://${req.headers.host}/twilio-stream" /></Connect></Response>`;
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
        this.turnState = { id: turnId, llmDone: true, pendingTts: 0 };
        this._speak(msg.greeting, turnId, Date.now());
      } else if (this.flow) {
        // No explicit greeting text — let the flow's own start node (usually
        // a 'greeting'-type node) generate the opening line itself, same as
        // every other node's turn.
        this._runNodeTurn(this.currentNodeId);
      }
      console.log(`[call-loop] context set — prompt: ${this.systemPrompt.length} chars, voice: ${this.voice}, ttsBackend: ${this.ttsBackend}`);
    }
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
    const turnId = ++this.turnSeq;
    this.activeTurn = turnId;
    this.turnState = { id: turnId, llmDone: false, pendingTts: 0 };
    const turnStartedAt = Date.now();
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
    this.turnState = { id: turnId, llmDone: false, pendingTts: 0 };
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
    await this._generateTurn(turnId, Date.now(), { isNodeEntry: true, suppressTransitionTool: isCallOpening });
  }

  // Shared by both a real caller turn and a flow auto-advance turn — the
  // only difference is whether a user message was already pushed to
  // this.history before calling in. When this.flow is set, the system
  // prompt/tools are scoped to just the current node (see
  // _buildNodeSystemPrompt/_buildTransitionTool) instead of the flat
  // this.systemPrompt every other turn uses, and the LLM signals when to
  // advance by calling the transition_flow tool rather than us guessing from
  // the model's prose.
  async _generateTurn(turnId, turnStartedAt, { isNodeEntry = false, suppressTransitionTool = false } = {}) {
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

    // Backchanneling — only for a real reply to something the caller just
    // said, not a flow auto-advance turn (isNodeEntry) where there's no
    // caller utterance to be acknowledging. Cleared the moment the LLM's
    // first token actually arrives (below) or the turn ends (finally,
    // below) so a stray filler never fires after the real response.
    let backchannelTimer = null;
    if (!isNodeEntry && this.backchannelEnabled) {
      backchannelTimer = setTimeout(() => {
        backchannelTimer = null;
        if (Math.random() < this.backchannelFrequency) this._maybeSpeakBackchannel(turnId);
      }, this.backchannelDelayMs);
    }

    try {
      const stream = anthropic.messages.stream({
        model: LLM_MODEL,
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
          console.log(`[call-loop] turn ${turnId} LLM TTFB: ${firstTokenAt - turnStartedAt}ms`);
        }
        assistantText += delta;
        chunker.push(delta);
      });

      const final = await stream.finalMessage();
      if (final.usage) this.cost.addLlmUsage(LLM_MODEL, final.usage.input_tokens, final.usage.output_tokens);
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
        // Unlike transition_flow (deferred until the turn finishes
        // speaking, see _maybeRetireTurn), a recorded field has no effect
        // on conversation flow state — it's just data — so it's safe to
        // persist immediately rather than waiting.
        for (const block of final.content) {
          if (block.type === 'tool_use' && block.name === 'record_field' && block.input?.field) {
            this.collectedData[block.input.field] = block.input.value;
            console.log(`[call-loop] recorded field "${block.input.field}" = "${block.input.value}"`);
          }
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
    this._speakCached(buf, turnId);
  }

  // Sends a pre-synthesized filler clip immediately — no network call, so
  // this is the one "speak" path with zero added latency. Still
  // participates in the same pendingTts/turnState bookkeeping every other
  // TTS path uses (see _speak/_speakHttpTts) so barge-in and
  // _maybeRetireTurn stay correct even when a filler is the only thing a
  // superseded turn ever said.
  _speakCached(buffer, turnId) {
    if (this.activeTurn !== turnId) return;
    if (this.clientWs.readyState !== WebSocket.OPEN) return;
    if (this.turnState?.id === turnId) this.turnState.pendingTts++;
    this.clientWs.send(buffer, { binary: true });
    if (this.turnState?.id === turnId) {
      this.turnState.pendingTts = Math.max(0, this.turnState.pendingTts - 1);
      this._maybeRetireTurn(turnId);
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
    let prompt =
      `You are a concise, friendly voice assistant on a phone call, currently in the ` +
      `"${node.id}" step of a structured conversation flow.\n\n`;
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
      prompt += `1. Ask for ALL of these together in ONE question: ${fields.join(', ')}. Don't ask one at a time, and don't re-ask anything the caller already volunteered earlier in the call.\n`;
      prompt += `2. The moment the caller gives you a field, call record_field for it — but ALWAYS also say something out loud to the caller in that same turn. Calling record_field is a silent background action, never a substitute for actually replying — never let a turn consist of only a tool call with nothing spoken.\n`;
      if (hasTimeField) {
        prompt += `3. If a date/time answer is vague ("afternoon", "next week"), propose ONE concrete slot inside their range and get a yes before treating it as captured.\n`;
      }
      prompt += `${hasTimeField ? '4' : '3'}. Names and numbers are easy to mishear — read back what you captured (e.g. "Got it, Alex, for 3pm — did I get that right?") before relying on it. Use the caller's correction if given, not your first guess.\n`;
      prompt += `${hasTimeField ? '5' : '4'}. Only once every field is recorded AND confirmed, say ONE summary sentence with all of them ("So that's Alex Morgan at 2pm tomorrow.") and THEN call transition_flow in the same turn — don't transition silently or without ever stating the final summary.\n`;
    }
    if (Object.keys(this.collectedData).length > 0) {
      prompt += `Already collected this call: ${JSON.stringify(this.collectedData)}\n`;
    }
    if (node.edges.length > 0) {
      prompt +=
        `\nWhen this step's goal has been met, call the transition_flow tool to move to the ` +
        `next step. If it hasn't been met yet, keep talking and don't call the tool.\n`;
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
    const AUTO_ADVANCE_TYPES = new Set(['function', 'knowledge_base', 'goodbye', 'transfer']);
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
      if (this.turnState.transition) this._applyTransition(this.turnState.transition);
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
        this._speakCached(warmupBuf, turnId);
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
  async _speakHttpTts(label, fetchPcm, text, turnId, turnStartedAt, format = 'pcm16') {
    if (!this._httpTtsAborts) this._httpTtsAborts = new Set();
    const controller = new AbortController();
    this._httpTtsAborts.add(controller);

    let carry = Buffer.alloc(0);
    let firstByteAt = null;

    const onChunk = (chunk) => {
      if (this.activeTurn !== turnId) return; // barge-in — stop forwarding
      if (this.clientWs.readyState !== WebSocket.OPEN) return;
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
      if (firstByteAt === null) {
        firstByteAt = Date.now();
        console.log(`[call-loop] TTS TTFB: ${firstByteAt - turnStartedAt}ms (turn latency end-to-end, ${label})`);
      }
      this.clientWs.send(buf, { binary: true, format: format === 'mulaw8k' ? 'mulaw8k' : undefined });
    };

    try {
      await fetchPcm(text, controller.signal, turnId, onChunk);
    } catch (err) {
      if (err.name !== 'AbortError') console.error(`[call-loop] ${label} stream error`, err);
    } finally {
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
            model_id: ELEVENLABS_MODEL,
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
          model_id: CARTESIA_MODEL,
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

  _onTtsMessage(data, isBinary) {
    if (isBinary) {
      // PCM16LE mono 24kHz chunk immediately following a chunk_meta message.
      if (this.clientWs.readyState === WebSocket.OPEN) {
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
    activeSessions.delete(this.id);
    console.log('[call-loop] client disconnected');
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
