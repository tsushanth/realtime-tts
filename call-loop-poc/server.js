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
import { resolveInboundCall, fetchKnowledgeItems, insertCallLog, updateCallLogByCallSid, updateCallLogById, findExpiredRecordings, acquireTwilioGlobalToken, findTenantIdByNumber, dispatchTenantWebhook, findTenantIdByCallSid, resolveAgentFlow } from './tenantLookup.js';
import { newAsyncContext, shouldInterruptAfterDeadline } from 'quickjs-emscripten';
import dns from 'node:dns/promises';
import net from 'node:net';

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

// Masks dead air during a real Cal.com round trip. Real bug reported on a
// live call: check_availability/book_appointment each carry up to an 8s
// timeout, and _handleCalendarTool awaited them with literally nothing
// spoken — the model's own "let me get that booked" line (its assistantText
// for that turn) finishes playing and then the line goes completely silent
// until the HTTP call resolves, easily several seconds, with no backchannel
// covering it since backchanneling only fires while waiting on the LLM, not
// while awaiting a tool call afterward. Same cached-clip approach as
// backchanneling — zero added latency since it's pre-synthesized, not
// generated live.
const CALENDAR_LOOKUP_FILLER_PHRASE = process.env.CALENDAR_LOOKUP_FILLER_PHRASE || 'One moment, let me check the calendar.';

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

// Wakes the TTS gateway's backend (RunPod serverless CPU worker) with a
// throwaway one-word synth. Live call finding (2026-09-18): turn 1 had a
// 15.9s TTS TTFB — the gateway WebSocket itself opens instantly (Fly), the
// stall is the worker spinning up BEHIND it. An outbound call rings for
// 5-30s and an inbound one has the answer webhook, so kicking this off at
// dial/answer time overlaps the cold start with time the caller is already
// waiting anyway. Throttled so a burst of calls doesn't stack pings.
let lastGatewayWarmAt = 0;
function warmTtsGateway(reason) {
  if (Date.now() - lastGatewayWarmAt < 60_000) return;
  lastGatewayWarmAt = Date.now();
  const startedAt = Date.now();
  const ws = new WebSocket(
    TTS_GATEWAY_WS_URL,
    TTS_GATEWAY_API_KEY ? { headers: { Authorization: `Bearer ${TTS_GATEWAY_API_KEY}` } } : undefined
  );
  const finish = (why) => {
    clearTimeout(timer);
    console.log(`[call-loop] tts gateway warm (${reason}) ${why} after ${Date.now() - startedAt}ms`);
    try { ws.close(); } catch { /* already closed */ }
  };
  const timer = setTimeout(() => finish('timed out'), 45_000);
  ws.on('open', () => ws.send(JSON.stringify({ type: 'synthesize', text: 'Hi.', voice: TTS_VOICE, speed: 1.0 })));
  ws.on('message', (_data, isBinary) => { if (isBinary) finish('ready'); });
  ws.on('error', (err) => finish(`failed (${err.message})`));
}

// Boot-time prewarm hits several providers at once and got 429s from
// ElevenLabs/Cartesia (seen right after a deploy) — leaving the cold-start
// warmup clip uncached, so the mask had nothing to play. Retry with backoff.
async function withRetry(fn, attempts = 4) {
  for (let i = 1; ; i++) {
    try { return await fn(); }
    catch (err) {
      if (i >= attempts) throw err;
      await new Promise((r) => setTimeout(r, 1500 * i + Math.random() * 500));
    }
  }
}

// The default self-hosted voice had no cached filler clips, so backchanneling
// silently did nothing on it. Synthesize them once per boot through the gateway.
function synthKokoroPcm(text) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(TTS_GATEWAY_WS_URL, TTS_GATEWAY_API_KEY ? { headers: { Authorization: `Bearer ${TTS_GATEWAY_API_KEY}` } } : undefined);
    const chunks = [];
    const timer = setTimeout(() => { try { ws.close(); } catch {} reject(new Error('timed out')); }, 60_000);
    const done = (err) => { clearTimeout(timer); try { ws.close(); } catch {} err ? reject(err) : resolve(Buffer.concat(chunks)); };
    ws.on('open', () => ws.send(JSON.stringify({ type: 'synthesize', text, voice: TTS_VOICE, speed: 1.0 })));
    ws.on('message', (data, isBinary) => {
      if (isBinary) { chunks.push(Buffer.from(data)); return; }
      try { const m = JSON.parse(data.toString()); if (m.type === 'done') done(); else if (m.type === 'error') done(new Error(m.message || 'tts error')); } catch { /* ignore */ }
    });
    ws.on('error', (err) => done(err));
  });
}

async function prewarmKokoroFillers() {
  for (const text of BACKCHANNEL_WORDS_DEFAULT) {
    try {
      const buf = await withRetry(() => synthKokoroPcm(text), 2);
      if (buf.length > 0) fillerCache.set(`kokoro::${TTS_VOICE}::${text}`, buf);
    } catch (err) {
      console.warn(`[call-loop] filler prewarm (kokoro, "${text}") failed:`, err.message);
    }
  }
  console.log('[call-loop] kokoro filler clips cached');
}

async function prewarmFillerCache() {
  const jobs = [];
  if (TTS_BACKEND === 'kokoro') jobs.push(prewarmKokoroFillers());
  for (const text of [...BACKCHANNEL_WORDS_DEFAULT, CALENDAR_LOOKUP_FILLER_PHRASE]) {
    if (ELEVENLABS_API_KEY) {
      jobs.push(
        withRetry(() => fetchElevenLabsPcmOnce(text))
          .then((buf) => fillerCache.set(`elevenlabs::${ELEVENLABS_VOICE_ID}::${text}`, buf))
          .catch((err) => console.warn(`[call-loop] filler prewarm (elevenlabs, "${text}") failed:`, err.message))
      );
    }
    if (CARTESIA_API_KEY && CARTESIA_VOICE_ID) {
      jobs.push(
        withRetry(() => fetchCartesiaPcmOnce(text))
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
      withRetry(() => fetchElevenLabsPcmOnce(KOKORO_WARMUP_PHRASE))
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
// Real call recording (2026-09-17) — on by default, matching Retell's own
// approach of not enforcing a disclosure announcement itself (consent is
// left to whoever configures the agent, per Retell's privacy policy). A
// <Connect><Stream> call isn't recorded by Twilio automatically the way a
// <Dial>/<Record> verb would be — has to be started explicitly via the REST
// API on the already-live call. RecordingStatusCallback below is how the
// finished recording's URL comes back (asynchronously, after the call ends).
const RECORD_REAL_CALLS = process.env.RECORD_REAL_CALLS !== 'false';

async function startCallRecording(callSid) {
  if (!TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) return;
  const auth64 = Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64');
  const params = new URLSearchParams({
    RecordingChannels: 'dual',
    RecordingStatusCallback: `https://${PUBLIC_HOST}/twilio/recording-status`,
    RecordingStatusCallbackEvent: 'completed',
  });
  const res = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Calls/${callSid}/Recordings.json`, {
    method: 'POST',
    headers: { Authorization: `Basic ${auth64}`, 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params,
  });
  if (!res.ok) {
    console.error(`[call-loop] failed to start recording for ${callSid}: HTTP ${res.status}`, await res.text().catch(() => ''));
  }
}

// Retention enforcement — actually DELETES the audio from Twilio once a
// tenant's own retention_days window passes (see Settings page's "Call
// Recording" section), not just hiding it from the UI; we never hold a copy
// ourselves (see calldesktech's own recording proxy comment — this whole
// feature deliberately doesn't store audio in a bucket we own), so this is
// the only real way the data actually stops existing. Runs periodically via
// setInterval below — a long-running Node process already, so no separate
// cron infra needed.
async function enforceRecordingRetention() {
  if (!TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) return;
  const expired = await findExpiredRecordings().catch((err) => {
    console.error('[call-loop] retention sweep query failed', err);
    return [];
  });
  if (expired.length === 0) return;
  console.log(`[call-loop] retention sweep: deleting ${expired.length} expired recording(s)`);
  const auth64 = Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64');
  for (const { recordingSid, callLogId } of expired) {
    try {
      const res = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Recordings/${recordingSid}.json`, {
        method: 'DELETE',
        headers: { Authorization: `Basic ${auth64}` },
      });
      // 404 means it's already gone (deleted manually, or a prior sweep run
      // partially succeeded) — still clear our own reference either way.
      if (res.ok || res.status === 404) {
        await updateCallLogById(callLogId, { recording_url: null, recording_sid: null });
      } else {
        console.error(`[call-loop] failed to delete recording ${recordingSid}: HTTP ${res.status}`);
      }
    } catch (err) {
      console.error(`[call-loop] failed to delete recording ${recordingSid}`, err);
    }
  }
}

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
// A reply that is entirely a bracketed/parenthesised note ("(The goodbye was already delivered.)")
// is the model narrating, not speaking; never voice it.
const isStageDirection = (t) => /^\s*[\(\[][^\)\]]*[\)\]]\s*[.!]?\s*$/.test(t);
// Optional per-call persona (place-test-call {persona}); keyed by the shopper's own CallSid.
const shopperPersonas = new Map();
const SHOPPER_PERSONA_RULES =
  ' You are NOT an AI — you are a real person on a phone call. Wait for the business to speak first and answer only what is asked, one thing at a time, in whatever order they ask; do not volunteer details early. ' +
  'Keep replies short and natural. If asked to confirm something that matches your goal, confirm it. Once your goal is done, or you are told you are being transferred, thank them briefly ONE time and say a single goodbye. ' +
  'Never break character or mention an AI, a test, or a script.';
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
// Guards against two logic_split nodes routing to each other — with no LLM
// turn in between to break the cycle, that would otherwise loop forever in
// a single synchronous call stack. Any real flow needs far fewer hops than
// this to reach a conversational node.
const LOGIC_SPLIT_MAX_HOPS = 10;
// Reminder Message Frequency's own cap — see _scheduleReminderIfConfigured.
const REMINDER_MAX_ATTEMPTS = 3;
// Real sandboxing for a 'code' node (see _executeCodeNode) — QuickJS
// compiled to WASM, same approach Retell's own Code node docs describe, run
// in a completely separate memory space from this process (not Node's own
// `vm` module, which shares the V8 heap/prototypes with the host and is a
// well-known non-boundary; not eval()/Function(), same problem). Limits
// mirror Retell's own documented ones.
const CODE_NODE_TIMEOUT_MS = 10000;
const CODE_NODE_MEMORY_LIMIT_BYTES = 16 * 1024 * 1024;
const CODE_NODE_MAX_SOURCE_CHARS = 20000;
const CODE_NODE_MAX_OUTPUT_CHARS = 15000;

// SSRF guard for the sandboxed fetch a code node's script can call — resolves
// the hostname and checks the RESOLVED address, not just the literal
// hostname string, so a hostname that resolves to a private/loopback address
// (e.g. via attacker-controlled DNS) can't bypass a string-only check.
function isPrivateOrLocalIp(ip) {
  if (net.isIPv4(ip)) {
    const [a, b] = ip.split('.').map(Number);
    if (a === 127 || a === 10 || a === 0) return true;
    if (a === 172 && b >= 16 && b <= 31) return true;
    if (a === 192 && b === 168) return true;
    if (a === 169 && b === 254) return true;
    return false;
  }
  if (net.isIPv6(ip)) {
    const lower = ip.toLowerCase();
    if (lower === '::1' || lower === '::') return true;
    if (lower.startsWith('fc') || lower.startsWith('fd')) return true; // unique local fc00::/7
    if (lower.startsWith('fe80')) return true; // link-local
    return false;
  }
  return true; // unrecognized format — block rather than risk it
}

async function sandboxSafeFetch(urlString, options) {
  let url;
  try {
    url = new URL(String(urlString));
  } catch {
    throw new Error('Invalid URL');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Only http:// and https:// URLs are allowed');
  }
  let addresses;
  try {
    addresses = await dns.lookup(url.hostname, { all: true });
  } catch (err) {
    throw new Error(`DNS lookup failed for "${url.hostname}": ${err.message}`);
  }
  if (addresses.length === 0 || addresses.some((a) => isPrivateOrLocalIp(a.address))) {
    throw new Error('Requests to private/local network addresses are not allowed');
  }
  const method = (options?.method || 'GET').toUpperCase();
  const headers = options?.headers && typeof options.headers === 'object' ? options.headers : undefined;
  const body = typeof options?.body === 'string' ? options.body : undefined;
  const res = await fetch(url, { method, headers, body, signal: AbortSignal.timeout(8000) });
  const text = await res.text();
  return { status: res.status, ok: res.ok, body: text.slice(0, 100000) };
}
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
// A caller reading a phone number pauses between digit groups and the speech model ends the
// turn at each pause. Given the agent just asked for a phone number, hold a short digit-only
// fragment so the rest of the number can join it before the agent replies.
const PHONE_ASK_RE = /\b(phone|cell|mobile|callback)\b[^.?!]{0,40}\bnumber\b|\bnumber\b[^.?!]{0,40}\b(reach|call) you\b|\bbest number\b/i;
const DIGIT_HOLD_MS = 3000;
function shouldHoldForDigits(lastAssistantText, callerText) {
  if (!lastAssistantText || !PHONE_ASK_RE.test(lastAssistantText)) return false;
  const stripped = callerText.replace(/[^0-9a-z]/gi, '');
  const digits = (callerText.match(/\d/g) || []).length;
  return digits >= 2 && stripped.length > 0 && digits / stripped.length >= 0.4 && digits < 7;
}
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
// Transcription Mode (flow-level Global Setting, not per-node — see the
// comment on _connectDeepgram below for why it can't honestly be per-node)
// maps directly onto this same real knob: lower = faster/more responsive
// turn-taking at the cost of more false turn-ends, higher = more patient/
// accurate at the cost of latency. 'balanced' matches today's existing
// default exactly, so a flow that never sets this sees zero behavior change.
const TRANSCRIPTION_MODE_THRESHOLDS = { fast: '0.5', balanced: DEEPGRAM_EOT_THRESHOLD, accurate: '0.75' };
function buildDeepgramUrl(isTwilio, eotThreshold) {
  const params =
    `model=flux-general-en&eot_threshold=${eotThreshold}&eot_timeout_ms=${DEEPGRAM_EOT_TIMEOUT_MS}` +
    `&numerals=true${DEEPGRAM_KEYWORDS ? `&keyterm=${encodeURIComponent(DEEPGRAM_KEYWORDS)}` : ''}`;
  return isTwilio
    ? `wss://api.deepgram.com/v2/listen?${params}&encoding=mulaw&sample_rate=8000`
    : `wss://api.deepgram.com/v2/listen?${params}&encoding=linear16&sample_rate=16000`;
}

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
    pendingCallContext.set(callSid, { isShopper: true, persona: shopperPersonas.get(callSid), createdAt: Date.now() });
  } else {
    const toNumber = req.query.routeAs || req.body.To;
    // ?direction=outbound (see /place-test-call below) resolves the DIALED
    // number's own outbound_agent_version_id instead of its inbound one —
    // a real "make an outbound call" button tests what that number is
    // actually configured to say when IT calls out, not what it says when
    // called.
    const direction = req.query.direction === 'outbound' ? 'outbound' : 'inbound';
    if (callSid) {
      const resolved = await resolveInboundCall(toNumber, direction).catch((err) => {
        console.error('[call-loop] tenant lookup failed', err);
        return null;
      });
      if (resolved && (resolved.ttsBackend || TTS_BACKEND) === 'kokoro') warmTtsGateway('answer');
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
          // The tenant's OWN number (what was dialed), as distinct from the
          // caller's (fromNumber) — needed as the "From" on an in-call SMS
          // (see 'sms' node type / _executeSmsNode), since texting the
          // caller has to come from a number that's actually theirs, not
          // whoever called in.
          tenantNumber: toNumber || null,
          direction,
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

// Twilio calls this once a recording started via startCallRecording()
// finishes processing — asynchronously, after the call itself has already
// ended (see close()'s own duration/transcript update, which happens
// separately and earlier). No auth on this one; Twilio doesn't sign
// RecordingStatusCallback requests the way it does some other webhooks, and
// there's nothing sensitive being READ here, only a URL being written
// against a callSid Twilio itself supplies.
app.post('/twilio/recording-status', express.urlencoded({ extended: false }), async (req, res) => {
  const { CallSid, RecordingStatus, RecordingUrl, RecordingSid } = req.body || {};
  if (CallSid && RecordingStatus === 'completed' && RecordingUrl) {
    // Twilio's RecordingUrl has no extension by default — appending .mp3
    // gets back a real playable media file instead of the JSON resource
    // representation the bare URL would otherwise return. RecordingSid is
    // stored separately (not parsed back out of the URL later) so the
    // retention sweep below can target Twilio's DELETE endpoint directly.
    await updateCallLogByCallSid(CallSid, { recording_url: `${RecordingUrl}.mp3`, recording_sid: RecordingSid || null });
  }
  res.sendStatus(200);
});

// The action= callback on _executeTransfer's <Dial> — Twilio POSTs here once
// the transfer leg resolves, with the real outcome (DialCallStatus) and, if
// answered, DialCallDuration (the answered portion only, not ring time).
// startedAt (round-tripped through the action URL's own query string, not
// server-side state) plus the request's own arrival time gives total
// elapsed; subtracting the answered duration leaves the ring/wait portion.
// Twilio expects a TwiML response here to know what to do next — the
// transfer is already over either way, so this just ends the call cleanly.
const DIAL_STATUS_MAP = { completed: 'answered', busy: 'busy', 'no-answer': 'no_answer', failed: 'failed', canceled: 'canceled' };
app.post('/twilio/dial-status', express.urlencoded({ extended: false }), async (req, res) => {
  const { CallSid, DialCallStatus, DialCallDuration } = req.body || {};
  const startedAt = Number(req.query.startedAt);
  if (CallSid && DialCallStatus && Number.isFinite(startedAt)) {
    const transferStatus = DIAL_STATUS_MAP[DialCallStatus] || DialCallStatus;
    const answeredMs = (Number(DialCallDuration) || 0) * 1000;
    const totalElapsedMs = Date.now() - startedAt;
    const waitMs = Math.max(0, totalElapsedMs - answeredMs);
    updateCallLogByCallSid(CallSid, { transfer_status: transferStatus, transfer_wait_ms: Math.round(waitMs) })
      .catch((err) => console.error('[call-loop] transfer status update failed', err));
    findTenantIdByCallSid(CallSid).then((tid) => dispatchTenantWebhook(tid, 'call.transferred', {
      call_id: CallSid, tenant_id: tid, outcome: 'transferred', transfer_status: transferStatus, transfer_wait_ms: Math.round(waitMs),
    })).catch(() => {});
  }
  res.type('text/xml').send('<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>');
});

// Proxies a Twilio recording's audio bytes through this server's own Twilio
// credentials — the browser can't fetch a Twilio recording URL directly, it
// requires HTTP Basic Auth with the account's SID/token, which obviously
// can't be handed to the client. calldesktech's own /api/calls/[id]/recording
// route calls this rather than exposing Twilio creds to calldesktech either.
// Same admin-secret guard as /place-test-call/-purchase-number.
// Async AMD result from Twilio (set up in /place-test-call). Result may
// arrive before the session has its flow, so it is stashed and replayed.
const pendingAnsweredBy = new Map(); // callSid -> answeredBy
app.post('/twilio/amd-status', express.urlencoded({ extended: false }), (req, res) => {
  const { CallSid, AnsweredBy } = req.body || {};
  res.sendStatus(204);
  if (!CallSid || !AnsweredBy) return;
  const session = [...activeSessions.values()].find((s) => s.callSid === CallSid);
  if (session?.flow) session.onAnsweredBy(AnsweredBy);
  else {
    pendingAnsweredBy.set(CallSid, AnsweredBy);
    setTimeout(() => pendingAnsweredBy.delete(CallSid), 60000).unref?.();
  }
});

app.get('/recording-audio', async (req, res) => {
  const auth = req.headers['authorization'] || '';
  if (!TEST_CALL_SECRET || auth !== `Bearer ${TEST_CALL_SECRET}`) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  const url = req.query.url;
  if (!url || typeof url !== 'string' || !url.startsWith('https://api.twilio.com/')) {
    return res.status(400).json({ error: 'url must be a real Twilio recording URL' });
  }
  if (!TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) {
    return res.status(500).json({ error: 'TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not configured' });
  }
  try {
    const auth64 = Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64');
    const twilioRes = await fetch(url, { headers: { Authorization: `Basic ${auth64}` } });
    if (!twilioRes.ok || !twilioRes.body) {
      return res.status(twilioRes.status).json({ error: 'Failed to fetch recording from Twilio' });
    }
    res.setHeader('Content-Type', twilioRes.headers.get('content-type') || 'audio/mpeg');
    for await (const chunk of twilioRes.body) res.write(chunk);
    res.end();
  } catch (err) {
    console.error('[call-loop] recording proxy failed', err);
    res.status(502).json({ error: err.message });
  }
});

// Real finding from a live test call (2026-09-17): Twilio's actual Result
// values (payment-connector-error, too-many-failed-attempts, etc. — see
// https://www.twilio.com/docs/voice/api/payment-resource#statuscallback)
// don't match a flow author's natural edge-condition wording ("payment
// failed or was canceled") closely enough for the model to reliably
// recognize a failure — reproduced live: it treated an unrecognized
// Result as "try again" and re-triggered the payment redirect twice in a
// row instead of routing to the flow's own failed-payment node. Same
// "compute, don't infer" fix as the missing-fields prompt work earlier —
// normalize Twilio's many possible Result strings into a plain
// success/failed boolean-ish field here, in code, rather than asking the
// model to pattern-match raw Twilio enum values it's never seen
// documented. The raw value is kept alongside it for debugging, just not
// as the field an edge condition is expected to reason about.
const PAYMENT_SUCCESS_RESULTS = new Set(['success']);

// <Pay>'s action callback — Twilio POSTs here once a payment session ends
// (success, failure, or the caller pressed * to cancel). Writes ONLY the
// outcome into the paused session's stashed collectedData — normalized
// status, last four digits, card brand — never PaymentCardNumber or any
// other raw cardholder field, even though Twilio's webhook payload can
// include one for certain transaction types (see CreatePayments' own
// Input field note: digits are redacted from Twilio's LOGS, which is a
// different guarantee than "never appears in this webhook body" — the
// responsibility not to let it into OUR history/logs/DB is on this
// handler, not Twilio).
// Reconnects the call into a fresh <Connect><Stream> either way; the
// payment node's own outgoing edges (flow-author-defined, e.g. "the
// payment succeeded" / "the payment did not succeed") take it from there
// once _tryResumeFromCallSid rehydrates and _runNodeTurn re-evaluates.
app.post('/twilio/pay-result', (req, res) => {
  const callSid = req.query.callSid || req.body.CallSid;
  const stashed = callSid ? pendingResumeSessions.get(callSid) : null;
  if (stashed) {
    const rawResult = req.body.Result || 'unknown';
    stashed.collectedData.payment_status = PAYMENT_SUCCESS_RESULTS.has(rawResult) ? 'succeeded' : 'failed';
    stashed.collectedData.payment_status_detail = rawResult; // raw Twilio value, for logs/debugging only — not what an edge condition should key off
    if (req.body.PaymentCardNumber) {
      // Last-resort guard: this field is meant to already be masked
      // ("XXXXXXXXXXXX1234") per Twilio's redaction guarantee, but never
      // trust that blindly — only ever keep a trailing run of <=4 digits,
      // regardless of what actually arrives here.
      const digitsOnly = String(req.body.PaymentCardNumber).replace(/\D/g, '');
      stashed.collectedData.payment_last4 = digitsOnly.slice(-4);
    }
    if (req.body.PaymentConfirmationCode) stashed.collectedData.payment_confirmation = req.body.PaymentConfirmationCode;
    console.log(`[call-loop] pay-result for ${callSid}: ${stashed.collectedData.payment_status} (raw: ${rawResult})`);
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
  const { toNumber, routeAs, record, shopper, direction, persona } = req.body || {};
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
    let fromNumber = routeAs;
    if (!fromNumber) {
      // Only shopper mode reaches here — no specific number to test, so ask
      // Twilio which number(s) this account owns and use the first one.
      // Real bug fixed alongside this: this fallback used to run
      // UNCONDITIONALLY, even when routeAs (the number a caller actually
      // wants to test FROM) was given — meaning a "make an outbound call"
      // button testing a specific number could silently place the call from
      // a DIFFERENT number instead, with no error or warning.
      const numbersRes = await fetch(
        `https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/IncomingPhoneNumbers.json?PageSize=1`,
        { headers: { Authorization: `Basic ${auth64}` } }
      );
      const numbersBody = await numbersRes.json();
      fromNumber = numbersBody.incoming_phone_numbers?.[0]?.phone_number;
      if (!fromNumber) {
        return res.status(500).json({ error: 'No Twilio phone number found on this account', detail: numbersBody });
      }
    }

    const outboundQuery = direction === 'outbound' ? '&direction=outbound' : '';
    const voiceUrl = shopper
      ? `https://${req.headers.host}/twilio/voice?mode=shopper`
      : `https://${req.headers.host}/twilio/voice?routeAs=${encodeURIComponent(routeAs)}${outboundQuery}`;
    const params = new URLSearchParams({ To: toNumber, From: fromNumber, Url: voiceUrl });
    // Opt-in only — a normal test call shouldn't silently start recording.
    // Twilio's own dual-channel recording (caller + callee on separate
    // tracks) gives a real downloadable wav for the ASR eval harnesses,
    // which need actual audio, not just a live transcript.
    if (record) {
      params.set('Record', 'true');
      params.set('RecordingChannels', 'dual');
    }
    // Voicemail detection is opt-in per flow (globalSettings.voicemailDetection).
    if (direction === 'outbound' && routeAs && !shopper) {
      const resolvedOut = await resolveInboundCall(routeAs, 'outbound').catch(() => null);
      const vm = resolvedOut?.flow?.globalSettings?.voicemailDetection;
      if (vm === 'hangup' || vm === 'leave_message') {
        params.set('MachineDetection', 'DetectMessageEnd');
        params.set('AsyncAmd', 'true');
        params.set('AsyncAmdStatusCallback', `https://${req.headers.host}/twilio/amd-status`);
        params.set('AsyncAmdStatusCallbackMethod', 'POST');
      }
    }
    // Global CPS cap (2026-09-17) — this is THE choke point for every real
    // Twilio call this app places (batch calling, single test calls,
    // mystery-shopper), so it's the right place to enforce the shared
    // account's platform-wide limit rather than per-caller. See
    // acquireTwilioGlobalToken's own comment in tenantLookup.js.
    if (TTS_BACKEND === 'kokoro') warmTtsGateway('dial');
    const gotToken = await acquireTwilioGlobalToken();
    if (!gotToken) {
      return res.status(429).json({ error: 'Rate limit wait timed out (platform-wide Twilio cap) — try again shortly' });
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
    console.log(`[call-loop] test call placed: ${fromNumber} -> ${toNumber} (routeAs=${routeAs}, direction=${direction || 'inbound'}), sid=${callBody.sid}`);

    // Tags a shopper call as internal/test call history for the tenant it
    // actually exercised, rather than leaving it invisible. Written directly
    // here at dial-time instead of relying on the normal inbound-call
    // webhook path — a real call between two of our own Twilio numbers
    // reproducibly only ever produced ONE observable CallSid/CallSession in
    // testing (2026-09-18), not the two independently-webhooked legs this
    // would otherwise need; the exact underlying Twilio mechanism wasn't
    // fully pinned down, so this sidesteps that uncertainty rather than
    // depending on it. Best-effort, fire-and-forget — never blocks the
    // response on this.
    if (shopper && typeof persona === 'string' && persona.trim()) shopperPersonas.set(callBody.sid, persona.trim().slice(0, 2500));
    if (shopper) {
      findTenantIdByNumber(toNumber)
        .then((tenantId) => {
          if (!tenantId) return; // shopper's target wasn't one of our own numbers (e.g. a competitor's)
          return insertCallLog({
            tenant_id: tenantId,
            retell_call_id: callBody.sid,
            caller_phone: fromNumber,
            to_number: toNumber,
            direction: 'inbound',
            voice_engine: 'poc',
            outcome: 'answered',
            duration_seconds: 0,
            is_internal_test: true,
          });
        })
        .catch((err) => console.error('[call-loop] shopper internal-test call log failed (non-fatal)', err));
    }

    res.json({ sid: callBody.sid, from: fromNumber, to: toNumber, status: callBody.status });
  } catch (err) {
    console.error('[call-loop] place-test-call failed', err);
    res.status(500).json({ error: err.message });
  }
});

// Populous US area codes essentially guaranteed to have inventory — same
// fallback list calldesktech's own purchase route used to lean on when it
// bought numbers through Retell instead of here.
const PURCHASE_FALLBACK_AREA_CODES = ['212', '415', '312', '404'];

// Buys a REAL Twilio number on THIS app's own Twilio account and points its
// Voice webhook at our own /twilio/voice, so it actually rings into this
// app's call handling. calldesktech used to buy poc-engine tenants' numbers
// through Retell's phone-number API instead — that number lives in Retell's
// own Twilio (sub)account, not this one, which is exactly why a real test
// call from it failed with Twilio's "not yet verified for your account":
// our own TWILIO_ACCOUNT_SID/AUTH_TOKEN (used by /place-test-call and for
// real inbound routing) never owned it. Same admin-secret guard as
// /place-test-call — this spends real money, not just reads state.
app.post('/purchase-number', express.json(), async (req, res) => {
  const auth = req.headers['authorization'] || '';
  if (!TEST_CALL_SECRET || auth !== `Bearer ${TEST_CALL_SECRET}`) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  if (!TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) {
    return res.status(500).json({ error: 'TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not configured' });
  }

  const { areaCode: requestedAreaCode } = req.body || {};
  const candidates = [requestedAreaCode, ...PURCHASE_FALLBACK_AREA_CODES, undefined].filter(
    (v, i, arr) => arr.indexOf(v) === i
  );
  const auth64 = Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64');

  try {
    let phoneNumber = null;
    let lastDetail = null;
    for (const areaCode of candidates) {
      const searchUrl = new URL(`https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/AvailablePhoneNumbers/US/Local.json`);
      searchUrl.searchParams.set('PageSize', '1');
      if (areaCode) searchUrl.searchParams.set('AreaCode', areaCode);
      const searchRes = await fetch(searchUrl, { headers: { Authorization: `Basic ${auth64}` } });
      const searchBody = await searchRes.json();
      if (!searchRes.ok) {
        lastDetail = searchBody;
        continue;
      }
      const candidate = searchBody.available_phone_numbers?.[0]?.phone_number;
      if (candidate) {
        phoneNumber = candidate;
        break;
      }
    }

    if (!phoneNumber) {
      return res.status(502).json({ error: 'No Twilio numbers available in any candidate area code', detail: lastDetail });
    }

    const voiceUrl = `https://${PUBLIC_HOST}/twilio/voice`;
    const purchaseParams = new URLSearchParams({
      PhoneNumber: phoneNumber,
      VoiceUrl: voiceUrl,
      VoiceMethod: 'POST',
    });
    const purchaseRes = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/IncomingPhoneNumbers.json`, {
      method: 'POST',
      headers: { Authorization: `Basic ${auth64}`, 'Content-Type': 'application/x-www-form-urlencoded' },
      body: purchaseParams,
    });
    const purchaseBody = await purchaseRes.json();
    if (!purchaseRes.ok) {
      return res.status(purchaseRes.status).json({ error: 'Twilio number purchase failed', detail: purchaseBody });
    }
    console.log(`[call-loop] purchased phone number: ${phoneNumber}`);
    res.status(201).json({ phone_number: phoneNumber });
  } catch (err) {
    console.error('[call-loop] purchase-number failed', err);
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
        systemPrompt: resolved.persona ? resolved.persona + SHOPPER_PERSONA_RULES : SHOPPER_SYSTEM_PROMPT,
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
      ...(resolved.tenantNumber ? { tenantNumber: resolved.tenantNumber } : {}),
      ...(resolved.calendar ? { calendar: resolved.calendar } : {}),
    }), false);

    // Real call logging + recording (2026-09-17) — this is the only place a
    // real poc-engine call becomes visible on the dashboard's Calls page at
    // all; unlike Retell, nothing external notifies calldesktech of this
    // call's lifecycle, since WE own the telephony end to end. Same
    // no-disclosure default Retell itself uses (consent is left to whoever
    // configures the agent, not a platform-enforced announcement — see
    // Retell's own privacy policy). Neither call blocks the greeting from
    // going out.
    session._tenantInterruptionSensitivity = resolved.interruptionSensitivity || null;
    if (resolved.tenantId) {
      session.direction = resolved.direction || 'inbound';
      session._tenantId = resolved.tenantId;
      dispatchTenantWebhook(resolved.tenantId, 'call.started', {
        call_id: callSid,
        tenant_id: resolved.tenantId,
        caller_phone: resolved.fromNumber || 'unknown',
        to_number: resolved.tenantNumber || null,
        direction: session.direction,
        started_at: new Date().toISOString(),
      });
      insertCallLog({
        tenant_id: resolved.tenantId,
        retell_call_id: callSid,
        caller_phone: resolved.fromNumber || 'unknown',
        to_number: resolved.tenantNumber || null,
        direction: session.direction,
        voice_engine: 'poc',
        outcome: 'answered',
        duration_seconds: 0,
      }).then((id) => { session._callLogId = id; }).catch((err) => console.error('[call-loop] call log insert failed', err));
      if (RECORD_REAL_CALLS && resolved.recordingEnabled !== false) {
        startCallRecording(callSid).catch((err) => console.error('[call-loop] recording start failed', err));
      }
    }
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
    // Subflows (2026-09-18) — real subroutine semantics, not a visual-only
    // grouping: entering a 'subflow_ref' node swaps flowNodesById to that
    // subflow's own (embedded-at-publish-time) node map and pushes a frame
    // here; reaching one of the subflow's own terminal nodes (no edges of
    // its own) pops back to the parent's node map and hands the model the
    // ORIGINAL subflow_ref node's real edges to choose from, exactly as if
    // the subflow_ref node itself were the one transitioning. See
    // _enterSubflow/_buildTransitionTool/_applyTransition.
    this._subflowStack = [];
    // Set via the {"type":"context"} message's `stripeCustomerId` field —
    // present only when this call belongs to a real billed tenant (browser
    // demo calls and flow-MCP test calls have none, and simply don't get
    // metered). See stripeMeter.js.
    this.stripeCustomerId = null;
    this.calendar = null; // { provider, apiKey, eventTypeId } — see tenantLookup.js / onClientMessage
    this._lastAvailableSlots = null; // label -> real ISO time, from the most recent check_availability — see _bookAppointment
    // Mystery-shopper flag (see MYSTERY_SHOPPER_DECISIONS.md) — enables the
    // closing-loop-detection hangup below, since a flow-less session
    // otherwise never ends a call on its own.
    this.isShopper = false;
    this.callSid = null; // set once the Twilio 'start' event arrives (browser calls never get one)
    this._callLogId = null; // set once insertCallLog resolves — see the 'start' handler and close() below
    this.direction = null;
    this._shopperClosingCount = 0;
    this._closing = false;
    // Set by _stashForResume() right before a mid-call TwiML detour (e.g.
    // Pay) redirects the live call — the 'stop' event that follows is this
    // session's transport actually closing, but NOT the call ending, so
    // close() must skip billing finalization and the live-call-registry
    // removal it normally does on a real hangup. See pendingResumeSessions.
    this._pausedForResume = false;
    // Set right before a press_digit node's detour (see _executePressDigit)
    // to the node id we should jump straight to once the reconnect lands —
    // must survive the detour via _stashForResume/_tryResumeFromCallSid
    // like every other cross-detour field, since the resumed call gets a
    // brand-new CallSession instance, not this one.
    this._pendingPressDigitTarget = null;
    // Real bug found 2026-09-18 auditing the press_digit resume path: a
    // resumed payment node had no equivalent guard. _maybeRetireTurn fires
    // _executePayment purely off turnState.nodeType === 'payment', with no
    // way to tell "just entered this node" from "sitting in it after
    // resuming from its OWN prior detour" — so the very turn generated on
    // reconnect (which _runNodeTurn always marks isNodeEntry:true, entry or
    // resume alike) retired and re-triggered ANOTHER real <Pay> redirect
    // every single time, unconditionally, regardless of what the model did
    // or said. This matches a real live symptom from this session ("entered
    // everything.. it is back to enter card number") that was never
    // actually root-caused at the time. Set true right before the detour in
    // _executePayment, checked in _maybeRetireTurn, reset in
    // _applyTransition — and stashed/rehydrated like every other
    // cross-detour field, since the resumed call gets a brand-new
    // CallSession instance.
    this._paymentAwaitingResume = false;
    this._nudgeAttempts = new Map(); // nodeId -> count, see _maybeRetireTurn's deadlock-nudge fix
    this._queuedUserText = null; // see _onUserTurnComplete's in-flight-turn guard
    this.clientWs.onDrained = () => this._onAudioDrained();
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
    this.tenantNumber = null; // the tenant's OWN dialed number, see 'sms' node / _executeSmsNode
    this._pendingResponseTimer = null; // see _scheduleUserTurn (Response Wait Time)
    this._reminderTimer = null; // see _scheduleReminderIfConfigured (Reminder Message Frequency)
    this._reminderAttempts = 0;
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

  // eotThreshold defaults to the global DEEPGRAM_EOT_THRESHOLD — overridden
  // when a flow's Transcription Mode differs (see the reconnect right after
  // `this.flow = msg.flow` is set below). Deliberately NOT a per-node
  // setting like Interruption Sensitivity/Response Wait Time/Reminder
  // Frequency: this connects once, before ANY flow node is even known (a
  // real phone call's Deepgram socket opens in the constructor, before the
  // 'context' WS message carrying the flow ever arrives), so it can only
  // honestly be a flow-level Global Setting — a per-node UI control here
  // would just silently do nothing for whichever node isn't current at
  // connection time, which is exactly the kind of cosmetic/fake setting
  // this codebase's "compute, don't infer" discipline explicitly avoids.
  _connectDeepgram(eotThreshold = DEEPGRAM_EOT_THRESHOLD) {
    if (!DEEPGRAM_API_KEY) return;
    const isTwilio = this.clientWs instanceof TwilioCallAdapter;
    const url = buildDeepgramUrl(isTwilio, eotThreshold);
    this._deepgramEotThreshold = eotThreshold;
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
          // Interruption Sensitivity (per-node tuning, params.interruptionSensitivity)
          // — real threshold on the SAME mechanism as the fix above, not a
          // separate cosmetic setting: 'high' (default) keeps today's
          // behavior (barge in on the very first transcribed word); 'low'
          // requires several words of real transcript before committing to
          // an interruption, so a short "um"/"okay" that Deepgram DOES
          // manage to transcribe still doesn't cut the assistant off —
          // exactly the class of false-positive the StartOfTurn-vs-Update
          // fix above already targets, just with a tunable bar instead of a
          // fixed one-word bar.
          if (this._pendingBargeIn && this._transcriptMeetsInterruptionThreshold(text)) {
            this._pendingBargeIn = false;
            this._bargeIn();
          }
        }
      } else if (msg.event === 'EndOfTurn') {
        // High-confidence, semantically-aware turn end (eot_threshold) — this
        // is what nova-2 + acoustic VAD couldn't do: it waits for a complete
        // thought, not just a gap in the audio.
        const text = msg.transcript?.trim();
        if (text) this._scheduleUserTurn(text);
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
      // Real calendar booking (2026-09-17) — see tenantLookup.js. Only a
      // tenant with a real Cal.com connection gets check_availability/
      // book_appointment offered as tools at all (see
      // _buildNodeSystemPrompt's tool list) — nothing changes for every
      // other tenant.
      if (msg.calendar && typeof msg.calendar === 'object') {
        this.calendar = msg.calendar;
      }
      // Live-monitoring metadata (see the registry / GET /active-calls) — a
      // real routed call passes both; anonymous browser demos pass neither.
      if (typeof msg.tenantId === 'string' && msg.tenantId.trim()) {
        this.tenantId = msg.tenantId.trim();
      }
      if (typeof msg.phoneNumber === 'string' && msg.phoneNumber.trim()) {
        this.phoneNumber = msg.phoneNumber.trim();
      }
      if (typeof msg.tenantNumber === 'string' && msg.tenantNumber.trim()) {
        this.tenantNumber = msg.tenantNumber.trim();
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
        const bcs = this.flow.globalSettings || {};
        if (bcs.calendarTools === false) this.calendar = null; // per-agent switch: no live calendar lookups or bookings
        if (typeof bcs.backchannelEnabled === 'boolean') this.backchannelEnabled = bcs.backchannelEnabled;
        if (typeof bcs.backchannelFrequency === 'number' && bcs.backchannelFrequency >= 0 && bcs.backchannelFrequency <= 1) this.backchannelFrequency = bcs.backchannelFrequency;
        if (typeof bcs.backchannelDelayMs === 'number' && bcs.backchannelDelayMs > 0) this.backchannelDelayMs = bcs.backchannelDelayMs;
        console.log(`[call-loop] flow set — ${this.flow.nodes.length} nodes, starting at "${this.currentNodeId}"`);
        this.send({ type: 'flow_state', currentNodeId: this.currentNodeId, nodeType: this.flowNodesById.get(this.currentNodeId)?.type, collectedData: this.collectedData });
        // Transcription Mode reconnect — this arrives essentially
        // immediately after the socket opens (same WS frame sequence, no
        // extra round trip), before the caller has had any real chance to
        // speak, so swapping the Deepgram connection here can't cut off a
        // word the way a mid-call reconnect could. Only actually reconnects
        // when the resolved mode differs from what's already connected
        // (i.e. never, for the common case of a flow that doesn't set this
        // at all) — zero behavior change unless a tenant explicitly opts in.
        if (this.callSid && pendingAnsweredBy.has(this.callSid)) {
      const ab = pendingAnsweredBy.get(this.callSid);
      pendingAnsweredBy.delete(this.callSid);
      setImmediate(() => this.onAnsweredBy(ab));
    }
    const maxSec = Number(this.flow.globalSettings?.maxCallDurationSec) || 0;
        if (maxSec > 0 && !this._maxDurationTimer) {
          this._maxDurationTimer = setTimeout(() => this._endCallGracefully(`maxCallDurationSec ${maxSec}s reached`), maxSec * 1000);
        }
        const requestedThreshold = TRANSCRIPTION_MODE_THRESHOLDS[this.flow.globalSettings?.transcriptionMode];
        if (requestedThreshold && requestedThreshold !== this._deepgramEotThreshold) {
          console.log(`[call-loop] transcription mode "${this.flow.globalSettings.transcriptionMode}" requested — reconnecting Deepgram (eot_threshold ${this._deepgramEotThreshold} -> ${requestedThreshold})`);
          this.dgConnection?.close();
          this._connectDeepgram(requestedThreshold);
        }
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

  // Response Wait Time (per-node tuning, params.responseWaitTimeMs) — real
  // delay inserted between Deepgram's EndOfTurn firing and actually
  // generating a reply, so a caller who pauses mid-thought (which can
  // trip semantic endpointing into firing early) has a window to keep
  // talking before the assistant jumps in. A LATER EndOfTurn arriving
  // before the timer fires cancels the pending one and reschedules with
  // the newer (more complete) text, rather than both firing.
  _scheduleUserTurn(text) {
    if (this._reminderTimer) { clearTimeout(this._reminderTimer); this._reminderTimer = null; }
    if (this._pendingResponseTimer) {
      clearTimeout(this._pendingResponseTimer);
      this._pendingResponseTimer = null;
    }
    if (this._pendingDigitText) { text = `${this._pendingDigitText} ${text}`; this._pendingDigitText = null; }
    const lastAssistant = [...this.history].reverse().find((m) => m.role === 'assistant' && typeof m.content === 'string');
    if (shouldHoldForDigits(lastAssistant?.content, text)) {
      console.log(`[call-loop] holding partial phone number "${text}" for the rest of the digits`);
      this._pendingDigitText = text;
      this._pendingResponseTimer = setTimeout(() => {
        this._pendingResponseTimer = null;
        const held = this._pendingDigitText; this._pendingDigitText = null;
        this._onUserTurnComplete(held);
      }, DIGIT_HOLD_MS);
      return;
    }
    const node = this.flow ? this.flowNodesById?.get(this.currentNodeId) : null;
    const resp = Number(this.flow?.globalSettings?.responsiveness);
    const globalWaitMs = Number.isFinite(resp) && resp >= 0 && resp <= 1 && this.flow?.globalSettings?.responsiveness != null ? Math.round((1 - resp) * 1500) : 0;
    const waitMs = Math.max(0, Math.min(10000, Number(node?.params?.responseWaitTimeMs) || globalWaitMs));
    if (waitMs > 0) console.log(`[call-loop] waiting ${waitMs}ms before replying (responsiveness/response wait)`);
    if (waitMs <= 0) {
      this._onUserTurnComplete(text);
      return;
    }
    this._pendingResponseTimer = setTimeout(() => {
      this._pendingResponseTimer = null;
      this._onUserTurnComplete(text);
    }, waitMs);
  }

  // Reminder Message Frequency (per-node tuning, params.reminderMessageFrequencySec)
  // — if the caller goes silent for that long after the assistant finishes
  // speaking, proactively check in ("Are you still there?") instead of
  // waiting forever. Capped at REMINDER_MAX_ATTEMPTS consecutive reminders
  // so a caller who's actually hung up without a clean signal doesn't get
  // nagged in an endless loop — after the cap, the call is left to whatever
  // normal hangup/timeout handling already exists elsewhere.
  _scheduleReminderIfConfigured() {
    if (this._reminderTimer) { clearTimeout(this._reminderTimer); this._reminderTimer = null; }
    const node = this.flow ? this.flowNodesById?.get(this.currentNodeId) : null;
    const freqSec = Number(node?.params?.reminderMessageFrequencySec) || 0;
    const endSec = Number(this.flow?.globalSettings?.endCallAfterSilenceSec) || 0;
    if (this._closing) return;
    if (endSec > 0 && !this._silenceSince) this._silenceSince = Date.now();
    const remindOk = freqSec > 0 && (this._reminderAttempts || 0) < REMINDER_MAX_ATTEMPTS;
    if (!remindOk && endSec <= 0) return;
    const endLeftMs = endSec > 0 ? Math.max(0, endSec * 1000 - (Date.now() - this._silenceSince)) : Infinity;
    const delayMs = remindOk ? Math.min(freqSec * 1000, endLeftMs) : endLeftMs;
    this._reminderTimer = setTimeout(() => {
      this._reminderTimer = null;
      if (this._closing || this.activeTurn !== 0) return; // caller/assistant already talking again
      if (endSec > 0 && this._silenceSince && Date.now() - this._silenceSince >= endSec * 1000 - 50) {
        console.log(`[call-loop] caller silent for ${endSec}s (endCallAfterSilenceSec) — hanging up`);
        this._closing = true;
        this.close();
        return;
      }
      this._reminderAttempts = (this._reminderAttempts || 0) + 1;
      console.log(`[call-loop] caller silent for ${freqSec}s — sending reminder (attempt ${this._reminderAttempts})`);
      this.history.push({
        role: 'user',
        content: '[System note: the caller has been silent for a while. Check in briefly — e.g. "Are you still there?" — without repeating your last message.]',
      });
      const turnId = ++this.turnSeq;
      this.activeTurn = turnId;
      this.turnState = { id: turnId, llmDone: false, pendingTts: 0, startedSpeaking: false, saidNothing: false };
      this._generateTurn(turnId, Date.now(), { isNodeEntry: false });
    }, delayMs);
  }

  async _onUserTurnComplete(userText) {
    // Voicemail greetings are recognisable from their words well before Twilio's
    // async machine detection reports, and this adds no delay for live callers.
    const vmMode = this.flow?.globalSettings?.voicemailDetection;
    if ((vmMode === 'hangup' || vmMode === 'leave_message') && (this._vmUserTurns = (this._vmUserTurns || 0) + 1) <= 3 &&
        /leave (a|your) (message|name)|after the (beep|tone)|at the tone|not available (right now|to take)|(voice ?mail|mailbox)|can'?t take your call|record your message/i.test(userText)) {
      // A greeting saying "after the beep" ends BEFORE the beep; speaking a
      // message now would talk over it and not be recorded, so wait it out.
      if (vmMode === 'leave_message') setTimeout(() => this.onAnsweredBy('machine_end_transcript'), 3500);
      else this.onAnsweredBy('machine_end_transcript');
      return;
    }
    this._reminderAttempts = 0; // real activity — the silence streak is over
    this._silenceSince = null;
    if (this._reminderTimer) { clearTimeout(this._reminderTimer); this._reminderTimer = null; }
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
    // Live-call finding (2026-09-18): with interruptions 'off', a caller's
    // words during the agent's message still started a NEW turn that
    // superseded the current one — cutting off its remaining sentences even
    // though barge-in was disabled, and answering each utterance
    // separately. When the current step can't be interrupted, hold what the
    // caller says exactly like the not-yet-spoken case below and replay it
    // once this turn finishes.
    const uninterruptible = this._resolveInterruptionSensitivity() === 'off';
    const audibleTail = this.clientWs.isSpeaking?.() === true; // audio still playing after synthesis finished
    const inFlight = this.activeTurn !== 0 && this.turnState?.id === this.activeTurn;
    if ((inFlight && (!this.turnState.startedSpeaking || uninterruptible)) || (uninterruptible && audibleTail)) {
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

    // An opening step whose only exit is "always" has nothing left to do once the caller
    // responds; move on deterministically instead of hoping the model calls transition_flow.
    const cur = this.flow ? this.flowNodesById.get(this.currentNodeId) : null;
    if (cur?.type === 'greeting' && cur.edges?.length === 1 && /^\s*always\s*$/i.test(String(cur.edges[0].condition || '')) && this.flowNodesById.has(cur.edges[0].target)) {
      const target = cur.edges[0].target;
      const nextType = this.flowNodesById.get(target).type;
      console.log(`[call-loop] "always" exit from "${cur.id}" -> "${target}"`);
      if (['knowledge_base', 'function', 'goodbye', 'transfer', 'payment', 'press_digit', 'sms', 'code', 'mcp', 'subflow_ref', 'agent_transfer', 'extract_variable', 'logic_split'].includes(nextType)) {
        this._applyTransition({ next_node_id: target });
        return;
      }
      this.currentNodeId = target;
    }

    await this._generateTurn(turnId, turnStartedAt);
  }

  // Generates an assistant turn for the *current flow node* with no new
  // caller utterance — used for the flow's opening line, and for node types
  // (function/knowledge_base/goodbye/transfer) that act as soon as the flow
  // enters them rather than waiting on the caller to say something first.
  async _runNodeTurn(nodeId) {
    // A logic_split node never talks and never calls the LLM — it's a pure
    // code-path branch over this.collectedData (see _evaluateLogicSplit).
    // This guard covers the two ways a logic_split can become "current":
    // as the flow's starting node, or as the node a resumed call lands back
    // on. Mid-flow arrivals (via transition_flow) are handled directly in
    // _applyTransition instead, since that's where hop-limited chaining
    // (two splits routing to each other) is enforced.
    const entryNode = this.flow ? this.flowNodesById.get(nodeId) : null;
    if (entryNode && entryNode.type === 'logic_split') {
      this.currentNodeId = nodeId;
      const target = this._evaluateLogicSplit(entryNode);
      if (!target) {
        console.warn(`[call-loop] logic_split "${nodeId}" matched no edge and has no default edge — flow stalled here`);
        return;
      }
      this._applyTransition({ next_node_id: target });
      return;
    }
    // A press_digit node never talks either — it plays real DTMF tones into
    // an already-connected call (via a TwiML detour, see _executePressDigit)
    // so the agent can navigate another system's phone tree, then continues
    // once reconnected. _runNodeTurn fires twice for it: once on fresh entry
    // (execute the detour, don't advance yet — _pendingPressDigitTarget is
    // unset) and once on the post-detour resume (advance to the edge target
    // now that the tones have played — _pendingPressDigitTarget is set,
    // carried across via _stashForResume/_tryResumeFromCallSid since the
    // resumed call gets a brand-new CallSession instance).
    if (entryNode && entryNode.type === 'press_digit') {
      this.currentNodeId = nodeId;
      if (this._pendingPressDigitTarget) {
        const target = this._pendingPressDigitTarget;
        this._pendingPressDigitTarget = null;
        this._applyTransition({ next_node_id: target });
        return;
      }
      await this._executePressDigit(entryNode);
      return;
    }
    // subflow_ref never talks or calls the LLM itself either — it's a pure
    // redirect into the referenced subflow's own node graph, same shape as
    // logic_split above. See _enterSubflow.
    if (entryNode && entryNode.type === 'subflow_ref') {
      await this._enterSubflow(entryNode);
      return;
    }
    // extract_variable never talks: it reads what the caller already said into
    // named variables, then moves on through its first edge.
    if (entryNode && entryNode.type === 'extract_variable') {
      this.currentNodeId = nodeId;
      await this._executeExtractVariable(entryNode);
      const target = entryNode.edges?.[0]?.target;
      if (!target) {
        console.warn(`[call-loop] extract_variable "${nodeId}" has no outgoing edge — flow stalled here`);
        return;
      }
      this._applyTransition({ next_node_id: target });
      return;
    }
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
  // globalSettings.fillerWords (true | string[]): one short phrase if a tool
  // node is still running after 1.2s and nothing has been spoken this turn.
  _startToolFiller(node, turnId) {
    const fw = this.flow?.globalSettings?.fillerWords;
    // On by default for slow tool steps; fillerWords: false turns it off.
    if (fw === false || !['function', 'code', 'mcp', 'knowledge_base'].includes(node?.type)) return null;
    const phrases = Array.isArray(fw) ? fw.filter((p) => typeof p === 'string' && p.trim()) : ['One moment.', 'Let me check that.'];
    if (phrases.length === 0) return null;
    return setTimeout(() => {
      if (this.activeTurn !== turnId || this._closing || this.turnState?.id !== turnId || this.turnState.startedSpeaking) return;
      const phrase = phrases[Math.floor(Math.random() * phrases.length)].trim();
      console.log(`[call-loop] turn ${turnId} tool filler: "${phrase}"`);
      this._speak(phrase, turnId, Date.now());
    }, 1200);
  }

  async _generateTurn(turnId, turnStartedAt, { isNodeEntry = false, suppressTransitionTool = false, isCallOpening = false, ranSubagentTools = new Set(), forceTransition = false } = {}) {
    if (!anthropic) {
      this.send({ type: 'error', message: 'ANTHROPIC_API_KEY not configured' });
      if (this.turnState?.id === turnId) this.turnState.llmDone = true;
      return;
    }

    const node = this.flow ? this.flowNodesById.get(this.currentNodeId) : null;
    if (node?.params?.model && !VALID_LLM_MODELS.has(node.params.model)) {
      console.warn(`[call-loop] node "${node.id}" params.model "${node.params.model}" not supported — using ${this.llmModel}`);
    }
    if (this.turnState?.id === turnId) {
      this.turnState.nodeType = node?.type || null;
      // Whose audio is (about to be) playing — see _resolveInterruptionSensitivity.
      this._audioNode = node || null;
      this.turnState.nodeParams = node?.params || null;
    }

    // Defensive: logic_split should always be intercepted by _runNodeTurn or
    // _applyTransition before reaching here (see LOGIC_SPLIT_MAX_HOPS), so
    // this should never actually fire — but a split has no prompt to build a
    // conversational turn from anyway, so if it ever does land here, treat
    // it as a stall rather than sending a broken/empty turn to the model.
    if (node?.type === 'logic_split' || node?.type === 'press_digit') {
      console.error(`[call-loop] _generateTurn reached a ${node.type} node ("${this.currentNodeId}") — this should have been intercepted earlier; stalling instead of calling the LLM`);
      if (this.turnState?.id === turnId) this.turnState.llmDone = true;
      return;
    }

    // A function node's side effect happens before it says anything, and
    // only once — on the turn that actually enters the node (via
    // _runNodeTurn). Without the isNodeEntry guard, every later turn where
    // the conversation just happens to still be sitting in this node (e.g.
    // the caller asking a follow-up before transitioning away) would re-run
    // the webhook, which is wrong — it's a one-time side effect of arriving
    // at the node, not a per-turn one.
    const fillerTimer = isNodeEntry ? this._startToolFiller(node, turnId) : null;
    try {
      if (node?.type === 'function' && isNodeEntry) await this._executeFunctionNode(node);
      if (node?.type === 'knowledge_base' && isNodeEntry) await this._executeKnowledgeBaseNode(node);
      if (node?.type === 'sms' && isNodeEntry) await this._executeSmsNode(node);
      if (node?.type === 'code' && isNodeEntry) await this._executeCodeNode(node);
      if (node?.type === 'mcp' && isNodeEntry) await this._executeMcpNode(node);
    } finally {
      if (fillerTimer) clearTimeout(fillerTimer);
    }

    // A transfer/goodbye step with a fixed message says exactly that, without the model.
    const fixedLine = isNodeEntry && (node?.type === 'transfer' || node?.type === 'goodbye' || node?.type === 'agent_transfer' || node?.type === 'greeting') && typeof node.params?.spokenMessage === 'string'
      ? this._interpolateFields(node.params.spokenMessage).trim() : '';
    if (fixedLine && this.turnState?.id === turnId) {
      this.history.push({ role: 'assistant', content: fixedLine });
      console.log(`[call-loop] [call ${this.callSid || this.id}] turn ${turnId} assistant: "${fixedLine}"`);
      this._speak(fixedLine, turnId, turnStartedAt);
      this.turnState.llmDone = true;
      this._maybeRetireTurn(turnId);
      return;
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
    // A node reached inside a subflow with no edges of its own is a
    // subflow-internal terminal, not a dead end — the valid "next" targets
    // are the original subflow_ref node's real edges back in the parent
    // flow. But if the subflow_ref itself has NO edges either (a subflow
    // whose whole point is to end the call itself — e.g. a shared "transfer
    // or take a callback" block invoked from many places, never returning
    // control anywhere), there's nothing to hand back to: don't attach a
    // transition tool with an empty enum, which some providers reject as
    // an invalid tool schema. The node just ends the call normally (every
    // node here is goodbye/transfer, both already call-ending types).
    const returnNode = this._subflowStack[this._subflowStack.length - 1]?.returnNode;
    const inSubflowAtTerminal = node && node.edges.length === 0 && this._subflowStack.length > 0 && returnNode?.edges.length > 0;
    if (node && !suppressTransitionTool && (node.edges.length > 0 || inSubflowAtTerminal)) {
      tools.push(this._buildTransitionTool(inSubflowAtTerminal ? returnNode : node));
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

      // Real calendar booking (2026-09-17) — only offered when this tenant
      // has a real Cal.com connection (see tenantLookup.js/onClientMessage),
      // and only on an extraction step (the same kind of node that already
      // gathers name/time) — never a standalone node type, matching how
      // record_field itself works: a tool available mid-turn, not a
      // one-shot node like 'function'/'transfer'/'payment'.
      if (this.calendar) {
        tools.push({
          name: 'check_availability',
          description:
            'Check real calendar availability for a given day. Call this whenever the caller mentions a day ' +
            'they want, BEFORE proposing or confirming any specific time — never guess or invent available times.',
          input_schema: {
            type: 'object',
            properties: {
              date: { type: 'string', description: 'The date to check, as YYYY-MM-DD' },
            },
            required: ['date'],
          },
        });
        tools.push({
          name: 'book_appointment',
          description:
            'Books the appointment for real on the calendar. Only call this AFTER check_availability has ' +
            'confirmed the exact time is open, and the caller has explicitly confirmed they want it. You MUST ' +
            'also have spelled the email address back letter-by-letter and gotten a yes — a mis-transcribed ' +
            'email books a real appointment no one can be reached to confirm or fix.',
          input_schema: {
            type: 'object',
            properties: {
              name: { type: 'string', description: "Caller's name" },
              email: { type: 'string', description: "Caller's email address for the booking confirmation" },
              // A label, not a timestamp you construct yourself — real bug
              // found live: asking the model to build its own ISO 8601
              // string got the UTC offset wrong and booked the wrong hour
              // while confidently confirming the right one out loud.
              selectedTime: { type: 'string', description: 'The EXACT time string from check_availability\'s results (e.g. "4:00 PM") — copy it exactly, do not reformat or compute a timestamp yourself.' },
            },
            required: ['name', 'email', 'selectedTime'],
          },
        });
      }
    }

    // 'subagent' node — attaches multiple pre-configured tools (any mix of
    // function/code/sms/mcp/transfer) to ONE node and lets the model decide
    // WHEN (if ever) to call each, mid-conversation, across as many turns as
    // it takes — unlike every other node type here, which maps to exactly
    // one auto-run behavior. Matches Retell's own Subagent node model: the
    // tools are configured by the flow author (server-side, already
    // {{field}}-interpolated by each executor same as their dedicated node
    // types), not supplied as arguments by the model — the model's only job
    // is picking which one to invoke and when. Reuses the SAME executors as
    // the dedicated function/code/sms/mcp/transfer node types via a
    // synthetic node built from each tool's config, rather than
    // duplicating their logic.
    if (node?.type === 'subagent') {
      for (const toolConfig of this._parseSubagentTools(node)) {
        // Live-call finding (2026-09-18): a tool that already ran in this
        // follow-up chain stayed offered, and the model re-called it (3x
        // `calc` in one turn) — a real double-fire risk for side-effecting
        // tools like sms/function. Each tool runs at most once per chain.
        if (ranSubagentTools.has(toolConfig.id)) continue;
        tools.push({
          name: `subagent_tool_${toolConfig.id}`,
          description: toolConfig.description || `Runs the "${toolConfig.id}" tool.`,
          input_schema: { type: 'object', properties: {}, required: [] },
        });
      }
    }

    let firstTokenAt = null;
    let assistantText = '';
    const chunker = new SentenceChunker((sentence) => {
      if (this.activeTurn !== turnId) {
        console.log(`[call-loop] turn ${turnId} sentence chunk dropped — activeTurn is now ${this.activeTurn}: "${sentence}"`);
        return;
      }
      if (isStageDirection(sentence)) {
        console.log(`[call-loop] turn ${turnId} dropped a stage direction instead of speaking it: "${sentence}"`);
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
        model: (VALID_LLM_MODELS.has(node?.params?.model) ? node.params.model : this.llmModel),
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
          console.log(`[call-loop] turn ${turnId} LLM TTFB: ${firstTokenAt - turnStartedAt}ms (model ${VALID_LLM_MODELS.has(node?.params?.model) ? node.params.model : this.llmModel})`);
        }
        assistantText += delta;
        chunker.push(delta);
      });

      const final = await stream.finalMessage();
      if (final.usage) {
        this.cost.addLlmUsage(this.llmModel, final.usage.input_tokens, final.usage.output_tokens);
        console.log(`[llm-usage] turn ${turnId} node=${node?.id}(${node?.type}) in=${final.usage.input_tokens} out=${final.usage.output_tokens} sysChars=${systemPrompt.length} histMsgs=${this.history.length} histChars=${JSON.stringify(this.history).length} tools=${tools.length}`);
      }
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

      // Real calendar booking (2026-09-17): unlike record_field, the model
      // needs to actually SEE the real API result and speak from it (real
      // available times, a real confirmation) — not just silently persist
      // a value. Rather than inventing Anthropic's native multi-round
      // tool_result protocol (a different message shape than anything else
      // here uses), this reuses the exact system-note-injection +
      // follow-up-turn pattern already proven for the deadlock-nudge fix:
      // push what actually happened as a synthetic note, generate a fresh
      // turn to react to it. Returns early — this turn's own (likely
      // empty or premature) text is superseded by the follow-up turn's.
      const calendarToolUse = final.content.find(
        (b) => b.type === 'tool_use' && (b.name === 'check_availability' || b.name === 'book_appointment')
      );
      if (calendarToolUse && this.activeTurn === turnId) {
        if (assistantText) this.history.push({ role: 'assistant', content: assistantText });
        await this._handleCalendarTool(calendarToolUse);
        return;
      }
      // Subagent tool call — same "execute, then generate a follow-up turn"
      // shape as the calendar tools above, just routed by id to whichever
      // of this node's configured tools the model actually picked.
      const subagentToolUse = node?.type === 'subagent'
        ? final.content.find((b) => b.type === 'tool_use' && b.name?.startsWith('subagent_tool_'))
        : null;
      if (subagentToolUse && this.activeTurn === turnId) {
        if (assistantText) this.history.push({ role: 'assistant', content: assistantText });
        const toolId = subagentToolUse.name.slice('subagent_tool_'.length);
        const toolConfig = this._parseSubagentTools(node).find((t) => t.id === toolId);
        if (toolConfig) {
          await this._executeSubagentTool(toolConfig);
          // A transfer tool ends the call itself (redirects it) — no
          // follow-up turn to generate, and this.activeTurn/turnState may
          // already be in a different state by the time it returns.
          if (toolConfig.kind === 'transfer') return;
          const followUpTurnId = ++this.turnSeq;
          this.activeTurn = followUpTurnId;
          this.turnState = { id: followUpTurnId, llmDone: false, pendingTts: 0, startedSpeaking: false, saidNothing: false };
          await this._generateTurn(followUpTurnId, Date.now(), { isNodeEntry: false, ranSubagentTools: new Set([...ranSubagentTools, toolId]) });
        } else {
          console.warn(`[call-loop] subagent node "${node.id}" — model called unknown tool id "${toolId}"`);
        }
        return;
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
        if (!assistantText && (node?.type === 'goodbye' || node?.type === 'transfer' || node?.type === 'agent_transfer')) {
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
        } else if (!toolUse && this.turnState?.id === turnId && !forceTransition && this._claimsHandoff(node, assistantText)) {
          // The agent told the caller they are being transferred but never moved to the transfer step.
          const tt0 = tools.find((t) => t.name === 'transition_flow');
          const targets = (node.edges || []).map((e) => e.target).filter((id) => ['transfer', 'agent_transfer'].includes(this.flowNodesById.get(id)?.type));
          if (tt0 && targets.length) {
            try {
              const tt = { ...tt0, input_schema: { ...tt0.input_schema, properties: { ...tt0.input_schema.properties, next_node_id: { ...tt0.input_schema.properties.next_node_id, enum: targets } } } };
              const last = this.history[this.history.length - 1];
              const msgs = last?.role === 'user' ? this.history : [...this.history, { role: 'user', content: '[System note: you told the caller you are transferring them. Do it now.]' }];
              const r = await anthropic.messages.create({
                model: (VALID_LLM_MODELS.has(node?.params?.model) ? node.params.model : this.llmModel),
                system: systemPrompt, max_tokens: 120, messages: msgs, tools: [tt], tool_choice: { type: 'tool', name: 'transition_flow' },
              });
              const tu = r.content.find((b) => b.type === 'tool_use');
              if (tu) { this.turnState.transition = tu.input; console.log(`[call-loop] forced handoff after spoken claim -> ${JSON.stringify(tu.input)}`); }
            } catch (err) {
              console.error('[call-loop] forced handoff failed', err.message);
            }
          }
        } else if (forceTransition && this.turnState?.id === turnId) {
          // The nudge got a spoken summary but no transition_flow call, which would leave the call
          // stuck on this node while the agent says it is moving on. Ask for the transition alone.
          const tt = tools.find((t) => t.name === 'transition_flow');
          if (tt) {
            try {
              const last = this.history[this.history.length - 1];
              const msgs = last?.role === 'user' ? this.history : [...this.history, { role: 'user', content: '[System note: choose the next step now.]' }];
              const r = await anthropic.messages.create({
                model: (VALID_LLM_MODELS.has(node?.params?.model) ? node.params.model : this.llmModel),
                system: systemPrompt, max_tokens: 120, messages: msgs, tools: [tt], tool_choice: { type: 'tool', name: 'transition_flow' },
              });
              const tu = r.content.find((b) => b.type === 'tool_use');
              if (tu && this.turnState?.id === turnId) {
                this.turnState.transition = tu.input;
                console.log(`[call-loop] forced transition after nudge -> ${JSON.stringify(tu.input)}`);
              }
            } catch (err) {
              console.error('[call-loop] forced transition failed', err.message);
            }
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

  _maybeSpeakBackchannel(turnId, only) {
    if (this.activeTurn !== turnId) return; // barge-in or turn already resolved
    const pool = only ? this.backchannelWords.filter((w) => only.includes(w)) : this.backchannelWords;
    if (!pool.length) return;
    const word = pool[Math.floor(Math.random() * pool.length)];
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
    // Agent Handbook (2026-09-18, builder parity Phase 2) — reference
    // material that applies across the WHOLE flow, not one node's own
    // instructions (node.prompt below). Included on every node's turn, same
    // as Retell's own "Global Prompt" being agent-wide rather than
    // per-step. Kept as its own clearly-labeled block so the model can tell
    // it apart from this step's actual task.
    if (gs.handbook?.trim()) {
      prompt += `Reference handbook (background knowledge for this business — consult as needed, don't recite it verbatim unless asked):\n${gs.handbook.trim()}\n\n`;
    }
    // Resumed from this node's own Pay detour (see _executePayment /
    // _paymentAwaitingResume) — this is the turn generated right after the
    // caller finished (or canceled/failed) Twilio's real card-entry flow.
    // The node's original prompt ("let them know you're transferring them
    // now") is wrong here — that already happened — and asking the model to
    // treat this as an ordinary "node entry" turn is what let it repeat
    // that line, sounding like the payment was starting over. React to the
    // real outcome instead.
    if (node.type === 'payment' && this._paymentAwaitingResume && this.collectedData.payment_status) {
      prompt +=
        `Step instructions (original, for context only — do NOT repeat this; the transfer to enter ` +
        `card details already happened): ${node.prompt}\n` +
        `The payment attempt just finished — the real outcome is payment_status = ` +
        `"${this.collectedData.payment_status}" (also see "Already collected" below). Tell the ` +
        `caller that outcome naturally in one short sentence, then immediately call transition_flow ` +
        `to move to this step's edge that matches it. Don't ask the caller anything, don't say ` +
        `you're transferring them again, and don't attempt the payment yourself — that already ran.\n`;
    } else if (isNodeEntry) {
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
      // Numbered with a running counter, not hand-tracked ternaries —
      // real bug caught writing the email-read-back step below: two
      // different steps both computed to the same number ('5') because
      // the ternary math wasn't updated when a step was inserted between
      // them. A counter can't drift out of sync with itself the way two
      // separate ternary expressions can.
      let stepNum = 1;
      prompt += `For this step, in order:\n`;
      if (missingFields.length > 0) {
        prompt += `${stepNum++}. You still need: ${missingFields.join(', ')}. Ask for ALL of these together in ONE question, in the SAME turn — don't ask one at a time. Do not ask about anything not in this list — it's already been captured (see "Already collected" below).\n`;
      } else {
        prompt += `${stepNum++}. Every field for this step is already captured (see "Already collected" below) — do not ask for any of them again. Move straight to confirming/summarizing.\n`;
      }
      prompt += `${stepNum++}. The moment the caller gives you a field, call record_field for it — but ALWAYS also say something out loud to the caller in that same turn. Calling record_field is a silent background action, never a substitute for actually replying — never let a turn consist of only a tool call with nothing spoken.\n`;
      if (hasTimeField) {
        prompt += `${stepNum++}. If a date/time answer is vague ("afternoon", "next week"), propose ONE concrete slot inside their range and get a yes before treating it as captured.\n`;
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
      prompt += `${stepNum++}. Names and numbers are easy to mishear. Before treating any field as final, you MUST ask the caller a direct yes/no question repeating back exactly what you captured (e.g. "Got it, Alex, for 3pm — did I get that right?"). Calling record_field is NOT confirmation — it only means you heard something. Wait for the caller to actually say yes (or correct you) before moving on.\n`;
      prompt += `${stepNum++}. Phone numbers and IDs read out digit by digit can arrive with gaps or fragments. If the caller gives a number in pieces across several replies, join the pieces in order and only ask for what is still missing; never ask them to repeat digits they already gave. Read the complete number back once. If the caller says it is wrong and repeats it, take their latest complete digit string and confirm it at most ONE more time; never confirm the same field more than twice — after that, accept the caller's latest version and move on.\n`;
      // Real call finding (2026-09-17): the generic "read back" rule above
      // didn't stop a mis-transcribed email (an extra letter added) from
      // going straight into a real booking with no confirmation at all —
      // spoken email addresses fail differently and more often than names:
      // spelled character-by-character, easy to drop or add one, and
      // unlike a name a caller usually can't hear something's wrong from
      // context. Worth its own explicit, harder-to-skip rule rather than
      // trusting it's covered by "numbers are easy to mishear."
      if (this.calendar && Object.keys(node.extract).includes('email')) {
        prompt += `${stepNum++}. Before calling book_appointment, you MUST spell the caller's email address back letter-by-letter (e.g. "That's A-L-E-X at gmail dot com, is that right?") and get an explicit yes. This is separate from the general read-back above — do it even if you already confirmed the name. Never call book_appointment on an unconfirmed email.\n`;
      }
      prompt += `${stepNum++}. Only after the caller has explicitly confirmed every field this way, say ONE summary sentence with all of them ("So that's Alex Morgan at 2pm tomorrow.") and THEN call transition_flow in the same turn — don't transition silently, without a prior yes/no confirmation, or without ever stating the final summary.\n`;
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
      // Transition Flexibility (2026-09-18, builder parity Phase 2) — how
      // literally to read an edge's condition text. Default (unset/
      // 'flexible') is today's existing behavior: use judgment on a
      // reasonably close match. 'strict' is a new, more conservative mode
      // for flows where a wrong transition is costly (e.g. routing to the
      // wrong department) and staying on the current step to ask a
      // clarifying question is preferable to guessing.
      prompt += gs.transitionFlexibility === 'strict'
        ? `Only call transition_flow when a condition below is clearly and unambiguously met by ` +
          `what the caller actually said — do not transition on a plausible guess or a partial ` +
          `match. If it's ambiguous which edge applies, ask a clarifying question instead of ` +
          `transitioning.\n`
        : `Use your judgment: if what the caller said reasonably matches the intent of a condition ` +
          `below, even if not a verbatim match, transition on it rather than demanding exact phrasing.\n`;
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
    // Fine-tuning examples — per-node few-shot guidance the flow author
    // writes for tricky/specific scenarios this step tends to hit (e.g. a
    // caller who gives a partial address, or asks to reschedule mid-booking).
    // Deliberately just prose handed to the model, not a structured
    // input/output pair format — this is guidance the model reads, not code
    // that executes, so free text the author writes naturally is more
    // useful than forcing a rigid schema on it.
    if (node.params?.fineTuningExamples?.trim()) {
      prompt +=
        `\nExample scenarios for this step (for guidance — adapt to what the caller actually says, ` +
        `don't recite these verbatim):\n${node.params.fineTuningExamples.trim()}\n`;
    }
    prompt += gs.allowInterruptions === false
      ? 'Complete your sentences before listening.\n'
      : 'Allow the caller to interrupt you.\n';
    prompt +=
      'Keep replies to 1-2 short sentences unless asked for more detail. Never use markdown, ' +
      'bullet points, or emoji — this is spoken audio. Always say at least one sentence out ' +
      'loud on every turn, even if you are also calling a tool — never respond with nothing. ' +
      'When reading a phone number back to the caller (e.g. to confirm a callback number), say ' +
      'the digits individually, grouped naturally (e.g. "four two five, six two eight, four ' +
      'eight eight seven") — never as one large number ("four billion..."); the same goes for ' +
      'any other long digit string like a confirmation code.';
    return this._applyVariables(prompt);
  }

  // subflow_ref is a pure redirect: it embeds a snapshot of another flow's
  // nodes (taken at publish time) and swaps them in as the active node map,
  // then immediately runs the subflow's own start node. Reaching one of the
  // subflow's own terminal (zero-edge) nodes pops back to the parent scope
  // and exposes the ORIGINAL subflow_ref node's real edges as the transition
  // tool (see the tool-attachment gate below and the fallback in
  // _applyTransition) — so a subflow can have multiple distinct exit paths,
  // not just one synthetic "return".
  async _enterSubflow(node) {
    let subflowNodes;
    try {
      subflowNodes = JSON.parse(node.params?.subflowNodes || '[]');
    } catch {
      console.error(`[call-loop] subflow_ref "${node.id}" has invalid embedded subflowNodes JSON`);
      return;
    }
    const startId = node.params?.subflowStartNodeId;
    if (!Array.isArray(subflowNodes) || subflowNodes.length === 0 || !startId) {
      console.error(`[call-loop] subflow_ref "${node.id}" has no embedded nodes — skipping`);
      return;
    }
    this._subflowStack.push({ parentNodesById: this.flowNodesById, returnNode: node });
    this.flowNodesById = new Map(subflowNodes.map((n) => [n.id, n]));
    await this._runNodeTurn(startId);
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
  _applyTransition({ next_node_id, extracted }, logicSplitHops = 0) {
    // A real transition means we're actually leaving whatever node we were
    // in — safe to clear unconditionally (harmless no-op unless we were
    // sitting in a payment node). Without this, a genuine LATER re-entry
    // into a payment node (e.g. a retry-payment edge) would stay
    // permanently suppressed by the first attempt's guard.
    this._paymentAwaitingResume = false;
    if (extracted && typeof extracted === 'object') {
      Object.assign(this.collectedData, extracted);
    }
    let nextNode = this.flowNodesById.get(next_node_id);
    // Not found in the current (possibly subflow-local) scope — if we're
    // inside a subflow, the model was actually choosing one of the return
    // node's real parent-flow edges, so pop back out and look again there.
    if (!nextNode && this._subflowStack.length > 0) {
      const frame = this._subflowStack.pop();
      this.flowNodesById = frame.parentNodesById;
      nextNode = this.flowNodesById.get(next_node_id);
    }
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

    // A logic_split routes again immediately, purely in code — no LLM turn,
    // no caller-facing side effect. Recursing here (rather than going
    // through _runNodeTurn) is what lets two splits chain straight through
    // in the same tick; the hop cap exists because two splits can route to
    // each other and nothing else here would ever stop that.
    if (nextNode.type === 'logic_split') {
      if (logicSplitHops >= LOGIC_SPLIT_MAX_HOPS) {
        console.error(`[call-loop] logic_split hop limit (${LOGIC_SPLIT_MAX_HOPS}) exceeded at "${next_node_id}" — likely two splits routing to each other; stopping here`);
        return;
      }
      const target = this._evaluateLogicSplit(nextNode);
      if (!target) {
        console.warn(`[call-loop] logic_split "${next_node_id}" matched no edge and has no default edge — flow stalled here`);
        this.currentNodeId = next_node_id;
        return;
      }
      this._applyTransition({ next_node_id: target }, logicSplitHops + 1);
      return;
    }

    // These node types act as soon as the flow enters them (run a webhook,
    // look something up, say goodbye and hang up, transfer the call) rather
    // than waiting for the caller to speak first.
    // subflow_ref belongs here too — like logic_split/press_digit, it never
    // waits on the caller; it immediately redirects into the subflow's own
    // start node (see _enterSubflow). Missing this meant a fresh arrival at
    // a subflow_ref node just sat on it waiting for the caller to speak,
    // and the caller's next turn then got generated directly against the
    // bare subflow_ref node (no prompt, only its OWN edges) instead of ever
    // entering the subflow — found via a real test call where a
    // subflow_ref was skipped over entirely.
    const AUTO_ADVANCE_TYPES = new Set(['function', 'knowledge_base', 'goodbye', 'transfer', 'payment', 'press_digit', 'sms', 'code', 'mcp', 'subflow_ref', 'agent_transfer', 'extract_variable']);
    // An 'extraction' node normally waits for the caller's next utterance —
    // correct when it still needs to ask something the caller hasn't
    // answered yet, since the current turn's own text already asked it
    // conversationally (see the header comment on _applyTransition). But
    // real bug, reproduced on a live call: a "booking" extraction node whose
    // extract fields (name, preferred_time, ...) were ALL already collected
    // — the model said "let me get that booked for you" (nothing left to
    // ask) and transitioned in, expecting to immediately call
    // check_availability/book_appointment itself. With no un-extracted
    // fields left, this just sat there waiting for the caller to speak,
    // dead silent for 46 real seconds until the caller said "hello?" out of
    // confusion. If every field this node would extract is already
    // satisfied, there's nothing left for the CALLER to answer — treat it
    // like the auto-advance types so the model gets a turn to act.
    const extractionFullySatisfied =
      nextNode.type === 'extraction' &&
      nextNode.extract &&
      Object.keys(nextNode.extract).every((field) => {
        const value = this.collectedData[field];
        return value !== undefined && value !== null && String(value).trim() !== '';
      });
    const hasFixedOpener = nextNode.type === 'greeting' && typeof nextNode.params?.spokenMessage === 'string' && nextNode.params.spokenMessage.trim() !== '';
    if (AUTO_ADVANCE_TYPES.has(nextNode.type) || extractionFullySatisfied || hasFixedOpener) {
      this._runNodeTurn(next_node_id);
    } else {
      this.currentNodeId = next_node_id;
      // A transition made without saying anything would leave the new step waiting
      // for a caller who has nothing to react to (both sides silent). Have it speak now.
      if ((nextNode.type === 'extraction' || nextNode.type === 'greeting') && this.turnState && !this.turnState.spokeText && !this._closing) {
        console.log(`[call-loop] silent transition into "${next_node_id}" — running its opening turn`);
        this._runNodeTurn(next_node_id);
        // The caller would otherwise wait out a second model call in silence; a cached one-word acknowledgement covers it.
        if (this.backchannelEnabled) this._maybeSpeakBackchannel(this.activeTurn, ['Got it.', 'Sure thing.']);
      }
    }
  }

  // Picks the first edge whose structured condition matches this.collectedData,
  // or the first conditionless edge as an explicit default/fallback. Returns
  // null if nothing matches and there's no default — the caller decides how
  // to handle a stalled split.
  _evaluateLogicSplit(node) {
    for (const edge of node.edges || []) {
      if (!edge.condition || typeof edge.condition !== 'object') {
        return edge.target; // conditionless edge = default
      }
      if (this._evaluateStructuredCondition(edge.condition, this.collectedData)) {
        return edge.target;
      }
    }
    return null;
  }

  // Real evaluation, not LLM judgment — the only place in this file a
  // condition string/object is actually compared rather than handed to the
  // model. Extracted fields have no real type system today (extract only
  // ever declares 'string'), so this coerces to numeric comparison
  // best-effort and otherwise falls back to a trimmed, case-insensitive
  // string comparison.
  _evaluateStructuredCondition({ field, operator, value }, data) {
    const actual = data ? data[field] : undefined;
    const actualNum = Number(actual);
    const valueNum = Number(value);
    const bothNumeric =
      actual !== undefined && actual !== '' && !Number.isNaN(actualNum) &&
      value !== undefined && value !== '' && !Number.isNaN(valueNum);
    const cmp = bothNumeric
      ? (actualNum < valueNum ? -1 : actualNum > valueNum ? 1 : 0)
      : String(actual ?? '').trim().toLowerCase().localeCompare(String(value ?? '').trim().toLowerCase());
    switch (operator) {
      case '==': return cmp === 0;
      case '!=': return cmp !== 0;
      case '>': return cmp > 0;
      case '<': return cmp < 0;
      case '>=': return cmp >= 0;
      case '<=': return cmp <= 0;
      default:
        console.warn(`[call-loop] logic_split: unknown operator "${operator}" — treating as no match`);
        return false;
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
      content: this._applyVariables(`[System note: knowledge base content for this step — answer using these facts when relevant, otherwise say you're not sure and offer to have someone follow up:\n${qa}]`),
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

  // In-call SMS — texts the caller (or an explicit number) mid-call, e.g. a
  // confirmation link or a payment link. Unlike press_digit/payment, this
  // never touches the live voice channel at all — POST /Messages.json is an
  // ordinary async REST call, same shape as _redirectForDetour's own fetch,
  // with nothing to detour or resume — so it follows the SAME isNodeEntry
  // pattern as function/knowledge_base nodes (see _generateTurn), not the
  // press_digit/payment pattern. A failed send is folded into history as a
  // system note, same as a failed function-node webhook, rather than
  // silently doing nothing — the model should know to react to it.
  async _executeSmsNode(node) {
    if (!TWILIO_ACCOUNT_SID || !TWILIO_AUTH_TOKEN) {
      console.warn(`[call-loop] sms node "${node.id}" requires Twilio credentials — skipping`);
      return;
    }
    const to = this._interpolateFields(node.params?.to || '').trim() || this.phoneNumber;
    const from = this.tenantNumber;
    const body = this._interpolateFields(node.params?.body || '');
    if (!to) {
      console.warn(`[call-loop] sms node "${node.id}" has no destination number (no params.to and no caller number on this session) — skipping`);
      return;
    }
    if (!from) {
      console.warn(`[call-loop] sms node "${node.id}" has no tenant number to send from on this session — skipping`);
      return;
    }
    if (!body.trim()) {
      console.warn(`[call-loop] sms node "${node.id}" has no message body — skipping`);
      return;
    }
    try {
      const auth = Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64');
      const res = await fetch(
        `https://api.twilio.com/2010-04-01/Accounts/${TWILIO_ACCOUNT_SID}/Messages.json`,
        {
          method: 'POST',
          headers: { Authorization: `Basic ${auth}`, 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ To: to, From: from, Body: body }).toString(),
          signal: AbortSignal.timeout(8000),
        }
      );
      const result = await res.json().catch(() => ({}));
      if (res.ok) {
        console.log(`[call-loop] sms node "${node.id}" sent -> ${to} (sid ${result.sid})`);
      } else {
        console.error(`[call-loop] sms node "${node.id}" failed: HTTP ${res.status} ${JSON.stringify(result)}`);
        this.history.push({
          role: 'user',
          content: `[System note: the text message failed to send — let the caller know and offer another way to get the info]`,
        });
      }
    } catch (err) {
      console.error(`[call-loop] sms node "${node.id}" request failed`, err);
      this.history.push({
        role: 'user',
        content: `[System note: the text message failed to send — let the caller know and offer another way to get the info]`,
      });
    }
  }

  // A 'code' node runs flow-author-authored JavaScript in a real sandbox
  // (see CODE_NODE_* constants above) — for calculations, data formatting,
  // or a lightweight HTTP lookup that doesn't need a whole function-node
  // webhook. No secrets from this process are ever exposed into the
  // sandbox: only `dv` (this.collectedData, matching Retell's own Code node
  // inputs) and a network-restricted `fetch`. Follows the same isNodeEntry
  // pattern as function/knowledge_base/sms — a one-time side effect on
  // entry, not re-run on every turn spent in the node.
  async _executeCodeNode(node) {
    const code = String(node.params?.code || '').slice(0, CODE_NODE_MAX_SOURCE_CHARS);
    if (!code.trim()) {
      console.warn(`[call-loop] code node "${node.id}" has no code — skipping`);
      return;
    }

    let context;
    try {
      context = await newAsyncContext();
      context.runtime.setMemoryLimit(CODE_NODE_MEMORY_LIMIT_BYTES);
      context.runtime.setInterruptHandler(shouldInterruptAfterDeadline(Date.now() + CODE_NODE_TIMEOUT_MS));

      // dv: read-only dynamic variables, all strings — mirrors Retell's own
      // Code node input shape.
      const dvResult = context.evalCode(`(${JSON.stringify(this.collectedData || {})})`);
      const dv = context.unwrapResult(dvResult);
      context.setProp(context.global, 'dv', dv);
      dv.dispose();

      // __hostFetch is asyncified (suspends the whole WASM module while the
      // real host fetch runs) but exposed to the sandboxed script as a
      // plain SYNCHRONOUS `fetch(url, options)` — no await, no Promise, no
      // module/top-level-await handling needed anywhere in this method.
      // This is the library's own documented "async on host, sync in
      // QuickJS" pattern; an earlier version of this tried to expose an
      // async `fetch` instead and hit real, verified problems (an unawaited
      // async IIFE's return value dumps as a raw pending-Promise state, and
      // `type: "module"` top-level-await mode never actually settles its
      // exports promise via evalCodeAsync) — this sync-facade design has no
      // such edge case and was the one actually verified working end-to-end
      // (calc+dv, object return, a real network fetch, the SSRF guard, and
      // the interrupt timeout) before landing.
      const hostFetchHandle = context.newAsyncifiedFunction('__hostFetch', async (urlHandle, optionsHandle) => {
        const urlStr = context.getString(urlHandle);
        let options = {};
        try { options = JSON.parse(context.getString(optionsHandle)); } catch { /* default {} */ }
        try {
          const result = await sandboxSafeFetch(urlStr, options);
          return context.newString(JSON.stringify(result));
        } catch (err) {
          return context.newString(JSON.stringify({ error: err.message }));
        }
      });
      hostFetchHandle.consume((fn) => context.setProp(context.global, '__hostFetch', fn));

      // QuickJS has no Intl, so local time in an IANA zone (DST-aware) is computed on the host.
      const hostLocalTimeHandle = context.newFunction('__hostLocalTime', (tzHandle) => {
        const tz = context.getString(tzHandle);
        try {
          const parts = new Intl.DateTimeFormat('en-US', { timeZone: tz, hour12: false, weekday: 'short', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
            .formatToParts(new Date()).reduce((o, p) => { o[p.type] = p.value; return o; }, {});
          const hour = Number(parts.hour) % 24;
          const minute = Number(parts.minute);
          return context.newString(JSON.stringify({
            timezone: tz, hour, minute, hourDecimal: hour + minute / 60,
            weekday: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].indexOf(parts.weekday),
            date: `${parts.year}-${parts.month}-${parts.day}`,
          }));
        } catch (err) {
          return context.newString(JSON.stringify({ error: `invalid timezone "${tz}"` }));
        }
      });
      hostLocalTimeHandle.consume((fn) => context.setProp(context.global, '__hostLocalTime', fn));

      const wrapped =
        `function fetch(url, options) {\n` +
        `  return JSON.parse(__hostFetch(String(url), JSON.stringify(options || {})));\n` +
        `}\n` +
        `function localTime(tz) {\n` +
        `  const r = JSON.parse(__hostLocalTime(String(tz || 'UTC')));\n` +
        `  if (r.error) throw new Error(r.error);\n` +
        `  return r;\n` +
        `}\n` +
        `(() => {\n${code}\n})();`;

      const evalResult = await context.evalCodeAsync(wrapped);
      const resultHandle = context.unwrapResult(evalResult);
      const value = context.dump(resultHandle);

      let serialized;
      try { serialized = JSON.stringify(value); } catch { serialized = String(value); }
      if (serialized && serialized.length > CODE_NODE_MAX_OUTPUT_CHARS) {
        serialized = serialized.slice(0, CODE_NODE_MAX_OUTPUT_CHARS) + '…(truncated)';
      }
      console.log(`[call-loop] code node "${node.id}" -> ${serialized}`);
      this.history.push({
        role: 'user',
        content: `[System note: code step returned ${serialized}]`,
      });
      // A plain-object return value (not an array/primitive) merges into
      // collectedData, same as a function node's webhook JSON result would
      // if it were treated that way — lets a later step reference a field
      // the code computed without the model having to parse it back out of
      // the system-note text.
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        Object.assign(this.collectedData, value);
      }
    } catch (err) {
      console.error(`[call-loop] code node "${node.id}" execution failed`, err);
      this.history.push({
        role: 'user',
        content: `[System note: the code step failed (${err.message}) — let the caller know something went wrong and offer to have someone follow up]`,
      });
    } finally {
      context?.dispose();
    }
  }

  // 'mcp' node — calls ONE pre-selected tool on a remote MCP (Model Context
  // Protocol) server the flow author configured (server URL, headers,
  // tool name/arguments), matching Retell's own MCP node model: the author
  // picks the tool at authoring time, not the LLM at call time. Hand-rolled
  // rather than the official @modelcontextprotocol/sdk — that package pulls
  // in a full server-framework dependency tree (express, hono, ajv, jose,
  // ...) for what's really three small JSON-RPC POSTs; this repo already
  // avoids SDKs in favor of raw fetch() for Twilio/Stripe, so the same
  // convention applies here. Stateless by design (fresh initialize on every
  // call) — this fires once per node entry, not a persistent chat session
  // with the MCP server, and the spec allows a session-less connection.
  async _executeMcpNode(node) {
    const serverUrl = node.params?.serverUrl?.trim();
    const toolName = node.params?.toolName?.trim();
    if (!serverUrl || !toolName) {
      console.warn(`[call-loop] mcp node "${node.id}" is missing serverUrl or toolName — skipping`);
      return;
    }
    let extraHeaders = {};
    try { extraHeaders = node.params?.headers ? JSON.parse(this._interpolateFields(node.params.headers)) : {}; }
    catch (err) { console.warn(`[call-loop] mcp node "${node.id}" has invalid JSON in params.headers — ignoring: ${err.message}`); }
    let toolArguments = {};
    try { toolArguments = node.params?.toolArguments ? JSON.parse(this._interpolateFields(node.params.toolArguments)) : {}; }
    catch (err) { console.warn(`[call-loop] mcp node "${node.id}" has invalid JSON in params.toolArguments — ignoring: ${err.message}`); }

    try {
      const baseHeaders = {
        'Content-Type': 'application/json',
        Accept: 'application/json, text/event-stream',
        ...extraHeaders,
      };

      const initRes = await this._mcpRequest(serverUrl, baseHeaders, {
        jsonrpc: '2.0', id: 1, method: 'initialize',
        params: {
          protocolVersion: '2025-06-18',
          capabilities: {},
          clientInfo: { name: 'call-loop-poc', version: '1.0.0' },
        },
      });
      if (initRes.body?.error) {
        throw new Error(`initialize failed: ${JSON.stringify(initRes.body.error)}`);
      }
      const sessionId = initRes.headers.get('mcp-session-id');
      const sessionHeaders = sessionId ? { ...baseHeaders, 'Mcp-Session-Id': sessionId, 'MCP-Protocol-Version': '2025-06-18' } : baseHeaders;

      // Notification — no id, no response body expected (spec: 202 Accepted).
      await this._mcpRequest(serverUrl, sessionHeaders, {
        jsonrpc: '2.0', method: 'notifications/initialized',
      }, true);

      const callRes = await this._mcpRequest(serverUrl, sessionHeaders, {
        jsonrpc: '2.0', id: 2, method: 'tools/call',
        params: { name: toolName, arguments: toolArguments },
      });
      if (callRes.body?.error) {
        throw new Error(`tools/call failed: ${JSON.stringify(callRes.body.error)}`);
      }

      // A tool result's content is an array of blocks (usually one text
      // block); try to parse that text as JSON for structured field
      // extraction, but fall back to the raw text if it isn't JSON — plenty
      // of real tools just return a plain string.
      const contentBlocks = callRes.body?.result?.content || [];
      const rawText = contentBlocks.map((b) => (b.type === 'text' ? b.text : '')).join('\n').trim();
      let parsed = null;
      try { parsed = JSON.parse(rawText); } catch { /* not JSON, keep raw text */ }

      const serialized = rawText.length > CODE_NODE_MAX_OUTPUT_CHARS
        ? rawText.slice(0, CODE_NODE_MAX_OUTPUT_CHARS) + '…(truncated)'
        : rawText;
      console.log(`[call-loop] mcp node "${node.id}" tool "${toolName}" -> ${serialized.slice(0, 300)}`);
      this.history.push({
        role: 'user',
        content: `[System note: MCP tool "${toolName}" returned ${serialized || '(empty result)'}]`,
      });
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        Object.assign(this.collectedData, parsed);
      }
    } catch (err) {
      console.error(`[call-loop] mcp node "${node.id}" failed`, err);
      this.history.push({
        role: 'user',
        content: `[System note: the MCP tool call failed (${err.message}) — let the caller know something went wrong and offer to have someone follow up]`,
      });
    }
  }

  // One JSON-RPC POST for the MCP client above. `expectNoBody` is for the
  // notifications/initialized call, which the spec says gets a 202 with no
  // JSON-RPC response — the connection's Content-Type header response can't
  // be parsed as JSON in that case, and shouldn't be.
  async _mcpRequest(serverUrl, headers, rpcBody, expectNoBody = false) {
    const res = await fetch(serverUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(rpcBody),
      signal: AbortSignal.timeout(10000),
    });
    if (expectNoBody) return { headers: res.headers, body: null };
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`MCP server HTTP ${res.status}: ${text.slice(0, 300)}`);
    }
    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('text/event-stream')) {
      // A streaming/SSE response — parse the last "data: {...}" frame as the
      // JSON-RPC result. Most single-tool-call servers respond synchronously
      // with plain JSON instead; this is a fallback for ones that don't.
      const text = await res.text();
      const dataLines = text.split('\n').filter((l) => l.startsWith('data:'));
      const lastData = dataLines[dataLines.length - 1]?.slice(5).trim();
      return { headers: res.headers, body: lastData ? JSON.parse(lastData) : null };
    }
    const body = await res.json().catch(() => null);
    return { headers: res.headers, body };
  }

  // Redirects the live call via Twilio's REST API — only possible on a real
  // phone call (TwilioCallAdapter exposes .callSid; a browser call has
  // nothing to redirect) and only with Twilio credentials configured.
  // Otherwise falls back to hanging up rather than silently doing nothing.
  // Hands the live call to another agent: same call, history and collected
  // data; only the flow (and its global settings) is swapped. Voice/TTS backend
  // stay the same as the agent that answered.
  // True when this node can hand off and the agent's words say it is doing so.
  _claimsHandoff(node, text) {
    if (!text || !node?.edges?.some((e) => ['transfer', 'agent_transfer'].includes(this.flowNodesById.get(e.target)?.type))) return false;
    const re = /\b(transfer(ring)?|connect(ing)?|put(ting)? you through|patch(ing)? you)\b[^.?!]{0,40}\byou\b|\bget(ting)? you (connected|transferred|through)\b|\bconnect(ing)? you (with|to)\b|\btransfer(ring)? you\b/i;
    // A question ("Would you like me to transfer you?") offers a hand-off; only a statement claims one.
    return text.split(/(?<=[.!?])\s+/).some((sentence) => !/\?\s*$/.test(sentence) && re.test(sentence));
  }

  async _executeAgentTransfer(params) {
    const targetId = params?.targetAgentId;
    this._agentTransferCount = (this._agentTransferCount || 0) + 1;
    const target = this._agentTransferCount <= 3 && this._tenantId && targetId ? await resolveAgentFlow(targetId, this._tenantId) : null;
    if (!target) {
      console.warn(`[call-loop] agent transfer to ${targetId || 'unset'} not possible (count=${this._agentTransferCount}, tenant=${this._tenantId || 'none'}) — hanging up`);
      this.close();
      return;
    }
    console.log(`[call-loop] agent transfer -> agent ${targetId}, ${target.flow.nodes.length} nodes, starting at "${target.flow.startNodeId}"`);
    this.collectedData.transferred_from_node = this.currentNodeId;
    this.flow = target.flow;
    this.flowNodesById = new Map(this.flow.nodes.map((n) => [n.id, n]));
    this._subflowStack = [];
    this.currentNodeId = this.flow.startNodeId;
    this.send({ type: 'flow_state', currentNodeId: this.currentNodeId, nodeType: this.flowNodesById.get(this.currentNodeId)?.type, collectedData: this.collectedData });
    this._runNodeTurn(this.currentNodeId);
  }

  async _executeExtractVariable(node) {
    const fields = Object.entries(node.extract || {}).filter(([k]) => k && k.trim());
    if (!anthropic || fields.length === 0) return;
    const properties = {};
    for (const [name, type] of fields) {
      const t = type === 'number' || type === 'boolean' ? type : 'string';
      properties[name] = { type: [t, 'null'] };
    }
    const text = this.history
      .filter((m) => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
      .map((m) => `${m.role === 'user' ? 'Caller' : 'Agent'}: ${m.content}`).join('\n');
    try {
      const res = await anthropic.messages.create({
        model: (VALID_LLM_MODELS.has(node?.params?.model) ? node.params.model : this.llmModel),
        max_tokens: 400,
        system: 'You extract named values from a phone call so far. Return a value for every field, or null when the conversation does not clearly state it. Never guess.' +
          (typeof node.prompt === 'string' && node.prompt.trim() ? `\nGuidance: ${node.prompt.trim()}` : ''),
        tools: [{ name: 'record_variables', description: 'Record the extracted values', input_schema: { type: 'object', properties, required: fields.map(([k]) => k) } }],
        tool_choice: { type: 'tool', name: 'record_variables' },
        messages: [{ role: 'user', content: `Conversation so far:\n\"\"\"\n${text || '(nothing said yet)'}\n\"\"\"` }],
      });
      const block = res.content.find((b) => b.type === 'tool_use');
      const got = {};
      for (const [name] of fields) {
        const v = block?.input?.[name];
        if (v !== undefined && v !== null && String(v).trim() !== '') { this.collectedData[name] = v; got[name] = v; }
      }
      console.log(`[call-loop] extract_variable "${node.id}" -> ${JSON.stringify(got)}`);
    } catch (err) {
      console.error(`[call-loop] extract_variable "${node.id}" failed:`, err.message);
    }
  }

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
      // Transfer Success Rate / Transfer Wait Time (2026-09-18) — action=
      // makes Twilio POST the real DialCallStatus (answered/busy/no-answer/
      // failed/canceled) and DialCallDuration once this leg resolves, to
      // /twilio/dial-status below. startedAt is round-tripped through the
      // URL itself rather than kept in any server-side state — simplest way
      // to correlate "how long from initiating the transfer to it
      // resolving" without a new Map to clean up.
      const startedAt = Date.now();
      const actionUrl = `https://${PUBLIC_HOST}/twilio/dial-status?startedAt=${startedAt}`;
      const twiml = `<?xml version="1.0" encoding="UTF-8"?><Response><Dial action="${actionUrl}" method="POST">${to}</Dial></Response>`;
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
      if (res.ok) { this.cost.addBillableEvent('transfer'); this._transferInitiated = true; }
      // Twilio's own <Dial> now owns the call — our media-stream WS leg will
      // get a 'stop' event and close() normally once that dial ends. The
      // action= callback (not this WS) is what learns the real outcome.
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
    // See the constructor's _paymentAwaitingResume comment — this is what
    // stops the resumed turn from re-triggering this same method again.
    this._paymentAwaitingResume = true;
    const ok = await this._redirectForDetour(payTwiml);
    if (!ok) {
      this._paymentAwaitingResume = false;
      console.error(`[call-loop] payment redirect failed for ${callSid} — hanging up rather than leaving the call stuck`);
      this.close();
    }
  }

  // Outbound DTMF — the agent pressing digits into ANOTHER system's phone
  // tree (e.g. navigating a pharmacy's IVR on a call this agent placed),
  // not routing an inbound caller's own keypresses (that's ordinary
  // conversational input, see _onDtmfDigit). <Play digits="..."> generates
  // real DTMF tones the far end hears as keypresses; unlike <Pay>, it has no
  // async outcome to wait for, so the reconnect goes in the SAME TwiML
  // response rather than needing a separate action/webhook round-trip.
  async _executePressDigit(node) {
    const callSid = this.clientWs?.callSid;
    if (!callSid) {
      console.warn('[call-loop] press_digit node requires a Twilio call (no callSid) — skipping');
      return;
    }
    const target = node.edges?.[0]?.target;
    if (!target) {
      console.warn(`[call-loop] press_digit node "${node.id}" has no edge to advance to — nothing to do`);
      return;
    }
    const resolvedDigits = this._interpolateFields(node.params?.digits || '');
    // Only real DTMF characters — 0-9, *, #, A-D, and w/W for pauses (see
    // Twilio's <Play digits> docs) — survive; this is also the XML-injection
    // guard for an attribute built from a flow-author + collectedData string.
    const sanitizedDigits = resolvedDigits.replace(/[^0-9*#A-Da-dwW]/g, '');
    if (!sanitizedDigits) {
      console.warn(`[call-loop] press_digit node "${node.id}" resolved to no valid digits ("${resolvedDigits}") — skipping the tone detour, advancing directly`);
      this._applyTransition({ next_node_id: target });
      return;
    }
    const twiml =
      `<?xml version="1.0" encoding="UTF-8"?><Response>` +
      `<Play digits="${sanitizedDigits}"/>` +
      `<Connect><Stream url="wss://${PUBLIC_HOST}/twilio-stream" /></Connect>` +
      `</Response>`;
    this._pendingPressDigitTarget = target;
    const ok = await this._redirectForDetour(twiml);
    if (!ok) {
      this._pendingPressDigitTarget = null;
      console.error(`[call-loop] press_digit redirect failed for ${callSid} — hanging up rather than leaving the call stuck`);
      this.close();
    }
  }

  // "{{field}}" -> this.collectedData[field], for a press_digit node's
  // digits param (e.g. "2{{account_number}}#") — the only templating this
  // codebase does, kept minimal on purpose: no expressions, no nesting,
  // just a literal lookup, since the result is sanitized to DTMF characters
  // immediately after anyway.
  // Values the agent's owner configured (globalSettings.variables), e.g. business_name and agent_name.
  // Only names with a configured, non-empty value are replaced; anything else stays for the model
  // (extracted fields such as {{patient_name}} keep working as before).
  _applyVariables(text) {
    const vars = this.flow?.globalSettings?.variables;
    if (!vars || typeof vars !== 'object') return text;
    return String(text).replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (match, name) => {
      const v = vars[name];
      return typeof v === 'string' && v.trim() !== '' ? v : match;
    });
  }

  _interpolateFields(template) {
    return String(template || '').replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (_match, field) => {
      const cfg = this.flow?.globalSettings?.variables?.[field];
      if (typeof cfg === 'string' && cfg.trim() !== '') return cfg;
      const value = this.collectedData?.[field];
      return value === undefined || value === null ? '' : String(value);
    });
  }

  // A subagent node's params.tools is a flow-author-written JSON array —
  // parsed defensively since it's hand-authored JSON in the flow editor,
  // same trust level as a code node's script or an mcp node's headers/
  // arguments JSON. Each entry needs at minimum an id (used to build the
  // Anthropic tool name and to route dispatch back to it) and a kind
  // matching one of the node types whose executor it reuses.
  _parseSubagentTools(node) {
    let raw;
    try {
      raw = JSON.parse(node.params?.tools || '[]');
    } catch (err) {
      console.warn(`[call-loop] subagent node "${node.id}" has invalid JSON in params.tools — no tools offered: ${err.message}`);
      return [];
    }
    if (!Array.isArray(raw)) return [];
    return raw.filter((t) => t && typeof t.id === 'string' && t.id.trim() && ['function', 'code', 'sms', 'mcp', 'transfer'].includes(t.kind));
  }

  // Dispatches a subagent tool call to the SAME executor its dedicated node
  // type uses, via a synthetic node built from the tool's own config —
  // reuses tested logic instead of re-implementing function/code/sms/mcp/
  // transfer execution a second time. Each of those executors already
  // pushes its own [System note: ...] into history on completion/failure
  // (see _executeFunctionNode/_executeCodeNode/_executeSmsNode/
  // _executeMcpNode), so the caller just needs to generate a follow-up turn
  // afterward — exactly _handleCalendarTool's own pattern.
  async _executeSubagentTool(toolConfig) {
    const syntheticNode = { id: toolConfig.id, type: toolConfig.kind, params: toolConfig, function: toolConfig.functionName, edges: [] };
    switch (toolConfig.kind) {
      case 'function': return this._executeFunctionNode(syntheticNode);
      case 'code': return this._executeCodeNode(syntheticNode);
      case 'sms': return this._executeSmsNode(syntheticNode);
      case 'mcp': return this._executeMcpNode(syntheticNode);
      case 'transfer': return this._executeTransfer(toolConfig);
      default: return undefined;
    }
  }

  // Real calendar booking (2026-09-17) — executes whichever calendar tool
  // the model called, then injects the REAL result as a system note and
  // generates a follow-up turn to speak from it (see the call site above
  // for why this reuses the nudge pattern instead of a native tool_result
  // round-trip). Never lets a calendar API error kill the call — always
  // produces some note for the model to react to, even a failure one.
  async _handleCalendarTool(toolUse) {
    // Fire-and-forget: plays after whatever the model already said this turn
    // finishes, covering the real API latency below instead of leaving dead
    // air. Silently a no-op if this session's backend/voice has no cached
    // clip (kokoro — see fillerCache's comment) rather than synthesizing
    // live, which would be just as slow as what it's meant to hide.
    const turnIdAtCall = this.activeTurn;
    const fillerBuf = fillerCache.get(this._fillerCacheKey(CALENDAR_LOOKUP_FILLER_PHRASE));
    if (fillerBuf) {
      this._speakCached(fillerBuf, turnIdAtCall).catch((err) => console.error('[call-loop] calendar filler send failed', err));
    }
    let note;
    try {
      if (toolUse.name === 'check_availability') {
        const slots = await this._checkAvailability(toolUse.input.date);
        note = slots.length > 0
          ? `Availability check for ${toolUse.input.date}: ${slots.join(', ')} are open. Propose one of these — do not invent times not in this list.`
          : `Availability check for ${toolUse.input.date}: nothing open that day. Ask the caller for a different day.`;
      } else {
        const result = await this._bookAppointment(toolUse.input);
        note = result.ok
          ? `Booking confirmed for ${toolUse.input.selectedTime} (confirmation ${result.uid}). Read this back to the caller and treat the field as captured.`
          : `Booking attempt failed (${result.error}). Apologize and ask the caller to pick a different time, or offer to take a message instead.`;
        if (result.ok) this.collectedData.booking_confirmed = result.uid;
      }
    } catch (err) {
      console.error(`[call-loop] calendar tool ${toolUse.name} failed`, err);
      note = 'The calendar system is temporarily unavailable. Apologize to the caller and offer to take their info for a callback instead.';
    }
    this.history.push({ role: 'user', content: `[System note: ${note}]` });
    const turnId = ++this.turnSeq;
    this.activeTurn = turnId;
    this.turnState = { id: turnId, llmDone: false, pendingTts: 0, startedSpeaking: false, saidNothing: false };
    await this._generateTurn(turnId, Date.now(), { isNodeEntry: false });
  }

  // GET /v2/slots — confirmed live against the real API (2026-09-17,
  // response shape verified directly rather than trusted from docs, which
  // described a different, incorrect path: /v2/slots/available with
  // startTime/endTime timestamp params — the real one takes eventTypeId
  // plus start/end as plain YYYY-MM-DD dates and returns
  // { data: { "<date>": [{ start: <ISO timestamp> }, ...] } }.
  //
  // Real bug reported on a live call: start/end as bare YYYY-MM-DD dates are
  // interpreted in UTC when no timeZone param is given (Cal.com's documented
  // default), so a Monday query's window is Monday 00:00 UTC to Monday 00:00
  // UTC the next day — for a Pacific business that's midnight to ~4-5pm
  // local depending on DST, silently dropping every slot later in the local
  // day. That's exactly why a caller could book 4:45pm but not 5pm: 4:45pm
  // Pacific is still within the UTC Monday window, 5pm already isn't. Passing
  // the flow's real timezone makes Cal.com filter the window in LOCAL days.
  async _checkAvailability(date) {
    const timezone = this.flow?.globalSettings?.timezone || 'America/Los_Angeles';
    const res = await fetch(
      `https://api.cal.com/v2/slots?eventTypeId=${this.calendar.eventTypeId}&start=${date}&end=${date}&timeZone=${encodeURIComponent(timezone)}`,
      { headers: { Authorization: `Bearer ${this.calendar.apiKey}`, 'cal-api-version': '2024-09-04' }, signal: AbortSignal.timeout(8000) }
    );
    if (!res.ok) throw new Error(`Cal.com slots request failed: HTTP ${res.status}`);
    const body = await res.json();
    const daySlots = body.data?.[date] || [];
    // Real bug, caught on a live call: book_appointment used to take a raw
    // ISO startTime the MODEL had to construct itself (given only a human-
    // readable "4:00 PM" from here) — it got the UTC offset wrong and
    // booked 9 AM instead of the requested 4 PM, while confidently telling
    // the caller "you're all set" for the right time. Classic don't-make-
    // the-LLM-do-exact-arithmetic mistake. Fix: keep the real ISO value
    // server-side, keyed by the exact label the model is given and is
    // expected to echo back — book_appointment looks it up here rather
    // than trusting a model-constructed timestamp at all.
    this._lastAvailableSlots = {};
    const labels = daySlots.map((s) => {
      const label = new Date(s.start).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: timezone });
      this._lastAvailableSlots[label] = s.start;
      return label;
    });
    return labels;
  }

  // POST /v2/bookings — confirmed live against the real API. Requires a
  // real attendee email (Cal.com's own required field, not our choice) —
  // a genuine hard problem for voice specifically (email capture by
  // spelling it out is error-prone) flagged in the design notes; not
  // solved here, just passed through as whatever the model captured.
  async _bookAppointment({ name, email, selectedTime }) {
    const realIsoTime = this._lastAvailableSlots?.[selectedTime];
    if (!realIsoTime) {
      // The model passed a time that either wasn't in the last
      // check_availability result, or check_availability was never called
      // this turn at all — refuse rather than guess at what it meant.
      return { ok: false, error: `"${selectedTime}" wasn't one of the times just checked — call check_availability again first` };
    }
    const res = await fetch('https://api.cal.com/v2/bookings', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.calendar.apiKey}`,
        'Content-Type': 'application/json',
        'cal-api-version': '2024-08-13',
      },
      body: JSON.stringify({
        eventTypeId: this.calendar.eventTypeId,
        start: realIsoTime,
        attendee: { name, email, timeZone: this.flow?.globalSettings?.timezone || 'America/Los_Angeles' },
      }),
      signal: AbortSignal.timeout(8000),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok || body.status === 'error') {
      return { ok: false, error: body.error?.message || `HTTP ${res.status}` };
    }
    return { ok: true, uid: body.data?.uid };
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
      pendingPressDigitTarget: this._pendingPressDigitTarget,
      paymentAwaitingResume: this._paymentAwaitingResume,
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
      _pendingPressDigitTarget: stashed.pendingPressDigitTarget ?? null,
      _paymentAwaitingResume: stashed.paymentAwaitingResume ?? false,
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
      if (this.turnState.nodeType === 'agent_transfer') {
        const params = this.turnState.nodeParams;
        const startedWaiting = Date.now();
        const go = () => {
          if (this._closed) return;
          if (this.clientWs?.isSpeaking?.() && Date.now() - startedWaiting < 12000) { setTimeout(go, 100); return; }
          this._executeAgentTransfer(params).catch((err) => console.error('[call-loop] agent transfer failed', err));
        };
        go();
        return;
      }
      if (this.turnState.nodeType === 'transfer') {
        // The redirect replaces the audio stream, so let queued speech finish first.
        const params = this.turnState.nodeParams;
        const startedWaiting = Date.now();
        const go = () => {
          if (this._closed) return;
          if (this.clientWs?.isSpeaking?.() && Date.now() - startedWaiting < 12000) { setTimeout(go, 100); return; }
          setTimeout(() => { if (!this._closed) this._executeTransfer(params); }, 400);
        };
        go();
        return;
      }
      // _paymentAwaitingResume guards against re-triggering the SAME <Pay>
      // detour on the resumed turn that follows it — that turn also reports
      // nodeType === 'payment' (currentNodeId hasn't moved yet), so without
      // this check every completed payment immediately re-charged. See the
      // constructor's comment for the full real-bug writeup. Once this is
      // true, fall through to the normal transition check below instead —
      // the resumed turn's own transition_flow call (guided by the
      // payment_status-aware prompt override in _buildNodeSystemPrompt) is
      // what actually moves the flow on from here.
      if (this.turnState.nodeType === 'payment' && !this._paymentAwaitingResume) {
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
        // Uninterruptible step with audio still playing out: hold until it
        // drains (_onAudioDrained replays it) instead of starting a new turn
        // whose audio would queue behind the tail.
        if (this._resolveInterruptionSensitivity() === 'off' && this.clientWs.isSpeaking?.()) return;
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
            this._generateTurn(nudgeTurnId, Date.now(), { isNodeEntry: false, forceTransition: allCaptured });
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
              ? `Sorry, I want to make sure I get this right — could you tell me ${missing.map((f) => f.replace(/_/g, ' ')).join(' and ')} one more time?`
              : `Sorry, let me just double check — ${fields.map((f) => `${f.replace(/_/g, ' ')}: ${this.collectedData[f]}`).join(', ')}. Is that all correct?`;
            const fallbackTurnId = ++this.turnSeq;
            this.activeTurn = fallbackTurnId;
            this.turnState = { id: fallbackTurnId, llmDone: true, pendingTts: 0, startedSpeaking: false, saidNothing: false };
            this.history.push({ role: 'assistant', content: fallback });
            this._speak(fallback, fallbackTurnId, Date.now());
          }
        }
      }
      // Reached only when nothing above started a new turn (activeTurn is
      // still the 0 this block set at entry) — the assistant is genuinely
      // done and now waiting on the caller. This is the real "silence
      // clock start" moment for Reminder Message Frequency.
      if (this.activeTurn === 0) {
        this._scheduleReminderIfConfigured();
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
    if (this.turnState?.id === turnId) this.turnState.spokeText = true;
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
    // The socket-open check above only catches a slow gateway CONNECTION.
    // The real cold start (RunPod worker spin-up, 15.9s on a live call) sits
    // behind an already-open socket, so also arm a timer: if no audio has
    // arrived ~1.5s after the session's first synthesize, play the warmup
    // clip anyway. Once per session, and only if nothing has spoken yet.
    if (!this._warmupTimerArmed && !this._ttsAudioSeen) {
      this._warmupTimerArmed = true;
      setTimeout(() => {
        if (this._ttsAudioSeen || this._warmupPlayed || this.activeTurn !== turnId) return;
        const buf = fillerCache.get(`warmup::elevenlabs::${ELEVENLABS_VOICE_ID}::${KOKORO_WARMUP_PHRASE}`);
        if (!buf) { console.warn('[call-loop] TTS slow to start but no warmup clip cached — dead air until the worker is ready'); return; }
        this._warmupPlayed = true;
        console.log(`[call-loop] turn ${turnId}: no TTS audio after 1.5s, playing warmup line`);
        this._speakCached(buf, turnId).catch((err) => console.error('[call-loop] warmup send failed', err));
      }, 1500);
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
        this._ttsAudioSeen = true;
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

  // Real threshold behind Interruption Sensitivity — see the Update-event
  // handler above. Reads the CURRENT node's params (falls back to the
  // platform default when unset).
  //
  // Default changed 'high' -> 'medium' (2026-09-17): a real mystery-shopper
  // call reproduced the exact failure mode this risked — the shopper said a
  // one-word "thanks" while the agent was still mid-sentence reading back
  // full booking details (date/time/phone number), 'high' cut the agent off
  // on that single word, and the rest of the confirmation only arrived
  // fragmented in a later turn after the caller had already said goodbye. A
  // real caller doing the same could hang up without ever hearing their
  // full confirmation. 'medium' (2 words) is still fast enough to feel
  // responsive but filters out exactly this class of short acknowledgment.
  // Precedence: this step's own setting > the flow's default > the tenant's
  // default (Settings) > platform default (medium). 'off' = never interrupt
  // (greetings, disclosures, payment prompts). "Allow interruptions" was only
  // ever a prompt hint ("complete your sentences") — enforced here too.
  // Judged by the node whose audio is still PLAYING, not the current node:
  // a flow transition lands as soon as TTS synthesis finishes, seconds before
  // the caller has heard the audio, so an 'off' step's tail was being judged
  // by the NEXT node's (default) sensitivity. Live call finding 2026-09-18.
  _resolveInterruptionSensitivity(node) {
    if (node === undefined) {
      node = (this.clientWs.isSpeaking?.() && this._audioNode) || (this.flow ? this.flowNodesById?.get(this.currentNodeId) : null);
    }
    const gs = this.flow?.globalSettings || {};
    if (gs.allowInterruptions === false) return 'off';
    return node?.params?.interruptionSensitivity || gs.interruptionSensitivity || this._tenantInterruptionSensitivity || 'medium';
  }

  _transcriptMeetsInterruptionThreshold(text) {
    const sensitivity = this._resolveInterruptionSensitivity();
    if (sensitivity === 'off') return false;
    const minWords = sensitivity === 'low' ? 3 : sensitivity === 'medium' ? 2 : 1;
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    return wordCount >= minWords;
  }

  // The phone adapter's paced audio queue just emptied — replay anything the
  // caller said while an uninterruptible step was still audibly speaking.
  _onAudioDrained() {
    if (!this._queuedUserText || this.activeTurn !== 0 || this._closing) return;
    const queued = this._queuedUserText;
    this._queuedUserText = null;
    this._onUserTurnComplete(queued);
  }

  _bargeIn() {
    // Nothing in flight AND nothing still audible = nothing to interrupt. A
    // finished turn whose audio is still draining IS interruptible (the
    // caller hears it) — this used to return early and ignore barge-in for
    // the whole playback tail of any turn whose synthesis had completed.
    if (this.activeTurn === 0 && this.clientWs.isSpeaking?.() !== true) return;
    console.log(`[call-loop] barge-in — cancelling turn ${this.activeTurn}`);
    this.activeTurn = 0; // no active turn is allowed to speak until the next final transcript
    this._silenceSince = null;
    if (this._reminderTimer) { clearTimeout(this._reminderTimer); this._reminderTimer = null; } // caller is clearly not silent
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

  // Waits (up to 10s) for the agent to finish its sentence, then hangs up.
  _endCallGracefully(reason, waited = 0) {
    if (this._closed) return;
    this._closing = true;
    if (this.activeTurn === 0 && this.clientWs.isSpeaking?.() !== true || waited >= 10000) {
      console.log(`[call-loop] ending call: ${reason}`);
      this.close();
      return;
    }
    setTimeout(() => this._endCallGracefully(reason, waited + 500), 500);
  }

  // Twilio async AMD result (see /twilio/amd-status). Acts only when the
  // flow opted in via globalSettings.voicemailDetection.
  onAnsweredBy(answeredBy) {
    this.collectedData.answered_by = answeredBy;
    const gs = this.flow?.globalSettings || {};
    const mode = gs.voicemailDetection;
    const isMachine = /^machine_end/.test(answeredBy) || answeredBy === 'fax';
    if (!isMachine || (mode !== 'hangup' && mode !== 'leave_message') || this._closing || this._closed) return;
    if (this._vmHandled) return;
    this._vmHandled = true;
    console.log(`[call-loop] answered_by=${answeredBy}, voicemailDetection=${mode}`);
    if (this.callSid) updateCallLogByCallSid(this.callSid, { outcome: 'voicemail' }).catch(() => {});
    const message = typeof gs.voicemailMessage === 'string' ? this._interpolateFields(gs.voicemailMessage).trim() : '';
    if (mode !== 'leave_message' || !message || answeredBy === 'fax') {
      this._bargeIn();
      this._endCallGracefully('voicemail detected');
      return;
    }
    this._bargeIn();
    if (this._reminderTimer) { clearTimeout(this._reminderTimer); this._reminderTimer = null; }
    this._closing = true; // no new LLM turns; retiring this goodbye turn hangs up
    this.history.push({ role: 'assistant', content: message });
    const turnId = ++this.turnSeq;
    this.activeTurn = turnId;
    this.turnState = { id: turnId, llmDone: true, pendingTts: 0, startedSpeaking: false, saidNothing: false, nodeType: 'goodbye' };
    this._speak(message, turnId, Date.now());
    setTimeout(() => this._endCallGracefully('voicemail message timeout', 0), 60000);
  }

  close() {
    if (this._closed) return;
    this._closed = true;
    if (this._maxDurationTimer) { clearTimeout(this._maxDurationTimer); this._maxDurationTimer = null; }
    if (this._reminderTimer) { clearTimeout(this._reminderTimer); this._reminderTimer = null; }
    if (this._pendingResponseTimer) { clearTimeout(this._pendingResponseTimer); this._pendingResponseTimer = null; }
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

    // Finalizes the call log row this session's 'start' handler created (see
    // insertCallLog there) — keyed by callSid rather than the possibly-not-
    // yet-resolved _callLogId, so a very short call can't race its own
    // insert. No-ops harmlessly if this session was never a real logged call
    // (browser tab, shopper mode, or Supabase creds unset).
    if (this.callSid) {
      const transcript = this.history
        .filter((m) => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
        .map((m) => ({ role: m.role, content: m.content }));
      const finalize = updateCallLogByCallSid(this.callSid, {
        duration_seconds: Math.round(voiceSeconds),
        transcript,
      }).catch((err) => console.error('[call-loop] call log finalize failed', err));
      const tenantId = this._tenantId;
      const answeredBy = this.collectedData?.answered_by;
      const machine = typeof answeredBy === 'string' && (/^machine_end/.test(answeredBy) || answeredBy === 'fax');
      const endedAt = new Date();
      Promise.all([finalize, this._runPostCallAnalysis(transcript).catch((err) => { console.error('[call-loop] post-call analysis failed', err); return null; })])
        .then(([, analysis]) => {
          if (!tenantId) return;
          const data = {
            call_id: this.callSid,
            tenant_id: tenantId,
            direction: this.direction || 'inbound',
            duration_seconds: Math.round(voiceSeconds),
            outcome: machine ? 'voicemail' : this._transferInitiated ? 'transferred' : 'answered',
            transcript,
            analysis: analysis || null,
            started_at: new Date(this._callStartedAt).toISOString(),
            ended_at: endedAt.toISOString(),
          };
          dispatchTenantWebhook(tenantId, 'call.completed', data);
          if (analysis) dispatchTenantWebhook(tenantId, 'call.analyzed', data);
        })
        .catch((err) => console.error('[call-loop] completion webhooks failed', err));
    }
  }
}


// Post-call analysis: extract the flow's globalSettings.postCallAnalysis.fields
// from the transcript into call_logs.analysis. Off unless fields are configured.
CallSession.prototype._runPostCallAnalysis = async function (transcript) {
  const rawFields = this.flow?.globalSettings?.postCallAnalysis?.fields;
  if (!anthropic || !this.callSid || !Array.isArray(rawFields) || transcript.length === 0) return null;
  const fields = rawFields.filter((f) => f && typeof f.name === 'string' && f.name.trim() && ['text', 'boolean', 'number', 'enum'].includes(f.type) && (f.type !== 'enum' || (Array.isArray(f.options) && f.options.length)));
  if (!fields.length) return null;
  const properties = {};
  for (const f of fields) {
    const description = f.description || undefined;
    if (f.type === 'enum') properties[f.name] = { type: ['string', 'null'], enum: [...f.options, null], description };
    else properties[f.name] = { type: [f.type === 'text' ? 'string' : f.type, 'null'], description };
  }
  const text = transcript.map((m) => `${m.role === 'user' ? 'Caller' : 'Agent'}: ${m.content}`).join('\n');
  const res = await anthropic.messages.create({
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 1024,
    system: 'You extract structured data from a phone call transcript. Return a value for every requested field, or null when the transcript does not support one. Never guess.',
    tools: [{ name: 'record_analysis', description: 'Record the extracted fields', input_schema: { type: 'object', properties, required: fields.map((f) => f.name) } }],
    tool_choice: { type: 'tool', name: 'record_analysis' },
    messages: [{ role: 'user', content: `Call transcript:\n\"\"\"\n${text}\n\"\"\"` }],
  });
  const block = res.content.find((b) => b.type === 'tool_use');
  if (!block) return null;
  const analysis = {};
  for (const f of fields) analysis[f.name] = block.input?.[f.name] ?? null;
  await updateCallLogByCallSid(this.callSid, { analysis });
  return analysis;
};

server.listen(PORT, () => {
  console.log(`[call-loop] listening on http://localhost:${PORT}`);
  console.log(`[call-loop] TTS gateway: ${TTS_GATEWAY_WS_URL}`);
  // Always prewarm regardless of the process-wide default — a per-session
  // context override can enable backchanneling even when it's off globally,
  // and the prewarm cost (a handful of short TTS calls, once, at startup)
  // is trivial either way.
  prewarmFillerCache();
  // Optional: keep the TTS worker warm between calls (costs idle worker time). Off unless TTS_KEEPALIVE_MINUTES is set.
  const keepaliveMin = Number(process.env.TTS_KEEPALIVE_MINUTES);
  if (TTS_BACKEND === 'kokoro' && keepaliveMin > 0) {
    setInterval(() => warmTtsGateway('keepalive'), keepaliveMin * 60 * 1000);
    console.log(`[call-loop] TTS keepalive every ${keepaliveMin} min`);
  }
  // Every 6 hours is frequent enough that a 30-day retention setting is
  // enforced within a fraction of a day of expiring, without hammering
  // Twilio/Supabase on every process restart the way "run once at boot,
  // then daily" would on a platform that redeploys/restarts often.
  enforceRecordingRetention().catch((err) => console.error('[call-loop] initial retention sweep failed', err));
  setInterval(() => enforceRecordingRetention().catch((err) => console.error('[call-loop] retention sweep failed', err)), 6 * 60 * 60 * 1000);
});
