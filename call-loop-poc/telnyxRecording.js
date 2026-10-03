// Telnyx call recording: start (TeXML REST), recording-completed webhook, retention delete and a playback proxy.
// Inert unless TELNYX_ENABLED=1 (server.js only reaches this for calls on the telnyx carrier / mounts the route when
// enabled). Every failure is soft: a recording problem must never affect the call. Twilio code never imports this file.
import { checkWebhookAuth, parseBody } from './telnyx.js';

const API_BASE = 'https://api.telnyx.com/v2';
const START_TIMEOUT_MS = 5000;
const START_RETRY_DELAYS_MS = [500, 1500]; // up to 3 attempts in total
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);
const DOWNLOAD_TIMEOUT_MS = 15000;

// Stored in calldesk_call_logs.recording_sid, so the retention sweep can tell a Telnyx recording from a Twilio one.
export const TELNYX_SID_PREFIX = 'telnyx:';
export const isTelnyxRecordingSid = (sid) => typeof sid === 'string' && sid.startsWith(TELNYX_SID_PREFIX);
export const telnyxRecordingId = (sid) => (isTelnyxRecordingSid(sid) ? sid.slice(TELNYX_SID_PREFIX.length) : null);

// Stored in calldesk_call_logs.recording_url. Telnyx's own download links expire (10 minutes), so the durable reference is
// the recording resource; the proxy resolves a fresh download link at play time.
const RECORDING_URL_RE = /^https:\/\/api\.telnyx\.com\/v2\/recordings\/([A-Za-z0-9-]{8,64})$/;
export const telnyxRecordingUrl = (id) => `${API_BASE}/recordings/${id}`;
export const parseTelnyxRecordingUrl = (url) => (typeof url === 'string' ? RECORDING_URL_RE.exec(url)?.[1] || null : null);

const sleepDefault = (ms) => new Promise((r) => setTimeout(r, ms));

// Account id used in the TeXML REST path: explicit env wins, else the AccountSid Telnyx sent with the voice webhook.
export function resolveTelnyxAccountSid(webhookAccountSid, env = process.env) {
  return env.TELNYX_ACCOUNT_SID || webhookAccountSid || null;
}

// Starts a dual-channel recording on the live call. Returns { started, reason?, status?, attempts }. Never throws.
export async function startTelnyxRecording({ callSid, accountSid, host, env = process.env, fetchImpl = globalThis.fetch, sleep = sleepDefault }) {
  try {
    if (env.TELNYX_ENABLED !== '1') return { started: false, reason: 'telnyx_disabled' };
    if (!env.TELNYX_API_KEY) return { started: false, reason: 'no_api_key' };
    if (!callSid || !accountSid) return { started: false, reason: 'no_call_or_account' };
    if (!host) return { started: false, reason: 'no_host' };
    if (!env.TELNYX_WEBHOOK_SECRET && !env.TELNYX_PUBLIC_KEY) return { started: false, reason: 'webhook_auth_not_configured' };

    // The callback goes through the same auth as /telnyx/voice; the shared secret travels as ?key= when configured.
    const key = env.TELNYX_WEBHOOK_SECRET ? `?key=${encodeURIComponent(env.TELNYX_WEBHOOK_SECRET)}` : '';
    const body = new URLSearchParams({
      RecordingChannels: 'dual',
      RecordingTrack: 'both',
      PlayBeep: 'false', // documented default is true; the Twilio path is silent, so this must be too
      RecordingStatusCallback: `https://${host}/telnyx/recording-status${key}`,
      RecordingStatusCallbackMethod: 'POST',
      RecordingStatusCallbackEvent: 'completed',
    });
    const url = `${API_BASE}/texml/Accounts/${encodeURIComponent(accountSid)}/Calls/${encodeURIComponent(callSid)}/Recordings.json`;

    let lastReason = 'error';
    let lastStatus;
    for (let attempt = 1; attempt <= START_RETRY_DELAYS_MS.length + 1; attempt++) {
      let retry = false;
      try {
        const res = await fetchImpl(url, {
          method: 'POST',
          headers: { Authorization: `Bearer ${env.TELNYX_API_KEY}`, 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json' },
          body,
          signal: AbortSignal.timeout(START_TIMEOUT_MS),
        });
        if (res.ok) return { started: true, attempts: attempt };
        lastStatus = res.status;
        lastReason = `http_${res.status}`;
        retry = RETRYABLE_STATUS.has(res.status);
      } catch (err) {
        lastReason = err?.name === 'TimeoutError' || err?.name === 'AbortError' ? 'timeout' : 'network_error';
        // A timed-out start may still have begun recording on Telnyx's side, so only retry when the request clearly failed.
        retry = lastReason === 'network_error';
      }
      if (!retry || attempt > START_RETRY_DELAYS_MS.length) break;
      await sleep(START_RETRY_DELAYS_MS[attempt - 1]);
    }
    return { started: false, reason: lastReason, status: lastStatus, attempts: START_RETRY_DELAYS_MS.length + 1 };
  } catch (err) {
    return { started: false, reason: 'unexpected_error', error: err?.message };
  }
}

// Logs a start result without ever including credentials.
export function logRecordingStart(callSid, result, log = console) {
  if (result.started) log.log?.(`[telnyx] recording started for ${callSid}${result.attempts > 1 ? ` (attempt ${result.attempts})` : ''}`);
  else log.warn?.(`[telnyx] recording not started for ${callSid}: ${result.reason}${result.status ? ` (HTTP ${result.status})` : ''} - call continues unrecorded`);
}

// ---------- recording-completed webhook ----------

const seen = new Map(); // recordingId -> expiry ms; bounded dedupe so a retried delivery is acknowledged without rework
const SEEN_TTL_MS = 60 * 60 * 1000;
const SEEN_MAX = 2000;
export function _resetSeenForTests() { seen.clear(); }
function markSeen(id, now) {
  for (const [k, exp] of seen) if (exp < now) seen.delete(k);
  if (seen.size >= SEEN_MAX) seen.delete(seen.keys().next().value);
  seen.set(id, now + SEEN_TTL_MS);
}

// Mounted behind express.raw (signature is over the raw bytes). deps: { updateCallLogByCallSid }.
export function createTelnyxRecordingStatusHandler(deps, env = process.env) {
  return async function telnyxRecordingStatus(req, res) {
    const rawBody = Buffer.isBuffer(req.body) ? req.body : Buffer.from(typeof req.body === 'string' ? req.body : '');
    const auth = checkWebhookAuth(req, rawBody, env);
    if (!auth.ok) {
      console.warn(`[telnyx] /telnyx/recording-status rejected: ${auth.reason}`);
      return res.status(auth.status).type('text/plain').send(auth.reason);
    }
    const body = parseBody(req, rawBody);
    const callSid = body.CallSid || body.call_sid;
    const recordingId = body.RecordingSid || body.recording_sid || body.recording_id;
    const status = String(body.RecordingStatus || body.recording_status || 'completed').toLowerCase();
    if (callSid && recordingId && status === 'completed' && /^[A-Za-z0-9-]{8,64}$/.test(String(recordingId))) {
      if (seen.has(recordingId) && seen.get(recordingId) > Date.now()) return res.sendStatus(200);
      try {
        // Same columns as the Twilio path. The URL is the durable recording resource (see top of file), not the
        // short-lived download link Telnyx sends in RecordingUrl.
        await deps.updateCallLogByCallSid(callSid, { recording_url: telnyxRecordingUrl(recordingId), recording_sid: `${TELNYX_SID_PREFIX}${recordingId}` });
        markSeen(recordingId, Date.now());
      } catch (err) {
        console.error('[telnyx] recording-status call log update failed', err);
        return res.sendStatus(500); // let Telnyx retry
      }
    }
    res.sendStatus(200);
  };
}

// ---------- retention delete ----------

// Returns true when the recording is gone (deleted now, or already absent), false otherwise. Never throws.
export async function deleteTelnyxRecording(recordingSid, { env = process.env, fetchImpl = globalThis.fetch } = {}) {
  const id = telnyxRecordingId(recordingSid);
  if (!id || !env.TELNYX_API_KEY) return false;
  try {
    const res = await fetchImpl(`${API_BASE}/recordings/${encodeURIComponent(id)}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${env.TELNYX_API_KEY}` },
      signal: AbortSignal.timeout(START_TIMEOUT_MS),
    });
    if (res.ok || res.status === 404) return true;
    console.error(`[telnyx] failed to delete recording ${id}: HTTP ${res.status}`);
    return false;
  } catch (err) {
    console.error(`[telnyx] failed to delete recording ${id}`, err?.message);
    return false;
  }
}

// ---------- playback proxy ----------

// Resolves a fresh download link via the recording resource and streams the bytes to `res`. The Telnyx API key is only
// sent to api.telnyx.com; the pre-signed download link is fetched without it. Returns true if it handled the response.
export async function proxyTelnyxRecording(recordingUrl, res, { env = process.env, fetchImpl = globalThis.fetch } = {}) {
  const id = parseTelnyxRecordingUrl(recordingUrl);
  if (!id) return false;
  if (!env.TELNYX_API_KEY) { res.status(500).json({ error: 'TELNYX_API_KEY not configured' }); return true; }
  try {
    const meta = await fetchImpl(`${API_BASE}/recordings/${encodeURIComponent(id)}`, {
      headers: { Authorization: `Bearer ${env.TELNYX_API_KEY}`, Accept: 'application/json' },
      signal: AbortSignal.timeout(START_TIMEOUT_MS),
    });
    if (!meta.ok) { res.status(meta.status === 404 ? 404 : 502).json({ error: 'Failed to fetch recording from Telnyx' }); return true; }
    const j = await meta.json();
    const dl = j?.data?.download_urls?.mp3 || j?.data?.download_urls?.wav;
    if (typeof dl !== 'string' || !dl.startsWith('https://')) { res.status(502).json({ error: 'Telnyx returned no download url' }); return true; }
    const audio = await fetchImpl(dl, { signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS) });
    if (!audio.ok || !audio.body) { res.status(502).json({ error: 'Failed to download recording from Telnyx' }); return true; }
    res.setHeader('Content-Type', audio.headers.get('content-type') || (dl.includes('.wav') ? 'audio/wav' : 'audio/mpeg'));
    for await (const chunk of audio.body) res.write(chunk);
    res.end();
  } catch (err) {
    console.error('[telnyx] recording proxy failed', err?.message);
    if (!res.headersSent) res.status(502).json({ error: 'recording proxy failed' }); else res.end();
  }
  return true;
}
