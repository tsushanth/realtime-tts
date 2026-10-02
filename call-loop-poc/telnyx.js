// Telnyx (TeXML) inbound plumbing: feature flag, webhook authentication, stream-URL signing,
// TeXML generation and the /telnyx/voice handler. Everything here is inert unless
// TELNYX_ENABLED=1, and the Twilio routes never import from this file.
import crypto from 'node:crypto';

export function isTelnyxEnabled(env = process.env) {
  return env.TELNYX_ENABLED === '1';
}

// ---------- webhook authentication ----------

// DER prefix that wraps a raw 32-byte Ed25519 public key into SubjectPublicKeyInfo.
const ED25519_SPKI_PREFIX = Buffer.from('302a300506032b6570032100', 'hex');
const SIGNATURE_TOLERANCE_SEC = 300;

// Telnyx signs webhooks with Ed25519: headers telnyx-signature-ed25519 (base64) and telnyx-timestamp
// (unix seconds); the signed message is `${timestamp}|${rawBody}`. Public key (base64) comes from the
// Telnyx portal (Mission Control > Keys & Credentials > Public Key) -> TELNYX_PUBLIC_KEY.
export function verifyTelnyxSignature({ rawBody, signature, timestamp, publicKeyB64, nowSec = Math.floor(Date.now() / 1000) }) {
  try {
    if (!rawBody || !signature || !timestamp || !publicKeyB64) return false;
    const ts = Number(timestamp);
    if (!Number.isFinite(ts) || Math.abs(nowSec - ts) > SIGNATURE_TOLERANCE_SEC) return false;
    const raw = Buffer.from(publicKeyB64, 'base64');
    if (raw.length !== 32) return false;
    const sig = Buffer.from(signature, 'base64');
    if (sig.length !== 64) return false;
    const key = crypto.createPublicKey({ key: Buffer.concat([ED25519_SPKI_PREFIX, raw]), format: 'der', type: 'spki' });
    const message = Buffer.concat([Buffer.from(`${timestamp}|`), Buffer.isBuffer(rawBody) ? rawBody : Buffer.from(rawBody)]);
    return crypto.verify(null, message, key, sig);
  } catch {
    return false;
  }
}

function safeEqual(a, b) {
  const x = Buffer.from(String(a));
  const y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

// Returns { ok, status, reason }. Requires at least one mechanism to be configured (fail closed):
//   TELNYX_PUBLIC_KEY      -> Ed25519 signature must verify
//   TELNYX_WEBHOOK_SECRET  -> `?key=<secret>` on the webhook URL must match
// Both set means both must pass. Telnyx documents signing for its v2 JSON webhooks; whether TeXML
// application webhooks carry the same headers is NOT confirmed, hence the shared-secret alternative.
export function checkWebhookAuth(req, rawBody, env = process.env) {
  const publicKey = env.TELNYX_PUBLIC_KEY;
  const secret = env.TELNYX_WEBHOOK_SECRET;
  if (!publicKey && !secret) return { ok: false, status: 503, reason: 'telnyx webhook auth not configured (set TELNYX_PUBLIC_KEY and/or TELNYX_WEBHOOK_SECRET)' };
  if (publicKey) {
    const ok = verifyTelnyxSignature({
      rawBody,
      signature: req.headers['telnyx-signature-ed25519'],
      timestamp: req.headers['telnyx-timestamp'],
      publicKeyB64: publicKey,
    });
    if (!ok) return { ok: false, status: 403, reason: 'bad or missing telnyx signature' };
  }
  if (secret) {
    const given = req.query?.key;
    if (typeof given !== 'string' || !safeEqual(given, secret)) return { ok: false, status: 403, reason: 'bad or missing webhook key' };
  }
  return { ok: true };
}

// ---------- stream URL signing (webhook -> websocket handoff) ----------

const STREAM_TOKEN_TTL_SEC = 120;
let ephemeralSecret = null;
function streamSecret(env) {
  const s = env.TELNYX_STREAM_SECRET || env.TELNYX_WEBHOOK_SECRET;
  if (s) return s;
  // Works on a single instance only; set TELNYX_STREAM_SECRET when running more than one machine.
  if (!ephemeralSecret) ephemeralSecret = crypto.randomBytes(32).toString('hex');
  return ephemeralSecret;
}

function hmac(callSid, exp, env) {
  return crypto.createHmac('sha256', streamSecret(env)).update(`${callSid}.${exp}`).digest('hex');
}

export function signStreamParams(callSid, env = process.env, nowSec = Math.floor(Date.now() / 1000)) {
  const exp = nowSec + STREAM_TOKEN_TTL_SEC;
  return { cs: callSid, exp: String(exp), sig: hmac(callSid, exp, env) };
}

// Returns the callSid when the signed params are valid and unexpired, else null.
export function verifyStreamParams(searchParams, env = process.env, nowSec = Math.floor(Date.now() / 1000)) {
  const cs = searchParams.get('cs');
  const exp = searchParams.get('exp');
  const sig = searchParams.get('sig');
  if (!cs || !exp || !sig || !/^\d+$/.test(exp) || Number(exp) < nowSec) return null;
  return safeEqual(sig, hmac(cs, exp, env)) ? cs : null;
}

// ---------- TeXML ----------

function xmlEsc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;');
}

export const TEXML_HANGUP = '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>';

// bidirectionalMode="rtp" + bidirectionalCodec="PCMU" is the mode measured on 2026-10-02 (8 kHz PCMU both ways).
// A trailing <Hangup/> ends the call once the stream closes instead of relying on end-of-document behaviour.
export function buildTelnyxStreamTexml({ host, callSid, env = process.env }) {
  const p = signStreamParams(callSid, env);
  const qs = `cs=${encodeURIComponent(p.cs)}&exp=${p.exp}&sig=${p.sig}`;
  const url = `wss://${host}/telnyx-stream?${qs}`;
  return (
    `<?xml version="1.0" encoding="UTF-8"?>` +
    `<Response><Connect><Stream url="${xmlEsc(url)}" bidirectionalMode="rtp" bidirectionalCodec="PCMU">` +
    `<Parameter name="callSid" value="${xmlEsc(callSid)}"/>` +
    `</Stream></Connect><Hangup/></Response>`
  );
}

// ---------- /telnyx/voice ----------

function parseBody(req, rawBody) {
  const type = String(req.headers['content-type'] || '');
  const text = rawBody ? rawBody.toString('utf8') : '';
  try {
    if (type.includes('json')) {
      const j = JSON.parse(text || '{}');
      return j && typeof j === 'object' ? (j.data?.payload && typeof j.data.payload === 'object' ? j.data.payload : j) : {};
    }
    return Object.fromEntries(new URLSearchParams(text));
  } catch {
    return {};
  }
}

// deps: { resolveInboundCall, pendingCallContext, contextWrittenCallSids } from server.js (injected so this is unit-testable).
export function createTelnyxVoiceHandler(deps, env = process.env) {
  return async function telnyxVoice(req, res) {
    const rawBody = Buffer.isBuffer(req.body) ? req.body : Buffer.from(typeof req.body === 'string' ? req.body : '');
    const auth = checkWebhookAuth(req, rawBody, env);
    if (!auth.ok) {
      console.warn(`[telnyx] /telnyx/voice rejected: ${auth.reason}`);
      return res.status(auth.status).type('text/plain').send(auth.reason);
    }
    const body = parseBody(req, rawBody);
    const callSid = body.CallSid || body.call_sid || body.call_control_id;
    const to = body.To || body.to;
    const from = body.From || body.from || null;
    if (!callSid || !to) {
      console.warn('[telnyx] /telnyx/voice missing CallSid/To - hanging up');
      return res.type('text/xml').send(TEXML_HANGUP);
    }

    let resolved = null;
    try {
      resolved = await deps.resolveInboundCall(to, 'inbound');
    } catch (err) {
      console.error('[telnyx] tenant lookup failed', err);
    }
    // Opt-in per number: only a number whose carrier is explicitly 'telnyx' runs here. Anything else
    // (unrouted, default 'twilio') is refused rather than silently served over the wrong carrier.
    if (!resolved || resolved.carrier !== 'telnyx') {
      console.warn(`[telnyx] ${to} is not routed with carrier=telnyx (resolved=${resolved ? `carrier:${resolved.carrier || 'twilio'}` : 'none'}) - hanging up`);
      return res.type('text/xml').send(TEXML_HANGUP);
    }

    if (!deps.contextWrittenCallSids.has(callSid)) {
      deps.contextWrittenCallSids.add(callSid);
      setTimeout(() => deps.contextWrittenCallSids.delete(callSid), 5 * 60_000).unref?.();
      deps.pendingCallContext.set(callSid, {
        ...resolved,
        fromNumber: from,
        tenantNumber: to,
        direction: 'inbound',
        createdAt: Date.now(),
      });
    }
    const host = env.TELNYX_PUBLIC_HOST || req.headers.host;
    res.type('text/xml').send(buildTelnyxStreamTexml({ host, callSid, env }));
  };
}
