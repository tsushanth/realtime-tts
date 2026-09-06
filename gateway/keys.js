// API key store — a JSON file on the Fly volume, not a database. Fine for low volume;
// revisit (real DB, per-key rate limits) if this outgrows a single Fly volume.
//
// Keys are stored HASHED (sha256), not plaintext — this is a self-serve signup flow
// with billing attached, so a raw key leak from this file must not be possible. The
// raw key is only ever returned once, at issuance, and never persisted.
//
// Each entry has a stable `id` separate from the raw `key` so callers (e.g. the
// ReadAloud backend) can store the id for later revocation without ever persisting
// the raw secret themselves.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const KEYS_PATH = process.env.KEYS_PATH || "/data/keys.json";

// The /developers marketing page advertises "no credit card required to try" —
// until this existed, that was false: a non-billing-enabled key was simply
// inert, rejected on every request. This is a real (if modest) free allowance,
// per-key, that works with zero payment method on file. Note this is
// enforceable per-KEY, not per-user — the gateway has no concept of a user,
// only individual key records, so a user creating multiple keys (backend caps
// this at 5 per user) could stack multiple free allowances. Acceptable at
// today's volume; revisit if abuse is observed.
const FREE_TIER_CHARS = parseInt(process.env.FREE_TIER_CHARS || "10000", 10);

// Signs/verifies short-lived session tokens for the direct-to-Modal fast path
// (see server.js's /tts/authorize and worker-modal-readaloud/app.py). Separate
// from ADMIN_SECRET deliberately — Modal only needs the narrow ability to
// verify a token and report usage, not full admin key-management power.
const SESSION_TOKEN_SECRET = process.env.MODAL_SESSION_SECRET;
const SESSION_TOKEN_TTL_MS = 60 * 1000; // just long enough to open the WS connection

function hash(key) {
  return crypto.createHash("sha256").update(key).digest("hex");
}

// Migrates any pre-hashing entry (plaintext `key`, no `keyHash`) in place —
// found live 2026-09-01: the file already had 5 real issued keys (test/infra,
// not end-user, but the mechanism has to handle real ones too) with no
// migration path once hashing shipped. Lazy-migrates on the next load rather
// than requiring a one-off SSH script against the Fly volume.
function migrateLegacyEntry(k) {
  if (k.keyHash) return k;
  return {
    id: k.id,
    keyHash: hash(k.key),
    keyPreview: k.key.slice(0, 10) + "…",
    label: k.label || "",
    created_at: k.created_at,
    revoked: !!k.revoked,
    // Grandfathered in — this key worked before the billing gate existed,
    // so migrating it must not silently lock out whatever's using it.
    billingEnabled: true,
    usageCharsSinceLastReport: 0,
    freeCharsUsed: 0,
    freeTierExhausted: false,
  };
}

function load() {
  let keys;
  try {
    keys = JSON.parse(fs.readFileSync(KEYS_PATH, "utf8"));
  } catch {
    return [];
  }
  const hadLegacy = keys.some((k) => !k.keyHash);
  if (hadLegacy) {
    keys = keys.map(migrateLegacyEntry);
    save(keys);
  }
  return keys;
}

function save(keys) {
  fs.mkdirSync(path.dirname(KEYS_PATH), { recursive: true });
  fs.writeFileSync(KEYS_PATH, JSON.stringify(keys, null, 2));
}

export function isValidKey(key) {
  if (!key) return false;
  const keys = load();
  const entry = keys.find((k) => k.keyHash === hash(key) && !k.revoked);
  return !!entry;
}

export function getIdForKey(key) {
  if (!key) return null;
  const keys = load();
  const entry = keys.find((k) => k.keyHash === hash(key) && !k.revoked);
  return entry?.id || null;
}

// Only billing-enabled keys may consume UNLIMITED paid GPU compute — see
// server.js. A key exists in one of two states: freshly issued and not yet
// enabled (the caller must explicitly opt it in — see setBillingEnabledById),
// or enabled. ReadAloud's issuance route enables its own keys immediately on
// creation; there's no per-key payment-method concept in this product today,
// so "billing enabled" means "this key is allowed to run," not "a card is on
// file" — the stronger version of this gate is future work, not implied by
// this flag's current callers.
export function isBillingEnabled(key) {
  if (!key) return false;
  const keys = load();
  const entry = keys.find((k) => k.keyHash === hash(key) && !k.revoked);
  return !!entry?.billingEnabled;
}

// Connection-level gate: can this key be used AT ALL right now? True if
// billing-enabled (unlimited, paid), or if it still has free-tier characters
// remaining (see FREE_TIER_CHARS). A key with zero free chars left and no
// billing enabled must be rejected before ever reaching the worker.
//
// Checks the `freeTierExhausted` flag FIRST, as a fast/cheap boolean instead
// of recomputing remaining budget — this matters for the direct-to-Modal path
// (see /tts/authorize in server.js), where usage is reported back
// asynchronously AFTER synthesis completes rather than metered inline as
// bytes flow through. Without a flag that flips the instant the threshold is
// crossed, several large requests could race past a "recompute remaining"
// check before any of their usage reports land. The flag bounds that: once
// set, every subsequent authorize call is rejected immediately, so the only
// exposure is whatever was already in flight at the moment it flipped, not
// an ongoing drain. See recordUsage() for where it gets set.
export function checkAccess(key) {
  if (!key) return { valid: false };
  const keys = load();
  const entry = keys.find((k) => k.keyHash === hash(key) && !k.revoked);
  if (!entry) return { valid: false };
  if (entry.billingEnabled) return { valid: true, allowed: true, billingEnabled: true };
  if (entry.freeTierExhausted) return { valid: true, allowed: false, billingEnabled: false, freeCharsRemaining: 0 };
  const freeCharsRemaining = Math.max(0, FREE_TIER_CHARS - (entry.freeCharsUsed || 0));
  return { valid: true, allowed: freeCharsRemaining > 0, billingEnabled: false, freeCharsRemaining };
}

// Per-request check, used right before forwarding a synthesize request to the
// worker — a key with SOME free chars left but fewer than this request needs
// must be rejected here (not mid-stream, and not by silently truncating the
// text), so the client gets a clear error instead of a surprise cutoff or
// unbilled overage.
export function canAffordRequest(key, chars) {
  if (!key) return false;
  const keys = load();
  const entry = keys.find((k) => k.keyHash === hash(key) && !k.revoked);
  if (!entry) return false;
  if (entry.billingEnabled) return true;
  const freeCharsRemaining = Math.max(0, FREE_TIER_CHARS - (entry.freeCharsUsed || 0));
  return chars <= freeCharsRemaining;
}

export function setBillingEnabledById(id, enabled) {
  const keys = load();
  const entry = keys.find((k) => k.id === id);
  if (!entry) return false;
  entry.billingEnabled = !!enabled;
  save(keys);
  return true;
}

export function issueKey(label) {
  const id = crypto.randomUUID();
  const key = `rtts_${crypto.randomBytes(24).toString("hex")}`;
  const keys = load();
  keys.push({
    id,
    keyHash: hash(key),
    keyPreview: key.slice(0, 10) + "…",
    label: label || "",
    created_at: new Date().toISOString(),
    revoked: false,
    billingEnabled: false,
    usageCharsSinceLastReport: 0,
    freeCharsUsed: 0,
    freeTierExhausted: false,
  });
  save(keys);
  return { id, key };
}

export function revokeKeyById(id) {
  const keys = load();
  const entry = keys.find((k) => k.id === id);
  if (!entry) return false;
  entry.revoked = true;
  save(keys);
  return true;
}

export function listKeys() {
  return load().map((k) => ({
    id: k.id,
    label: k.label,
    created_at: k.created_at,
    revoked: k.revoked,
    billingEnabled: !!k.billingEnabled,
    key_preview: k.keyPreview,
    freeCharsRemaining: k.billingEnabled ? null : Math.max(0, FREE_TIER_CHARS - (k.freeCharsUsed || 0)),
  }));
}

function applyUsage(entry, chars) {
  if (entry.billingEnabled) {
    entry.usageCharsSinceLastReport = (entry.usageCharsSinceLastReport || 0) + chars;
    return;
  }
  // Free-tier usage isn't billed and never reported to Stripe — tracked
  // separately so drainUsage()'s output stays exactly "what to invoice."
  entry.freeCharsUsed = (entry.freeCharsUsed || 0) + chars;
  if (entry.freeCharsUsed >= FREE_TIER_CHARS) entry.freeTierExhausted = true;
}

export function recordUsage(key, chars) {
  if (!key || !chars) return;
  const keys = load();
  const entry = keys.find((k) => k.keyHash === hash(key) && !k.revoked);
  if (!entry) return;
  applyUsage(entry, chars);
  save(keys);
}

// Same as recordUsage, but looked up by key ID rather than the raw key —
// used by the /admin/usage/report callback from Modal (see
// worker-modal-readaloud/app.py), which only ever sees the ID embedded in a
// session token, never the raw key itself.
export function recordUsageById(id, chars) {
  if (!id || !chars) return false;
  const keys = load();
  const entry = keys.find((k) => k.id === id && !k.revoked);
  if (!entry) return false;
  applyUsage(entry, chars);
  save(keys);
  return true;
}

// --- Session tokens for the direct-to-Modal fast path ---
// Deliberately NOT bound to a specific text/character count — the free-tier
// check at issuance time is a coarse "is this key currently allowed to
// connect at all" boolean (see checkAccess's freeTierExhausted fast path),
// and the actual billing-relevant number comes from Modal's own async report
// of what it really synthesized, not anything the client declares upfront.
// A short expiry is the only thing bounding how long a token is usable for.
export function createSessionToken(id) {
  if (!SESSION_TOKEN_SECRET) throw new Error("MODAL_SESSION_SECRET is not configured");
  const exp = Date.now() + SESSION_TOKEN_TTL_MS;
  const payload = Buffer.from(JSON.stringify({ id, exp })).toString("base64url");
  const sig = crypto.createHmac("sha256", SESSION_TOKEN_SECRET).update(payload).digest("base64url");
  return `${payload}.${sig}`;
}

// Returns the key id if the token is validly signed and unexpired, else null.
// Verification only — deliberately does not also check billing/free-tier
// status again, since that was already checked at issuance a few seconds
// earlier; re-checking here would just be the same TOCTOU race in a
// different spot, not a real improvement over the issuance-time flag.
export function verifySessionToken(token) {
  if (!SESSION_TOKEN_SECRET || !token) return null;
  const [payload, sig] = String(token).split(".");
  if (!payload || !sig) return null;
  const expectedSig = crypto.createHmac("sha256", SESSION_TOKEN_SECRET).update(payload).digest("base64url");
  const sigBuf = Buffer.from(sig);
  const expectedBuf = Buffer.from(expectedSig);
  if (sigBuf.length !== expectedBuf.length || !crypto.timingSafeEqual(sigBuf, expectedBuf)) return null;
  let parsed;
  try {
    parsed = JSON.parse(Buffer.from(payload, "base64url").toString());
  } catch {
    return null;
  }
  if (!parsed.id || typeof parsed.exp !== "number" || Date.now() > parsed.exp) return null;
  return parsed.id;
}

// Returns accumulated usage per key since the last drain and resets the
// counters — the caller (ReadAloud's usage-reporting job) is responsible for
// turning this into Stripe usage records. A failed report on that side loses
// the batch; acceptable at this volume, revisit with a durable outbox if not.
export function drainUsage() {
  const keys = load();
  const result = keys
    .filter((k) => k.usageCharsSinceLastReport > 0)
    .map((k) => ({ id: k.id, chars: k.usageCharsSinceLastReport }));
  for (const k of keys) k.usageCharsSinceLastReport = 0;
  save(keys);
  return result;
}
