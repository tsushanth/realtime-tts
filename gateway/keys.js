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

// Only billing-enabled keys may consume paid GPU compute — see server.js. A
// key exists in one of two states: freshly issued and not yet enabled (the
// caller must explicitly opt it in — see setBillingEnabledById), or enabled.
// ReadAloud's issuance route enables its own keys immediately on creation;
// there's no per-key payment-method concept in this product today, so
// "billing enabled" means "this key is allowed to run," not "a card is on
// file" — the stronger version of this gate is future work, not implied by
// this flag's current callers.
export function isBillingEnabled(key) {
  if (!key) return false;
  const keys = load();
  const entry = keys.find((k) => k.keyHash === hash(key) && !k.revoked);
  return !!entry?.billingEnabled;
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
  }));
}

export function recordUsage(key, chars) {
  if (!key || !chars) return;
  const keys = load();
  const entry = keys.find((k) => k.keyHash === hash(key) && !k.revoked);
  if (!entry) return;
  entry.usageCharsSinceLastReport = (entry.usageCharsSinceLastReport || 0) + chars;
  save(keys);
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
