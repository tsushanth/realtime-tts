// Minimal API key store — a JSON file on the Fly volume, not a database. Fine for
// manual issuance at low volume; revisit (real DB, hashed keys, rate limits per key)
// before this is a self-serve signup flow. Keys are stored in PLAINTEXT — acceptable
// only because issuance is manual/trusted for now. Hash them before self-serve signup.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const KEYS_PATH = process.env.KEYS_PATH || "/data/keys.json";

function load() {
  try {
    return JSON.parse(fs.readFileSync(KEYS_PATH, "utf8"));
  } catch {
    return [];
  }
}

function save(keys) {
  fs.mkdirSync(path.dirname(KEYS_PATH), { recursive: true });
  fs.writeFileSync(KEYS_PATH, JSON.stringify(keys, null, 2));
}

export function isValidKey(key) {
  if (!key) return false;
  const keys = load();
  const entry = keys.find((k) => k.key === key && !k.revoked);
  return !!entry;
}

export function issueKey(label) {
  const key = `rtts_${crypto.randomBytes(24).toString("hex")}`;
  const keys = load();
  keys.push({ key, label: label || "", created_at: new Date().toISOString(), revoked: false });
  save(keys);
  return key;
}

export function revokeKey(key) {
  const keys = load();
  const entry = keys.find((k) => k.key === key);
  if (!entry) return false;
  entry.revoked = true;
  save(keys);
  return true;
}

export function listKeys() {
  return load().map((k) => ({ ...k, key: k.key.slice(0, 10) + "…" }));
}
