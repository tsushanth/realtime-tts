// node --test gateway/tests/owner.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
process.env.MODAL_SESSION_SECRET = "s3cret";
process.env.KEYS_PATH = path.join(os.tmpdir(), `keys-owner-${process.pid}.json`);
const keys = await import("../keys.js");

test("token without owner has no uid (legacy behaviour)", () => {
  const { id, key } = keys.issueKey("legacy");
  assert.equal(keys.getOwnerForKey(key), null);
  const t = keys.createSessionToken(id, keys.getOwnerForKey(key));
  assert.equal(keys.verifySessionToken(t), id);
  assert.deepEqual(keys.verifySessionClaims(t), { id, uid: null });
});
test("key issued with owner embeds uid; verifySessionToken still returns the id", () => {
  const { id, key } = keys.issueKey("mine", "user-1");
  const t = keys.createSessionToken(id, keys.getOwnerForKey(key));
  assert.deepEqual(keys.verifySessionClaims(t), { id, uid: "user-1" });
  assert.equal(keys.verifySessionToken(t), id);
});
test("setOwnerById backfills once and never overwrites", () => {
  const { id } = keys.issueKey("old");
  assert.equal(keys.setOwnerById(id, "u1"), "set");
  assert.equal(keys.setOwnerById(id, "u1"), "unchanged");
  assert.equal(keys.setOwnerById(id, "u2"), "conflict");
  assert.equal(keys.setOwnerById("nope", "u1"), null);
});
test("tampered uid fails signature", () => {
  const { id } = keys.issueKey("x", "user-1");
  const t = keys.createSessionToken(id, "user-1");
  const [p, sig] = t.split(".");
  const forged = Buffer.from(JSON.stringify({ ...JSON.parse(Buffer.from(p, "base64url")), uid: "user-2" })).toString("base64url");
  assert.equal(keys.verifySessionClaims(`${forged}.${sig}`), null);
});
