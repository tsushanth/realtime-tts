// Structured audit logging for security-relevant gateway events: auth attempts
// (success/failure), API key creation/revocation/ownership/billing changes, voice
// creation/deletion, and other admin actions. Ground-floor compliance groundwork —
// not a real audit trail (no tamper-evidence, no durable storage beyond whatever
// captures stdout on Fly), but enough to reconstruct "who did what, when" from logs.
//
// Hard rule, same as the rest of this codebase: NEVER include a raw API key, admin
// secret, session token, or any other secret value in a log line. Callers pass a key
// `id` (safe, non-secret) or a short non-reversible preview, never the raw key/token.
//
// One JSON object per line on stdout (Fly captures/ships this). Kept intentionally
// dependency-free — this doesn't warrant a logging library at current scale.
export function auditLog(event, fields = {}) {
  const record = {
    ts: new Date().toISOString(),
    audit: true,
    event,
    ...fields,
  };
  console.log(JSON.stringify(record));
}
