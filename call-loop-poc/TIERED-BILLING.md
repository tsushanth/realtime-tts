# Per-agent billing tier (engine side)

The engine only records which tier an agent version is on. It does not bill tiers.
Billing for tiered calls is done by the web app's daily job, which reads
`calldesk_call_logs` and reports voice seconds to Stripe Billing Meters per customer.

## What the engine does

1. `tenantLookup.resolveInboundCall` reads `calldesk_agent_versions.tier` for the
   version serving the call, in the same query as `llm_model` / `tts_model`.
   Allowed values are `lite`, `standard`, `pro`; anything else (or NULL) means no tier.
   If the column does not exist yet, the query is retried without it (model choice kept),
   then without the model columns too. A call never fails because of the tier.
2. The resolved object carries `tier` only when set, and `buildTenantContextMessage`
   passes it as `tier` only when set. Calls with no tier produce the same objects and
   messages as before.
3. `CallSession.tier` holds it for the call (also kept across a detour/resume).
4. The call log row gets `tier` when the row is created and again when it is finalized at
   hangup. Calls with no tier write exactly what they wrote before. If a write that includes
   `tier` is rejected (column not migrated), it is retried once without `tier`, so the row,
   duration and transcript are never lost.

`stripeMeter.js` is unchanged: it still reports legacy usage records exactly as before.

## Prerequisite

The web team's migration adds nullable `calldesk_call_logs.tier` and
`calldesk_agent_versions.tier`. The engine works before and after it is applied.

## Verify in production after enabling

- Set a test agent version's `tier`, place a call, then check the call's row:
  `select retell_call_id, tier, duration_seconds from calldesk_call_logs order by created_at desc limit 5;`
  The tiered call has `tier` set and a non-zero duration; untiered calls have NULL.
- Engine logs should show no `[tenant-lookup] calldesk_agent_versions query failed` lines
  once the column exists, and no `[call-log] ... failed` lines.
- Confirm the daily job's Stripe meter events for that customer match the call's seconds.

## Rollback

Set the version's `tier` back to NULL (calls stop carrying a tier immediately), or redeploy
the previous engine image. The extra column is nullable and ignored by older engine versions.
