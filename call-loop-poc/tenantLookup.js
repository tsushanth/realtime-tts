import crypto from 'node:crypto';
import { safeFetch } from './ssrfGuard.js';
import { isCallAudioEnabled, encodeCallAudioForContext } from './callAudio.js';
// Resolves a real inbound Twilio call to the tenant that owns the dialed
// number — the gap that made every phone call get the exact same static
// config regardless of which number was called, and made per-tenant
// billing impossible for real calls. Talks straight to
// Supabase's PostgREST API (same shared project calldesktech uses) rather
// than pulling in the full supabase-js client for a handful of read-only
// lookups.
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

async function pg(table, query) {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) return null;
  const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${query}`, {
    headers: {
      apikey: SUPABASE_SERVICE_ROLE_KEY,
      Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
    },
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    console.error(`[tenant-lookup] ${table} query failed: HTTP ${res.status}`);
    return null;
  }
  return res.json();
}

// Global CPS cap for the shared Twilio account (2026-09-17) — this is the
// single choke point every Twilio call placement in this file goes through
// (batch calling, single test calls, mystery-shopper), so it's the right
// place to enforce the platform-wide limit: Twilio is ONE account shared
// across every tenant, and this app runs multiple Fly machines, so an
// in-memory counter wouldn't be safe. See calldesktech's
// supabase/migrations/023_rate_limiters.sql for the shared atomic-token-
// bucket implementation both repos call into.
//
// Verified against the real account (2026-09-18, Twilio Console -> Voice ->
// Settings -> General -> "Calls per second (CPS)"): Current CPS
// configuration is 1 — Twilio's own default, never raised. Capacity=1 (no
// burst) matches this exactly rather than guessing a buffer above it;
// Twilio's own text says it queues/throttles excess REST requests to this
// rate rather than hard-rejecting them, but relying on their queue instead
// of our own would just move the backlog somewhere we can't see it. Raise
// both once/if the account's real CPS is increased (Console has an "Edit
// CPS" control — this is a paid, non-trial account, so that's a real,
// available lever if throughput ever needs to go up).
const TWILIO_GLOBAL_CAPACITY = Number(process.env.TWILIO_GLOBAL_BURST ?? 1);
const TWILIO_GLOBAL_REFILL_PER_SEC = Number(process.env.TWILIO_GLOBAL_CPS ?? 1);

export async function acquireTwilioGlobalToken(maxWaitMs = 30000) {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) return true; // fail open, same reasoning as pg()
  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${SUPABASE_URL}/rest/v1/rpc/calldesk_try_acquire_token`, {
        method: 'POST',
        headers: {
          apikey: SUPABASE_SERVICE_ROLE_KEY,
          Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          p_key: 'twilio-global',
          p_capacity: TWILIO_GLOBAL_CAPACITY,
          p_refill_per_sec: TWILIO_GLOBAL_REFILL_PER_SEC,
          p_cost: 1,
        }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) {
        console.error(`[rate-limiter] RPC failed: HTTP ${res.status} — failing open for this attempt`);
        return true;
      }
      const acquired = await res.json();
      if (acquired === true) return true;
    } catch (err) {
      console.error('[rate-limiter] RPC failed — failing open for this attempt', err);
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return false;
}

// Used by /place-test-call's shopper branch to tag a call log as internal
// test traffic when the shopper's target happens to be one of our own
// tenants' real numbers (2026-09-17) — see that route's own comment for why
// this writes the log directly rather than relying on the normal inbound-
// call webhook path for it.
export async function findTenantIdByNumber(number) {
  if (!number) return null;
  const rows = await pg('calldesk_phone_numbers', `number=eq.${encodeURIComponent(number)}&select=tenant_id&limit=1`);
  return rows?.[0]?.tenant_id || null;
}

// Real call logging for poc-engine calls (2026-09-17): unlike Retell, which
// notifies calldesktech of a call's lifecycle via its own webhook, THIS
// engine owns the telephony lifecycle directly — nothing else logs a
// poc-engine call, so it has to happen here. Same shared Supabase project,
// direct PostgREST insert/update rather than a round trip through
// calldesktech, matching pg()'s own reasoning above.
export async function insertCallLog(row) {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) return null;
  // `tier` is a newer nullable column: if a write that carries it is rejected (column not migrated yet), retry once without it
  // so the call log row is never lost because of billing attribution.
  if (row && 'tier' in row) {
    const { tier, ...rest } = row;
    const id = await insertCallLogOnce(row, true);
    return id === undefined ? insertCallLogOnce(rest) : id;
  }
  return insertCallLogOnce(row);
}

// Returns the id, null on failure; with `tierRetry`, undefined means "rejected with a 4xx, caller may retry without tier".
async function insertCallLogOnce(row, tierRetry = false) {
  try {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/calldesk_call_logs`, {
      method: 'POST',
      headers: {
        apikey: SUPABASE_SERVICE_ROLE_KEY,
        Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
        'Content-Type': 'application/json',
        Prefer: 'return=representation',
      },
      body: JSON.stringify(row),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) {
      console.error(`[call-log] insert failed: HTTP ${res.status}`, await res.text().catch(() => ''));
      if (tierRetry && res.status >= 400 && res.status < 500) return undefined;
      return null;
    }
    const inserted = await res.json();
    return inserted?.[0]?.id || null;
  } catch (err) {
    console.error('[call-log] insert failed', err);
    return null;
  }
}

export async function updateCallLogById(id, patch) {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY || !id) return;
  try {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/calldesk_call_logs?id=eq.${encodeURIComponent(id)}`, {
      method: 'PATCH',
      headers: {
        apikey: SUPABASE_SERVICE_ROLE_KEY,
        Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patch),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) {
      console.error(`[call-log] update by id failed: HTTP ${res.status}`, await res.text().catch(() => ''));
    }
  } catch (err) {
    console.error('[call-log] update by id failed', err);
  }
}

export async function updateCallLogByCallSid(callSid, patch) {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY || !callSid) return;
  if (patch && 'tier' in patch) {
    const { tier, ...rest } = patch;
    const ok = await updateCallLogByCallSidOnce(callSid, patch);
    if (!ok) await updateCallLogByCallSidOnce(callSid, rest); // column not migrated yet: still finalize the row
    return;
  }
  await updateCallLogByCallSidOnce(callSid, patch);
}

// true = written, false = rejected or errored.
async function updateCallLogByCallSidOnce(callSid, patch) {
  try {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/calldesk_call_logs?retell_call_id=eq.${encodeURIComponent(callSid)}`, {
      method: 'PATCH',
      headers: {
        apikey: SUPABASE_SERVICE_ROLE_KEY,
        Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(patch),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) {
      console.error(`[call-log] update failed: HTTP ${res.status}`, await res.text().catch(() => ''));
      return false;
    }
    return true;
  } catch (err) {
    console.error('[call-log] update failed', err);
    return false;
  }
}

// Recording retention enforcement (2026-09-17, see Settings page's "Call
// Recording" section) — returns [{ recordingSid, callLogId }] for every
// poc-engine recording whose tenant has set a retention_days window AND
// whose call is now older than it. Deliberately two flat queries + a JS
// filter rather than one clever PostgREST query: a per-row date comparison
// against a PER-TENANT setting isn't expressible as a single filter, and
// the volume here (recordings with a retention window set at all) is small
// enough that this is simpler and easier to reason about than fighting
// PostgREST's query syntax for it.
export async function findExpiredRecordings() {
  const tenants = await pg('calldesk_tenants', 'select=id,settings');
  const retentionByTenant = new Map();
  for (const t of tenants || []) {
    const days = Number(t.settings?.recording_retention_days);
    if (days > 0) retentionByTenant.set(t.id, days);
  }
  if (retentionByTenant.size === 0) return [];

  const logs = await pg(
    'calldesk_call_logs',
    'voice_engine=eq.poc&recording_sid=not.is.null&select=id,tenant_id,recording_sid,created_at&order=created_at.asc&limit=500'
  );
  const now = Date.now();
  return (logs || [])
    .filter((row) => {
      const retentionDays = retentionByTenant.get(row.tenant_id);
      if (!retentionDays) return false;
      return now - new Date(row.created_at).getTime() > retentionDays * 24 * 60 * 60 * 1000;
    })
    .map((row) => ({ recordingSid: row.recording_sid, callLogId: row.id }));
}

// Returns null when the number isn't routed to anything (unknown number, or
// its inbound slot is unset) — callers fall back to the old static
// single-tenant behavior rather than erroring the call.
//
// `direction` ('inbound' | 'outbound') picks which of the number's two
// routing slots to resolve — calldesk_phone_numbers has always stored both
// inbound_agent_version_id and outbound_agent_version_id (the Numbers page
// UI lets a tenant set both independently), but this function only ever
// read the inbound one until now, so an outbound-placed test call always
// ran the number's INBOUND flow regardless of what was actually configured
// for outbound. Found auditing what a real "make an outbound call" button
// would need to call correctly.
// Per-tenant jingle / sound-effect assets (see callAudio.js and the 2026-09-29 design spec). The
// rows are queried on every call (cheap, and picks up dashboard changes immediately), but the
// audio bytes are cached per process: an asset's storage object is immutable for its row id (the
// dashboard creates a fresh row rather than overwriting — no versioning, per the spec), so an id
// is safe to cache forever. Bounded so a long-lived machine can't grow without limit.
const CALL_AUDIO_BUCKET = 'call-audio-assets';
const CALL_AUDIO_CACHE_MAX = 200;
const callAudioBytesCache = new Map(); // asset id -> Buffer

async function downloadCallAudio(row) {
  const hit = callAudioBytesCache.get(row.id);
  if (hit) return hit;
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY || !row.mulaw8k_storage_path) return null;
  const res = await fetch(
    `${SUPABASE_URL}/storage/v1/object/${CALL_AUDIO_BUCKET}/${row.mulaw8k_storage_path.split('/').map(encodeURIComponent).join('/')}`,
    {
      headers: { apikey: SUPABASE_SERVICE_ROLE_KEY, Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}` },
      signal: AbortSignal.timeout(5000),
    }
  );
  if (!res.ok) {
    console.error(`[tenant-lookup] call audio download failed for asset ${row.id}: HTTP ${res.status}`);
    return null;
  }
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length === 0) return null;
  if (callAudioBytesCache.size >= CALL_AUDIO_CACHE_MAX) callAudioBytesCache.delete(callAudioBytesCache.keys().next().value);
  callAudioBytesCache.set(row.id, buf);
  return buf;
}

// Returns the context-message payload for callAudio.js (or undefined). Never throws and never
// blocks the call on a failure: a broken/missing asset just means that sound doesn't play.
export async function fetchCallAudioAssets(tenantId) {
  if (!tenantId || !isCallAudioEnabled()) return undefined;
  try {
    const rows = await pg(
      'tenant_call_audio_assets',
      `tenant_id=eq.${encodeURIComponent(tenantId)}&enabled=eq.true&select=id,asset_type,name,description,mulaw8k_storage_path,enabled&order=created_at.asc`
    );
    if (!rows?.length) return undefined;
    const withAudio = await Promise.all(
      rows.map(async (r) => {
        try {
          return { ...r, audio: await downloadCallAudio(r) };
        } catch (err) {
          console.error(`[tenant-lookup] call audio asset ${r.id} unavailable`, err);
          return { ...r, audio: null };
        }
      })
    );
    return encodeCallAudioForContext(withAudio);
  } catch (err) {
    console.error('[tenant-lookup] call audio lookup failed', err);
    return undefined;
  }
}

export const VALID_TIERS = ['lite', 'standard', 'pro'];
// Anything other than the three known tiers is treated as no tier.
export function normalizeTier(value) {
  return typeof value === 'string' && VALID_TIERS.includes(value.trim().toLowerCase()) ? value.trim().toLowerCase() : undefined;
}

export async function resolveInboundCall(toNumber, direction = 'inbound') {
  if (!toNumber) return null;
  const versionColumn = direction === 'outbound' ? 'outbound_agent_version_id' : 'inbound_agent_version_id';
  // Optional per-number carrier ('twilio' default | 'telnyx'). Only requested when TELNYX_ENABLED=1, so a Twilio-only
  // deployment issues exactly the same query as before. Same retry pattern as llm_model below: pg() returns null on ANY
  // failed query (e.g. the carrier column does not exist yet), so retry once without it - a call must still route.
  const phoneBaseQuery = `number=eq.${encodeURIComponent(toNumber)}&select=tenant_id,${versionColumn}`;
  let numbers;
  if (process.env.TELNYX_ENABLED === '1') {
    numbers = await pg('calldesk_phone_numbers', `${phoneBaseQuery},carrier`);
    if (numbers === null) numbers = await pg('calldesk_phone_numbers', phoneBaseQuery);
  } else {
    numbers = await pg('calldesk_phone_numbers', phoneBaseQuery);
  }
  const numberRow = numbers?.[0];
  const agentVersionId = numberRow?.[versionColumn];
  if (!agentVersionId) {
    console.warn(`[tenant-lookup] no ${direction} routing for ${toNumber} — falling back to static config`);
    return null;
  }

  // llm_model / tts_model are the version's optional model choice (calldesktech migration 062). pg() returns null on ANY failed
  // query, which would make this number unroutable, so if the query that names the new columns fails (for example before the
  // migration is applied) retry once with the original column list: model choice is optional, a call must still connect.
  const BASE_COLS = 'voice_engine,tts_backend,flow_id,agent_id';
  // Billing tier (calldesktech: calldesk_agent_versions.tier, nullable 'lite'|'standard'|'pro'). Same pattern again: if the tier
  // column does not exist yet the query fails, so retry without it (keeping the model choice), then without the models too.
  // voice_id (the builder's voice choice, saved by the web app on publish) is requested in the same query. Column sets are tried in
  // order until one is accepted; each later set drops something that may not exist yet, so a call always connects:
  //   1 everything  2 no tier (keeps models + voice)  3 base + voice (no model columns)
  //   4 no voice_id (the pre-voice query)  5 no voice_id, no tier  6 base columns only
  const colSets = [
    `${BASE_COLS},llm_model,tts_model,tier,voice_id`,
    `${BASE_COLS},llm_model,tts_model,voice_id`,
    `${BASE_COLS},voice_id`,
    `${BASE_COLS},llm_model,tts_model,tier`,
    `${BASE_COLS},llm_model,tts_model`,
    BASE_COLS,
  ];
  let versions = null;
  for (const cols of colSets) {
    versions = await pg('calldesk_agent_versions', `id=eq.${agentVersionId}&select=${cols}`);
    if (versions !== null) break;
  }
  const version = versions?.[0];
  const tier = normalizeTier(version?.tier);
  // 'retell' versions are handled entirely on Retell's side (this call
  // wouldn't even reach call-loop-poc's Twilio number for those) — only
  // 'poc' versions need a flow handed to this engine.
  if (!version || version.voice_engine !== 'poc' || !version.flow_id) {
    console.warn(`[tenant-lookup] ${toNumber} -> version ${agentVersionId} is not a poc-engine version with a flow — falling back`);
    return null;
  }

  const [flows, businesses, calendars, tenants, callAudio] = await Promise.all([
    pg('calldesk_conversation_flows', `id=eq.${version.flow_id}&select=nodes,global_settings`),
    pg('calldesk_businesses', `tenant_id=eq.${numberRow.tenant_id}&select=stripe_customer_id`),
    pg('calldesk_calendar_connections', `tenant_id=eq.${numberRow.tenant_id}&select=provider,api_key,event_type_id`),
    pg('calldesk_tenants', `id=eq.${numberRow.tenant_id}&select=settings`),
    fetchCallAudioAssets(numberRow.tenant_id),
  ]);
  const flowRow = flows?.[0];
  if (!flowRow?.nodes?.length) {
    console.warn(`[tenant-lookup] flow ${version.flow_id} has no nodes — falling back`);
    return null;
  }

  const nodes = await attachKnowledgeBaseIds(flowRow.nodes, version.agent_id);

  return {
    tenantId: numberRow.tenant_id,
    // Key present only when the number's carrier column says 'telnyx'; absent (= twilio) for column missing, null or anything else,
    // so the Twilio path's resolved object is unchanged.
    ...(numberRow.carrier === 'telnyx' ? { carrier: 'telnyx' } : {}),
    flow: {
      nodes,
      startNodeId: flowRow.global_settings?.startNodeId || nodes[0].id,
      globalSettings: flowRow.global_settings || {},
    },
    ttsBackend: version.tts_backend || undefined,
    // Optional per-version model choice (undefined = the engine defaults). The engine ignores a model it cannot use.
    llmModel: version.llm_model || undefined,
    ttsModel: version.tts_model || undefined,
    // The builder's voice choice (calldesk_agent_versions.voice_id). Key present only when set, so a version without one resolves
    // exactly as before. The session validates it against the backend before use (see voiceSelection.js).
    ...(typeof version.voice_id === 'string' && version.voice_id.trim() ? { voiceId: version.voice_id.trim() } : {}),
    // Per-tenant jingle + sound effects (undefined for a tenant with none, or when the global
    // CALL_AUDIO_ASSETS_ENABLED kill switch is off) — see callAudio.js.
    callAudio,
    // Only present for a version with a valid billing tier, so an untiered call's resolved object is unchanged.
    ...(tier ? { tier } : {}),
    stripeCustomerId: businesses?.[0]?.stripe_customer_id || undefined,
    // Real calendar booking (2026-09-17) — present only for a tenant that's
    // actually connected one; check_availability/book_appointment simply
    // aren't offered as tools when this is undefined (see server.js).
    calendar: calendars?.[0]
      ? { provider: calendars[0].provider, apiKey: calendars[0].api_key, eventTypeId: calendars[0].event_type_id }
      : undefined,
    // Real per-tenant recording control (2026-09-17, see Settings page's
    // "Call Recording" section) — defaults to recording ON (matches
    // Retell's own default), only skipped when explicitly turned off.
    recordingEnabled: tenants?.[0]?.settings?.recording_enabled !== 'false',
    // Tenant-wide default for how easily a caller can barge in (high/medium/low/off) — see server.js _transcriptMeetsInterruptionThreshold for the full precedence.
    interruptionSensitivity: ['high', 'medium', 'low', 'off'].includes(tenants?.[0]?.settings?.interruption_sensitivity) ? tenants[0].settings.interruption_sensitivity : undefined,
  };
}

// Real Q&A retrieval for a knowledge_base node (see _executeKnowledgeBaseNode
// in server.js) — no embeddings/semantic search, just the KB's actual
// content handed to the model to pick from. Bounded at 50 items: enough for
// every real KB seeded so far, and an unbounded fetch folded whole into the
// prompt would blow the context budget on a large one.
export async function fetchKnowledgeItems(knowledgeBaseId) {
  if (!knowledgeBaseId) return [];
  const items = await pg(
    'calldesk_knowledge_items',
    `knowledge_base_id=eq.${knowledgeBaseId}&select=question,answer&limit=50`
  );
  return items || [];
}

// A knowledge_base flow node has no column of its own for which knowledge
// base it reads from — that association lives on calldesk_knowledge_bases
// via agent_id. Stamping the real id onto each such node here (rather than
// requiring a flow schema change) is what lets server.js's
// _executeKnowledgeBaseNode actually query real content instead of running
// the node as a plain prompt with nothing behind it (the disclosed gap).
async function attachKnowledgeBaseIds(nodes, agentId) {
  if (!nodes.some((n) => n.type === 'knowledge_base') || !agentId) return nodes;
  const kbs = await pg('calldesk_knowledge_bases', `agent_id=eq.${agentId}&select=id&limit=1`);
  const knowledgeBaseId = kbs?.[0]?.id;
  if (!knowledgeBaseId) return nodes;
  return nodes.map((n) =>
    n.type === 'knowledge_base' ? { ...n, params: { ...n.params, knowledgeBaseId } } : n
  );
}

// Same delivery contract as calldesktech's dispatchWebhookEvent (src/lib/webhooks.ts):
// body {event, created_at, data}, X-CallDesk-Signature = sha256=HMAC(secret, body).
export async function dispatchTenantWebhook(tenantId, event, data) {
  if (!tenantId) return;
  try {
    const hooks = await pg('calldesk_webhooks', `select=id,url,secret&tenant_id=eq.${encodeURIComponent(tenantId)}&enabled=eq.true&events=cs.${encodeURIComponent(`{"${event}"}`)}`);
    if (!hooks?.length) return;
    const body = JSON.stringify({ event, created_at: new Date().toISOString(), data });
    await Promise.all(hooks.map(async (wh) => {
      try {
        // wh.url is tenant-controlled: ssrfGuard refuses private/internal destinations (see ssrfGuard.js).
        const res = await safeFetch(wh.url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'CallDesk-Webhooks/1',
            'X-CallDesk-Event': event,
            'X-CallDesk-Signature': `sha256=${crypto.createHmac('sha256', wh.secret).update(body).digest('hex')}`,
          },
          body,
          timeoutMs: 8000,
        });
        console.log(`[webhook] ${wh.id} ${event} -> HTTP ${res.status}`);
      } catch (err) {
        console.error(`[webhook] ${wh.id} ${event} failed: ${err.message}`);
      }
    }));
  } catch (err) {
    console.error('[webhook] dispatch failed', err);
  }
}

export async function findTenantIdByCallSid(callSid) {
  const rows = await pg('calldesk_call_logs', `select=tenant_id&retell_call_id=eq.${encodeURIComponent(callSid)}&limit=1`);
  return rows?.[0]?.tenant_id || null;
}

// Latest published poc-engine version of another agent in the SAME tenant, as a runnable flow (for the agent_transfer node).
export async function resolveAgentFlow(agentId, tenantId) {
  const agents = await pg('calldesk_agents', `id=eq.${encodeURIComponent(agentId)}&tenant_id=eq.${encodeURIComponent(tenantId)}&select=id`);
  if (!agents?.length) return null;
  const versions = await pg('calldesk_agent_versions', `agent_id=eq.${encodeURIComponent(agentId)}&voice_engine=eq.poc&flow_id=not.is.null&order=version_number.desc&limit=1&select=flow_id,agent_id`);
  const version = versions?.[0];
  if (!version) return null;
  const flows = await pg('calldesk_conversation_flows', `id=eq.${version.flow_id}&select=nodes,global_settings`);
  const flowRow = flows?.[0];
  if (!flowRow?.nodes?.length) return null;
  const nodes = await attachKnowledgeBaseIds(flowRow.nodes, version.agent_id);
  return { flow: { nodes, startNodeId: flowRow.global_settings?.startNodeId || nodes[0].id, globalSettings: flowRow.global_settings || {} } };
}
