// Resolves a real inbound Twilio call to the tenant that owns the dialed
// number — the gap that made every phone call get the exact same static
// config regardless of which number was called, and made per-tenant
// billing (see stripeMeter.js) impossible for real calls. Talks straight to
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

// Returns null when the number isn't routed to anything (unknown number, or
// its inbound slot is unset) — callers fall back to the old static
// single-tenant behavior rather than erroring the call.
export async function resolveInboundCall(toNumber) {
  if (!toNumber) return null;
  const numbers = await pg(
    'calldesk_phone_numbers',
    `number=eq.${encodeURIComponent(toNumber)}&select=tenant_id,inbound_agent_version_id`
  );
  const numberRow = numbers?.[0];
  if (!numberRow?.inbound_agent_version_id) {
    console.warn(`[tenant-lookup] no inbound routing for ${toNumber} — falling back to static config`);
    return null;
  }

  const versions = await pg(
    'calldesk_agent_versions',
    `id=eq.${numberRow.inbound_agent_version_id}&select=voice_engine,tts_backend,flow_id,agent_id`
  );
  const version = versions?.[0];
  // 'retell' versions are handled entirely on Retell's side (this call
  // wouldn't even reach call-loop-poc's Twilio number for those) — only
  // 'poc' versions need a flow handed to this engine.
  if (!version || version.voice_engine !== 'poc' || !version.flow_id) {
    console.warn(`[tenant-lookup] ${toNumber} -> version ${numberRow.inbound_agent_version_id} is not a poc-engine version with a flow — falling back`);
    return null;
  }

  const [flows, businesses] = await Promise.all([
    pg('calldesk_conversation_flows', `id=eq.${version.flow_id}&select=nodes,global_settings`),
    pg('calldesk_businesses', `tenant_id=eq.${numberRow.tenant_id}&select=stripe_customer_id`),
  ]);
  const flowRow = flows?.[0];
  if (!flowRow?.nodes?.length) {
    console.warn(`[tenant-lookup] flow ${version.flow_id} has no nodes — falling back`);
    return null;
  }

  const nodes = await attachKnowledgeBaseIds(flowRow.nodes, version.agent_id);

  return {
    flow: {
      nodes,
      startNodeId: flowRow.global_settings?.startNodeId || nodes[0].id,
      globalSettings: flowRow.global_settings || {},
    },
    ttsBackend: version.tts_backend || undefined,
    stripeCustomerId: businesses?.[0]?.stripe_customer_id || undefined,
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
