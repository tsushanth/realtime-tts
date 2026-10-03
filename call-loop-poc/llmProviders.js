// Multi-provider LLM support for the turn-generation call (CallSession._generateTurn in server.js).
//
// Anthropic stays the native, default path and is untouched. Any OTHER model is reached through an OpenAI-compatible
// "chat completions" endpoint (OpenAI, Google's Gemini compatibility endpoint, Groq, xAI, DeepSeek, Mistral, a self-hosted
// vLLM, ...). The tools are defined once, in Anthropic's shape ({name, description, input_schema}), and this module
// translates them, then converts the streamed answer BACK into the same {content: [{type:'text'|'tool_use'}], usage} shape
// the rest of server.js already reads from Anthropic's stream.finalMessage(). Everything downstream (record_field,
// transition_flow, play_sound_effect, ...) therefore never knows which provider answered.
//
// Swapping a model, no code change:
//   * per call:   context message  { "model": "gpt-6-luna" }
//   * per node:   node.params.model
//   * globally:   LLM_MODEL env (the default); unusable values fall back to Haiku with a warning
//   * new model:  LLM_EXTRA_MODELS env (JSON array, see README) plus that provider's key env var
// A model is "usable" only when its key env var is set, so a model can never be selected that cannot answer.
// If a non-Anthropic provider fails BEFORE it has spoken, the turn is retried on Haiku (see generateWithFallback).

export const DEFAULT_ANTHROPIC_MODEL = 'claude-haiku-4-5-20251001';

const GEMINI_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions';

// Prices are USD per million tokens, from the providers' own pricing pages (retrieved 2026-10-01/02); re-verify before relying on them.
const BUILTIN_MODELS = [
  { id: 'claude-haiku-4-5-20251001', provider: 'anthropic', price: { in: 1, out: 5 } },
  { id: 'claude-sonnet-4-6', provider: 'anthropic', price: { in: 3, out: 15 } },
  {
    id: 'gpt-6-luna', provider: 'openai', endpoint: 'https://api.openai.com/v1/chat/completions', keyEnv: 'OPENAI_API_KEY',
    price: { in: 0.1, out: 0.5 },
    // Found live 2026-09-28/29: Luna rejects `max_tokens`; reasoning_effort 'none' is required for tool calling on Chat
    // Completions and is the only voice-safe latency setting.
    quirks: { maxTokensParam: 'max_completion_tokens', reasoningEffort: 'none', streamUsage: true },
  },
  // Gemini 3.x: tool calls carry extra_content.google.thought_signature, which MUST be echoed back on the assistant message
  // of the follow-up request or Google answers 400 "Function call is missing a thought_signature" (quirks.preserveExtraContent).
  // gemini-2.5-flash-lite was removed: Google returns 404 "no longer available to new users" (checked 2026-10-02).
  {
    id: 'gemini-3.1-flash-lite', provider: 'gemini', endpoint: GEMINI_ENDPOINT,
    keyEnv: 'GEMINI_API_KEY', price: { in: 0.25, out: 1.5 }, status: 'untested', quirks: { preserveExtraContent: true },
  },
  {
    id: 'gemini-3.5-flash-lite', provider: 'gemini', endpoint: GEMINI_ENDPOINT,
    keyEnv: 'GEMINI_API_KEY', price: { in: 0.3, out: 2.5 }, status: 'untested', quirks: { preserveExtraContent: true },
  },
];

const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,79}$/;
const PROVIDER_RE = /^[a-z][a-z0-9_-]{0,31}$/;
const ENV_RE = /^[A-Z][A-Z0-9_]{0,63}$/;
const MAX_TOKENS_PARAMS = new Set(['max_tokens', 'max_completion_tokens']);
const EXTRA_BODY_MAX_BYTES = 2000;
// Keys a spec's extraBody may never override: they are owned by the adapter.
const RESERVED_BODY_KEYS = new Set(['model', 'messages', 'stream', 'tools', 'tool_choice', 'stream_options', 'max_tokens', 'max_completion_tokens']);

function endpointOk(url) {
  try {
    const u = new URL(url);
    if (u.protocol === 'https:') return true;
    return u.protocol === 'http:' && (u.hostname === 'localhost' || u.hostname === '127.0.0.1');
  } catch { return false; }
}

// Returns a normalized spec, or { error } explaining why a raw entry is rejected.
export function normalizeSpec(raw) {
  if (!raw || typeof raw !== 'object') return { error: 'not an object' };
  if (typeof raw.id !== 'string' || !ID_RE.test(raw.id)) return { error: 'id must match ' + ID_RE };
  if (typeof raw.provider !== 'string' || !PROVIDER_RE.test(raw.provider)) return { error: 'provider must match ' + PROVIDER_RE };
  const native = raw.provider === 'anthropic';
  const price = raw.price || {};
  if (![price.in, price.out].every((n) => typeof n === 'number' && Number.isFinite(n) && n >= 0)) return { error: 'price.in and price.out (USD per million tokens) are required numbers' };
  const spec = { id: raw.id, provider: raw.provider, native, price: { in: price.in, out: price.out }, status: raw.status === 'untested' ? 'untested' : 'ok', quirks: {} };
  if (native) return spec;
  if (typeof raw.endpoint !== 'string' || !endpointOk(raw.endpoint)) return { error: 'endpoint must be an https URL (http only for localhost)' };
  if (typeof raw.keyEnv !== 'string' || !ENV_RE.test(raw.keyEnv)) return { error: 'keyEnv must be an UPPER_SNAKE env var name' };
  spec.endpoint = raw.endpoint;
  spec.keyEnv = raw.keyEnv;
  const q = raw.quirks && typeof raw.quirks === 'object' ? raw.quirks : {};
  if (q.maxTokensParam !== undefined) {
    if (!MAX_TOKENS_PARAMS.has(q.maxTokensParam)) return { error: 'quirks.maxTokensParam must be max_tokens or max_completion_tokens' };
    spec.quirks.maxTokensParam = q.maxTokensParam;
  }
  if (q.reasoningEffort !== undefined) {
    if (typeof q.reasoningEffort !== 'string' || !/^[a-z]{1,16}$/.test(q.reasoningEffort)) return { error: 'quirks.reasoningEffort must be a short lowercase word' };
    spec.quirks.reasoningEffort = q.reasoningEffort;
  }
  if (q.streamUsage !== undefined) spec.quirks.streamUsage = q.streamUsage === true;
  if (q.preserveExtraContent !== undefined) spec.quirks.preserveExtraContent = q.preserveExtraContent === true;
  if (q.extraBody !== undefined) {
    if (!q.extraBody || typeof q.extraBody !== 'object' || Array.isArray(q.extraBody)) return { error: 'quirks.extraBody must be an object' };
    if (JSON.stringify(q.extraBody).length > EXTRA_BODY_MAX_BYTES) return { error: 'quirks.extraBody too large' };
    const bad = Object.keys(q.extraBody).find((k) => RESERVED_BODY_KEYS.has(k));
    if (bad) return { error: `quirks.extraBody may not set "${bad}"` };
    spec.quirks.extraBody = q.extraBody;
  }
  return spec;
}

// Built-ins plus LLM_EXTRA_MODELS (a JSON array of the same shape). A bad entry is skipped with a warning, never fatal.
export function buildRegistry(env = process.env, log = console) {
  const reg = new Map();
  for (const m of BUILTIN_MODELS) {
    const s = normalizeSpec(m);
    if (s.error) throw new Error(`built-in model ${m.id}: ${s.error}`); // a programming error, caught by the tests
    reg.set(s.id, s);
  }
  const extra = (env.LLM_EXTRA_MODELS || '').trim();
  if (extra) {
    let list;
    try { list = JSON.parse(extra); } catch (e) { log.warn(`[llm] LLM_EXTRA_MODELS is not valid JSON (${e.message}); ignoring it`); list = []; }
    if (!Array.isArray(list)) { log.warn('[llm] LLM_EXTRA_MODELS must be a JSON array; ignoring it'); list = []; }
    for (const raw of list) {
      const s = normalizeSpec(raw);
      if (s.error) { log.warn(`[llm] skipping LLM_EXTRA_MODELS entry ${JSON.stringify(raw?.id)}: ${s.error}`); continue; }
      reg.set(s.id, s); // same id as a built-in replaces it (e.g. to change a price or endpoint)
    }
  }
  return reg;
}

// A model can answer only if its provider is reachable: Anthropic needs its client, every other provider needs its key env var.
export function isUsable(spec, env = process.env, hasAnthropic = true) {
  if (!spec) return false;
  if (spec.native) return !!hasAnthropic;
  return typeof env[spec.keyEnv] === 'string' && env[spec.keyEnv].trim() !== '';
}

export function resolveDefaultModel(reg, requested, env = process.env, hasAnthropic = true, log = console) {
  const want = (requested || '').trim() || DEFAULT_ANTHROPIC_MODEL;
  const spec = reg.get(want);
  if (isUsable(spec, env, hasAnthropic)) return want;
  log.warn(`[llm] default model "${want}" is ${spec ? `not usable (set ${spec.keyEnv || 'ANTHROPIC_API_KEY'})` : 'not registered'}; using ${DEFAULT_ANTHROPIC_MODEL}`);
  return DEFAULT_ANTHROPIC_MODEL;
}

export function describeModels(reg, env = process.env, hasAnthropic = true) {
  return [...reg.values()].map((s) => `${s.id} (${s.provider}${s.status === 'untested' ? ', untested' : ''}${isUsable(s, env, hasAnthropic) ? '' : ', unavailable: no key'})`).join('; ');
}

// Side calls that force a specific Anthropic tool (the forced transition retries, variable extraction) always run on an
// Anthropic model: a non-Anthropic id sent to Anthropic's API would just fail.
export function auxAnthropicModel(reg, selectedId) {
  const s = reg.get(selectedId);
  return s && s.native ? s.id : DEFAULT_ANTHROPIC_MODEL;
}

// USD per token for the cost tracker.
export function ratesFor(reg, id) {
  const s = reg.get(id);
  return s ? { input: s.price.in / 1_000_000, output: s.price.out / 1_000_000 } : null;
}

// Anthropic tool schema -> OpenAI function-calling schema.
export function toOpenAiTools(anthropicTools) {
  return anthropicTools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input_schema } }));
}

export function buildChatBody(spec, { messages, tools = [], maxTokens = 300 }) {
  const q = spec.quirks || {};
  const body = {
    ...(q.extraBody || {}),
    model: spec.id,
    stream: true,
    messages,
    [q.maxTokensParam || 'max_tokens']: maxTokens,
  };
  if (q.streamUsage) body.stream_options = { include_usage: true };
  if (tools.length > 0) body.tools = toOpenAiTools(tools);
  if (q.reasoningEffort) body.reasoning_effort = q.reasoningEffort;
  return body;
}

const REQUEST_TIMEOUT_MS = 30_000;
const EXTRA_CONTENT_MAX_BYTES = 16_000;

// Provider-opaque per-tool-call data (Gemini's thought_signature). Only a plain JSON object of sane size is kept; anything
// else (string, array, null, oversized) is ignored so a malformed value can never break the follow-up request.
function sanitizeExtraContent(v) {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return undefined;
  try { return JSON.stringify(v).length <= EXTRA_CONTENT_MAX_BYTES ? v : undefined; } catch { return undefined; }
}

// One streaming chat-completion request. Calls onText(delta) as text arrives. Returns { text, toolCalls, usage }.
export async function streamChatCompletion({ spec, apiKey, messages, tools, onText = () => {}, fetchImpl = fetch, timeoutMs = REQUEST_TIMEOUT_MS }) {
  const res = await fetchImpl(spec.endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(buildChatBody(spec, { messages, tools })),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok || !res.body) {
    const errText = await res.text().catch(() => '');
    // Providers sometimes echo a (masked) key fragment in their error body; never let key-shaped text reach the logs.
    const safe = String(errText).slice(0, 300).replace(/\b(sk|key|gsk|xai|AIza)[-_A-Za-z0-9*]{6,}/g, '$1-***');
    const err = new Error(`[llm] ${spec.provider} chat completion failed: ${res.status} ${safe}`);
    err.status = res.status;
    throw err;
  }
  let text = '';
  const calls = new Map(); // index -> {id, name, argsText}
  let usage = null;
  let buffer = '';
  for await (const chunk of res.body) {
    buffer += Buffer.from(chunk).toString('utf8');
    let nl;
    while ((nl = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') continue;
      let evt;
      try { evt = JSON.parse(data); } catch { continue; }
      if (evt.usage) usage = evt.usage;
      const delta = evt.choices?.[0]?.delta;
      if (!delta) continue;
      if (delta.content) { text += delta.content; onText(delta.content); }
      for (const tc of delta.tool_calls || []) {
        const idx = tc.index ?? 0;
        if (!calls.has(idx)) calls.set(idx, { id: tc.id || `call_${idx}`, name: '', argsText: '' });
        const entry = calls.get(idx);
        if (tc.id) entry.id = tc.id;
        if (tc.function?.name) entry.name = tc.function.name;
        if (tc.function?.arguments) entry.argsText += tc.function.arguments;
        if (spec.quirks?.preserveExtraContent) {
          const extra = sanitizeExtraContent(tc.extra_content);
          if (extra) entry.extra_content = { ...(entry.extra_content || {}), ...extra };
        }
      }
    }
  }
  return { text, toolCalls: [...calls.values()].filter((c) => c.name), usage: { input_tokens: usage?.prompt_tokens ?? 0, output_tokens: usage?.completion_tokens ?? 0 } };
}

// One whole turn on a non-Anthropic provider, returned in Anthropic's finalMessage() shape.
//
// Unlike Anthropic, these endpoints often answer a tool-calling turn with NO spoken text (content:null), which would drop
// the flow's "always say something out loud too" requirement. If a turn produced tool calls but no text, issue ONE follow-up
// request with synthetic tool results appended and tools disabled, asking for the spoken half. That costs an extra round trip
// on those turns (a measured reason these models respond slower than Haiku), but keeps every downstream consumer unchanged.
export async function runOpenAiCompatibleTurn({ spec, apiKey, system, messages, tools, onText, fetchImpl }) {
  if (!apiKey) throw new Error(`[llm] no API key configured for provider "${spec.provider}" (${spec.keyEnv})`);
  const baseMessages = [{ role: 'system', content: system }, ...messages];
  const first = await streamChatCompletion({ spec, apiKey, messages: baseMessages, tools, onText, fetchImpl });

  const content = [];
  for (const { name, argsText, extra_content } of first.toolCalls) {
    let input = {};
    try { input = argsText ? JSON.parse(argsText) : {}; } catch { input = {}; }
    // extra_content is provider-opaque (Gemini thought_signature); Anthropic-shaped consumers ignore the extra field.
    content.push({ type: 'tool_use', name, input, ...(extra_content ? { extra_content } : {}) });
  }
  if (first.text) {
    content.unshift({ type: 'text', text: first.text });
    return { content, usage: first.usage };
  }
  if (first.toolCalls.length === 0) return { content, usage: first.usage }; // nothing to speak or record

  const followUp = [
    ...baseMessages,
    { role: 'assistant', content: null, tool_calls: first.toolCalls.map((tc) => ({
      id: tc.id, type: 'function', function: { name: tc.name, arguments: tc.argsText },
      // Gemini 3 400s without its thought_signature echoed back verbatim; parallel calls may have it on the first only.
      ...(tc.extra_content ? { extra_content: tc.extra_content } : {}),
    })) },
    ...first.toolCalls.map((tc) => ({ role: 'tool', tool_call_id: tc.id, content: 'recorded' })),
  ];
  const second = await streamChatCompletion({ spec, apiKey, messages: followUp, tools: [], onText, fetchImpl });
  if (second.text) content.unshift({ type: 'text', text: second.text });
  return { content, usage: { input_tokens: first.usage.input_tokens + second.usage.input_tokens, output_tokens: first.usage.output_tokens + second.usage.output_tokens } };
}

// Runs the selected model; if a NON-native provider throws before anything was spoken, retries the turn on Haiku so a provider
// outage or a bad key degrades to "slightly more expensive", not to a dead call. Once speech has started the error is rethrown
// (a half-spoken turn cannot be restarted cleanly).
export async function generateWithFallback({ selected, runSelected, runFallback, hasSpoken = () => false, log = console }) {
  if (selected.native) return { final: await runSelected(selected), usedModel: selected.id, fellBack: false };
  try {
    return { final: await runSelected(selected), usedModel: selected.id, fellBack: false };
  } catch (err) {
    if (hasSpoken()) throw err;
    log.error(`[llm] ${selected.id} (${selected.provider}) failed before speaking (${String(err.message).slice(0, 200)}); falling back to ${DEFAULT_ANTHROPIC_MODEL}`);
    return { final: await runFallback(DEFAULT_ANTHROPIC_MODEL), usedModel: DEFAULT_ANTHROPIC_MODEL, fellBack: true };
  }
}
