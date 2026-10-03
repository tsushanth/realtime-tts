import { describe, it, expect, vi } from 'vitest';
import {
  DEFAULT_ANTHROPIC_MODEL, normalizeSpec, buildRegistry, isUsable, resolveDefaultModel, describeModels, auxAnthropicModel, ratesFor,
  toOpenAiTools, buildChatBody, streamChatCompletion, runOpenAiCompatibleTurn, generateWithFallback,
} from '../llmProviders.js';
import { CallCostTracker } from '../costTracker.js';

const quiet = { warn: vi.fn(), error: vi.fn(), log: vi.fn() };
const sse = (events) => events.map((e) => `data: ${typeof e === 'string' ? e : JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n';
// A fetch stub returning scripted SSE bodies in order; records every request body.
function fakeFetch(...bodies) {
  const calls = [];
  const impl = vi.fn(async (url, init) => {
    calls.push({ url, headers: init.headers, body: JSON.parse(init.body) });
    const b = bodies[Math.min(calls.length - 1, bodies.length - 1)];
    if (b instanceof Error) throw b;
    if (typeof b === 'object' && b.status) return { ok: false, status: b.status, text: async () => b.text || '' };
    return { ok: true, status: 200, body: (async function* () { for (const part of b) yield Buffer.from(part); })(), text: async () => '' };
  });
  impl.calls = calls;
  return impl;
}
const delta = (d, extra = {}) => ({ choices: [{ delta: d }], ...extra });
const luna = () => buildRegistry({}, quiet).get('gpt-6-luna');
const TOOLS = [{ name: 'record_field', description: 'Save a field', input_schema: { type: 'object', properties: { field: { type: 'string' }, value: { type: 'string' } }, required: ['field', 'value'] } }];

describe('registry', () => {
  it('has Haiku and Sonnet as native, Luna and Gemini as OpenAI-compatible', () => {
    const reg = buildRegistry({}, quiet);
    expect(reg.get('claude-haiku-4-5-20251001').native).toBe(true);
    expect(reg.get('claude-sonnet-4-6').native).toBe(true);
    expect(reg.get('gpt-6-luna')).toMatchObject({ native: false, provider: 'openai', keyEnv: 'OPENAI_API_KEY' });
    expect(reg.has('gemini-2.5-flash-lite')).toBe(false); // dead for new users (404)
    expect(reg.get('gemini-3.1-flash-lite')).toMatchObject({ provider: 'gemini', status: 'untested', keyEnv: 'GEMINI_API_KEY', price: { in: 0.25, out: 1.5 }, quirks: { preserveExtraContent: true } });
    expect(reg.get('gemini-3.5-flash-lite')).toMatchObject({ provider: 'gemini', status: 'untested', keyEnv: 'GEMINI_API_KEY', price: { in: 0.3, out: 2.5 }, quirks: { preserveExtraContent: true } });
  });

  it('a model is usable only when its key is set (Anthropic needs its client)', () => {
    const reg = buildRegistry({}, quiet);
    expect(isUsable(reg.get('gpt-6-luna'), {})).toBe(false);
    expect(isUsable(reg.get('gpt-6-luna'), { OPENAI_API_KEY: '  ' })).toBe(false);
    expect(isUsable(reg.get('gpt-6-luna'), { OPENAI_API_KEY: 'k' })).toBe(true);
    expect(isUsable(reg.get(DEFAULT_ANTHROPIC_MODEL), {}, true)).toBe(true);
    expect(isUsable(reg.get(DEFAULT_ANTHROPIC_MODEL), {}, false)).toBe(false);
    expect(isUsable(undefined, {})).toBe(false);
  });

  it('LLM_EXTRA_MODELS adds a model with no code change, and may replace a built-in', () => {
    const extra = JSON.stringify([
      { id: 'grok-fast', provider: 'xai', endpoint: 'https://api.x.ai/v1/chat/completions', keyEnv: 'XAI_API_KEY', price: { in: 0.2, out: 0.5 }, quirks: { streamUsage: true } },
      { id: 'gpt-6-luna', provider: 'openai', endpoint: 'https://api.openai.com/v1/chat/completions', keyEnv: 'OPENAI_API_KEY', price: { in: 0.2, out: 0.75 } },
    ]);
    const reg = buildRegistry({ LLM_EXTRA_MODELS: extra }, quiet);
    expect(reg.get('grok-fast')).toMatchObject({ provider: 'xai', price: { in: 0.2, out: 0.5 } });
    expect(reg.get('gpt-6-luna').price).toEqual({ in: 0.2, out: 0.75 });
  });

  it('skips bad extras with a warning and never throws', () => {
    const warn = vi.fn();
    const bad = JSON.stringify([
      { id: 'a', provider: 'x', endpoint: 'http://evil.example.com/v1', keyEnv: 'K', price: { in: 1, out: 1 } },        // plain http to a remote host
      { id: 'b', provider: 'x', endpoint: 'https://ok.example.com', keyEnv: 'lowercase', price: { in: 1, out: 1 } },    // bad env name
      { id: 'c', provider: 'x', endpoint: 'https://ok.example.com', keyEnv: 'K', price: { in: 'free', out: 1 } },       // bad price
      { id: 'd', provider: 'x', endpoint: 'https://ok.example.com', keyEnv: 'K', price: { in: 1, out: 1 }, quirks: { extraBody: { stream: false } } }, // reserved key
      { id: 'e', provider: 'x', endpoint: 'https://ok.example.com', keyEnv: 'K', price: { in: 1, out: 1 }, quirks: { maxTokensParam: 'limit' } },
      { id: 'good', provider: 'x', endpoint: 'http://localhost:8000/v1/chat/completions', keyEnv: 'LOCAL_KEY', price: { in: 0, out: 0 } },
    ]);
    const reg = buildRegistry({ LLM_EXTRA_MODELS: bad }, { warn });
    expect([...reg.keys()].filter((k) => ['a', 'b', 'c', 'd', 'e'].includes(k))).toEqual([]);
    expect(reg.has('good')).toBe(true);
    expect(warn).toHaveBeenCalledTimes(5);
    expect(() => buildRegistry({ LLM_EXTRA_MODELS: '{not json' }, { warn: vi.fn() })).not.toThrow();
    expect(() => buildRegistry({ LLM_EXTRA_MODELS: '{"id":"x"}' }, { warn: vi.fn() })).not.toThrow();
  });

  it('the default model falls back to Haiku when the requested one is unknown or has no key', () => {
    const reg = buildRegistry({}, quiet);
    const warn = vi.fn();
    expect(resolveDefaultModel(reg, undefined, {}, true, { warn })).toBe(DEFAULT_ANTHROPIC_MODEL);
    expect(resolveDefaultModel(reg, 'gpt-6-luna', { OPENAI_API_KEY: 'k' }, true, { warn })).toBe('gpt-6-luna');
    expect(resolveDefaultModel(reg, 'gpt-6-luna', {}, true, { warn })).toBe(DEFAULT_ANTHROPIC_MODEL);
    expect(resolveDefaultModel(reg, 'no-such-model', {}, true, { warn })).toBe(DEFAULT_ANTHROPIC_MODEL);
    expect(warn).toHaveBeenCalledTimes(2);
  });

  it('side calls that force an Anthropic tool always run on an Anthropic model', () => {
    const reg = buildRegistry({}, quiet);
    expect(auxAnthropicModel(reg, 'claude-sonnet-4-6')).toBe('claude-sonnet-4-6');
    expect(auxAnthropicModel(reg, 'gpt-6-luna')).toBe(DEFAULT_ANTHROPIC_MODEL);
    expect(auxAnthropicModel(reg, 'unknown')).toBe(DEFAULT_ANTHROPIC_MODEL);
  });

  it('describes availability without leaking keys', () => {
    const d = describeModels(buildRegistry({}, quiet), { OPENAI_API_KEY: 'sk-secret' }, true);
    expect(d).toContain('gpt-6-luna (openai)');
    expect(d).toContain('gemini-3.1-flash-lite (gemini, untested, unavailable: no key)');
    expect(d).not.toContain('sk-secret');
  });
});

describe('cost rates', () => {
  it('prices registry models per token, and the cost tracker uses them', () => {
    const reg = buildRegistry({}, quiet);
    expect(ratesFor(reg, 'gpt-6-luna')).toEqual({ input: 0.1 / 1e6, output: 0.5 / 1e6 });
    expect(ratesFor(reg, 'nope')).toBeNull();
    const t = new CallCostTracker({ ttsBackend: 'kokoro' });
    t.addLlmUsage('gpt-6-luna', 10_000, 600);
    t.addLlmUsage('claude-haiku-4-5-20251001', 10_000, 600);
    expect(t.breakdown().claude).toBeCloseTo(10_000 * 0.1e-6 + 600 * 0.5e-6 + 10_000 * 1e-6 + 600 * 5e-6, 8);
  });
});

describe('request building', () => {
  it('translates Anthropic tools to OpenAI functions', () => {
    expect(toOpenAiTools(TOOLS)).toEqual([{ type: 'function', function: { name: 'record_field', description: 'Save a field', parameters: TOOLS[0].input_schema } }]);
  });

  it('applies Luna quirks: max_completion_tokens, reasoning_effort none, usage in the stream', () => {
    const body = buildChatBody(luna(), { messages: [{ role: 'user', content: 'hi' }], tools: TOOLS });
    expect(body).toMatchObject({ model: 'gpt-6-luna', stream: true, max_completion_tokens: 300, reasoning_effort: 'none', stream_options: { include_usage: true } });
    expect(body.max_tokens).toBeUndefined();
    expect(body.tools).toHaveLength(1);
  });

  it('uses max_tokens and no extras for a plain model, omits tools when there are none, and cannot be overridden by extraBody', () => {
    const spec = normalizeSpec({ id: 'plain', provider: 'p', endpoint: 'https://x.example.com', keyEnv: 'K', price: { in: 1, out: 1 }, quirks: { extraBody: { temperature: 0.2 } } });
    const body = buildChatBody(spec, { messages: [], tools: [] });
    expect(body).toMatchObject({ model: 'plain', max_tokens: 300, temperature: 0.2 });
    expect(body.tools).toBeUndefined();
    expect(body.reasoning_effort).toBeUndefined();
    expect(body.stream_options).toBeUndefined();
  });
});

describe('streaming parser', () => {
  it('collects streamed text, forwarding each delta, and the usage chunk', async () => {
    const f = fakeFetch([sse([delta({ content: 'Hello' }), delta({ content: ' there' }), { choices: [], usage: { prompt_tokens: 120, completion_tokens: 8 } }])]);
    const seen = [];
    const r = await streamChatCompletion({ spec: luna(), apiKey: 'k', messages: [], tools: [], onText: (d) => seen.push(d), fetchImpl: f });
    expect(r.text).toBe('Hello there');
    expect(seen).toEqual(['Hello', ' there']);
    expect(r.usage).toEqual({ input_tokens: 120, output_tokens: 8 });
    expect(f.calls[0].headers.Authorization).toBe('Bearer k');
  });

  it('reassembles tool-call arguments that arrive split across chunks and across SSE lines', async () => {
    const whole = sse([
      delta({ tool_calls: [{ index: 0, id: 'call_1', function: { name: 'record_field', arguments: '{"field":"na' } }] }),
      delta({ tool_calls: [{ index: 0, function: { arguments: 'me","value":"Mich' } }] }),
      delta({ tool_calls: [{ index: 0, function: { arguments: 'ael"}' } }] }),
    ]);
    // split the raw bytes mid-line to prove the line buffer works
    const f = fakeFetch([whole.slice(0, 57), whole.slice(57, 130), whole.slice(130)]);
    const r = await streamChatCompletion({ spec: luna(), apiKey: 'k', messages: [], tools: TOOLS, fetchImpl: f });
    expect(r.toolCalls).toEqual([{ id: 'call_1', name: 'record_field', argsText: '{"field":"name","value":"Michael"}' }]);
  });

  it('scrubs key-shaped text out of provider error messages', async () => {
    const f = fakeFetch({ status: 401, text: 'Incorrect API key provided: sk-proj-abcdEFGH1234********wxyz and AIzaSyDfakefakefake' });
    const err = await streamChatCompletion({ spec: luna(), apiKey: 'k', messages: [], tools: [], fetchImpl: f }).catch((e) => e);
    expect(err.status).toBe(401);
    expect(err.message).not.toMatch(/abcdEFGH|Dfakefake/);
    expect(err.message).toContain('sk-***');
  });

  it('throws with the status on an HTTP error', async () => {
    const f = fakeFetch({ status: 429, text: 'rate limited' });
    await expect(streamChatCompletion({ spec: luna(), apiKey: 'k', messages: [], tools: [], fetchImpl: f })).rejects.toMatchObject({ status: 429, message: expect.stringContaining('rate limited') });
  });
});

describe('runOpenAiCompatibleTurn', () => {
  it('text plus a tool call in one answer: one request, content in Anthropic shape (text first)', async () => {
    const f = fakeFetch([sse([
      delta({ content: 'Got it, Michael.' }),
      delta({ tool_calls: [{ index: 0, id: 'c1', function: { name: 'record_field', arguments: '{"field":"name","value":"Michael"}' } }] }),
      { choices: [], usage: { prompt_tokens: 100, completion_tokens: 20 } },
    ])]);
    const r = await runOpenAiCompatibleTurn({ spec: luna(), apiKey: 'k', system: 'sys', messages: [{ role: 'user', content: 'x' }], tools: TOOLS, onText: () => {}, fetchImpl: f });
    expect(f.calls).toHaveLength(1);
    expect(f.calls[0].body.messages[0]).toEqual({ role: 'system', content: 'sys' });
    expect(r.content).toEqual([{ type: 'text', text: 'Got it, Michael.' }, { type: 'tool_use', name: 'record_field', input: { field: 'name', value: 'Michael' } }]);
    expect(r.usage).toEqual({ input_tokens: 100, output_tokens: 20 });
  });

  it('a tool call with NO speech triggers one follow-up request with synthetic tool results, tools off, and sums usage', async () => {
    const f = fakeFetch(
      [sse([delta({ tool_calls: [{ index: 0, id: 'c1', function: { name: 'record_field', arguments: '{"field":"name","value":"Michael"}' } }] }), { choices: [], usage: { prompt_tokens: 100, completion_tokens: 10 } }])],
      [sse([delta({ content: 'Thanks, Michael. What time works?' }), { choices: [], usage: { prompt_tokens: 130, completion_tokens: 9 } }])],
    );
    const spoken = [];
    const r = await runOpenAiCompatibleTurn({ spec: luna(), apiKey: 'k', system: 's', messages: [{ role: 'user', content: 'x' }], tools: TOOLS, onText: (d) => spoken.push(d), fetchImpl: f });
    expect(f.calls).toHaveLength(2);
    const second = f.calls[1].body;
    expect(second.tools).toBeUndefined();
    expect(second.messages.at(-2)).toMatchObject({ role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'record_field' } }] });
    expect(second.messages.at(-1)).toEqual({ role: 'tool', tool_call_id: 'c1', content: 'recorded' });
    expect(spoken.join('')).toBe('Thanks, Michael. What time works?');
    expect(r.content[0]).toEqual({ type: 'text', text: 'Thanks, Michael. What time works?' });
    expect(r.content[1]).toMatchObject({ type: 'tool_use', name: 'record_field' });
    expect(r.usage).toEqual({ input_tokens: 230, output_tokens: 19 });
  });

  it('an empty answer (no text, no tools) returns empty content without a second request', async () => {
    const f = fakeFetch([sse([{ choices: [], usage: { prompt_tokens: 5, completion_tokens: 0 } }])]);
    const r = await runOpenAiCompatibleTurn({ spec: luna(), apiKey: 'k', system: 's', messages: [], tools: TOOLS, onText: () => {}, fetchImpl: f });
    expect(f.calls).toHaveLength(1);
    expect(r.content).toEqual([]);
  });

  it('malformed tool arguments become an empty input instead of throwing', async () => {
    const f = fakeFetch([sse([delta({ content: 'ok' }), delta({ tool_calls: [{ index: 0, id: 'c1', function: { name: 'record_field', arguments: '{oops' } }] })])]);
    const r = await runOpenAiCompatibleTurn({ spec: luna(), apiKey: 'k', system: 's', messages: [], tools: TOOLS, onText: () => {}, fetchImpl: f });
    expect(r.content[1]).toEqual({ type: 'tool_use', name: 'record_field', input: {} });
  });

  it('refuses without a key, naming the env var but not any value', async () => {
    await expect(runOpenAiCompatibleTurn({ spec: luna(), apiKey: undefined, system: 's', messages: [], tools: [], onText: () => {} })).rejects.toThrow(/OPENAI_API_KEY/);
  });
});

describe('generateWithFallback', () => {
  const reg = buildRegistry({}, quiet);
  const haiku = reg.get(DEFAULT_ANTHROPIC_MODEL);
  it('a native model is run as-is and its errors are NOT retried elsewhere', async () => {
    const runFallback = vi.fn();
    await expect(generateWithFallback({ selected: haiku, runSelected: async () => { throw new Error('anthropic down'); }, runFallback, log: quiet })).rejects.toThrow('anthropic down');
    expect(runFallback).not.toHaveBeenCalled();
  });
  it('a provider that fails before speaking is retried on Haiku, and the result says so', async () => {
    const out = await generateWithFallback({ selected: luna(), runSelected: async () => { throw new Error('429'); }, runFallback: async (m) => ({ content: [], model: m }), hasSpoken: () => false, log: quiet });
    expect(out).toMatchObject({ usedModel: DEFAULT_ANTHROPIC_MODEL, fellBack: true, final: { model: DEFAULT_ANTHROPIC_MODEL } });
    expect(quiet.error).toHaveBeenCalled();
  });
  it('a provider that fails AFTER speaking has started rethrows (a half-spoken turn cannot be restarted)', async () => {
    const runFallback = vi.fn();
    await expect(generateWithFallback({ selected: luna(), runSelected: async () => { throw new Error('stream cut'); }, runFallback, hasSpoken: () => true, log: quiet })).rejects.toThrow('stream cut');
    expect(runFallback).not.toHaveBeenCalled();
  });
  it('a healthy non-native provider is used without fallback', async () => {
    const out = await generateWithFallback({ selected: luna(), runSelected: async () => ({ content: [{ type: 'text', text: 'hi' }] }), runFallback: vi.fn(), log: quiet });
    expect(out).toMatchObject({ usedModel: 'gpt-6-luna', fellBack: false });
  });
});

describe('extra_content / thought_signature (Gemini 3)', () => {
  const gem = () => buildRegistry({}, quiet).get('gemini-3.5-flash-lite');
  const SIG = { google: { thought_signature: 'SIG-abc/123==' } };
  const toolDelta = (extra) => delta({ tool_calls: [{ index: 0, id: 'c1', ...(extra === undefined ? {} : { extra_content: extra }), function: { name: 'record_field', arguments: '{"field":"name","value":"M"}' } }] });
  const run = (spec, f) => runOpenAiCompatibleTurn({ spec, apiKey: 'k', system: 's', messages: [{ role: 'user', content: 'x' }], tools: TOOLS, onText: () => {}, fetchImpl: f });

  it('captures the signature from a streamed tool call and replays it verbatim on the synthetic second request', async () => {
    const f = fakeFetch([sse([toolDelta(SIG)])], [sse([delta({ content: 'Thanks.' })])]);
    const r = await run(gem(), f);
    expect(f.calls).toHaveLength(2);
    expect(f.calls[1].body.messages.at(-2).tool_calls[0]).toEqual({ id: 'c1', type: 'function', function: { name: 'record_field', arguments: '{"field":"name","value":"M"}' }, extra_content: SIG });
    expect(r.content.find((b) => b.type === 'tool_use').extra_content).toEqual(SIG);
  });

  it('parallel calls: only the call that carried a signature gets one back', async () => {
    const f = fakeFetch([sse([
      delta({ tool_calls: [{ index: 0, id: 'a', extra_content: SIG, function: { name: 'record_field', arguments: '{}' } }] }),
      delta({ tool_calls: [{ index: 1, id: 'b', function: { name: 'record_field', arguments: '{}' } }] }),
    ])], [sse([delta({ content: 'ok' })])]);
    await run(gem(), f);
    const tcs = f.calls[1].body.messages.at(-3).tool_calls;
    expect(tcs[0].extra_content).toEqual(SIG);
    expect('extra_content' in tcs[1]).toBe(false);
  });

  it('Gemini parallel calls that all stream at index 0 become separate calls (not glued arguments)', async () => {
    const f = fakeFetch([sse([
      delta({ tool_calls: [{ index: 0, id: 'a', extra_content: SIG, function: { name: 'record_field', arguments: '{"field":"name","value":"M"}' } }] }),
      delta({ tool_calls: [{ index: 0, id: 'b', function: { name: 'record_field', arguments: '{"field":"phone","value":"415"}' } }] }),
      delta({ tool_calls: [{ index: 0, function: { name: 'record_field', arguments: '{"field":"time","value":"9"}' } }] }),
    ])], [sse([delta({ content: 'ok' })])]);
    const r = await run(gem(), f);
    const uses = r.content.filter((b) => b.type === 'tool_use');
    expect(uses.map((u) => u.input.field)).toEqual(['name', 'phone', 'time']);
    const tcs = f.calls[1].body.messages.at(-4).tool_calls;
    expect(tcs).toHaveLength(3);
    expect(new Set(tcs.map((t) => t.id)).size).toBe(3);
    expect(tcs[0].extra_content).toEqual(SIG);
    expect('extra_content' in tcs[1]).toBe(false);
    expect(f.calls[1].body.messages.filter((m) => m.role === 'tool')).toHaveLength(3);
  });

  it('without the quirk (other providers) same-index deltas still merge into one call, as before', async () => {
    const f = fakeFetch([sse([
      delta({ tool_calls: [{ index: 0, id: 'a', function: { name: 'record_field', arguments: '{"field":"name",' } }] }),
      delta({ tool_calls: [{ index: 0, function: { name: 'record_field', arguments: '"value":"M"}' } }] }),
    ])], [sse([delta({ content: 'ok' })])]);
    const r = await run(luna(), f);
    expect(r.content.filter((b) => b.type === 'tool_use')).toEqual([{ type: 'tool_use', name: 'record_field', input: { field: 'name', value: 'M' } }]);
  });

  it('extra_content split across deltas is merged', async () => {
    const f = fakeFetch([sse([
      delta({ tool_calls: [{ index: 0, id: 'a', function: { name: 'record_field', arguments: '{"field"' } }] }),
      delta({ tool_calls: [{ index: 0, extra_content: SIG, function: { arguments: ':"n","value":"v"}' } }] }),
    ])], [sse([delta({ content: 'ok' })])]);
    const r = await run(gem(), f);
    expect(f.calls[1].body.messages.at(-2).tool_calls[0].extra_content).toEqual(SIG);
    expect(r.content.find((b) => b.type === 'tool_use').input).toEqual({ field: 'n', value: 'v' });
  });

  it('non-Gemini providers never carry extra_content, even if the stream sends it', async () => {
    const f = fakeFetch([sse([toolDelta(SIG)])], [sse([delta({ content: 'Thanks.' })])]);
    const r = await run(luna(), f);
    expect(JSON.stringify(f.calls[1].body)).not.toContain('extra_content');
    expect(r.content.find((b) => b.type === 'tool_use')).toEqual({ type: 'tool_use', name: 'record_field', input: { field: 'name', value: 'M' } });
  });

  it('malformed extra_content is ignored', async () => {
    for (const bad of ['str', 7, ['x'], null, { big: 'x'.repeat(20000) }]) {
      const f = fakeFetch([sse([toolDelta(bad)])], [sse([delta({ content: 'ok' })])]);
      const r = await run(gem(), f);
      expect(JSON.stringify(f.calls[1].body)).not.toContain('extra_content');
      expect('extra_content' in r.content.find((b) => b.type === 'tool_use')).toBe(false);
    }
  });

  it('the quirk is normalized from LLM_EXTRA_MODELS and defaults off', () => {
    const base = { id: 'x', provider: 'p', endpoint: 'https://e.example/v1', keyEnv: 'K', price: { in: 1, out: 1 } };
    expect(normalizeSpec({ ...base, quirks: { preserveExtraContent: true } }).quirks.preserveExtraContent).toBe(true);
    expect(normalizeSpec(base).quirks.preserveExtraContent).toBeUndefined();
  });
});
