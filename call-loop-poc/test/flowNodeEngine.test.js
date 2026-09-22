// Tests the flow-node execution engine's deterministic, non-LLM logic:
// CallSession.prototype._evaluateStructuredCondition, _evaluateLogicSplit, and
// _executePressDigit (server.js). Follows test/applyLanguage.test.js's established pattern —
// minimal stub objects carrying only the fields the method reads/writes, then
// `CallSession.prototype.<method>.call(stub, ...args)` — rather than extracting pure functions,
// since these methods are real CallSession instance methods that read `this.collectedData`,
// `this.flowNodesById`, etc., and diverging into a reimplementation would risk drifting from the
// real shipped code.
import { describe, it, expect, vi } from 'vitest';
import { CallSession } from '../server.js';

describe('_evaluateStructuredCondition', () => {
  // Pure with respect to `this` — takes { field, operator, value } and a data object as explicit
  // args and reads nothing off `this` (verified against the real implementation) — but called via
  // .call(stub, ...) anyway to match the file's established pattern and stay honest that it's a
  // real prototype method.
  const evalCond = (cond, data) => CallSession.prototype._evaluateStructuredCondition.call({}, cond, data);

  it('numeric string vs number: coerces both sides to numbers when both parse as numeric', () => {
    // actual "5" (string) vs value 5 (number) — Number("5") = 5, Number(5) = 5, both numeric => numeric compare, not string compare.
    expect(evalCond({ field: 'age', operator: '==', value: 5 }, { age: '5' })).toBe(true);
    expect(evalCond({ field: 'age', operator: '>', value: 3 }, { age: '5' })).toBe(true);
    expect(evalCond({ field: 'age', operator: '<', value: 10 }, { age: '5' })).toBe(true);
  });

  it('numeric string comparison is numeric, not lexicographic ("9" < "10")', () => {
    // Lexicographically "10" < "9", but numerically 10 > 9 — confirms real numeric coercion.
    expect(evalCond({ field: 'n', operator: '>', value: '9' }, { n: '10' })).toBe(true);
    expect(evalCond({ field: 'n', operator: '<', value: '9' }, { n: '10' })).toBe(false);
  });

  it('non-numeric values fall back to trimmed, case-insensitive string comparison', () => {
    expect(evalCond({ field: 'name', operator: '==', value: '  Bob  ' }, { name: 'bob' })).toBe(true);
    expect(evalCond({ field: 'name', operator: '!=', value: 'alice' }, { name: 'Bob' })).toBe(true);
    expect(evalCond({ field: 'name', operator: '<', value: 'bob' }, { name: 'alice' })).toBe(true);
  });

  it('each operator: ==, !=, >, <, >=, <=', () => {
    expect(evalCond({ field: 'n', operator: '==', value: 5 }, { n: 5 })).toBe(true);
    expect(evalCond({ field: 'n', operator: '!=', value: 5 }, { n: 6 })).toBe(true);
    expect(evalCond({ field: 'n', operator: '>', value: 5 }, { n: 6 })).toBe(true);
    expect(evalCond({ field: 'n', operator: '<', value: 5 }, { n: 4 })).toBe(true);
    expect(evalCond({ field: 'n', operator: '>=', value: 5 }, { n: 5 })).toBe(true);
    expect(evalCond({ field: 'n', operator: '<=', value: 5 }, { n: 5 })).toBe(true);
    // Same values, operator that should be false
    expect(evalCond({ field: 'n', operator: '>', value: 5 }, { n: 5 })).toBe(false);
  });

  it('unknown operator logs a warning and returns false (treated as no match)', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(evalCond({ field: 'n', operator: '~=', value: 5 }, { n: 5 })).toBe(false);
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('missing field in collectedData: actual is undefined, falls back to string comparison against "" ', () => {
    // actual undefined -> actualNum = Number(undefined) = NaN -> not bothNumeric -> string path;
    // String(undefined ?? '') = ''.
    expect(evalCond({ field: 'missing', operator: '==', value: '' }, {})).toBe(true);
    expect(evalCond({ field: 'missing', operator: '==', value: 'something' }, {})).toBe(false);
    expect(evalCond({ field: 'missing', operator: '!=', value: 'something' }, {})).toBe(true);
  });

  it('missing field vs a numeric value: not bothNumeric (actual is undefined) so falls back to string compare', () => {
    // actual undefined fails the `actual !== undefined` check for bothNumeric even though
    // Number(5) parses fine — confirms the empty-field guard, not just NaN-ness of actual.
    expect(evalCond({ field: 'missing', operator: '==', value: 5 }, {})).toBe(false);
  });

  it('empty-string actual/value are explicitly excluded from numeric coercion', () => {
    // actual === '' fails the bothNumeric guard even though Number('') === 0 (not NaN).
    expect(evalCond({ field: 'n', operator: '==', value: 0 }, { n: '' })).toBe(false); // string '' vs '0' string compare, not equal
    expect(evalCond({ field: 'n', operator: '==', value: '' }, { n: '' })).toBe(true); // '' vs '' string compare
  });

  it('data itself null/undefined: actual is undefined, no throw', () => {
    expect(evalCond({ field: 'x', operator: '==', value: '' }, null)).toBe(true);
    expect(evalCond({ field: 'x', operator: '==', value: '' }, undefined)).toBe(true);
  });
});

describe('_evaluateLogicSplit', () => {
  const evalSplit = (node, collectedData) =>
    CallSession.prototype._evaluateLogicSplit.call(
      { collectedData, _evaluateStructuredCondition: CallSession.prototype._evaluateStructuredCondition },
      node
    );

  it('returns the target of the first edge whose structured condition matches', () => {
    const node = {
      edges: [
        { condition: { field: 'status', operator: '==', value: 'pending' }, target: 'nodeA' },
        { condition: { field: 'status', operator: '==', value: 'done' }, target: 'nodeB' },
      ],
    };
    expect(evalSplit(node, { status: 'done' })).toBe('nodeB');
  });

  it('a conditionless edge (no condition, or condition not an object) is the default/fallback and short-circuits earlier edges', () => {
    const node = {
      edges: [
        { condition: null, target: 'defaultNode' },
        { condition: { field: 'status', operator: '==', value: 'anything' }, target: 'neverReached' },
      ],
    };
    // Conditionless edge is first in list => wins immediately regardless of order semantics.
    expect(evalSplit(node, { status: 'anything' })).toBe('defaultNode');
  });

  it('a conditionless edge placed after conditioned edges only wins if none of the earlier ones match', () => {
    const node = {
      edges: [
        { condition: { field: 'status', operator: '==', value: 'match-me' }, target: 'matched' },
        { condition: undefined, target: 'fallback' },
      ],
    };
    expect(evalSplit(node, { status: 'match-me' })).toBe('matched');
    expect(evalSplit(node, { status: 'no-match' })).toBe('fallback');
  });

  it('a string condition (not an object) is treated the same as conditionless — also a default', () => {
    const node = { edges: [{ condition: 'legacy string condition', target: 'defaultViaString' }] };
    expect(evalSplit(node, {})).toBe('defaultViaString');
  });

  it('returns null when nothing matches and there is no default edge', () => {
    const node = {
      edges: [{ condition: { field: 'status', operator: '==', value: 'pending' }, target: 'nodeA' }],
    };
    expect(evalSplit(node, { status: 'done' })).toBe(null);
  });

  it('returns null for a node with no edges at all', () => {
    expect(evalSplit({ edges: [] }, {})).toBe(null);
    expect(evalSplit({}, {})).toBe(null);
  });

  it('missing field in collectedData is evaluated (not skipped) via _evaluateStructuredCondition\'s own missing-field handling', () => {
    const node = {
      edges: [{ condition: { field: 'never_collected', operator: '==', value: '' }, target: 'matchesEmpty' }],
    };
    expect(evalSplit(node, {})).toBe('matchesEmpty');
  });
});

describe('_executePressDigit', () => {
  // Minimal stub carrying only what _executePressDigit reads/writes (verified against the real
  // implementation): clientWs.callSid, flow.globalSettings.variables (via _interpolateFields),
  // collectedData (via _interpolateFields), _pendingPressDigitTarget, _redirectForDetour,
  // _applyTransition, close.
  function makeStub({ callSid = 'CA123', collectedData = {}, variables } = {}) {
    return {
      clientWs: { callSid },
      flow: { globalSettings: { variables } },
      collectedData,
      _pendingPressDigitTarget: null,
      _applyVariables: CallSession.prototype._applyVariables,
      _interpolateFields: CallSession.prototype._interpolateFields,
      _redirectForDetour: vi.fn(async () => true),
      _applyTransition: vi.fn(),
      close: vi.fn(),
    };
  }

  it('no callSid (not a real Twilio call): skips entirely, no transition, no redirect', async () => {
    const stub = makeStub({ callSid: null });
    const node = { id: 'pd1', params: { digits: '1' }, edges: [{ target: 'next' }] };
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await CallSession.prototype._executePressDigit.call(stub, node);
    expect(stub._redirectForDetour).not.toHaveBeenCalled();
    expect(stub._applyTransition).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('no edge to advance to: skips, does nothing (no redirect, no transition)', async () => {
    const stub = makeStub();
    const node = { id: 'pd1', params: { digits: '1' }, edges: [] };
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await CallSession.prototype._executePressDigit.call(stub, node);
    expect(stub._redirectForDetour).not.toHaveBeenCalled();
    expect(stub._applyTransition).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it('only the FIRST edge is used as the advance target — extra edges are ignored, not an enforced single-edge error', async () => {
    const stub = makeStub();
    const node = {
      id: 'pd1',
      params: { digits: '' }, // empty digits -> takes the "skip tone detour" path, easiest to assert target on
      edges: [{ target: 'firstEdge' }, { target: 'secondEdge' }],
    };
    await CallSession.prototype._executePressDigit.call(stub, node);
    expect(stub._applyTransition).toHaveBeenCalledWith({ next_node_id: 'firstEdge' });
  });

  it('sanitizes digits: only 0-9 * # A-D a-d w W survive; other characters (including XML-injection attempts) are stripped', async () => {
    const stub = makeStub();
    const node = {
      id: 'pd1',
      params: { digits: '1<Hangup/>2*#wAd9' },
      edges: [{ target: 'next' }],
    };
    await CallSession.prototype._executePressDigit.call(stub, node);
    expect(stub._redirectForDetour).toHaveBeenCalledTimes(1);
    const twiml = stub._redirectForDetour.mock.calls[0][0];
    // '<' and '/' and '>' are stripped, but the letters inside "Hangup" (a, n, g, u, p) are plain
    // alphanumerics — only 'a' survives since it's a valid DTMF letter (A-D case-insensitive);
    // the rest (n, g, u, p) are stripped too since they're not in the DTMF character set.
    expect(twiml).toContain('digits="1a2*#wAd9"');
    expect(twiml).not.toContain('<Hangup');
    expect(twiml).not.toContain('/>2');
  });

  it('interpolates {{field}} from collectedData into the digits template before sanitizing', async () => {
    const stub = makeStub({ collectedData: { account_number: '5551234' } });
    const node = {
      id: 'pd1',
      params: { digits: '2{{account_number}}#' },
      edges: [{ target: 'next' }],
    };
    await CallSession.prototype._executePressDigit.call(stub, node);
    const twiml = stub._redirectForDetour.mock.calls[0][0];
    expect(twiml).toContain('digits="25551234#"');
  });

  it('globalSettings.variables take precedence over collectedData for the same field name', async () => {
    const stub = makeStub({
      collectedData: { pin: '0000' },
      variables: { pin: '9999' },
    });
    const node = { id: 'pd1', params: { digits: '{{pin}}' }, edges: [{ target: 'next' }] };
    await CallSession.prototype._executePressDigit.call(stub, node);
    const twiml = stub._redirectForDetour.mock.calls[0][0];
    expect(twiml).toContain('digits="9999"');
  });

  it('empty/whitespace digits param resolving to no valid characters: skips the tone detour and transitions straight to the target', async () => {
    const stub = makeStub();
    const node = { id: 'pd1', params: { digits: '!!!' }, edges: [{ target: 'next' }] };
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await CallSession.prototype._executePressDigit.call(stub, node);
    expect(stub._redirectForDetour).not.toHaveBeenCalled();
    expect(stub._applyTransition).toHaveBeenCalledWith({ next_node_id: 'next' });
    warnSpy.mockRestore();
  });

  it('missing params.digits entirely behaves the same as empty digits: skips detour, transitions directly', async () => {
    const stub = makeStub();
    const node = { id: 'pd1', params: {}, edges: [{ target: 'next' }] };
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await CallSession.prototype._executePressDigit.call(stub, node);
    expect(stub._redirectForDetour).not.toHaveBeenCalled();
    expect(stub._applyTransition).toHaveBeenCalledWith({ next_node_id: 'next' });
    warnSpy.mockRestore();
  });

  it('sets _pendingPressDigitTarget to the edge target before redirecting', async () => {
    const stub = makeStub();
    const node = { id: 'pd1', params: { digits: '1' }, edges: [{ target: 'awaitingTarget' }] };
    stub._redirectForDetour = vi.fn(async () => {
      // Assert mid-flight, before the redirect "completes", that the pending target was already set.
      expect(stub._pendingPressDigitTarget).toBe('awaitingTarget');
      return true;
    });
    await CallSession.prototype._executePressDigit.call(stub, node);
  });

  it('redirect failure: clears _pendingPressDigitTarget and closes the call instead of leaving it stuck', async () => {
    const stub = makeStub();
    stub._redirectForDetour = vi.fn(async () => false);
    const node = { id: 'pd1', params: { digits: '1' }, edges: [{ target: 'next' }] };
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    await CallSession.prototype._executePressDigit.call(stub, node);
    expect(stub._pendingPressDigitTarget).toBe(null);
    expect(stub.close).toHaveBeenCalledTimes(1);
    errSpy.mockRestore();
  });

  it('redirect success: does not close the call, leaves _pendingPressDigitTarget set for the detour to consume later', async () => {
    const stub = makeStub();
    const node = { id: 'pd1', params: { digits: '1' }, edges: [{ target: 'next' }] };
    await CallSession.prototype._executePressDigit.call(stub, node);
    expect(stub.close).not.toHaveBeenCalled();
    expect(stub._pendingPressDigitTarget).toBe('next');
  });
});
