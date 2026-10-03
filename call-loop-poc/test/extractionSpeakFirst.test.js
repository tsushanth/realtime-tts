// Latency fix (2026-10-03): Claude Haiku answered a caller's field with ONLY a record_field tool_use block (no text) on
// ~95% of first extraction turns, forcing _maybeRetireTurn to run a second sequential "nudge" model call (+1-2 s on the
// first reply). The extraction node prompt and the record_field tool description now tell the model to speak FIRST.
// These tests pin (1) the hint is present for extraction nodes only, (2) the tool description carries it, and (3) the
// nudge that remains as a safety net starts immediately (no timer/debounce between the silent turn and the next call).
import { describe, it, expect, vi, afterEach } from 'vitest';
import { CallSession, EXTRACTION_SPEAK_FIRST_HINT, RECORD_FIELD_TOOL_DESCRIPTION } from '../server.js';

const P = CallSession.prototype;

function promptStub(overrides = {}) {
  return {
    flow: { globalSettings: {} }, lang: null, collectedData: {}, calendar: null, isShopper: false, expressiveDelivery: false,
    _subflowStack: [], _applyVariables: (x) => x, _buildNodeSystemPrompt: P._buildNodeSystemPrompt, ...overrides,
  };
}
const extractionNode = { id: 'collect', type: 'extraction', prompt: 'Book it.', params: {}, extract: { caller_name: 'name', phone: 'phone' }, edges: [{ target: 'bye', condition: 'done' }] };
const plainNode = { id: 'chat', type: 'conversation', prompt: 'Chat.', params: {}, edges: [{ target: 'bye', condition: 'done' }] };

describe('extraction speak-first hint (prompt assembly)', () => {
  it('is present, as the last node-prompt line, for an extraction node', () => {
    const prompt = P._buildNodeSystemPrompt.call(promptStub(), extractionNode, false);
    expect(prompt).toContain(EXTRACTION_SPEAK_FIRST_HINT);
    expect(prompt.trimEnd().endsWith(EXTRACTION_SPEAK_FIRST_HINT)).toBe(true);
    expect(EXTRACTION_SPEAK_FIRST_HINT).toMatch(/FIRST/);
    expect(EXTRACTION_SPEAK_FIRST_HINT).toMatch(/record_field/);
  });
  it('is absent for a non-extraction node', () => {
    const prompt = P._buildNodeSystemPrompt.call(promptStub(), plainNode, false);
    expect(prompt).not.toContain(EXTRACTION_SPEAK_FIRST_HINT);
    expect(prompt).not.toContain('record_field');
  });
  it('reaches the full turn system prompt for extraction nodes only', () => {
    expect(P._buildTurnSystemPrompt.call(promptStub(), extractionNode, false)).toContain(EXTRACTION_SPEAK_FIRST_HINT);
    expect(P._buildTurnSystemPrompt.call(promptStub(), plainNode, false)).not.toContain(EXTRACTION_SPEAK_FIRST_HINT);
  });
  it('the record_field tool description tells the model to write text before the call', () => {
    expect(RECORD_FIELD_TOOL_DESCRIPTION).toMatch(/BEFORE calling this tool/);
    expect(RECORD_FIELD_TOOL_DESCRIPTION).toMatch(/never call it as the first thing/);
  });
});

describe('silent-turn nudge safety net starts with no added delay', () => {
  afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

  function retireStub(collected) {
    const gen = vi.fn();
    const stub = {
      turnSeq: 4, activeTurn: 4, currentNodeId: 'collect', history: [{ role: 'user', content: 'My name is Laura Park' }],
      collectedData: collected, _nudgeAttempts: new Map(), _queuedUserText: null, _paymentAwaitingResume: false, _closed: false,
      turnState: { id: 4, llmDone: true, pendingTts: 0, startedSpeaking: false, saidNothing: true },
      flowNodesById: new Map([['collect', extractionNode]]),
      _resolveInterruptionSensitivity: () => 'on', _scheduleReminderIfConfigured: vi.fn(), _generateTurn: gen,
    };
    return { stub, gen };
  }

  it('a silent tool-only turn with fields missing starts the nudge call synchronously, without advancing any timer', () => {
    vi.useFakeTimers();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    const { stub, gen } = retireStub({ caller_name: 'Laura Park' });
    P._maybeRetireTurn.call(stub, 4);
    expect(gen).toHaveBeenCalledTimes(1); // no vi.advanceTimersByTime: nothing waits on a timer
    expect(vi.getTimerCount()).toBe(0);
    expect(stub.history.at(-1).content).toMatch(/ask for whatever is still missing/);
    expect(stub._nudgeAttempts.get('collect')).toBe(1);
    expect(gen.mock.calls[0][2]).toMatchObject({ isNodeEntry: false, forceTransition: false });
  });

  it('a turn that DID speak is never nudged', () => {
    vi.useFakeTimers();
    const { stub, gen } = retireStub({ caller_name: 'Laura Park' });
    stub.turnState.saidNothing = false;
    P._maybeRetireTurn.call(stub, 4);
    expect(gen).not.toHaveBeenCalled();
  });
});
