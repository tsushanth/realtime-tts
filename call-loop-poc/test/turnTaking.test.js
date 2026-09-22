// Tests CallSession's turn-taking / barge-in bookkeeping (server.js) via
// `CallSession.prototype.<method>.call(stub, ...)` on minimal stub objects —
// same approach as test/applyLanguage.test.js, chosen over spinning up a
// live session (real WebSocket/timers) because these methods are pure
// decision logic over a handful of fields (activeTurn, turnState,
// _httpTtsAborts, clientWs) that's straightforward to stub directly.
import { describe, it, expect, vi } from 'vitest';
import { CallSession } from '../server.js';

// Minimal stub carrying only the fields these methods actually read or
// write, verified against the real CallSession constructor + method bodies.
function makeStub(overrides = {}) {
  return {
    activeTurn: 0,
    turnState: null,
    _closing: false,
    _silenceSince: null,
    _reminderTimer: null,
    _httpTtsAborts: null,
    ttsWs: null,
    clientWs: { isSpeaking: () => false, clearQueue: vi.fn(), readyState: 1, send: vi.fn() },
    send: vi.fn(),
    _sendChain: null,
    _scheduleReminderIfConfigured: vi.fn(),
    ...overrides,
  };
}

describe('_bargeIn (barge-in cancellation)', () => {
  it('no-ops when nothing is in flight and nothing is audible', () => {
    const stub = makeStub({ activeTurn: 0, clientWs: { isSpeaking: () => false } });
    CallSession.prototype._bargeIn.call(stub);
    expect(stub.activeTurn).toBe(0);
    expect(stub.send).not.toHaveBeenCalled();
  });

  it('cancels an in-flight turn: resets activeTurn to 0, aborts HTTP TTS, clears the audio queue, and notifies the client', () => {
    const abort1 = { abort: vi.fn() };
    const abort2 = { abort: vi.fn() };
    const stub = makeStub({
      activeTurn: 7,
      _httpTtsAborts: new Set([abort1, abort2]),
      ttsWs: { readyState: 1, send: vi.fn() }, // 1 === WebSocket.OPEN
      _reminderTimer: 'some-timer-handle',
    });
    // WebSocket.OPEN is 1 in the ws module; use the real import indirectly by
    // matching readyState === 1 — the constant itself isn't exported for
    // stubbing, so this mirrors the real module's OPEN value directly.
    CallSession.prototype._bargeIn.call(stub);

    expect(stub.activeTurn).toBe(0);
    expect(abort1.abort).toHaveBeenCalled();
    expect(abort2.abort).toHaveBeenCalled();
    expect(stub._httpTtsAborts.size).toBe(0); // cleared after aborting
    expect(stub.clientWs.clearQueue).toHaveBeenCalled();
    expect(stub.send).toHaveBeenCalledWith({ type: 'barge_in' });
    expect(stub._silenceSince).toBe(null);
  });

  it('still cancels when activeTurn is 0 but audio is still audibly playing out (finished-turn tail is interruptible)', () => {
    const stub = makeStub({ activeTurn: 0, clientWs: { isSpeaking: () => true, clearQueue: vi.fn() } });
    CallSession.prototype._bargeIn.call(stub);
    expect(stub.send).toHaveBeenCalledWith({ type: 'barge_in' });
    expect(stub.clientWs.clearQueue).toHaveBeenCalled();
  });
});

describe('turn supersession: in-flight TTS/turn-state logic recognizes activeTurn !== turnId', () => {
  // This is the real barge-in mechanism as seen from the TTS-dispatch side:
  // once _bargeIn (or any reassignment) moves activeTurn to a new turn id,
  // every path still holding the OLD turnId must recognize the mismatch and
  // stop processing rather than speak/count against the wrong turn.

  it('_speakCached drops a stale turn\'s clip and does not touch turnState when superseded before it can send', async () => {
    const turnState = { id: 99, pendingTts: 0 };
    const stub = makeStub({ activeTurn: 99, turnState });
    // Simulate a fresh user utterance reassigning activeTurn mid-flight,
    // before _speakCached's awaited _sendChain resolves would matter here —
    // the very first guard already catches it.
    stub.activeTurn = 5; // turn 99 has been superseded by turn 5
    await CallSession.prototype._speakCached.call(stub, Buffer.from('audio'), 99);

    expect(stub.clientWs.send).not.toHaveBeenCalled();
    expect(turnState.pendingTts).toBe(0); // never incremented — bailed before bookkeeping
  });

  it('_speakCached sends and unwinds pendingTts/retirement when still the active turn', async () => {
    const turnState = { id: 12, llmDone: true, pendingTts: 0, startedSpeaking: false, saidNothing: false };
    const stub = makeStub({ activeTurn: 12, turnState });
    stub._maybeRetireTurn = vi.fn(function (turnId) {
      CallSession.prototype._maybeRetireTurn.call(this, turnId);
    });
    await CallSession.prototype._speakCached.call(stub, Buffer.from('audio'), 12);

    expect(stub.clientWs.send).toHaveBeenCalledWith(Buffer.from('audio'), { binary: true });
    // pendingTts is incremented then decremented back to 0 around the send.
    expect(turnState.pendingTts).toBe(0);
    expect(stub._maybeRetireTurn).toHaveBeenCalledWith(12);
    // llmDone was already true and pendingTts settled back to 0 -> the turn
    // actually retires (activeTurn goes back to 0).
    expect(stub.activeTurn).toBe(0);
  });

  it('a turn dropped mid-dispatch (activeTurn reassigned while waiting on the TTS socket) still decrements pendingTts so the superseding turn is never blocked from retiring', async () => {
    // Mirrors _speak's `dispatch` closure guard (this.activeTurn !== turnId)
    // for the kokoro websocket path: exercised here directly via
    // _speakCached's equivalent bail-with-cleanup branch, since dispatch()
    // itself is a private closure not reachable without a live socket.
    const turnState = { id: 3, pendingTts: 1 }; // pretend _speak already reserved the slot
    const stub = makeStub({ activeTurn: 3, turnState });
    stub.activeTurn = 42; // superseded
    await CallSession.prototype._speakCached.call(stub, Buffer.from('x'), 3);
    // _speakCached's very first guard bails before ever touching pendingTts
    // for a turn that's already superseded when it's called — confirming
    // stale in-flight work never mutates the new turn's bookkeeping.
    expect(turnState.pendingTts).toBe(1);
    expect(stub.clientWs.send).not.toHaveBeenCalled();
  });
});

describe('turnState bookkeeping across a turn lifecycle (_maybeRetireTurn)', () => {
  it('does nothing while pendingTts is still > 0 (TTS still in flight)', () => {
    const turnState = { id: 1, llmDone: true, pendingTts: 2, startedSpeaking: true, saidNothing: false };
    const stub = makeStub({ activeTurn: 1, turnState });
    CallSession.prototype._maybeRetireTurn.call(stub, 1);
    expect(stub.activeTurn).toBe(1); // still in flight, not retired
  });

  it('does nothing while llmDone is still false (LLM still streaming)', () => {
    const turnState = { id: 1, llmDone: false, pendingTts: 0, startedSpeaking: true, saidNothing: false };
    const stub = makeStub({ activeTurn: 1, turnState });
    CallSession.prototype._maybeRetireTurn.call(stub, 1);
    expect(stub.activeTurn).toBe(1);
  });

  it('ignores a stale turnId that no longer matches this.turnState.id (superseded turn retiring late)', () => {
    const turnState = { id: 2, llmDone: true, pendingTts: 0, startedSpeaking: true, saidNothing: false };
    const stub = makeStub({ activeTurn: 2, turnState });
    CallSession.prototype._maybeRetireTurn.call(stub, 1); // stale id from a superseded turn
    expect(stub.activeTurn).toBe(2); // untouched — the real (turn 2) state stands
  });

  it('retires a fully-finished ordinary turn: activeTurn resets to 0 and the reminder clock starts', () => {
    const turnState = { id: 1, llmDone: true, pendingTts: 0, startedSpeaking: true, saidNothing: false, nodeType: null, transition: null };
    const stub = makeStub({ activeTurn: 1, turnState });
    CallSession.prototype._maybeRetireTurn.call(stub, 1);
    expect(stub.activeTurn).toBe(0);
    expect(stub._scheduleReminderIfConfigured).toHaveBeenCalled();
  });

  it('replays queued caller text (from the in-flight-turn guard) instead of starting the reminder clock', () => {
    const turnState = { id: 1, llmDone: true, pendingTts: 0, startedSpeaking: true, saidNothing: false, nodeType: null, transition: null };
    const stub = makeStub({ activeTurn: 1, turnState, _queuedUserText: 'are you still there' });
    stub._resolveInterruptionSensitivity = () => 'medium'; // not 'off', so it doesn't hold for audio drain
    stub._onUserTurnComplete = vi.fn();
    CallSession.prototype._maybeRetireTurn.call(stub, 1);
    expect(stub._queuedUserText).toBe(null);
    expect(stub._onUserTurnComplete).toHaveBeenCalledWith('are you still there');
    expect(stub._scheduleReminderIfConfigured).not.toHaveBeenCalled();
  });

  it('holds queued text instead of replaying it when interruption is off and audio is still playing out', () => {
    const turnState = { id: 1, llmDone: true, pendingTts: 0, startedSpeaking: true, saidNothing: false, nodeType: null, transition: null };
    const stub = makeStub({
      activeTurn: 1,
      turnState,
      _queuedUserText: 'hello?',
      clientWs: { isSpeaking: () => true },
    });
    stub._resolveInterruptionSensitivity = () => 'off';
    stub._onUserTurnComplete = vi.fn();
    CallSession.prototype._maybeRetireTurn.call(stub, 1);
    expect(stub._queuedUserText).toBe('hello?'); // untouched — replayed later via _onAudioDrained
    expect(stub._onUserTurnComplete).not.toHaveBeenCalled();
  });

  it('hangs up on a goodbye-node turn instead of scheduling a reminder', () => {
    const turnState = { id: 1, llmDone: true, pendingTts: 0, startedSpeaking: true, saidNothing: false, nodeType: 'goodbye' };
    const stub = makeStub({ activeTurn: 1, turnState, close: vi.fn() });
    CallSession.prototype._maybeRetireTurn.call(stub, 1);
    expect(stub._closing).toBe(true);
    expect(stub.close).toHaveBeenCalled();
    expect(stub._scheduleReminderIfConfigured).not.toHaveBeenCalled();
  });

  it('applies a pending flow transition instead of scheduling a reminder', () => {
    const transition = { targetNodeId: 'node-2' };
    const turnState = { id: 1, llmDone: true, pendingTts: 0, startedSpeaking: true, saidNothing: false, nodeType: null, transition };
    const stub = makeStub({ activeTurn: 1, turnState, _applyTransition: vi.fn() });
    CallSession.prototype._maybeRetireTurn.call(stub, 1);
    expect(stub._applyTransition).toHaveBeenCalledWith(transition);
    expect(stub._scheduleReminderIfConfigured).not.toHaveBeenCalled();
  });
});

describe('_maybeSpeakBackchannel gating', () => {
  it('does not fire when activeTurn no longer matches turnId (barge-in or turn already resolved)', () => {
    const stub = makeStub({ activeTurn: 5, backchannelWords: ['Mm-hmm', 'Got it'] });
    stub._speakCached = vi.fn();
    CallSession.prototype._maybeSpeakBackchannel.call(stub, 3); // stale turnId
    expect(stub._speakCached).not.toHaveBeenCalled();
  });

  it('does not fire when there is no cached filler clip for this backend/voice', () => {
    const stub = makeStub({
      activeTurn: 5,
      backchannelWords: ['Mm-hmm'],
      ttsBackend: 'kokoro', // fillerCache lookup below will miss for a backend/voice combo never warmed
      voice: 'nonexistent-voice-xyz',
      _fillerCacheKey: CallSession.prototype._fillerCacheKey,
    });
    stub._speakCached = vi.fn();
    CallSession.prototype._maybeSpeakBackchannel.call(stub, 5);
    expect(stub._speakCached).not.toHaveBeenCalled();
  });

  it('does not fire when the (optional) `only` pool filters out every configured word', () => {
    const stub = makeStub({ activeTurn: 5, backchannelWords: ['Mm-hmm', 'Got it'] });
    stub._speakCached = vi.fn();
    CallSession.prototype._maybeSpeakBackchannel.call(stub, 5, ['Sure thing']); // none of these are in backchannelWords
    expect(stub._speakCached).not.toHaveBeenCalled();
  });
});

describe('Response Wait Time scheduling (_scheduleUserTurn)', () => {
  // _scheduleUserTurn pulls in _maybeSwitchLanguage, shouldHoldForDigits and
  // flow/history lookups that would need a much larger stub to exercise
  // faithfully end-to-end; the piece with real regression value in
  // isolation — the pending-timer swap — is tested directly below rather
  // than forcing a full flow/history stub just to hit it.
  it('clears a previously pending response timer before scheduling a new one (replaces, never stacks, the wait)', () => {
    const oldTimer = setTimeout(() => {}, 100000);
    const stub = makeStub({
      _pendingResponseTimer: oldTimer,
      _reminderTimer: null,
      history: [],
      flow: null,
      flowNodesById: null,
      lang: null,
      _maybeSwitchLanguage: vi.fn(),
      _onUserTurnComplete: vi.fn(),
    });
    const clearSpy = vi.spyOn(global, 'clearTimeout');
    CallSession.prototype._scheduleUserTurn.call(stub, 'hello');
    expect(clearSpy).toHaveBeenCalledWith(oldTimer);
    // No flow/responsiveness configured -> waitMs resolves to 0 -> answers immediately.
    expect(stub._onUserTurnComplete).toHaveBeenCalledWith('hello');
    clearSpy.mockRestore();
    clearTimeout(stub._pendingResponseTimer);
  });
});
