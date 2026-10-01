// CallSession wiring for jingle / sound-effect playback (callAudio.js has the pure-logic tests).
// Follows flowNodeEngine.test.js's stub pattern: real CallSession.prototype methods called against
// minimal stub `this` objects, so the shipped code is what's exercised.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import WebSocket from 'ws';

process.env.NODE_ENV = 'test';
const { CallSession, sampleAudioEvents, samplePlacedSidByTo } = await import('../server.js');
const { parseCallAudioContext } = await import('../callAudio.js');

const b64 = (bytes) => Buffer.from(bytes).toString('base64');
const RAW = {
  jingle: { name: 'intro', audio: b64([1, 2, 3, 4]) },
  effects: [{ name: 'chime', description: 'after a booking', audio: b64([9, 9]) }],
};

function stubSession(callAudio) {
  const sent = [];
  return {
    sent,
    callAudio,
    _jinglePlayed: false,
    _sfxLastPlayedAt: new Map(),
    clientWs: { readyState: WebSocket.OPEN, send: (data, opts) => sent.push({ data, opts }) },
    _speakCached: vi.fn(async () => {}),
  };
}

describe('_playIntroJingle', () => {
  it('sends the jingle as mulaw8k binary, once', () => {
    const s = stubSession(parseCallAudioContext(RAW, {}));
    CallSession.prototype._playIntroJingle.call(s);
    CallSession.prototype._playIntroJingle.call(s);
    expect(s.sent).toHaveLength(1);
    expect([...s.sent[0].data]).toEqual([1, 2, 3, 4]);
    expect(s.sent[0].opts).toEqual({ binary: true, format: 'mulaw8k' });
  });
  it('does nothing without a jingle (effects-only tenant) or without callAudio at all', () => {
    const effectsOnly = stubSession(parseCallAudioContext({ effects: RAW.effects }, {}));
    CallSession.prototype._playIntroJingle.call(effectsOnly);
    const none = stubSession(null);
    CallSession.prototype._playIntroJingle.call(none);
    expect(effectsOnly.sent).toHaveLength(0);
    expect(none.sent).toHaveLength(0);
  });
  it('does nothing when the socket is closed', () => {
    const s = stubSession(parseCallAudioContext(RAW, {}));
    s.clientWs.readyState = WebSocket.CLOSED;
    CallSession.prototype._playIntroJingle.call(s);
    expect(s.sent).toHaveLength(0);
  });
});

describe('_playSoundEffect', () => {
  beforeEach(() => vi.spyOn(console, 'log').mockImplementation(() => {}));
  afterEach(() => vi.restoreAllMocks());

  it('dispatches the cached buffer for a known name via _speakCached as mulaw8k', () => {
    const s = stubSession(parseCallAudioContext(RAW, {}));
    CallSession.prototype._playSoundEffect.call(s, 'chime', 7);
    expect(s._speakCached).toHaveBeenCalledTimes(1);
    const [buf, turnId, format] = s._speakCached.mock.calls[0];
    expect([...buf]).toEqual([9, 9]);
    expect(turnId).toBe(7);
    expect(format).toBe('mulaw8k');
  });
  it('ignores an unknown name (hallucinated enum value)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const s = stubSession(parseCallAudioContext(RAW, {}));
    CallSession.prototype._playSoundEffect.call(s, 'nope', 1);
    expect(s._speakCached).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalled();
  });
  it('suppresses the same effect replaying within the repeat window, but allows it after', () => {
    const s = stubSession(parseCallAudioContext(RAW, {}));
    CallSession.prototype._playSoundEffect.call(s, 'chime', 1);
    CallSession.prototype._playSoundEffect.call(s, 'chime', 2);
    expect(s._speakCached).toHaveBeenCalledTimes(1);
    s._sfxLastPlayedAt.set('chime', Date.now() - 60_000);
    CallSession.prototype._playSoundEffect.call(s, 'chime', 3);
    expect(s._speakCached).toHaveBeenCalledTimes(2);
  });
});

describe('_playSoundEffect in a SAMPLE call (once per call)', () => {
  beforeEach(() => vi.spyOn(console, 'log').mockImplementation(() => {}));
  afterEach(() => vi.restoreAllMocks());

  it('plays each effect at most once even after the repeat window has passed: a sample demonstrates one moment', () => {
    const s = { ...stubSession(parseCallAudioContext(RAW, {})), _sampleTo: '+15550000001' };
    CallSession.prototype._playSoundEffect.call(s, 'chime', 1);
    s._sfxLastPlayedAt.set('chime', Date.now() - 120_000); // well past the 10s window
    CallSession.prototype._playSoundEffect.call(s, 'chime', 2);
    expect(s._speakCached).toHaveBeenCalledTimes(1);
  });
  it('a normal (tenant) call is unaffected: the same effect may play again after the window', () => {
    const s = stubSession(parseCallAudioContext(RAW, {}));
    CallSession.prototype._playSoundEffect.call(s, 'chime', 1);
    s._sfxLastPlayedAt.set('chime', Date.now() - 120_000);
    CallSession.prototype._playSoundEffect.call(s, 'chime', 2);
    expect(s._speakCached).toHaveBeenCalledTimes(2);
  });
  it('a different effect in the same sample call still plays', () => {
    const two = parseCallAudioContext({ ...RAW, effects: [...RAW.effects, { name: 'other', description: 'd', audio: b64([5]) }] }, {});
    const s = { ...stubSession(two), _sampleTo: '+15550000001' };
    CallSession.prototype._playSoundEffect.call(s, 'chime', 1);
    CallSession.prototype._playSoundEffect.call(s, 'other', 2);
    expect(s._speakCached).toHaveBeenCalledTimes(2);
  });
});

describe('_speakCached format passthrough', () => {
  const mk = (format) => {
    const sent = [];
    const s = {
      activeTurn: 1,
      turnState: { id: 1, pendingTts: 0 },
      _sendChain: null,
      clientWs: { readyState: WebSocket.OPEN, send: (d, o) => sent.push(o) },
      _maybeRetireTurn: () => {},
    };
    return { s, sent, run: () => CallSession.prototype._speakCached.call(s, Buffer.from([1]), 1, format) };
  };
  it('adds format only when given, leaving the existing backchannel/filler call shape untouched', async () => {
    const a = mk(undefined);
    await a.run();
    expect(a.sent[0]).toEqual({ binary: true });
    const b = mk('mulaw8k');
    await b.run();
    expect(b.sent[0]).toEqual({ binary: true, format: 'mulaw8k' });
  });
});

describe('_startGreetingTurn (flow-less sessions, e.g. a sample call\'s demo agent)', () => {
  const mk = (callAudio) => {
    const order = [];
    const s = {
      callAudio, _jinglePlayed: false, history: [], turnSeq: 0, activeTurn: 0, turnState: null,
      clientWs: { readyState: WebSocket.OPEN, send: () => order.push('jingle-sent') },
      _speak: vi.fn((text) => order.push(`speak:${text}`)),
    };
    s._playIntroJingle = CallSession.prototype._playIntroJingle.bind(s);
    return { s, order };
  };
  beforeEach(() => vi.spyOn(console, 'log').mockImplementation(() => {}));
  afterEach(() => vi.restoreAllMocks());

  it('sends the jingle BEFORE speaking the greeting', () => {
    const { s, order } = mk(parseCallAudioContext(RAW, {}));
    CallSession.prototype._startGreetingTurn.call(s, 'Thanks for calling.');
    expect(order).toEqual(['jingle-sent', 'speak:Thanks for calling.']);
  });
  it('still records the greeting in history and opens a real turn for it (barge-in keeps working)', () => {
    const { s } = mk(null);
    CallSession.prototype._startGreetingTurn.call(s, 'Hello there');
    expect(s.history).toEqual([{ role: 'assistant', content: 'Hello there' }]);
    expect(s.activeTurn).toBe(1);
    expect(s.turnState).toMatchObject({ id: 1, llmDone: true, pendingTts: 0 });
    expect(s._speak).toHaveBeenCalledWith('Hello there', 1, expect.any(Number));
  });
  it('a session with no callAudio just speaks the greeting (zero behavior change)', () => {
    const { s, order } = mk(null);
    CallSession.prototype._startGreetingTurn.call(s, 'Hi');
    expect(order).toEqual(['speak:Hi']);
  });
});

describe('recording what actually played (sampleAudioEvents)', () => {
  const SIDX = 'CA' + '9'.repeat(32);
  beforeEach(() => { sampleAudioEvents.clear(); vi.spyOn(console, 'log').mockImplementation(() => {}); });
  afterEach(() => vi.restoreAllMocks());

  const mk = (over = {}) => ({
    ...stubSession(parseCallAudioContext(RAW, {})),
    callSid: SIDX, _callStartedAt: Date.now() - 5000,
    ...over,
  });

  it('records the jingle with its offset from the start of the call', () => {
    const s = mk();
    CallSession.prototype._playIntroJingle.call(s);
    const ev = sampleAudioEvents.get(SIDX);
    expect(ev).toHaveLength(1);
    expect(ev[0]).toMatchObject({ kind: 'jingle', name: 'intro' });
    expect(ev[0].atMs).toBeGreaterThanOrEqual(4900);
  });
  it('records a sound effect that actually plays, and NOT one that was suppressed or unknown', () => {
    const s = mk();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    CallSession.prototype._playSoundEffect.call(s, 'chime', 1);
    CallSession.prototype._playSoundEffect.call(s, 'chime', 2); // within the repeat window -> suppressed
    CallSession.prototype._playSoundEffect.call(s, 'nope', 3); // unknown
    const ev = sampleAudioEvents.get(SIDX);
    expect(ev.map((e) => `${e.kind}:${e.name}`)).toEqual(['effect:chime']);
  });
  describe('a sample call has TWO phone legs with different call sids', () => {
    // The generator polls /call-status with the sid of the SHOPPER's outbound leg, but the jingle/effects play on the
    // demo agent's INBOUND leg (its own sid). Events must be filed under the sid the caller actually polls.
    const TO = '+15550000001', PLACED = 'CA' + '1'.repeat(32), CALLEE = 'CA' + '2'.repeat(32);
    beforeEach(() => samplePlacedSidByTo.clear());
    it('files events under the shopper leg\'s sid when this session is the sample callee for that number', () => {
      samplePlacedSidByTo.set(TO, PLACED);
      const s = mk({ callSid: CALLEE, _sampleTo: TO });
      CallSession.prototype._playIntroJingle.call(s);
      CallSession.prototype._playSoundEffect.call(s, 'chime', 1);
      expect(sampleAudioEvents.get(PLACED).map((e) => `${e.kind}:${e.name}`)).toEqual(['jingle:intro', 'effect:chime']);
      expect(sampleAudioEvents.has(CALLEE)).toBe(false);
    });
    it('falls back to the session\'s own sid if no placement is known (never loses the events)', () => {
      const s = mk({ callSid: CALLEE, _sampleTo: TO });
      CallSession.prototype._playIntroJingle.call(s);
      expect(sampleAudioEvents.get(CALLEE)).toHaveLength(1);
    });
    it('a non-sample session (a tenant call) is unaffected: events stay under its own sid', () => {
      samplePlacedSidByTo.set(TO, PLACED);
      const s = mk({ callSid: CALLEE }); // no _sampleTo
      CallSession.prototype._playIntroJingle.call(s);
      expect(sampleAudioEvents.get(CALLEE)).toHaveLength(1);
      expect(sampleAudioEvents.has(PLACED)).toBe(false);
    });
  });

  it('does nothing (and does not crash) for a session with no callSid, e.g. a browser demo', () => {
    const s = mk({ callSid: undefined });
    CallSession.prototype._playIntroJingle.call(s);
    expect(sampleAudioEvents.size).toBe(0);
  });
  it('the map is bounded so a long-lived server cannot grow without limit', () => {
    for (let i = 0; i < 80; i++) {
      const sid = 'CA' + i.toString(16).padStart(32, '0');
      CallSession.prototype._playIntroJingle.call({ ...stubSession(parseCallAudioContext(RAW, {})), callSid: sid, _callStartedAt: Date.now() });
    }
    expect(sampleAudioEvents.size).toBeLessThanOrEqual(50);
  });
});

describe('_followUpAfterSilentSoundEffect (a turn whose ONLY output was the sound-effect tool call)', () => {
  // Real bug, first full sample call after the quota fix: on the booking turn the model emitted only the tool call and no
  // words. The chime played, nobody spoke, and the caller (waiting for the business to speak) sat in silence for 100s.
  const sfx = { type: 'tool_use', name: 'play_sound_effect', input: { name: 'chime' } };
  const mk = (over = {}) => ({
    history: [{ role: 'user', content: 'caller said something' }], turnSeq: 4, activeTurn: 4, turnState: null,
    _generateTurn: vi.fn(async () => {}), ...over,
  });
  beforeEach(() => vi.spyOn(console, 'log').mockImplementation(() => {}));
  afterEach(() => vi.restoreAllMocks());

  it('asks the model to speak: adds a system note and generates a follow-up turn with the effect tool suppressed', async () => {
    const s = mk();
    const handled = await CallSession.prototype._followUpAfterSilentSoundEffect.call(s, [sfx], '', 4);
    expect(handled).toBe(true);
    expect(s.history.at(-1)).toMatchObject({ role: 'user' });
    expect(s.history.at(-1).content).toMatch(/^\[System note:/);
    expect(s.history.at(-1).content).toMatch(/say your (spoken )?reply|out loud/i);
    expect(s._generateTurn).toHaveBeenCalledTimes(1);
    expect(s._generateTurn.mock.calls[0][2]).toMatchObject({ suppressSoundEffectTool: true });
    expect(s.activeTurn).toBe(5); // a fresh turn, so the reply is spoken as a normal turn
    expect(s.turnState).toMatchObject({ id: 5, llmDone: false });
  });
  it('does nothing when the turn DID speak (the normal case: words plus the chime)', async () => {
    const s = mk();
    expect(await CallSession.prototype._followUpAfterSilentSoundEffect.call(s, [sfx], "You're all set.", 4)).toBe(false);
    expect(s._generateTurn).not.toHaveBeenCalled();
  });
  it('does nothing when there was no sound-effect call at all', async () => {
    const s = mk();
    expect(await CallSession.prototype._followUpAfterSilentSoundEffect.call(s, [{ type: 'text', text: '' }], '', 4)).toBe(false);
    expect(s._generateTurn).not.toHaveBeenCalled();
  });
  it('leaves a turn that ALSO called another tool to that tool\'s own handling (calendar, transition, ...)', async () => {
    const s = mk();
    expect(await CallSession.prototype._followUpAfterSilentSoundEffect.call(s, [sfx, { type: 'tool_use', name: 'book_appointment', input: {} }], '', 4)).toBe(false);
    expect(s._generateTurn).not.toHaveBeenCalled();
  });
  it('does nothing if the caller already barged in (the turn is no longer the active one)', async () => {
    const s = mk({ activeTurn: 0 });
    expect(await CallSession.prototype._followUpAfterSilentSoundEffect.call(s, [sfx], '', 4)).toBe(false);
    expect(s._generateTurn).not.toHaveBeenCalled();
  });
  it('treats whitespace-only text as silence', async () => {
    const s = mk();
    expect(await CallSession.prototype._followUpAfterSilentSoundEffect.call(s, [sfx], '  \n ', 4)).toBe(true);
  });
});
