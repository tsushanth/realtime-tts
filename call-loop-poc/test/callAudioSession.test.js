// CallSession wiring for jingle / sound-effect playback (callAudio.js has the pure-logic tests).
// Follows flowNodeEngine.test.js's stub pattern: real CallSession.prototype methods called against
// minimal stub `this` objects, so the shipped code is what's exercised.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import WebSocket from 'ws';

process.env.NODE_ENV = 'test';
const { CallSession } = await import('../server.js');
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
