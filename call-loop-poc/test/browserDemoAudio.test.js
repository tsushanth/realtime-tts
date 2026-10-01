// In-browser demo: jingle + confirmation chime through the same callAudio mechanism as phone calls, converted for the
// browser player (PCM16 mono @24kHz). Pure unit tests: no network, no keys, no live calls.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import WebSocket from 'ws';
import { TwilioCallAdapter } from '../twilioAdapter.js';

process.env.NODE_ENV = 'test';
process.env.ELEVENLABS_API_KEY = 'fake-test-key'; // so ttsBackend=elevenlabs is accepted; never called
const { CallSession, clipForClient } = await import('../server.js');
const {
  mulaw8kToPcm16At24k, browserPcmForClip, loadBrowserDemoCallAudio, isBrowserDemoAudioEnabled,
  parseCallAudioContext, buildPlaySoundEffectTool, DEMO_EFFECT_NAME, BROWSER_PCM_RATE,
} = await import('../callAudio.js');

// Reference G.711 mu-law encode/decode, independent of the module under test.
function encodeMuLaw(sample) {
  const sign = sample < 0 ? 0x80 : 0;
  let s = Math.min(Math.abs(sample), 32635) + 0x84;
  let exp = 7;
  for (let mask = 0x4000; (s & mask) === 0 && exp > 0; mask >>= 1) exp--;
  return ~(sign | (exp << 4) | ((s >> (exp + 3)) & 0x0f)) & 0xff;
}
function decodeMuLaw(byte) {
  const b = ~byte & 0xff;
  const mag = (((b & 0x0f) << 3) + 0x84) << ((b >> 4) & 0x07);
  return b & 0x80 ? 0x84 - mag : mag - 0x84;
}
const sine = (freq, seconds, amp) => Buffer.from(Array.from({ length: Math.round(8000 * seconds) }, (_, i) => encodeMuLaw(Math.round(amp * Math.sin((2 * Math.PI * freq * i) / 8000)))));
const toInt16 = (buf) => Int16Array.from({ length: buf.length / 2 }, (_, i) => buf.readInt16LE(i * 2));
const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);

describe('mulaw8kToPcm16At24k', () => {
  it('produces 3x the samples (6x the bytes) at the browser rate, even-length PCM16', () => {
    const out = mulaw8kToPcm16At24k(sine(440, 1, 8000));
    expect(BROWSER_PCM_RATE).toBe(24000);
    expect(out.length).toBe(8000 * 3 * 2);
    expect(out.length % 2).toBe(0);
    expect(out.length / 2 / BROWSER_PCM_RATE).toBeCloseTo(1, 6); // exactly 1 s of audio at 24 kHz
  });
  it('keeps the tone: same frequency and level (resample, not a speed or pitch change)', () => {
    const src = sine(1000, 1, 8000);
    const pcm = toInt16(mulaw8kToPcm16At24k(src));
    const mid = pcm.slice(2400, pcm.length - 2400); // skip filter edges
    let crossings = 0;
    for (let i = 1; i < mid.length; i++) if (mid[i - 1] < 0 && mid[i] >= 0) crossings++;
    const seconds = mid.length / 24000;
    expect(crossings / seconds).toBeGreaterThan(995);
    expect(crossings / seconds).toBeLessThan(1005);
    const srcRms = rms(Array.from(src, decodeMuLaw).slice(800, -800));
    expect(rms(Array.from(mid)) / srcRms).toBeGreaterThan(0.97);
    expect(rms(Array.from(mid)) / srcRms).toBeLessThan(1.03);
  });
  it('does not clip: a loud tone stays within a hair of the source peak and nothing saturates int16', () => {
    const src = sine(700, 0.5, 30000);
    const srcPeak = Math.max(...Array.from(src, (b) => Math.abs(decodeMuLaw(b))));
    const pcm = toInt16(mulaw8kToPcm16At24k(src));
    const peak = Math.max(...Array.from(pcm, Math.abs));
    expect(peak).toBeLessThan(32767);
    expect(peak).toBeLessThan(srcPeak * 1.05);
    expect(pcm.filter((v) => v === 32767 || v === -32768)).toHaveLength(0);
  });
  it('removes the 8kHz images (energy above 4kHz is negligible next to a 1kHz tone)', () => {
    const pcm = toInt16(mulaw8kToPcm16At24k(sine(1000, 1, 8000)));
    // The image of 1kHz after zero-stuffing sits at 7kHz and 9kHz: correlate against 7kHz.
    let re = 0, im = 0;
    for (let i = 2400; i < pcm.length - 2400; i++) {
      re += pcm[i] * Math.cos((2 * Math.PI * 7000 * i) / 24000);
      im += pcm[i] * Math.sin((2 * Math.PI * 7000 * i) / 24000);
    }
    const n = pcm.length - 4800;
    const image = Math.hypot(re, im) / n * 2;
    expect(image).toBeLessThan(8000 * 0.01); // < 1% of the tone amplitude (-40 dB)
  });
  it('handles empty input and is memoized per buffer', () => {
    expect(mulaw8kToPcm16At24k(Buffer.alloc(0)).length).toBe(0);
    const b = sine(500, 0.1, 5000);
    expect(browserPcmForClip(b)).toBe(browserPcmForClip(b));
  });
});

describe('the committed demo assets (assets/demo-call-audio.json)', () => {
  const payload = loadBrowserDemoCallAudio({});
  it('load, are small, and have the expected durations', () => {
    expect(payload).not.toBeNull();
    const jingle = Buffer.from(payload.jingle.audio, 'base64');
    const chime = Buffer.from(payload.effects[0].audio, 'base64');
    expect(jingle.length).toBe(32000); // 4 s mu-law@8kHz
    expect(chime.length).toBe(16000); // 2 s
    expect(jingle.length + chime.length).toBeLessThan(60_000);
  });
  it('convert for the browser to exact durations with no clipping and audible level', () => {
    for (const [b64, seconds] of [[payload.jingle.audio, 4], [payload.effects[0].audio, 2]]) {
      const src = Buffer.from(b64, 'base64');
      const out = mulaw8kToPcm16At24k(src);
      const pcm = toInt16(out);
      expect(pcm.length / BROWSER_PCM_RATE).toBeCloseTo(seconds, 6);
      const srcPeak = Math.max(...Array.from(src, (b) => Math.abs(decodeMuLaw(b))));
      const peak = Math.max(...Array.from(pcm, Math.abs));
      expect(peak).toBeLessThan(32767);
      expect(peak).toBeLessThan(srcPeak * 1.1); // interpolation overshoot only
      expect(pcm.filter((v) => v === 32767 || v === -32768)).toHaveLength(0);
      expect(rms(Array.from(pcm))).toBeGreaterThan(300); // not silent
    }
  });
  it('offer exactly one effect, whose description says when to play it', () => {
    const parsed = parseCallAudioContext(payload, {});
    expect(parsed.jingle).toBeInstanceOf(Buffer);
    expect([...parsed.effects.keys()]).toEqual([DEMO_EFFECT_NAME]);
    const tool = buildPlaySoundEffectTool(parsed);
    expect(tool.name).toBe('play_sound_effect');
    expect(tool.input_schema.properties.name.description).toMatch(/confirm a specific appointment or booking/);
  });
});

describe('loadBrowserDemoCallAudio fails open', () => {
  const tmp = mkdtempSync(join(tmpdir(), 'demo-audio-'));
  const file = (name, content) => { const p = join(tmp, name); writeFileSync(p, content); return pathToFileURL(p); };
  const good = Buffer.alloc(100, 1).toString('base64');

  it('returns null when the flag is off ("false" and "0"), on by default', () => {
    expect(isBrowserDemoAudioEnabled({})).toBe(true);
    expect(loadBrowserDemoCallAudio({ BROWSER_DEMO_AUDIO_ENABLED: 'false' })).toBeNull();
    expect(loadBrowserDemoCallAudio({ BROWSER_DEMO_AUDIO_ENABLED: '0' })).toBeNull();
  });
  it('returns null for a missing file, bad JSON, a missing clip, empty audio, or an oversized clip', () => {
    expect(loadBrowserDemoCallAudio({}, pathToFileURL(join(tmp, 'nope.json')))).toBeNull();
    expect(loadBrowserDemoCallAudio({}, file('bad.json', '{not json'))).toBeNull();
    expect(loadBrowserDemoCallAudio({}, file('nochime.json', JSON.stringify({ intro_jingle: { audio: good } })))).toBeNull();
    expect(loadBrowserDemoCallAudio({}, file('empty.json', JSON.stringify({ intro_jingle: { audio: '' }, confirmation_chime: { audio: good } })))).toBeNull();
    const huge = Buffer.alloc(100_001, 1).toString('base64');
    expect(loadBrowserDemoCallAudio({}, file('huge.json', JSON.stringify({ intro_jingle: { audio: huge }, confirmation_chime: { audio: good } })))).toBeNull();
  });
  it('a valid file loads', () => {
    const p = loadBrowserDemoCallAudio({}, file('ok.json', JSON.stringify({ intro_jingle: { audio: good }, confirmation_chime: { audio: good } })));
    expect(p.jingle.name).toBe('intro');
    expect(p.effects).toHaveLength(1);
  });
});

describe('existing kill switches still win over the demo assets', () => {
  const payload = loadBrowserDemoCallAudio({});
  it('CALL_AUDIO_ASSETS_ENABLED=false -> nothing at all', () => {
    expect(parseCallAudioContext(payload, { CALL_AUDIO_ASSETS_ENABLED: 'false' })).toBeNull();
  });
  it('CALL_AUDIO_EFFECTS_ENABLED=false -> jingle only, no effect tool', () => {
    const p = parseCallAudioContext(payload, { CALL_AUDIO_EFFECTS_ENABLED: 'false' });
    expect(p.jingle).not.toBeNull();
    expect(p.effects.size).toBe(0);
    expect(buildPlaySoundEffectTool(p)).toBeNull();
  });
});

describe('clipForClient', () => {
  const mu = sine(440, 0.1, 5000);
  it('phone: raw mu-law with the mulaw8k hint (unchanged behavior)', () => {
    const ws = Object.create(TwilioCallAdapter.prototype);
    const c = clipForClient(ws, mu);
    expect(c.data).toBe(mu);
    expect(c.opts).toEqual({ binary: true, format: 'mulaw8k' });
  });
  it('browser: PCM16@24kHz with no format hint', () => {
    const c = clipForClient({ readyState: WebSocket.OPEN }, mu);
    expect(c.data.length).toBe(mu.length * 6);
    expect(c.opts).toEqual({ binary: true });
  });
});

describe('browser session wiring', () => {
  afterEach(() => vi.restoreAllMocks());
  const fakeWs = () => ({ readyState: WebSocket.OPEN, sent: [], send(d, o) { this.sent.push({ d, o }); }, close() {} });
  const ctx = (extra) => JSON.stringify({ type: 'context', systemPrompt: 'You are a demo receptionist.', ttsBackend: 'elevenlabs', ttsModel: 'eleven_v4_turbo', ...extra });
  const open = (ws, extra) => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    const s = new CallSession(ws);
    s.onClientMessage(ctx(extra), false);
    return s;
  };

  it('demoAudio:true attaches the demo jingle + effect and marks the session once-per-effect', () => {
    const s = open(fakeWs(), { demoAudio: true });
    expect(s.callAudio.jingle.length).toBe(32000);
    expect([...s.callAudio.effects.keys()]).toEqual([DEMO_EFFECT_NAME]);
    expect(s._browserDemo).toBe(true);
    expect(s.ttsModel).toBe('eleven_v4_turbo');
    s.close();
  });
  it('without demoAudio the session has no call audio (today\'s behavior)', () => {
    const s = open(fakeWs(), {});
    expect(s.callAudio).toBeNull();
    s.close();
  });
  it('a tenant-scoped context never gets the public demo assets', () => {
    const s = open(fakeWs(), { demoAudio: true, tenantId: 't_1' });
    expect(s.callAudio).toBeNull();
    s.close();
  });
  it('a Twilio session ignores demoAudio', () => {
    const s = open(Object.assign(Object.create(TwilioCallAdapter.prototype), { readyState: WebSocket.OPEN, send() {}, close() {} }), { demoAudio: true });
    expect(s.callAudio).toBeNull();
    s.close();
  });
  it('the intro jingle goes to the browser as PCM16@24kHz binary (not mu-law), exactly once', () => {
    const ws = fakeWs();
    const s = open(ws, { demoAudio: true });
    CallSession.prototype._playIntroJingle.call(s);
    CallSession.prototype._playIntroJingle.call(s);
    expect(ws.sent).toHaveLength(1);
    expect(ws.sent[0].d.length).toBe(32000 * 6); // 4 s * 24000 * 2 bytes
    expect(ws.sent[0].o).toEqual({ binary: true });
    s.close();
  });
  it('the effect is sent via _speakCached as PCM with no format hint, and only once per browser session', () => {
    const s = open(fakeWs(), { demoAudio: true });
    s._speakCached = vi.fn(async () => {});
    s._playSoundEffect(DEMO_EFFECT_NAME, 3);
    s._sfxLastPlayedAt.set(DEMO_EFFECT_NAME, Date.now() - 120_000);
    s._playSoundEffect(DEMO_EFFECT_NAME, 4);
    expect(s._speakCached).toHaveBeenCalledTimes(1);
    const [data, turnId, format] = s._speakCached.mock.calls[0];
    expect(data.length).toBe(16000 * 6);
    expect(turnId).toBe(3);
    expect(format).toBeUndefined();
    s.close();
  });
  it('flag off at attach time: no jingle, no effect, no tool (fails open)', () => {
    const prev = process.env.BROWSER_DEMO_AUDIO_ENABLED;
    process.env.BROWSER_DEMO_AUDIO_ENABLED = 'false';
    try {
      const ws = fakeWs();
      const s = open(ws, { demoAudio: true });
      expect(s.callAudio).toBeNull();
      CallSession.prototype._playIntroJingle.call(s);
      expect(ws.sent).toHaveLength(0);
      s.close();
    } finally {
      if (prev === undefined) delete process.env.BROWSER_DEMO_AUDIO_ENABLED; else process.env.BROWSER_DEMO_AUDIO_ENABLED = prev;
    }
  });
});
