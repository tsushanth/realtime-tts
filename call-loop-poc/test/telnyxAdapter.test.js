// Unit tests for TelnyxCallAdapter against a fake websocket (no network, no timers left running).
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EventEmitter } from 'node:events';
import { TelnyxCallAdapter } from '../telnyxAdapter.js';
import { TwilioCallAdapter } from '../twilioAdapter.js';

class FakeWs extends EventEmitter {
  constructor() {
    super();
    this.readyState = 1; // OPEN
    this.bufferedAmount = 0;
    this.sent = [];
    this.closed = false;
  }
  send(d) { this.sent.push(JSON.parse(d)); }
  close() { this.closed = true; this.readyState = 3; this.emit('close'); }
  inbound(obj) { this.emit('message', Buffer.from(typeof obj === 'string' ? obj : JSON.stringify(obj))); }
}

const START = {
  event: 'start',
  sequence_number: '1',
  stream_id: 'stream-1',
  start: { call_control_id: 'v2:abc', call_session_id: 'sess-1', media_format: { encoding: 'PCMU', sample_rate: 8000, channels: 1 } },
};
const mediaMsgs = (ws) => ws.sent.filter((m) => m.event === 'media');

describe('TelnyxCallAdapter', () => {
  let ws, a;
  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    ws = new FakeWs();
    a = new TelnyxCallAdapter(ws, { callSid: 'CS-1' });
  });
  afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

  it('is a TwilioCallAdapter subclass (server.js format checks key off that) with carrier=telnyx', () => {
    expect(a).toBeInstanceOf(TwilioCallAdapter);
    expect(a.carrier).toBe('telnyx');
    expect(new TwilioCallAdapter(new FakeWs()).carrier).toBeUndefined(); // Twilio path untouched
  });

  it('start records ids, opens the adapter and emits start with the URL-carried callSid', () => {
    const onStart = vi.fn();
    a.on('start', onStart);
    expect(a.readyState).toBe(0);
    ws.inbound(START);
    expect(a.readyState).toBe(1);
    expect(a.streamId).toBe('stream-1');
    expect(a.callControlId).toBe('v2:abc');
    expect(a.callSessionId).toBe('sess-1');
    expect(a.mediaFormat.encoding).toBe('PCMU');
    expect(onStart).toHaveBeenCalledWith('CS-1');
  });

  it('falls back to a custom parameter, then call_control_id, for callSid; a duplicate start is ignored', () => {
    const w2 = new FakeWs();
    const b = new TelnyxCallAdapter(w2);
    const onStart = vi.fn();
    b.on('start', onStart);
    w2.inbound({ ...START, start: { ...START.start, custom_parameters: { callSid: 'FROM-PARAM' } } });
    w2.inbound(START);
    expect(onStart).toHaveBeenCalledTimes(1);
    expect(onStart).toHaveBeenCalledWith('FROM-PARAM');
    const w3 = new FakeWs();
    const c = new TelnyxCallAdapter(w3);
    const s3 = vi.fn();
    c.on('start', s3);
    w3.inbound(START);
    expect(s3).toHaveBeenCalledWith('v2:abc');
  });

  it('inbound media is forwarded as the raw mulaw bytes (binary=true); outbound-track frames are dropped', () => {
    ws.inbound(START);
    const got = [];
    a.on('message', (d, bin) => got.push([d, bin]));
    const pcmu = Buffer.from(Array.from({ length: 160 }, (_, i) => i));
    ws.inbound({ event: 'media', stream_id: 'stream-1', media: { track: 'inbound', chunk: '2', timestamp: '5', payload: pcmu.toString('base64') } });
    ws.inbound({ event: 'media', media: { track: 'outbound', payload: pcmu.toString('base64') } });
    expect(got).toHaveLength(1);
    expect(got[0][1]).toBe(true);
    expect(Buffer.compare(got[0][0], pcmu)).toBe(0);
  });

  it('outbound mulaw8k audio is sent as 160-byte PCMU frames, one per 20 ms, no streamSid field', () => {
    ws.inbound(START);
    const audio = Buffer.alloc(160 * 5, 0x7f);
    a.send(audio, { binary: true, format: 'mulaw8k' });
    expect(a.isSpeaking()).toBe(true);
    expect(mediaMsgs(ws)).toHaveLength(0);
    vi.advanceTimersByTime(20);
    expect(mediaMsgs(ws)).toHaveLength(1);
    vi.advanceTimersByTime(80);
    const frames = mediaMsgs(ws);
    expect(frames).toHaveLength(5);
    for (const f of frames) {
      expect(Buffer.from(f.media.payload, 'base64').length).toBe(160);
      expect(f.streamSid).toBeUndefined();
    }
    vi.advanceTimersByTime(40);
    expect(a.isSpeaking()).toBe(false);
  });

  it('PCM16 @24k falls back through the shared resampler/encoder into 160-byte frames', () => {
    ws.inbound(START);
    const pcm = new Int16Array(24000 * 0.1); // 100 ms @ 24 kHz -> 800 mulaw bytes -> 5 frames
    a.send(pcm.buffer, { binary: true });
    vi.advanceTimersByTime(200);
    expect(mediaMsgs(ws)).toHaveLength(5);
  });

  it('audio produced before start is held, then paced out (not burst) after start', () => {
    a.send(Buffer.alloc(160 * 3, 1), { binary: true, format: 'mulaw8k' });
    expect(ws.sent).toHaveLength(0);
    ws.inbound(START);
    expect(mediaMsgs(ws)).toHaveLength(0);
    vi.advanceTimersByTime(20);
    expect(mediaMsgs(ws)).toHaveLength(1);
    vi.advanceTimersByTime(40);
    expect(mediaMsgs(ws)).toHaveLength(3);
  });

  it('clearQueue drops queued audio and sends {event:"clear"} to Telnyx', () => {
    ws.inbound(START);
    a.send(Buffer.alloc(160 * 10, 1), { binary: true, format: 'mulaw8k' });
    vi.advanceTimersByTime(40);
    expect(mediaMsgs(ws)).toHaveLength(2);
    a.clearQueue();
    expect(a.isSpeaking()).toBe(false);
    expect(ws.sent.at(-1)).toEqual({ event: 'clear' });
    vi.advanceTimersByTime(200);
    expect(mediaMsgs(ws)).toHaveLength(2);
  });

  it('backpressure: no frames are sent while the socket buffer is over the limit, none are lost', () => {
    ws.inbound(START);
    ws.bufferedAmount = 1_000_000;
    a.send(Buffer.alloc(160 * 3, 1), { binary: true, format: 'mulaw8k' });
    vi.advanceTimersByTime(200);
    expect(mediaMsgs(ws)).toHaveLength(0);
    ws.bufferedAmount = 0;
    vi.advanceTimersByTime(100);
    expect(mediaMsgs(ws)).toHaveLength(3);
  });

  it('dtmf event emits the digit', () => {
    const d = vi.fn();
    a.on('dtmf', d);
    ws.inbound({ event: 'dtmf', stream_id: 's', occurred_at: 'x', sequence_number: '5', dtmf: { digit: '7' } });
    ws.inbound({ event: 'dtmf', dtmf: {} });
    expect(d).toHaveBeenCalledTimes(1);
    expect(d).toHaveBeenCalledWith('7');
  });

  it('mark events are surfaced; sendMark writes a mark frame', () => {
    ws.inbound(START);
    const m = vi.fn();
    a.on('mark', m);
    ws.inbound({ event: 'mark', mark: { name: 'm1' } });
    expect(m).toHaveBeenCalledWith('m1');
    a.sendMark('m2');
    expect(ws.sent.at(-1)).toEqual({ event: 'mark', mark: { name: 'm2' } });
  });

  it('stop emits close; a socket close emits close and marks CLOSED', () => {
    const c = vi.fn();
    a.on('close', c);
    ws.inbound(START);
    ws.inbound({ event: 'stop', stream_id: 'stream-1' });
    expect(c).toHaveBeenCalledTimes(1);
    ws.emit('close');
    expect(a.readyState).toBe(3);
  });

  it('malformed frames never throw and emit nothing', () => {
    const any = vi.fn();
    a.on('message', any); a.on('dtmf', any); a.on('start', any); a.on('close', any);
    for (const bad of ['not json', '{', 'null', '42', '"str"', '[]', '{"event":"media"}', '{"event":"media","media":{"payload":5}}', '{"event":"media","media":{"payload":""}}', '{"event":"start","start":"x"}x', '{"event":"unknown"}', '{"event":"error","payload":{"code":100003}}']) {
      expect(() => ws.inbound(bad)).not.toThrow();
    }
    expect(any).not.toHaveBeenCalled();
    // an error event is logged, not thrown
    expect(console.error).toHaveBeenCalled();
  });

  it('a send to a socket that throws or is closed does not crash pacing', () => {
    ws.inbound(START);
    ws.send = () => { throw new Error('boom'); };
    a.send(Buffer.alloc(320, 1), { binary: true, format: 'mulaw8k' });
    expect(() => vi.advanceTimersByTime(100)).not.toThrow();
    ws.readyState = 3;
    a.send(Buffer.alloc(320, 1), { binary: true, format: 'mulaw8k' });
    expect(() => vi.advanceTimersByTime(100)).not.toThrow();
  });

  it('close() waits for the paced queue to drain, then closes the socket', () => {
    ws.inbound(START);
    a.send(Buffer.alloc(160 * 3, 1), { binary: true, format: 'mulaw8k' });
    a.close();
    expect(ws.closed).toBe(false);
    vi.advanceTimersByTime(20 * 4 + 400);
    expect(mediaMsgs(ws)).toHaveLength(3);
    expect(ws.closed).toBe(true);
  });

  it('control JSON (transcripts etc.) is dropped, not sent to the phone', () => {
    ws.inbound(START);
    a.send(JSON.stringify({ type: 'transcript' }));
    expect(ws.sent).toHaveLength(0);
  });
});
