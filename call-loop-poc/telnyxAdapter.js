// Telnyx TeXML bidirectional media streaming, presented as the same interface
// CallSession already uses for a Twilio call (see twilioAdapter.js).
//
// Deliberately a SUBCLASS of TwilioCallAdapter rather than a copy: the audio
// side is identical (mu-law @ 8 kHz, 160 B / 20 ms frames, the PCM16 -> mulaw
// fallback in send(), isSpeaking(), close() draining the paced queue), and
// server.js decides provider-specific audio formats with
// `instanceof TwilioCallAdapter` (mulaw8k TTS output, mulaw Deepgram URL).
// Subclassing keeps all of that working with ZERO edits to twilioAdapter.js,
// so the Twilio path cannot change. Only the WIRE PROTOCOL differs and is
// overridden here:
//
//   Telnyx -> us   {event:'connected'} | {event:'start', stream_id, start:{call_control_id,
//                  call_session_id, media_format:{encoding,sample_rate,channels}, ...}} |
//                  {event:'media', media:{track,chunk,timestamp,payload}, stream_id} |
//                  {event:'dtmf', dtmf:{digit}} | {event:'mark', mark:{name}} |
//                  {event:'stop'} | {event:'error', payload:{code,title,detail}}
//   us -> Telnyx   {event:'media', media:{payload}} | {event:'clear'} | {event:'mark', mark:{name}}
//
// Source: developers.telnyx.com/docs/voice/programmable-voice/media-streaming
// (see TELNYX.md for what is confirmed vs measured vs unverified).
import { WebSocket } from 'ws';
import { TwilioCallAdapter } from './twilioAdapter.js';

const FRAME_BYTES_8K = 160; // 20 ms of PCMU @ 8 kHz
// If the socket's own send buffer holds more than this, skip a pacing tick
// instead of piling more frames on (slow network / stalled peer).
const MAX_BUFFERED_BYTES = 64 * 1024;

export class TelnyxCallAdapter extends TwilioCallAdapter {
  // opts.callSid: the TeXML CallSid carried through the stream URL by the
  // /telnyx/voice webhook (it is the key pendingCallContext was stored under).
  constructor(telnyxWs, opts = {}) {
    super(telnyxWs);
    this.carrier = 'telnyx';
    this.callSid = opts.callSid || null;
    this.streamId = null;
    this.callControlId = null;
    this.callSessionId = null;
    this.mediaFormat = null;
    this._started = false;
    this._pendingFrames = []; // frames produced before 'start' (kept raw, paced out after start)
  }

  _onTwilioMessage(data) {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return; // malformed frame: ignore, never crash the call
    }
    if (!msg || typeof msg !== 'object') return;

    switch (msg.event) {
      case 'connected':
        return;
      case 'start': {
        if (this._started) return; // a duplicate start must not re-greet
        this._started = true;
        const start = msg.start && typeof msg.start === 'object' ? msg.start : {};
        this.streamId = msg.stream_id || null;
        this.callControlId = start.call_control_id || null;
        this.callSessionId = start.call_session_id || null;
        this.mediaFormat = start.media_format || null;
        // Custom <Parameter>s: Telnyx documents that they appear in the start message but not
        // under which key. Look in the plausible places; the signed URL query is the primary route.
        const custom = start.custom_parameters || start.customParameters || start.stream_params || msg.custom_parameters || null;
        if (!this.callSid && custom && typeof custom.callSid === 'string') this.callSid = custom.callSid;
        if (!this.callSid) this.callSid = this.callControlId;
        const enc = this.mediaFormat?.encoding;
        if (enc && String(enc).toUpperCase() !== 'PCMU') {
          console.error(`[telnyx] stream media_format is ${enc}, expected PCMU - audio will be garbled (set bidirectionalCodec="PCMU")`);
        }
        this.readyState = WebSocket.OPEN;
        console.log(`[telnyx] stream started: ${this.streamId} (callSid: ${this.callSid}, callControlId: ${this.callControlId ? 'present' : 'missing'})`);
        this._flushQueue();
        this.emit('start', this.callSid);
        return;
      }
      case 'media': {
        const m = msg.media;
        if (!m || typeof m.payload !== 'string' || m.payload.length === 0) return;
        // With track=both_tracks Telnyx also echoes our own outbound audio; only the caller's goes to STT.
        if (m.track && m.track !== 'inbound') return;
        this.emit('message', Buffer.from(m.payload, 'base64'), true);
        return;
      }
      case 'dtmf': {
        const digit = msg.dtmf?.digit;
        if (digit) this.emit('dtmf', String(digit));
        return;
      }
      case 'mark':
        this.emit('mark', msg.mark?.name);
        return;
      case 'stop':
        console.log('[telnyx] stream stopped');
        this.emit('close');
        return;
      case 'error':
        // e.g. 100003 malformed_frame, 100004 invalid_media, 100005 rate_limit_reached
        console.error(`[telnyx] stream error event: ${JSON.stringify(msg.payload || {})}`);
        return;
      default:
        return;
    }
  }

  _sendMediaFrames(mulawBuf) {
    const frames = [];
    for (let i = 0; i < mulawBuf.length; i += FRAME_BYTES_8K) frames.push(mulawBuf.subarray(i, i + FRAME_BYTES_8K));
    if (this.readyState !== WebSocket.OPEN) {
      this._pendingFrames.push(...frames);
      return;
    }
    this._pacedQueue.push(...frames);
    this._startPacing();
  }

  // Telnyx rate-limits media submissions (error 100005), so frames produced before 'start' are
  // paced out like any others instead of being dumped in one burst.
  _flushQueue() {
    if (this._pendingFrames.length === 0) return;
    this._pacedQueue.push(...this._pendingFrames);
    this._pendingFrames = [];
    this._startPacing();
  }

  // Barge-in: drop what is still queued locally AND tell Telnyx to stop what it already holds.
  clearQueue() {
    this._pacedQueue = [];
    this._pendingFrames = [];
    this._sendJson({ event: 'clear' });
  }

  // Outbound mark (not used by CallSession today; here so playout tracking can be added).
  sendMark(name) {
    this._sendJson({ event: 'mark', mark: { name: String(name) } });
  }

  _sendJson(obj) {
    if (this.twilioWs.readyState !== WebSocket.OPEN) return false;
    try {
      this.twilioWs.send(JSON.stringify(obj));
      return true;
    } catch (err) {
      console.error('[telnyx] send failed', err.message);
      return false;
    }
  }

  _startPacing() {
    if (this._paceTimer) return;
    this._paceTimer = setInterval(() => {
      if (this.twilioWs.readyState !== WebSocket.OPEN) {
        // Peer is gone: nothing more can be delivered.
        this._pacedQueue = [];
      } else if ((this.twilioWs.bufferedAmount || 0) > MAX_BUFFERED_BYTES) {
        return; // backpressure: skip this tick, keep the frame queued
      }
      const frame = this._pacedQueue.shift();
      if (!frame) {
        clearInterval(this._paceTimer);
        this._paceTimer = null;
        if (this._closeRequested) this._doClose();
        if (this.onDrained) this.onDrained();
        return;
      }
      this._sendJson({ event: 'media', media: { payload: frame.toString('base64') } });
    }, 20);
  }
}
