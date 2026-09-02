// Twilio Media Streams speaks a different wire protocol than our browser
// client: JSON frames with base64 mulaw@8kHz audio, not raw PCM16 binary.
// Rather than teach CallSession two protocols, this adapter translates in
// both directions and presents an interface CallSession already knows how
// to talk to — same shape as the `ws` WebSocket it normally holds
// (.send(data, opts), .readyState) plus the events the outer wiring in
// server.js listens for (.on('message', ...), .on('close', ...)).
import { EventEmitter } from 'node:events';
import { WebSocket } from 'ws';

// Standard G.711 mu-law codec (no external deps) — same algorithm as the
// classic public-domain reference implementation.
const BIAS = 0x84;
const CLIP = 32635;

function linearToMuLaw(sample) {
  let sign = 0;
  if (sample < 0) {
    sample = -sample;
    sign = 0x80;
  }
  if (sample > CLIP) sample = CLIP;
  sample += BIAS;

  let exponent = 7;
  for (let expMask = 0x4000; (sample & expMask) === 0 && exponent > 0; expMask >>= 1) {
    exponent--;
  }
  const mantissa = (sample >> (exponent + 3)) & 0x0f;
  return ~(sign | (exponent << 4) | mantissa) & 0xff;
}

function muLawToLinear(muLawByte) {
  const b = ~muLawByte & 0xff;
  const sign = b & 0x80;
  const exponent = (b >> 4) & 0x07;
  const mantissa = b & 0x0f;
  let sample = ((mantissa << 3) + BIAS) << exponent;
  sample -= BIAS;
  return sign ? -sample : sample;
}

function decodeMuLawBuffer(buf) {
  const out = new Int16Array(buf.length);
  for (let i = 0; i < buf.length; i++) out[i] = muLawToLinear(buf[i]);
  return out;
}

function encodeMuLawBuffer(int16) {
  const out = Buffer.alloc(int16.length);
  for (let i = 0; i < int16.length; i++) out[i] = linearToMuLaw(int16[i]);
  return out;
}

// Naive nearest-neighbor resample — matches the same approach the browser
// client already uses (downsampleTo16k in public/index.html). Fine for
// speech at these ratios; not broadcast-quality, not meant to be.
function resampleInt16(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) out[i] = input[Math.floor(i * ratio)];
  return out;
}

const FRAME_BYTES_8K = 160; // 20ms @ 8kHz mulaw — Twilio's own outbound frame size

export class TwilioCallAdapter extends EventEmitter {
  constructor(twilioWs) {
    super();
    this.twilioWs = twilioWs;
    this.streamSid = null;
    this.readyState = WebSocket.CONNECTING;
    this._outQueue = [];
    this._pacedQueue = [];
    this._paceTimer = null;

    twilioWs.on('message', (data) => this._onTwilioMessage(data));
    twilioWs.on('close', () => {
      this.readyState = WebSocket.CLOSED;
      this.emit('close');
    });
    twilioWs.on('error', (err) => this.emit('error', err));
  }

  _onTwilioMessage(data) {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }

    if (msg.event === 'start') {
      this.streamSid = msg.start.streamSid;
      // Needed for a flow's 'transfer' node type — redirecting a live call
      // via Twilio's REST API requires the CallSid, not the media stream's
      // own Sid (they're different identifiers).
      this.callSid = msg.start.callSid;
      this.readyState = WebSocket.OPEN;
      console.log(`[twilio] stream started: ${this.streamSid} (callSid: ${this.callSid})`);
      this._flushQueue();
      this.emit('start', this.callSid);
    } else if (msg.event === 'media') {
      const mulaw = Buffer.from(msg.media.payload, 'base64');
      const pcm8k = decodeMuLawBuffer(mulaw);
      const pcm16k = resampleInt16(pcm8k, 8000, 16000);
      this.emit('message', Buffer.from(pcm16k.buffer), true);
    } else if (msg.event === 'stop') {
      console.log('[twilio] stream stopped');
      this.emit('close');
    }
  }

  // Mirrors ws.WebSocket's send(data, {binary}) signature — CallSession
  // calls this exact shape for both control JSON (no opts) and raw TTS
  // audio (opts.binary === true), unaware it's not talking to a real
  // browser WebSocket.
  send(data, opts) {
    if (opts && opts.binary) {
      // PCM16LE @ 24kHz from our cascaded TTS leg -> mulaw @ 8kHz for Twilio.
      // The TTS gateway socket is opened with binaryType='arraybuffer'
      // (see server.js), so `data` here is a raw ArrayBuffer, not a Node
      // Buffer — they don't share a shape (ArrayBuffer has no .length/
      // .buffer/.byteOffset), so handle both rather than assume one.
      const int16 = data instanceof ArrayBuffer
        ? new Int16Array(data)
        : new Int16Array(data.buffer, data.byteOffset, data.length / 2);
      const down = resampleInt16(int16, 24000, 8000);
      const mulaw = encodeMuLawBuffer(down);
      this._sendMediaFrames(mulaw);
    }
    // Text/JSON control messages (transcript, tts_event, etc.) are for a
    // browser UI — the phone has no equivalent channel, so they're
    // intentionally dropped here rather than forwarded anywhere.
  }

  // Twilio expects media frames paced at real time (one 160-byte/20ms frame
  // roughly every 20ms) — dumping a whole response's worth of frames in one
  // tight loop the instant they arrive (which is what OpenAI Realtime's
  // audio deltas invite, since each delta can be a few seconds of audio at
  // once) overruns Twilio's playback buffer and comes out as static/garbled
  // audio rather than an error. Queue and drip-feed instead of sending
  // synchronously.
  _sendMediaFrames(mulawBuf) {
    if (this.readyState !== WebSocket.OPEN) {
      for (let i = 0; i < mulawBuf.length; i += FRAME_BYTES_8K) {
        this._outQueue.push({ event: 'media', streamSid: this.streamSid, media: { payload: mulawBuf.subarray(i, i + FRAME_BYTES_8K).toString('base64') } });
      }
      return;
    }
    for (let i = 0; i < mulawBuf.length; i += FRAME_BYTES_8K) {
      this._pacedQueue.push(mulawBuf.subarray(i, i + FRAME_BYTES_8K));
    }
    this._startPacing();
  }

  // Barge-in: drop whatever's still queued rather than let stale audio keep
  // playing out over the call after the caller's already interrupted.
  clearQueue() {
    this._pacedQueue = [];
  }

  _startPacing() {
    if (this._paceTimer) return;
    this._paceTimer = setInterval(() => {
      const frame = this._pacedQueue.shift();
      if (!frame) {
        clearInterval(this._paceTimer);
        this._paceTimer = null;
        return;
      }
      this.twilioWs.send(JSON.stringify({ event: 'media', streamSid: this.streamSid, media: { payload: frame.toString('base64') } }));
    }, 20);
  }

  _flushQueue() {
    for (const payload of this._outQueue) {
      payload.streamSid = this.streamSid;
      this.twilioWs.send(JSON.stringify(payload));
    }
    this._outQueue = [];
  }

  close() {
    try {
      this.twilioWs.close();
    } catch {
      // already closed
    }
  }
}
