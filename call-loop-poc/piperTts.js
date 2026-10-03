// Direct WebSocket client for the owned Piper TTS service (Fly app piper-tts-sjc, source: realtime-tts/worker-piper-fly/server.py).
//
// Protocol (read from that source):
//   connect  wss://<host>/tts   with  Authorization: Bearer <static AUTH_TOKEN>
//   client -> {"type":"synthesize","text":"..","voice":"custom:en-us-ljspeech","speed":1.0,"format":"mulaw_8000"|"pcm_24000"}
//   client -> {"type":"stop"}                        (cancel the in-flight synth)
//   server -> {"type":"chunk_meta",...} then ONE binary frame per Piper sentence
//   server -> {"type":"done"} | {"type":"cancelled"} | {"type":"error","message":".."}
// A socket serves one synth at a time and can be reused for the next request. The server counts open sockets against MAX_CONNECTIONS
// (beyond it: an {"type":"error"} message and close 1013), so this module caps open sockets per engine instance (PIPER_MAX_STREAMS).
//
// Env (read lazily, never hardcoded secrets):
//   PIPER_TTS_TOKEN                 bearer token (required; the backend is treated as unconfigured without it)
//   PIPER_TTS_URL                   default wss://piper-tts-sjc.fly.dev/tts  (a Fly private-network URL also works)
//   PIPER_MAX_STREAMS               default 8   open sockets (busy + warm idle) per engine instance; overflow -> caller falls back
//   PIPER_CONNECT_TIMEOUT_MS        default 1500
//   PIPER_FIRST_AUDIO_TIMEOUT_MS    default 4000  request sent -> first binary frame
//   PIPER_REQUEST_TIMEOUT_MS        default 15000 request sent -> done
//   PIPER_IDLE_MS                   default 20000 how long a warm idle socket is kept
import WebSocket from 'ws';

export const PIPER_DEFAULT_URL = 'wss://piper-tts-sjc.fly.dev/tts';

const num = (v, d) => { const n = Number(v); return Number.isFinite(n) && n > 0 ? n : d; };

export function piperConfig(env = process.env) {
  return {
    url: env.PIPER_TTS_URL || PIPER_DEFAULT_URL,
    token: env.PIPER_TTS_TOKEN || '',
    maxStreams: Math.floor(num(env.PIPER_MAX_STREAMS, 8)),
    connectTimeoutMs: num(env.PIPER_CONNECT_TIMEOUT_MS, 1500),
    firstAudioTimeoutMs: num(env.PIPER_FIRST_AUDIO_TIMEOUT_MS, 4000),
    requestTimeoutMs: num(env.PIPER_REQUEST_TIMEOUT_MS, 15000),
    idleMs: num(env.PIPER_IDLE_MS, 20000),
  };
}

export const piperConfigured = (env = process.env) => !!env.PIPER_TTS_TOKEN;

export class PiperError extends Error {
  constructor(code, message) { super(message); this.name = 'PiperError'; this.code = code; }
}
const abortError = () => { const e = new Error('aborted'); e.name = 'AbortError'; return e; };

// Open sockets (busy + warm idle) are capped at cfg.maxStreams. Busy streams are never evicted; a warm idle socket of any session is
// closed to make room. `cfg` is a function so tests/env changes are honoured.
export class PiperPool {
  constructor(getCfg = piperConfig) {
    this.getCfg = getCfg;
    this.busy = 0;
    this.idle = new Map(); // sessionKey -> [{ ws, timer }]
  }

  get idleCount() { let n = 0; for (const l of this.idle.values()) n += l.length; return n; }
  get open() { return this.busy + this.idleCount; }

  _dropIdle(sessionKey, entry) {
    clearTimeout(entry.timer);
    const l = this.idle.get(sessionKey);
    if (l) { const i = l.indexOf(entry); if (i >= 0) l.splice(i, 1); if (!l.length) this.idle.delete(sessionKey); }
    try { entry.ws.close(); } catch { /* already closed */ }
  }

  // Reserves a slot and returns { ws, reused }, or throws PiperError('overflow' | 'connect').
  async acquire(sessionKey) {
    const cfg = this.getCfg();
    if (this.busy >= cfg.maxStreams) throw new PiperError('overflow', `piper stream cap reached (${cfg.maxStreams})`);
    const list = this.idle.get(sessionKey);
    while (list && list.length) {
      const entry = list.pop();
      clearTimeout(entry.timer);
      if (!list.length) this.idle.delete(sessionKey);
      if (entry.ws.readyState === WebSocket.OPEN) { this.busy++; return { ws: entry.ws, reused: true }; }
    }
    // evict warm idle sockets (oldest session first) until a new socket fits under the cap
    while (this.open >= cfg.maxStreams && this.idleCount > 0) {
      const [k, l] = this.idle.entries().next().value;
      this._dropIdle(k, l[0]);
    }
    this.busy++;
    try {
      return { ws: await this._connect(cfg), reused: false };
    } catch (err) {
      this.busy--;
      throw err;
    }
  }

  _connect(cfg) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const ws = new WebSocket(cfg.url, { headers: { Authorization: `Bearer ${cfg.token}` }, handshakeTimeout: cfg.connectTimeoutMs });
      ws.binaryType = 'nodebuffer';
      const fail = (msg) => {
        if (settled) return;
        settled = true; clearTimeout(timer);
        try { ws.terminate(); } catch { /* ignore */ }
        reject(new PiperError('connect', msg));
      };
      const timer = setTimeout(() => fail('connect timeout'), cfg.connectTimeoutMs);
      ws.once('open', () => { if (settled) return; settled = true; clearTimeout(timer); resolve(ws); });
      ws.once('error', (e) => fail(`connect error: ${e.message}`));
      ws.once('close', (code) => fail(`closed before open (${code})`));
    });
  }

  // Returns the slot. reusable=true parks the socket warm for the session; otherwise it is closed.
  release(sessionKey, ws, reusable) {
    this.busy = Math.max(0, this.busy - 1);
    if (!reusable || ws.readyState !== WebSocket.OPEN) { try { ws.close(); } catch { /* ignore */ } return; }
    const cfg = this.getCfg();
    const entry = { ws, timer: null };
    entry.timer = setTimeout(() => this._dropIdle(sessionKey, entry), cfg.idleMs);
    entry.timer.unref?.();
    ws.once('close', () => this._dropIdle(sessionKey, entry));
    if (!this.idle.has(sessionKey)) this.idle.set(sessionKey, []);
    this.idle.get(sessionKey).push(entry);
  }

  // Opens a warm socket ahead of the first utterance. Best effort: never throws, never exceeds the cap.
  async prewarm(sessionKey) {
    try {
      if (this.idle.get(sessionKey)?.length) return;
      const lease = await this.acquire(sessionKey);
      this.release(sessionKey, lease.ws, true);
    } catch { /* prewarm is optional */ }
  }

  closeSession(sessionKey) {
    for (const entry of [...(this.idle.get(sessionKey) || [])]) this._dropIdle(sessionKey, entry);
  }
}

export const piperPool = new PiperPool();

// Synthesizes `text`, calling onChunk(Buffer) per binary frame in order. Resolves { frames, firstAudioMs, reused }.
// Throws PiperError (overflow | connect | timeout | server | closed) or an AbortError when `signal` fires (a `stop` is sent and the
// socket is closed, not reused). err.frames = frames already delivered before the failure.
export async function piperSynthesize({ pool = piperPool, sessionKey, text, voice, format, signal, onChunk, cfg = pool.getCfg() }) {
  if (signal?.aborted) throw abortError();
  const lease = await pool.acquire(sessionKey);
  const { ws } = lease;
  let frames = 0;
  let firstAudioMs = null;
  const t0 = Date.now();
  return new Promise((resolve, reject) => {
    let finished = false;
    let firstTimer = null;
    let reqTimer = null;
    const finish = (err, reusable) => {
      if (finished) return;
      finished = true;
      clearTimeout(firstTimer); clearTimeout(reqTimer);
      ws.off('message', onMessage); ws.off('close', onClose); ws.off('error', onError);
      signal?.removeEventListener('abort', onAbort);
      pool.release(sessionKey, ws, reusable);
      if (err) { err.frames = frames; reject(err); } else resolve({ frames, firstAudioMs, reused: lease.reused });
    };
    const onAbort = () => {
      try { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'stop' })); } catch { /* ignore */ }
      finish(abortError(), false);
    };
    const onClose = (code) => finish(new PiperError('closed', `socket closed mid-request (${code})`), false);
    const onError = (e) => finish(new PiperError('closed', `socket error: ${e.message}`), false);
    const onMessage = (data, isBinary) => {
      if (isBinary) {
        if (frames === 0) { firstAudioMs = Date.now() - t0; clearTimeout(firstTimer); }
        frames++;
        try { onChunk(Buffer.from(data)); } catch (e) { finish(e, false); }
        return;
      }
      let msg;
      try { msg = JSON.parse(data.toString()); } catch { return; }
      if (msg.type === 'done') finish(null, true);
      else if (msg.type === 'error') finish(new PiperError('server', `piper error: ${msg.message || 'unknown'}`), false);
      else if (msg.type === 'cancelled') finish(new PiperError('server', 'piper cancelled the request'), false);
      // chunk_meta: informational
    };
    ws.on('message', onMessage); ws.on('close', onClose); ws.on('error', onError);
    signal?.addEventListener('abort', onAbort, { once: true });
    firstTimer = setTimeout(() => finish(new PiperError('timeout', `no audio within ${cfg.firstAudioTimeoutMs}ms`), false), cfg.firstAudioTimeoutMs);
    reqTimer = setTimeout(() => finish(new PiperError('timeout', `request exceeded ${cfg.requestTimeoutMs}ms`), false), cfg.requestTimeoutMs);
    try {
      ws.send(JSON.stringify({ type: 'synthesize', text, voice, speed: 1.0, format }));
    } catch (e) {
      finish(new PiperError('closed', `send failed: ${e.message}`), false);
    }
  });
}
