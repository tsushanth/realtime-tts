// ReadAloud JS client. Zero deps; needs global fetch + WebSocket (Node >= 22, browsers).

export class ReadAloudError extends Error {
  constructor(message) { super(message); this.name = new.target.name; }
}
export class ApiError extends ReadAloudError {
  constructor(message, status) { super(message); this.status = status; }
}
export class AuthError extends ApiError {}
export class QuotaError extends ApiError {}
export class CapacityError extends ApiError {
  constructor(message = 'at capacity, retry shortly', status, retryAfter) {
    super(message, status);
    this.retryAfter = retryAfter;
  }
}
export class VoiceError extends ApiError {}

function fromMessage(message) {
  const low = String(message).toLowerCase();
  if (low.includes('capacity')) return new CapacityError(message);
  if (low.includes('unknown voice')) return new VoiceError(message);
  return new ApiError(message);
}

function fromStatus(status, message, retryAfter) {
  if (status === 401) return new AuthError(message, status);
  if (status === 402) return new QuotaError(message, status);
  if (status === 503) return new CapacityError(message, status, retryAfter);
  if ((status === 400 || status === 404) && /voice/i.test(message)) return new VoiceError(message, status);
  return new ApiError(message, status);
}

async function errText(res) {
  const text = await res.text().catch(() => '');
  try { const b = JSON.parse(text); return String(b.error ?? b.message ?? text); } catch { return text.slice(0, 200) || `HTTP ${res.status}`; }
}

const abortError = (signal) => signal.reason ?? new DOMException('Aborted', 'AbortError');

export class ReadAloud {
  /** @param {{apiKey: string, engine?: string, apiBase?: string, fetch?: typeof fetch, WebSocket?: any}} opts */
  constructor({ apiKey, engine = 'piper', apiBase = 'https://api.readaloudai.org', fetch: f, WebSocket: WS } = {}) {
    if (!apiKey) throw new TypeError('apiKey is required');
    this.apiKey = apiKey;
    this.engine = engine;
    this.apiBase = apiBase.replace(/\/+$/, '');
    this._fetch = f ?? ((...a) => globalThis.fetch(...a));
    this._WS = WS ?? globalThis.WebSocket;
  }

  /** Exchange the API key for a short-lived token: {token, url, http_url?} */
  async authorize(signal) {
    const res = await this._fetch(`${this.apiBase}/tts/authorize`, {
      method: 'POST', signal,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ key: this.apiKey, engine: this.engine }),
    });
    if (!res.ok) throw fromStatus(res.status, await errText(res));
    return res.json();
  }

  /** Async iterable of Uint8Array PCM chunks. Abort the signal (or `break`) to cancel. */
  async *stream(text, { voice = 'default', speed = 1.0, format = 'pcm_24000', signal } = {}) {
    signal?.throwIfAborted();
    const auth = await this.authorize(signal);
    yield* this._wsStream(auth, text, { voice, speed, format, signal });
  }

  async *_wsStream(auth, text, { voice, speed, format, signal }) {
    signal?.throwIfAborted();
    const ws = new this._WS(`${auth.url}?token=${encodeURIComponent(auth.token)}`);
    ws.binaryType = 'arraybuffer';

    const queue = [];           // items: {chunk} | {done} | {error}
    let wake = null;
    const push = (item) => { queue.push(item); wake?.(); wake = null; };
    let finished = false;
    const finish = (item) => { if (!finished) { finished = true; push(item); } };

    ws.onopen = () => ws.send(JSON.stringify({ type: 'synthesize', text, voice, speed, format }));
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return push({ chunk: new Uint8Array(ev.data) });
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === 'error') finish({ error: fromMessage(m.message ?? 'unknown error') });
      else if (m.type === 'done' || m.type === 'cancelled') finish({ done: true });
    };
    ws.onerror = () => finish({ error: new ApiError('WebSocket error') });
    ws.onclose = (ev) => finish(ev.code === 1013 ? { error: new CapacityError() }
      : { error: new ApiError(`connection closed unexpectedly (code ${ev.code})`) });
    const onAbort = () => finish({ error: abortError(signal) });
    signal?.addEventListener('abort', onAbort, { once: true });

    try {
      for (;;) {
        while (!queue.length) await new Promise((r) => { wake = r; });
        const item = queue.shift();
        if (item.chunk) yield item.chunk;
        else if (item.error) throw item.error;
        else return;
      }
    } finally {
      signal?.removeEventListener('abort', onAbort);
      try { if (ws.readyState === 1) ws.send(JSON.stringify({ type: 'stop' })); } catch {}
      try { ws.close(); } catch {}
    }
  }

  async _httpResponse(auth, text, { voice, speed, format, signal }) {
    if (!auth.http_url) throw new ApiError('HTTP streaming is not available for this engine; use stream()');
    const res = await this._fetch(auth.http_url, {
      method: 'POST', signal,
      headers: { authorization: `Bearer ${auth.token}`, 'content-type': 'application/json' },
      body: JSON.stringify({ text, voice, speed, format }),
    });
    if (!res.ok) {
      const ra = Number(res.headers.get('retry-after'));
      throw fromStatus(res.status, await errText(res), Number.isFinite(ra) && ra > 0 ? ra : undefined);
    }
    return res;
  }

  /**
   * Stream audio bytes from the HTTP streaming endpoint (POST http_url, Bearer token, chunked
   * response). Piper only; throws ApiError otherwise. Chunks are arbitrary byte slices, not
   * sentence-aligned. `break` or abort cancels the request.
   */
  async *streamHttp(text, { voice = 'default', speed = 1.0, format = 'pcm_24000', signal } = {}) {
    const res = await this._httpResponse(await this.authorize(signal), text, { voice, speed, format, signal });
    if (!res.body) { yield new Uint8Array(await res.arrayBuffer()); return; }
    const reader = res.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) return;
        if (value?.length) yield value;
      }
    } finally {
      try { await reader.cancel(); } catch {}
    }
  }

  /** Whole clip as one Uint8Array. Uses the HTTP endpoint when offered (Piper), else the WebSocket. */
  async convert(text, { voice = 'default', speed = 1.0, format = 'pcm_24000', signal } = {}) {
    const auth = await this.authorize(signal);
    const o = { voice, speed, format, signal };
    if (!auth.http_url) return concat(await collect(this._wsStream(auth, text, o)));
    const res = await this._httpResponse(auth, text, o);
    return new Uint8Array(await res.arrayBuffer());
  }
}

async function collect(it) { const out = []; for await (const c of it) out.push(c); return out; }

function concat(chunks) {
  const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/** Wrap raw PCM16LE in a WAV container. */
export function pcmToWav(pcm, sampleRate = 24000, channels = 1) {
  const out = new Uint8Array(44 + pcm.length);
  const v = new DataView(out.buffer);
  const tag = (o, s) => { for (let i = 0; i < 4; i++) out[o + i] = s.charCodeAt(i); };
  tag(0, 'RIFF'); v.setUint32(4, 36 + pcm.length, true); tag(8, 'WAVE'); tag(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, channels, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * channels * 2, true);
  v.setUint16(32, channels * 2, true); v.setUint16(34, 16, true);
  tag(36, 'data'); v.setUint32(40, pcm.length, true);
  out.set(pcm, 44);
  return out;
}
