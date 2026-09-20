export class ReadAloudError extends Error {}
export class ApiError extends ReadAloudError { status?: number; constructor(message: string, status?: number); }
export class AuthError extends ApiError {}
export class QuotaError extends ApiError {}
export class CapacityError extends ApiError { retryAfter?: number; }
export class VoiceError extends ApiError {}

export type Engine = 'piper' | 'kokoro';
export type AudioFormat = 'pcm_24000' | 'pcm_8000' | 'mulaw_8000' | 'alaw_8000' | (string & {});
/** 'default', a built-in voice name, or 'custom:<id>'. */
export type Voice = 'default' | `custom:${string}` | (string & {});

export interface ReadAloudOptions {
  apiKey: string;
  engine?: Engine;
  apiBase?: string;
  /** Override for testing / non-global environments. */
  fetch?: typeof fetch;
  WebSocket?: typeof WebSocket;
}

export interface SynthesizeOptions {
  voice?: Voice;
  speed?: number;
  format?: AudioFormat;
  signal?: AbortSignal;
}

export interface Authorization { token: string; url: string; http_url?: string; }

export class ReadAloud {
  constructor(opts: ReadAloudOptions);
  authorize(signal?: AbortSignal): Promise<Authorization>;
  stream(text: string, opts?: SynthesizeOptions): AsyncGenerator<Uint8Array, void, undefined>;
  /** HTTP chunked streaming (Piper only; throws ApiError if the server offers no http_url). */
  streamHttp(text: string, opts?: SynthesizeOptions): AsyncGenerator<Uint8Array, void, undefined>;
  convert(text: string, opts?: SynthesizeOptions): Promise<Uint8Array>;
}

export function pcmToWav(pcm: Uint8Array, sampleRate?: number, channels?: number): Uint8Array;
