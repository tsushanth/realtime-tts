# call-loop-poc

Retell-style low-latency voice call loop, built on top of this repo's own realtime TTS
(see `../README.md` / `../DECISIONS.md`). Adds the three pieces that weren't built yet:
streaming STT (Deepgram), streaming LLM (Anthropic Claude), and orchestration
(sentence-boundary chunking so TTS starts before the LLM finishes the full reply, plus
barge-in that kills in-flight audio the instant the caller starts talking again).

```
browser mic --WS--> this server --WS--> Deepgram (STT)
                                --HTTP streaming--> Anthropic (LLM)
                                --WS--> realtime-tts gateway (TTS, this repo's ../gateway)
```

## Run

1. Have a TTS backend reachable — either:
   - local dev worker: `cd ../worker && python server.py` (ws://127.0.0.1:8765), then
     `cd ../gateway && WORKER_WS_URL=ws://127.0.0.1:8765 node server.js` (gateway on :8080), or
   - the live gateway: `TTS_GATEWAY_WS_URL=wss://realtime-tts-gateway.fly.dev/tts` (CPU
     Serverless fallback unless the GPU pod is on — see `../README.md` "Current real status").

2. `DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=... npm start` (defaults to the local gateway
   at `ws://127.0.0.1:8080/tts` on port 8090).

3. Open `http://localhost:8090`, click "Start call", allow mic access, talk. Interrupt the
   assistant mid-reply to test barge-in — it should cut off within one Deepgram VAD tick.

## Per-call context (for embedding this in a multi-tenant product)

By default every call gets the same hardcoded assistant (`SYSTEM_PROMPT` / `TTS_VOICE`
constants in `server.js`). To give a specific call its own business prompt, voice, and
opening line — e.g. one call-loop-poc server handling calls for many different tenants —
send this as the very first message after the WS connects, before any audio:

```json
{ "type": "context", "systemPrompt": "You are the AI receptionist for Bright Smile Dental...",
  "voice": "af_heart", "greeting": "Thanks for calling Bright Smile Dental, how can I help?" }
```

`voice` must be a valid voice id for whatever TTS backend the gateway is pointed at
(Kokoro voice ids look like `af_heart`, `am_adam`). `greeting` is optional — if sent, the
assistant speaks it immediately as turn 1, before the caller says anything, and it's a
real interruptible turn (barge-in works on it same as any other response). Sending
`context` after the first turn still updates the prompt/voice going forward, but won't
retroactively change anything already said.

## What's measured

Server console logs two latency numbers per turn:
- `LLM TTFB` — time from end-of-user-speech to the first LLM token
- `TTS TTFB` — time from end-of-user-speech to the first audio chunk_meta back from the
  TTS gateway (i.e. the number that actually matters for "does this feel like a real call")

## Known POC gaps (deliberately not built)

- No telephony (Twilio) — browser mic only. Adding phone calls means SIP/PSTN ingress
  feeding the same `/call` WS in mulaw@8kHz instead of the browser's PCM16@16kHz — a
  format/resample change, not an architecture change.
- No conversation memory beyond the in-process `session.history` array — resets on
  reconnect.
- No interruption grace period — barge-in fires on Deepgram's `SpeechStarted` VAD event
  immediately, which is aggressive; real products tune this (a brief "let them finish"
  window) — see the caveat about turn-taking tuning in the original response.
- Uses `ScriptProcessorNode` (deprecated but universally supported) for mic capture
  instead of an AudioWorklet — fine for a POC, worth swapping for production.
