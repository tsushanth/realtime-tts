# Telnyx carrier option (v1, opt-in per phone number)

Status: implemented and unit/smoke tested locally. **Not deployed, and never exercised on a live call.** Whether a real
caller hears the agent over Telnyx has not been verified (see "Not verified" below).

## How it works

A phone number is served by exactly one carrier. The default is Twilio and nothing about the Twilio path changes.

```
Twilio:  caller -> Twilio number -> POST /twilio/voice  -> <Connect><Stream wss://host/twilio-stream> -> TwilioCallAdapter
Telnyx:  caller -> Telnyx number -> POST /telnyx/voice  -> <Connect><Stream wss://host/telnyx-stream?cs&exp&sig bidirectionalMode=rtp> -> TelnyxCallAdapter
```

Both adapters present the same interface to `CallSession` (`.send`, `.readyState`, `message`/`close`/`dtmf`/`start` events,
`clearQueue()`, `isSpeaking()`). `TelnyxCallAdapter` is a subclass of `TwilioCallAdapter`: audio handling (mu-law 8 kHz,
160 byte / 20 ms frames, PCM16 fallback resampler, drain-before-close) is shared, only the wire protocol is overridden.
`twilioAdapter.js` is unmodified. No separate shared codec module was extracted because subclassing needs none.

Carrier selection: `calldesk_phone_numbers.carrier` (`'twilio'` default, `'telnyx'`). `resolveInboundCall()` reads it only
when `TELNYX_ENABLED=1`, using the same retry pattern as `llm_model`: if the query naming `carrier` fails (column not yet
migrated) it is retried without it and the number is treated as Twilio. `/telnyx/voice` serves a call only when the resolved
carrier is `telnyx`; any other number gets `<Hangup/>` (it never silently runs over the wrong carrier). The Twilio route
does not look at `carrier`.

## Environment variables (all optional; nothing set = unchanged Twilio-only behaviour)

| Var | Purpose |
|---|---|
| `TELNYX_ENABLED=1` | Mounts `POST /telnyx/voice` and the `/telnyx-stream` websocket path, and makes tenant lookup read `carrier`. |
| `TELNYX_PUBLIC_KEY` | Base64 Ed25519 public key from the Telnyx portal. When set, `/telnyx/voice` requires a valid `telnyx-signature-ed25519` over `${telnyx-timestamp}|${raw body}` (5 min tolerance). |
| `TELNYX_WEBHOOK_SECRET` | Shared secret appended to the webhook URL as `?key=...`; compared in constant time. When both this and the public key are set, both must pass. |
| `TELNYX_STREAM_SECRET` | HMAC secret for the signed websocket URL (falls back to `TELNYX_WEBHOOK_SECRET`, then to a random per-process secret, which only works on a single machine; set it if you run more than one). |
| `TELNYX_PUBLIC_HOST` | Host to put in the `wss://` URL (defaults to the request `Host` header). |

At least one of `TELNYX_PUBLIC_KEY` / `TELNYX_WEBHOOK_SECRET` must be set or `/telnyx/voice` answers 503 (fail closed).
Telnyx documents Ed25519 signing for its webhooks but I could not confirm that TeXML application webhooks carry those
headers, so the shared-secret option exists as the dependable alternative. The websocket URL is separately protected by an
HMAC (`cs` = call id, `exp` = now+120 s, `sig`), so a socket can only be opened for a call the webhook accepted.

## Switching a number to Telnyx

1. Apply the migration below (or skip it: until it exists no number can be `telnyx`).
2. `update calldesk_phone_numbers set carrier = 'telnyx' where number = '+1XXXXXXXXXX';` (the number must be a number you
   own on Telnyx, with the agent version routed exactly as for Twilio).
3. Back to Twilio: set it to `'twilio'` or null.

Draft migration (NOT applied):

```sql
alter table calldesk_phone_numbers
  add column if not exists carrier text
  check (carrier is null or carrier in ('twilio', 'telnyx'));
-- null is treated as 'twilio'. No default and no backfill, so existing rows are unchanged.
```

## Telnyx-side setup

1. Portal > Voice > Programmable Voice > TeXML Applications > create an application.
2. Voice webhook URL: `https://<your-host>/telnyx/voice?key=<TELNYX_WEBHOOK_SECRET>`, method POST. (If you rely on
   `TELNYX_PUBLIC_KEY` instead, copy the public key from the portal's API keys page into the env var.)
3. Assign the Telnyx number to that TeXML application (Numbers > the number > Voice settings).
4. Set the Fly/host env vars above, set the number's `carrier` to `telnyx`, then place one real test call (see next steps).

## What the TeXML returns

```xml
<Response><Connect>
  <Stream url="wss://HOST/telnyx-stream?cs=CALLSID&amp;exp=...&amp;sig=..." bidirectionalMode="rtp" bidirectionalCodec="PCMU">
    <Parameter name="callSid" value="CALLSID"/>
  </Stream></Connect><Hangup/></Response>
```

## Feature support: Telnyx v1 vs Twilio

| Feature (node / tool) | Twilio | Telnyx v1 | Telnyx behaviour |
|---|---|---|---|
| Inbound conversation (STT, LLM, TTS, barge-in, flows, knowledge base, function/code/MCP/calendar, agent_transfer, jingle/SFX) | yes | yes (adapter + smoke tested, not live) | same engine |
| Caller DTMF into the flow (`dtmf` event) | yes | yes | same |
| Barge-in / flush queued audio | local queue drop | local drop + `{"event":"clear"}` | |
| `transfer` node and `transfer` subagent tool | yes (Twilio REST `<Dial>`) | **no** | logs "transfer is not supported on telnyx yet", then hangs up (the same outcome as the existing "cannot transfer" branch) |
| `payment` node (`<Pay>`) | yes | **no** | logs; sets `payment_status=failed`, `payment_status_detail=unsupported_on_telnyx` and re-runs the node turn so the flow can tell the caller and route on |
| `press_digit` node (send DTMF to another system) | yes | **no** | logs; skips tones and advances along the node's first edge |
| `sms` node and `sms` subagent tool | yes (Twilio Messages API) | **no** | logs and skips |
| Call recording / retention | yes | **no** | logs "not supported on telnyx yet"; call is not recorded |
| `_redirectForDetour` / session resume | yes | **no** | returns false |
| Outbound calls, batch calling, test calls, shopper/demo/sample-callee modes, number purchase | yes | **no** (inbound only; those routes are Twilio-only and untouched) | n/a |
| Stripe usage metering, call log, post-call analysis, webhooks | yes | yes (call id = TeXML `CallSid`; the call log row does not record the carrier) | |
| Cost: telephony line | none (unchanged) | yes: `$0.0032 + $0.002 + $0.0035` per min, 60 s minimum on the carrier leg | `CallCostTracker.setCarrier('telnyx')` |

Cost note: the 60 s minimum is applied as `max(call seconds, 60)` billed per minute-fraction; whether Telnyx bills per second
after the minimum is an assumption.

## Wire protocol used (Telnyx -> us / us -> Telnyx)

- in: `connected`, `start` (`stream_id`, `start.call_control_id`, `start.call_session_id`, `start.media_format`), `media`
  (`media.track`, `media.payload` base64 PCMU), `dtmf` (`dtmf.digit`), `mark` (`mark.name`), `stop`, `error`
  (`payload.code/title/detail`: 100002 unknown, 100003 malformed frame, 100004 invalid media, 100005 rate limit).
- out: `{"event":"media","media":{"payload":"<b64 PCMU, 160 B>"}}` paced every 20 ms, `{"event":"clear"}`,
  `{"event":"mark","mark":{"name":..}}` (available as `adapter.sendMark`, unused by CallSession today).
- Frames from `track` other than `inbound` are dropped; malformed JSON and unknown events are ignored; a send on a dead
  socket or a socket buffer over 64 KB skips ticks instead of throwing.

## Sources

- https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming (event shapes, clear/mark semantics, codecs, chunk sizes)
- https://developers.telnyx.com/docs/voice/programmable-voice/texml-verbs/stream (Stream attributes `bidirectionalMode` mp3|rtp,
  `bidirectionalCodec`, `bidirectionalSamplingRate`, `track`, `statusCallback`, `enableReconnect`; `Parameter name/value`)
- https://developers.telnyx.com/docs/development/api-fundamentals/webhooks/receiving-webhooks (Ed25519 headers, `timestamp|body`,
  public key in the portal, "TeXML callbacks can be form-encoded")
- https://developers.telnyx.com/docs/voice/programmable-voice/voice-api-webhooks
- Measured 2026-10-02 (inbound test, not from docs): start event shape, 20 ms / 160 B PCMU inbound media at about 49 frames/s,
  raw base64 PCMU 160 B outbound frames accepted without error.

## Not verified (read before trusting this)

1. **A real caller hearing the agent over Telnyx has not been tested.** Outbound frames were accepted without error in the
   2026-10-02 test but audibility was not checked. Telnyx docs describe the rtp-mode payload as a "base64 encoded RTP
   stream"; the adapter sends raw PCMU bytes (what the measured test did). If the caller hears nothing or noise, the payload
   format is the first suspect.
2. The docs say "Media payloads can only be submitted once per second" and chunks may be 20 ms to 30 s, which reads like the
   mp3 mode rule; sending 160 B every 20 ms was accepted in the test but could trip error 100005 under load or in a long call.
   A `TELNYX_OUT_CHUNK` coalescing option was not built.
3. Whether the query string on the `<Stream url>` is preserved on the websocket upgrade, and under which key `<Parameter>`
   values appear in the `start` message, is not documented. The adapter prefers the signed query (`cs`), then looks for
   `callSid` in `start.custom_parameters` / `customParameters` / `stream_params`, then falls back to `call_control_id`
   (which would NOT match the stored context, so the call would run without tenant context). The 2026-10-02 test used a
   `?hit=` query, suggesting the query survives, but this was not re-checked here. Note the upgrade is rejected (401) if the
   query is lost.
4. Whether TeXML application webhooks are Ed25519 signed is not confirmed (hence `TELNYX_WEBHOOK_SECRET`). Telnyx's
   documented signed message is over the raw body; the route preserves raw bytes, but signature verification here is tested
   only against a locally generated key, not a Telnyx-produced signature.
5. Whether closing the websocket (or the trailing `<Hangup/>`) ends the PSTN leg promptly is untested; behaviour of the
   `stop` event on hangup (always sent?) is untested, the adapter also closes on socket close.
6. `clear` semantics are from the docs (stops playback, clears the queue, returns queued marks) but untested live.
7. Telnyx `L16`/`OPUS`/other codecs are not supported by this adapter (PCMU only); the docs list them for the
   Programmable Voice API and the TeXML attribute list differs, so codec options were not pursued.
8. Telnyx billing figures in `costTracker.js` were supplied by the requester, not re-verified against Telnyx's pricing page.
9. Multi-machine deployments need `TELNYX_STREAM_SECRET` set (the random fallback differs per process).

## Tests

- `test/telnyxAdapter.test.js`: fake-websocket adapter tests (decode, 160 B/20 ms pacing, clear, backpressure, dtmf, mark, stop, malformed frames, drain-before-close).
- `test/telnyxRouting.test.js`: routing (telnyx only when enabled + carrier=telnyx, twilio/null/missing column refused), webhook auth, Ed25519 verification, signed stream URL.
- `test/telnyxDisabled.test.js`: default behaviour (route absent, original query, original TwiML).
- `test/telnyxSoftFail.test.js`: unsupported features soft-fail and never call Twilio; carrier-aware cost.
- `scripts/telnyxSmoke.mjs`: local end-to-end smoke (real `server.js` on a spare port, fake Supabase, fake Telnyx client). With no STT/LLM/TTS
  keys the session cannot reply to speech, so the audio written back is the tenant's intro jingle from the fake database; caller audio reaches the session but is not transcribed.
