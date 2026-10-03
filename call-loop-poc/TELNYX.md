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
| `TELNYX_PUBLIC_HOST` | Host to put in the `wss://` URL (defaults to the request `Host` header). Also the host of the recording callback (falls back to `PUBLIC_HOST`). |
| `TELNYX_API_KEY` | Telnyx v2 API key (Bearer). Needed to start, play back and delete recordings. Never sent to the browser or to non-api.telnyx.com hosts. Unset = recording soft-fails. |
| `TELNYX_ACCOUNT_SID` | Optional override for the account id used in the TeXML REST path; default is the `AccountSid` Telnyx sends with the voice webhook. |

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

## Call recording

Gate (identical to Twilio): `RECORD_REAL_CALLS` is not `false` and the tenant's `recording_enabled` setting is not `'false'`;
plus `TELNYX_ENABLED=1`, `TELNYX_API_KEY`, a Telnyx account id and webhook auth configured. Otherwise nothing is sent to Telnyx.

- Start (at stream start, non-blocking): `POST https://api.telnyx.com/v2/texml/Accounts/{AccountSid}/Calls/{CallSid}/Recordings.json`
  form-encoded with `RecordingChannels=dual`, `RecordingTrack=both`, `PlayBeep=false` (Telnyx's default is a beep; Twilio is silent),
  `RecordingStatusCallback=https://<host>/telnyx/recording-status?key=<TELNYX_WEBHOOK_SECRET>`, event `completed`. 5 s timeout,
  up to 3 attempts on network errors / 429 / 5xx (not on timeouts, which may already have started a recording, and not on 4xx).
  Any failure only logs; the call is never affected.
- Completed webhook: `POST /telnyx/recording-status` uses the same auth as `/telnyx/voice` (Ed25519 and/or `?key=`). Writes
  `recording_url = https://api.telnyx.com/v2/recordings/<RecordingSid>` and `recording_sid = telnyx:<RecordingSid>` on the call log row
  (same columns as Twilio). Telnyx's own download link is deliberately not stored: it expires (10 minutes). A repeated delivery
  is acknowledged without a second write; a DB write error returns 500 so Telnyx can retry.
- Playback: `GET /recording-audio?url=<recording_url>` (same admin bearer as today). A `https://api.telnyx.com/v2/recordings/<id>` url
  is resolved server-side via `GET /v2/recordings/{id}` to a fresh `download_urls.mp3` and streamed; the API key is only sent to api.telnyx.com.
- Retention: the existing 6-hourly sweep deletes `telnyx:` rows with `DELETE /v2/recordings/{id}` (404 = already gone) and then clears
  `recording_url` / `recording_sid`, exactly like Twilio. A failed delete keeps the reference for the next sweep.
- Still not supported on Telnyx: nothing recording-related is left unsupported except the no-key / no-account-id / no-webhook-auth
  cases above (soft-fail, logged). Recording of Twilio-placed calls, outbound Telnyx calls and transfers is out of scope (no Telnyx outbound).

Portal setup: nothing extra. The callback URL is supplied per recording in the start request, so no recording webhook needs to be
configured in the portal. Create an API key (Mission Control > Keys & Credentials > API Keys) and set it as `TELNYX_API_KEY`.
Optionally set the account's recording storage in the portal (default: Telnyx-managed storage).

### Live test plan (one real call)

1. Env on the host: `TELNYX_ENABLED=1`, `TELNYX_API_KEY`, `TELNYX_WEBHOOK_SECRET`, `TELNYX_PUBLIC_HOST` (public host), the usual Telnyx number setup; tenant recording setting on and a `recording_retention_days` of your choice.
2. Call the Telnyx number; talk for 15-20 s on both sides (say distinct words, let the agent answer), hang up.
3. Logs, during the call: expect `[telnyx] recording started for <CallSid>`. A line `recording not started ... http_NNN` means the start failed: 404 suggests the account id or call id in the path is wrong, 401/403 the key, 422 a parameter.
4. Logs, within about a minute after hangup: no `/telnyx/recording-status rejected` line. If absent entirely, the callback never arrived (check the `?key=` and public host).
5. DB: the `calldesk_call_logs` row for that CallSid has `recording_url = https://api.telnyx.com/v2/recordings/<uuid>` and `recording_sid = telnyx:<uuid>`.
6. Playback: `curl -H "Authorization: Bearer $TEST_CALL_SECRET" "https://<host>/recording-audio?url=<recording_url>" -o r.mp3`; open it: confirm it is a 2-channel file (caller on one channel, agent on the other), no beep at the start, audio complete. If the proxy returns 404 while the callback arrived, `RecordingSid` is not the v2 recording id (see Not verified #10).
7. Dashboard: the call's recording plays in the Calls page.
8. Retention: set `recording_retention_days` to 1, backdate that row's `created_at` by 2 days, restart (the sweep runs at boot) and confirm the row's recording columns are cleared and `GET /v2/recordings/<uuid>` returns 404 in Telnyx.

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
| Call recording / retention | yes | **yes, when `TELNYX_API_KEY` is set** (not live-tested) | same per-tenant setting and `RECORD_REAL_CALLS` as Twilio; dual channel, no beep; see "Call recording" below. Without `TELNYX_API_KEY` (or with no Telnyx account id) it logs "recording not started for <call>: <reason> - call continues unrecorded" and the call is unaffected |
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
- https://developers.telnyx.com/api-reference/texml-rest-commands/request-recording-for-a-call (start recording on a live call: params, defaults, `sid`)
- https://developers.telnyx.com/docs/voice/programmable-voice/texml-verbs/record (format mp3|wav, channels, "Recording URLs are valid for 10 minutes", callback fields)
- https://developers.telnyx.com/api-reference/call-recordings/retrieve-a-call-recording and .../delete-a-call-recording (GET/DELETE /recordings/{id}, `download_urls`)
- https://telnyx.com/pricing/voice-api (recording $0.002/min, storage $0)
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

10. Recording (all from developers.telnyx.com text, none exercised): (a) that the TeXML callback `RecordingSid` equals the v2 `/recordings/{id}` id used for playback and delete is an assumption; (b) that `CallSid` in the voice webhook is accepted as `{call_sid}` in the TeXML REST path and that `AccountSid` is present in the voice webhook; (c) whether the recording callback carries Ed25519 headers (the `?key=` secret is what protects it); (d) the exact callback payload (form-encoded fields `CallSid, RecordingSid, RecordingUrl, RecordingStatus, RecordingDuration, RecordingChannels` come from a search summary of the Record verb page, not the REST page); (e) the retrieve endpoint's `download_urls.mp3` shape and that its links expire is only documented for the Record verb (10 minutes); (f) whether recording a stream-connected (`<Connect><Stream>`) call captures both parties; (g) whether `PlayBeep=false` is honoured; (h) rate: Telnyx lists call recording at $0.002/min with $0 storage on telnyx.com/pricing/voice-api (not wired into `costTracker.js`, so call cost does not include it).
11. The sweep's delete query is limited to 500 rows oldest-first, shared between carriers (existing behaviour).

## Tests

- `test/telnyxRecording.test.js`: start request shape, soft failure/retry rules, webhook auth + idempotency + call-log write, retention routing/deletion, playback proxy, Twilio proxy unchanged.

- `test/telnyxAdapter.test.js`: fake-websocket adapter tests (decode, 160 B/20 ms pacing, clear, backpressure, dtmf, mark, stop, malformed frames, drain-before-close).
- `test/telnyxRouting.test.js`: routing (telnyx only when enabled + carrier=telnyx, twilio/null/missing column refused), webhook auth, Ed25519 verification, signed stream URL.
- `test/telnyxDisabled.test.js`: default behaviour (route absent, original query, original TwiML).
- `test/telnyxSoftFail.test.js`: unsupported features soft-fail and never call Twilio; carrier-aware cost.
- `scripts/telnyxSmoke.mjs`: local end-to-end smoke (real `server.js` on a spare port, fake Supabase, fake Telnyx client). With no STT/LLM/TTS
  keys the session cannot reply to speech, so the audio written back is the tenant's intro jingle from the fake database; caller audio reaches the session but is not transcribed.
