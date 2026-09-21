# Voice cloning via API key — DRAFT (not published, not announced)

Status: implemented behind a dark feature flag, on branch `voice-api` in both `realtime-tts`
(worktree `rt-voiceapi`) and `ReadAloudAI` (worktree `ra-voiceapi`). Not deployed to production.
Do not link this from the developers page or announce it until the open items at the bottom are
resolved by the owner.

## What this adds

Today, `voice: "custom:<id>"` voices can only be created by a signed-in website user going through
`backend/src/routes/voiceStudioRouter.ts` (Supabase auth). An API-key developer had no way in — the
developers page says "Early access — email us" for exactly this reason. This adds a second front
door that reaches the *same* business logic (consent recording, intake calls, upload chunking,
ownership, rate limits) via `Authorization: Bearer <api key>` instead of a browser session.

## Design

**Where the endpoints live:** on the gateway (`api.readaloudai.org`, realtime-tts's `gateway/`), which
validates the API key exactly like `/tts/authorize` does (`keys.isValidKey`, `keys.checkAccess`,
`keys.isBillingEnabled`), resolves an owning identity (`keys.getOwnerForKey(key) || keys.getIdForKey(key)`),
and forwards the request to a **new API-key auth path on the ReadAloudAI backend**
(`backend/src/routes/voiceStudioApiKey.ts`, mounted at `/internal/voice-studio-api`, reachable only
from the gateway via a shared secret). That path calls the *exact same* `createVoiceStudioRouter(...)`
function the Supabase web flow uses (`voiceStudioRouter.ts`) — same consent validation, same intake
calls, same upload-chunking code, same ownership/rate-limit rules — with a different `authenticate()`
and two new opt-in hooks (`featureFlagId`, `requireConsentStatement`) added to that shared router.

Why: the gateway already owns key validation and billing state; the backend already owns the
voice-studio business logic, consent records and intake wiring. Building this on the backend with a
second auth path (rather than re-implementing intake calls in the gateway, or re-validating API keys
on the backend) reuses the most and creates the least drift — a bug fix or gate change in the shared
router logic fixes both front doors at once. The gateway does not talk to intake directly.

```
API caller --Bearer <api key>--> gateway /v1/voices/*
  (gateway: isValidKey, isBillingEnabled, resolve identity)
  --forward w/ shared secret + identity headers--> backend /internal/voice-studio-api/*
    (backend: createVoiceStudioRouter — same code the web flow uses)
    --Bearer INTAKE_SECRET--> voice-pipeline intake.py (unchanged)
```

A voice created via the API and one created via the website are indistinguishable to the serving
layer: both end up as `owner_user_id` in intake's `consent.json` and `user_ids` in Piper's
`owner.json`, exactly as today.

## Endpoints (gateway, `api.readaloudai.org`)

All require `Authorization: Bearer <api key>` and a **billing-enabled** key (see Quotas below).

| Method | Path | Mirrors intake.py | Notes |
|---|---|---|---|
| POST | `/v1/voices` | `POST /voices` | body includes the consent object (below) |
| GET | `/v1/voices` | `GET /voices` | lists only voices owned by this key's resolved identity |
| GET | `/v1/voices/:id` | `GET /voices/{id}` | status: created/training/ready/rejected/deployed |
| PUT | `/v1/voices/:id/dataset` | `PUT /voices/{id}/dataset` | single-shot zip, small datasets only |
| PUT | `/v1/voices/:id/dataset/parts/:n` | `.../dataset/parts/{n}` | chunked upload, part n |
| GET | `/v1/voices/:id/dataset/parts` | `.../dataset/parts` | list uploaded parts |
| POST | `/v1/voices/:id/dataset/commit` | `.../dataset/commit` | join parts, start training |
| GET | `/v1/voices/:id/samples/:n` | `GET .../samples/{n}` | fixed preview sentence (0-4) |
| POST | `/v1/voices/:id/preview` | `POST .../preview` | `{text}` (<=300 chars) -> wav |
| POST | `/v1/voices/:id/deploy` | `POST .../deploy` | -> `{voice: "custom:<id>"}` |
| DELETE | `/v1/voices/:id` | `DELETE /voices/{id}` | revoke, removes from serving + storage |

Response shapes match `voiceStudioRouter.ts`'s `publicVoice()` (id, status, speaker_name, created_at,
warnings, stats, error, voice) — the same JSON a website user's dashboard gets, so one SDK type covers
both.

## Consent for an API caller

There's no UI click to stand in for "I attest...". `POST /v1/voices` body:

```json
{
  "speaker_name": "Jane Doe",
  "attested_by": "Jane Doe",
  "consent": true,
  "consent_text_version": "2026-09-v1",
  "consent_statement": "I am authorized to consent on behalf of the speaker named in this request, and that speaker has agreed to have their voice cloned and used to synthesize new speech through this service."
}
```

- `consent_text_version` — same version pin the web flow already used; reject if it's stale (the
  wording changed).
- `consent_statement` — **new for the API path only** (`requireConsentStatement: true`). The caller
  must echo this exact string verbatim, fetched fresh from `GET /v1/voices/enabled` right before
  showing/collecting it, not hardcoded — same principle as APIs that require echoing exact terms text
  to prove the caller actually saw it, not just typed a version number. `speaker_name` +
  `attested_by` (who is actually attesting — may differ from the account holder) are recorded with a
  server timestamp and, when available, the source IP, in the identical `consent.json` shape intake.py
  already writes (`recorded_at`, `client_ip`). This makes the artifact provably equivalent to the web
  flow's: same fields, same storage, same audit trail — just with `consent_statement` replacing the
  "I clicked a checkbox next to rendered text" step, and additionally requiring the exact wording to be
  echoed, which the checkbox flow gets for free from the DOM but an API call does not.

This is *a* reasonable design, not a legal sign-off — get counsel's eyes on whether echoing
`consent_statement` is sufficiently meaningful before this ships live, same caveat as the web flow's
"policy drafts need legal review" note.

## Upload mechanism

**Both**, chosen per size, same as the web client already does for the same underlying reasons
(Modal's ~150s timeout on big uploads, a 75MB zip measured at ~360KB/s = 178s earlier in this
project):

- **Small datasets (roughly under ~20-30MB, or a caller confident their connection won't take
  anywhere near 150s):** one `PUT /v1/voices/:id/dataset` with the zip as the raw body. Simplest
  path for a backend script on a fast connection.
- **Anything larger, or any caller that wants resumability:** the same chunked-parts protocol as the
  web client — `PUT /v1/voices/:id/dataset/parts/{n}` with each part ≤16MB, then
  `POST /v1/voices/:id/dataset/commit` with `{parts: N, bytes: total}`. Parts are idempotent (re-send
  a lost one) and may be sent in any order or in parallel.

Sketch of what `sdk/python` and `sdk/js` methods would look like (not implemented in this task):

```python
# sdk/python
voice = client.voices.create(speaker_name=..., attested_by=..., consent_statement=...)
client.voices.upload_dataset(voice.id, "dataset.zip")  # auto-picks single-shot vs chunked by size
status = client.voices.get(voice.id)
wav = client.voices.preview(voice.id, "Hello, this is a test.")
deployed = client.voices.deploy(voice.id)   # -> voice="custom:v-xxxx"
client.voices.delete(voice.id)
```
```js
// sdk/js
const voice = await client.voices.create({ speakerName, attestedBy, consentStatement });
await client.voices.uploadDataset(voice.id, fileStreamOrBuffer); // same auto-chunk rule
const status = await client.voices.get(voice.id);
await client.voices.deploy(voice.id);
```

## Rate limits, quotas, cost exposure

**Implemented as engineering safety defaults, not final pricing — flagged below as open.**

- Only **billing-enabled** gateway keys may call `/v1/voices/*` at all. A free-tier key gets a clear
  `402` telling it to enable billing. Training spends real GPU money (~$0.40-1/voice per the
  intake.py comment); an unpaid key training GPU jobs is a direct abuse vector the web flow doesn't
  have (Supabase signup has more friction than "get an API key").
- **3 active voices per key** (same cap as the web flow's per-user default), enforced the same way
  the web router already enforces it (counts non-rejected voices via `intake.list`).
- Per-hour request caps on the API-key router, tighter than the web defaults since a script can
  hammer much faster than a human clicking: `create` 5/hr, `preview` 30/hr, `parts` 400/hr (matches
  web). Keyed by the resolved identity (uid, or key id when no uid is bound).

**Open for the owner:** these numbers (3 voices, $-per-voice exposure, whether billing-enabled alone
is a sufficient gate or a stricter tier/deposit is needed, whether the per-key cap should differ from
the per-user web cap when a key has no bound uid) are defaults that unblock shipping the code, not a
pricing decision. Needs sign-off before the feature flag goes on for any real customer.

## Feature flag

Off by default, same posture as `VOICE_STUDIO_ENABLED_USERS`: `VOICE_STUDIO_API_ENABLED_KEYS` on the
backend is empty by default (every `/v1/voices/*` call 404s regardless of key validity). It's an
allowlist of **gateway key ids** (not uids — a key may have no bound uid, and the flag is meant to be
turned on per-key, deliberately), or `"*"`. Independent of `VOICE_STUDIO_ENABLED_USERS`: a key's owner
being web-allowlisted does not imply the key is API-allowlisted.

## What's still needed before this can ship dark-to-live

1. Owner's decision on the quota/pricing numbers above.
2. Legal review of the `consent_statement` mechanism (same open item the web flow already has).
3. `sdk/python` / `sdk/js` implementation of the sketch above (not built in this task).
4. Set `GATEWAY_FORWARD_SECRET` (same value on both the gateway and backend), `READALOUD_BACKEND_URL`
   on the gateway, and `VOICE_STUDIO_API_ENABLED_KEYS` on the backend in each environment — none of
   these are set in production, so the surface is inert (`GET /v1/voices` returns 501 from the
   gateway until `READALOUD_BACKEND_URL`/`GATEWAY_FORWARD_SECRET` are set) until someone deliberately
   configures it.
5. Live end-to-end verification with a real throwaway voice, trained/deployed/deleted through this
   path, run separately (owner has first-hand authorization to spend the GPU cost for that
   verification in their own session; this draft's author stopped short of it per that scoping split).
