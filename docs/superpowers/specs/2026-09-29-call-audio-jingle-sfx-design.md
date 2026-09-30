# Call audio: jingles & sound effects in live calls

Date: 2026-09-29
Status: Approved (design), pending implementation plan

## Intent

Calldesk wants live production phone calls handled by its in-house engine
(`call-loop-poc`) to be able to play:

1. A fixed intro **jingle** near the start of the call.
2. Situational **sound effects**, triggered by the LLM agent's own judgment
   mid-conversation (e.g. a confirmation chime after booking an appointment).

Both use ReadAloudAI's Sound Effects / Music generation APIs as the
underlying audio-generation capability, but generation happens offline
during tenant configuration, never during a live call.

**Scope boundary:** this feature is available only to tenants whose calls
run through `call-loop-poc`, Calldesk's own in-house telephony engine.
Tenants on Retell (a third-party managed platform Calldesk doesn't control
the audio pipeline for) are structurally excluded — no code path in this
design touches `src/lib/retell.ts` or Retell call handling.

## Non-goals

- Live/on-the-fly audio generation during a call (latency + uptime risk;
  explicitly rejected in favor of pre-generate-and-cache).
- Retell support.
- Asset versioning/history — saving a new asset overwrites the previous one.
- A shared/fixed sound library for all tenants — every tenant's jingle and
  SFX set is independently configurable.

## Architecture

Two independent pieces:

- **Asset pipeline** (offline, dashboard-triggered): a tenant configures
  jingle/SFX assets via a new Calldesk dashboard section. Each asset is
  generated once via ReadAloudAI's API, converted to mu-law@8kHz, and
  stored. No ReadAloudAI call ever happens during a live call.
- **Runtime playback** (inside `call-loop-poc`): pre-converted assets are
  loaded into memory once per call (mirroring the existing `fillerCache`
  pattern) and played through the Twilio adapter's existing paced-frame
  writer — the same code path that already streams TTS output and filler
  phrases to the caller.

## Data model

New Supabase table `tenant_call_audio_assets`:

| column | type | notes |
|---|---|---|
| `id` | uuid | PK |
| `tenant_id` | uuid | FK to tenants table |
| `asset_type` | text | `jingle` \| `sound_effect` |
| `name` | text | short slug, e.g. `booking_confirmed_chime` |
| `description` | text | free text; surfaced to the LLM as the `play_sound_effect` tool's parameter description so it can decide when to use it |
| `mulaw8k_storage_path` | text | pre-converted, playback-ready audio |
| `source_readaloud_job_id` | text | traceability back to the ReadAloudAI generation job |
| `enabled` | boolean | default true |
| `created_at` | timestamptz | |

Constraints:
- At most one enabled `jingle` row per tenant, enforced at the application
  layer (not a DB constraint) when saving.
- Any number of enabled `sound_effect` rows per tenant.
- No versioning: regenerating an asset overwrites its storage object and
  updates the existing row (or the UI simply creates a fresh row and
  disables the old one — implementation detail for the plan).

Storage: existing Supabase Storage bucket pattern, keyed by
`tenant_id/asset_id.raw`.

## Generation & caching pipeline

1. Tenant enters a prompt in the Calldesk dashboard's new "Call Audio"
   settings section.
2. Calldesk backend calls ReadAloudAI's `generate_sound_effect` (or Music)
   API using the existing gateway-identity-bridge auth pattern already used
   for MCP tool calls.
3. On confirmation, the backend downloads the generated clip, converts it
   to mu-law@8kHz mono via ffmpeg (same conversion `twilioAdapter.js`
   already performs on PCM16@24kHz TTS output), uploads the converted bytes
   to Supabase Storage, and writes the `tenant_call_audio_assets` row.
4. This pipeline runs only at configuration time — never during a live call.

## Runtime playback & tool-calling

- **Global kill switch:** an env var on `call-loop-poc`
  (`CALL_AUDIO_ASSETS_ENABLED`, default `true`) gates the entire feature
  independent of per-tenant asset configuration. When `false`, no jingle
  plays and the `play_sound_effect` tool is never added, regardless of what
  assets exist. This is an emergency org-wide disable, separate from
  per-tenant enablement.
- **Asset loading:** tenant's enabled assets are queried once per call
  (alongside the existing tenant-context resolution in
  `tenantLookup.js`/`server.js` around the `pendingCallContext` handoff) and
  held in memory for the call's duration, mirroring `fillerCache`'s shape
  but tenant-scoped.
- **Jingle:** if the feature is enabled and the tenant has an enabled
  `jingle` asset, its cached mu-law frames are sent via
  `adapter.send(buf, {binary:true, format:'mulaw8k'})` right after the
  `TwilioCallAdapter` is ready, before/alongside the greeting. Reuses
  `_sendMediaFrames`/`_startPacing` unchanged.
- **Situational SFX:** if the feature is enabled and the tenant has any
  enabled `sound_effect` assets, a `play_sound_effect` tool is added to the
  per-turn tools array — same conditional-inclusion pattern already used for
  `check_availability`/`book_appointment` (gated on `this.calendar`). One
  enum value per asset `name`; each asset's `description` becomes that
  enum value's description, so the LLM decides naturally from conversation
  context, same as it already does for `transition_flow`.
- **Tool handler:** looks up the asset's cached frames (already in memory)
  and calls `adapter.send()` — mirrors the existing `_startToolFiller`
  "play cached audio mid-turn" pattern, just backed by tenant assets
  instead of filler phrases.
- **Barge-in:** reuses the adapter's existing `clearQueue()`/`isSpeaking()`
  — if the caller talks over a jingle/SFX, it's cut off exactly like TTS
  output is today. No new interruption logic needed.

## Testing & rollout

- **Unit:**
  - `play_sound_effect` tool is included in the tools array only when the
    tenant has enabled SFX assets AND the global flag is on.
  - Tool-call handler dispatches the correct cached buffer for a given
    asset name.
  - Intro-jingle-on-connect fires only when the tenant has an enabled
    jingle AND the global flag is on.
  - mu-law conversion round-trip correctness (known WAV → mu-law → compare
    against reference bytes).
- **Integration:** a real outbound test call (same pattern as ReadAloudAI's
  production e2e tests) against a test tenant configured with a jingle and
  one SFX, asserting via call recording/transcript that the jingle plays
  near call start and the SFX plays when the scripted conversation reaches
  the trigger condition.
- **Rollout:** per-tenant gating is inherent — a tenant with zero asset
  rows sees zero behavior change, so the code path can ship to all
  `call-loop-poc` tenants at once; actual rollout is just enabling assets
  per tenant in the dashboard. The `CALL_AUDIO_ASSETS_ENABLED` flag is the
  emergency-only, org-wide override on top of that.
- **Retell tenants:** entirely unaffected; this code lives only in
  `call-loop-poc` and the Calldesk dashboard's tenant-settings UI.
