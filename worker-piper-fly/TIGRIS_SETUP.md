# Tigris voice storage setup

One-time setup for a fresh environment. Already done for `piper-tts-sjc` production
as of 2026-09-22 (Task 3 of the production-readiness plan) — see the exact state below.

## What's live in `piper-tts-sjc`

- Tigris bucket: `piper-voices-prod` (region matches `primary_region` in `fly.toml`, `sjc`).
- Fly secrets set on the app (`fly secrets list -a piper-tts-sjc`):
  - `TIGRIS_BUCKET=piper-voices-prod`
  - `TIGRIS_ENDPOINT_URL=https://fly.storage.tigris.dev`
  - Fly's `fly storage create` also auto-injected its own generic S3-compatible secrets
    alongside the two above: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
    `AWS_ENDPOINT_URL_S3`, `AWS_REGION`, `BUCKET_NAME`. `boto3` in `voice_storage.py`
    reads `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` from the environment automatically —
    no extra code needed for credential wiring.
- `worker-piper-fly/Dockerfile` installs `boto3==1.35.36` and copies `voice_storage.py`
  into the image (no separate `requirements.txt` in this service — deps are pinned
  directly in the Dockerfile's `pip install` line).
- Deployed via `fly deploy -a piper-tts-sjc` from `worker-piper-fly/` — confirmed healthy
  post-deploy via `curl https://piper-tts-sjc.fly.dev/health`.
- Existing voice catalog backfilled into Tigris by re-running `publish_voice.py` against
  the now-Tigris-backed admin endpoint (see "Backfill" below).
- Machine ceiling set via `fly scale count 4 -a piper-tts-sjc` (Task 4) - confirmed via
  `fly scale show -a piper-tts-sjc` showing `COUNT 4`. See step 5 below; this is not
  automatic and must be set explicitly per app.

## Steps for a fresh environment / new app

1. `cd worker-piper-fly && fly storage create --name <bucket-name>` — creates a Tigris
   bucket and prints access credentials once. Save them immediately; Tigris does not show
   the secret key again. This also auto-sets the generic `AWS_*`/`BUCKET_NAME` secrets on
   the target app.
2. `fly secrets set TIGRIS_BUCKET=<bucket-name> TIGRIS_ENDPOINT_URL=https://fly.storage.tigris.dev -a <app-name>`
   (the `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` are already set by step 1; only add
   them explicitly if provisioning credentials by hand instead of via `fly storage create`).
3. Confirm `boto3` is installed in the image (see the Dockerfile's `pip install` line) and
   redeploy: `fly deploy -a <app-name>` so the new secrets and code take effect.
4. Backfill existing voices (see below).
5. **Set the real machine ceiling: `fly scale count 4 -a <app-name>`.** `fly.toml`'s
   `max_machines_running = 4` is not actually honored by flyctl (confirmed on v0.4.95 and
   v0.4.106 - it never appears in `fly config show`; see `fly.toml`'s comment and
   `LOAD_TEST.md`). `fly scale count` is the real, confirmed-working mechanism that
   governs machine count on this account, and a fresh app defaults to a count of 1. Skip
   this and autoscaling is silently inert - the app will never run more than 1 machine no
   matter what `fly.toml` says. Verify with `fly scale show -a <app-name>` (expect
   `COUNT 4`).

## Backfill

Re-publish the known voice catalog through the now-Tigris-backed `/admin/voices/<vid>`
endpoint — this naturally writes every already-published voice into Tigris without a
separate local-to-Tigris copy script, since `publish_voice.py` re-pulls source audio from
the canonical Modal `house-voices` volume and re-PUTs it:

```bash
cd voices
PIPER_ADMIN_URL=https://piper-tts-sjc.fly.dev \
VOICES_ADMIN_TOKEN=$(cat ~/.config/voice-pipeline/piper_admin_token) \
python3 publish_voice.py --tier A,B --allow-tier-b --republish
```

Verify counts:

```bash
curl -H "Authorization: Bearer $(cat ~/.config/voice-pipeline/piper_admin_token)" \
  https://piper-tts-sjc.fly.dev/admin/voices
```

and confirm the voice count matches `voices/catalog.json`'s published (tier A + tier B)
entries. Also spot-check at least one voice end-to-end with `/tts/authorize` + a real
synthesis call, to prove the Tigris round-trip (not just that the admin endpoint returned
200).
