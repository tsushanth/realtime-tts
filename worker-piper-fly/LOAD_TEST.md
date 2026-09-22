# Load-testing the autoscaled Piper worker

Two ways to authenticate the test WebSocket connections, use whichever fits how you're
running this:

## Option A: real customer-shaped session token (preferred for a full end-to-end check)

1. Mint a temporary billing-enabled test key (see `eval/README.md`'s established
   mint/use/revoke pattern) - never use a real customer key for load testing.
2. Authorize once to get a session token + WS URL:
   `curl -s -XPOST https://api.readaloudai.org/tts/authorize -d '{"key":"<test-key>","engine":"piper"}'`
3. Use the returned session token as `<token>` in Step 4 below. Note this token is
   short-lived (60s TTL, see `gateway/keys.js`) - mint it immediately before running the
   load test, not ahead of time.
4. Revoke the temporary test key when done:
   `curl -s -XDELETE https://api.readaloudai.org/admin/keys -d '{"id":"<key-id>"}'`

## Option B: static AUTH_TOKEN (what this repo's own tooling can do without the backend)

The worker also accepts its static `AUTH_TOKEN` secret directly as the `token` query
param on the public `wss://piper-tts-sjc.fly.dev/tts` endpoint (see `server.py`'s
`tts()` handler - `static_ok` check). This is the same secret used for the private
`.internal` path; from an operator machine that already has `fly` access to this app,
fetch it without printing other secrets:

```bash
fly ssh console -a piper-tts-sjc -C 'sh -c "echo \$AUTH_TOKEN"'
```

## Running the test

1. In one terminal, watch machine count live: `watch -n2 'fly status -a piper-tts-sjc'`
2. Run the load test above a single machine's known ceiling (`MAX_CONNECTIONS=4` per
   `fly.toml`, and `soft_limit = 3` per machine) to force a scale-up:
   `python3 tests/load_test.py wss://piper-tts-sjc.fly.dev/tts <token> --concurrency 20 --requests 60`
3. Confirm in the `fly status` terminal that machine count actually increased above 1
   during the run, and dropped back to 1 a few minutes after the run ends (Fly's default
   scale-down cooldown).
4. Confirm zero errors in the load test's own output - "at capacity" errors during a
   scale-up transition are exactly the failure mode this piece exists to prevent.
5. If using Option A, revoke the temporary test key (see above).

## Known caveat, observed 2026-09-22

Running this for real against `piper-tts-sjc` surfaced two environment issues that blocked
a clean pass and are worth checking before you assume a failed run means the config is wrong:

- **`fly.toml`'s `max_machines_running` key is not parsed by every flyctl version.** The
  version in use here (v0.4.95) silently drops it - `fly config show` and the live machine
  config never show it, confirmed by `strings`-ing the flyctl binary and finding no
  `MaxMachinesRunning` field at all. If your `fly config show -a piper-tts-sjc` output is
  missing `max_machines_running`, that's why; you likely need a newer flyctl, or to
  provision machine slots explicitly with `fly scale count <n> -a piper-tts-sjc` instead.
- **Org-wide Fly machine quota.** `fly scale count 4` and a rolling redeploy both failed
  with `Your organization has reached its machine limit. Please contact billing@fly.io` -
  this Fly org runs ~30 other apps sharing one quota. Autoscaling for this app cannot
  provision machines beyond whatever headroom exists org-wide; check with
  `fly scale count <n> -a piper-tts-sjc -y` (dry-run the plan first) before relying on a
  scale-up actually happening in production.

Under this constraint, a 20-concurrent/60-request run against the 2 machines that were
already up (not 1, and never scaled to more) completed 30/60 with 30 "timed out during
opening handshake" errors, p50=3359ms/p90=9242ms for the ones that completed. That is a
real, reproducible failure mode at current org headroom, not a hypothetical - raise the
org's machine quota before treating this service as safe under a comparable traffic spike.
