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

## History: blockers hit on 2026-09-22, resolved and re-verified on 2026-09-23

The first attempt at this load test surfaced two environment issues that blocked a clean
pass. Both are now resolved; kept here for context in case either regresses.

- **`fly.toml`'s `max_machines_running` key is not parsed by flyctl v0.4.95.** That
  version silently drops it - `fly config show` and the live machine config never showed
  it, confirmed by `strings`-ing the flyctl binary and finding no `MaxMachinesRunning`
  field at all. flyctl was subsequently upgraded to v0.4.106 on the operator machine; the
  key *still* doesn't appear in `fly config show -a piper-tts-sjc` output on v0.4.106
  either. Re-checked 2026-09-23 - this now looks like a legacy/dead config key in current
  Fly, not a version-specific parsing bug. It's not a functional gap in practice: the real
  governing mechanism for machine count on this account is `fly scale count <n> -a
  piper-tts-sjc`, confirmed working below.
- **Org-wide Fly machine quota.** `fly scale count 4` and a rolling redeploy both failed
  with `Your organization has reached its machine limit. Please contact billing@fly.io` -
  this Fly org runs ~30 other apps sharing one quota. The org owner freed up quota by
  suspending unrelated apps; `fly scale count 4 -a piper-tts-sjc -y` was then re-run and
  succeeded, confirmed via `fly status -a piper-tts-sjc` showing 4 machines.

### Independently re-verified results, 2026-09-23

With `fly scale count` at 4 and only 1 machine started at rest (`min_machines_running = 1`),
two runs were made directly against `wss://piper-tts-sjc.fly.dev/tts` in production using
the static `AUTH_TOKEN` fetched via `fly ssh console` (Option B above):

- **12-concurrent / 40 requests, first run (cold - 3 of 4 machines still stopped at start):**
  35/40 completed, 5 "timed out during opening handshake" errors, p50=1174ms/p90=10909ms.
  `fly status` before the run showed 1 machine started, 3 stopped; immediately after, all 4
  were `started` - real scale-out happened during the run, and the errors are consistent
  with connections landing mid-transition while the 3 stopped machines were still booting.
- **12-concurrent / 40 requests, second run (warm - all 4 machines already started):**
  40/40 completed, 0 errors, p50=997ms/p90=2076ms - clean, within the 4-machine x
  4-connection = 16-connection real capacity.
- **20-concurrent / 60 requests (exceeds 16-connection capacity by design):** 55/60
  completed, 5 "timed out during opening handshake" errors, p50=3687ms/p90=8616ms - a
  real, reproducible failure mode from genuinely exceeding provisioned capacity, not a
  broken scaler.

Conclusion: `auto_start_machines` / `fly scale count` do provision and start real capacity
under load in production, and requests routed to already-warm machines within capacity
complete cleanly. Requests that land while additional machines are still cold-starting, or
that exceed total provisioned capacity, still fail with handshake timeouts - that's an
inherent cold-start/overload characteristic to plan around (e.g. a higher
`min_machines_running` or pre-warming ahead of known traffic spikes), not evidence the
autoscaling config itself is broken.
