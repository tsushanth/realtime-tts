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
2. Before load-testing, confirm the machine ceiling is actually raised - `fly scale show
   -a piper-tts-sjc` should show `COUNT 4`. `fly.toml`'s `max_machines_running` key is not
   honored by flyctl (see below); `fly scale count 4 -a piper-tts-sjc` is what actually
   provisions the extra machine slots, and it must be run at least once per app (see
   `TIGRIS_SETUP.md`). If it's still at the default of 1, none of this will scale.
3. **In-capacity pass/fail run** - this is the one whose result should be zero errors.
   Run within the real provisioned capacity (4 machines x `MAX_CONNECTIONS=4` = 16
   connections):
   `python3 tests/load_test.py wss://piper-tts-sjc.fly.dev/tts <token> --concurrency 12 --requests 40`
   - Pass: confirm in the `fly status` terminal that machine count actually increased
     above 1 during the run (if it started cold), and confirm zero errors in the load
     test's own output once machines are warm - "at capacity" or handshake-timeout errors
     on a warm run are exactly the failure mode this piece exists to prevent. A cold run
     (most machines stopped at the start) may show a handful of handshake timeouts while
     the stopped machines are still booting; that's expected mid-transition behavior, not
     a failure - re-run once warm to get the clean pass/fail signal.
   - Fail: zero machine-count increase during a cold run, or any errors once all
     machines are already warm/started.
4. **Overload probe (separate, expected-imperfect run)** - deliberately exceeds the
   16-connection real capacity to characterize behavior beyond provisioned capacity, not
   to get a clean result:
   `python3 tests/load_test.py wss://piper-tts-sjc.fly.dev/tts <token> --concurrency 20 --requests 60`
   - Pass criteria here are different from Step 3: some handshake-timeout errors are
     expected and fine (this run intentionally asks for more than 4 machines can serve).
     A completely stuck test, 0 completions, or a crash would indicate a real problem;
     partial completions with timeouts (e.g. the ~55/60 recorded below) do not.
5. If using Option A, revoke the temporary test key (see above).

## History: blockers hit on 2026-09-22, resolved and re-verified on 2026-09-23

Before any of this load test could even run, a stale `[mounts]` volume block had to be
removed from `fly.toml` (see that file's comment near `[[vm]]`) - a Fly volume can only
attach to one machine, which structurally blocks Fly from creating additional machines at
all. That was caught and fixed during Task 4's initial config change, before the first
load test attempt below, so it is a separate, already-resolved prerequisite - not one of
the two blockers this section covers.

With the volume already removed, the first attempt at actually running this load test
surfaced two further, unrelated environment issues that blocked a clean pass. Both are
now resolved; kept here for context in case either regresses.

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
- **Overload probe, 20-concurrent / 60 requests (exceeds 16-connection capacity by
  design):** 55/60 completed, 5 "timed out during opening handshake" errors,
  p50=3687ms/p90=8616ms - a real, reproducible failure mode from genuinely exceeding
  provisioned capacity, not a broken scaler. This is the expected, non-clean result for
  this specific run (see Step 4 above) - do not treat it as a regression.

Conclusion: `auto_start_machines` / `fly scale count` do provision and start real capacity
under load in production, and requests routed to already-warm machines within capacity
complete cleanly. Requests that land while additional machines are still cold-starting, or
that exceed total provisioned capacity, still fail with handshake timeouts - that's an
inherent cold-start/overload characteristic to plan around (e.g. a higher
`min_machines_running` or pre-warming ahead of known traffic spikes), not evidence the
autoscaling config itself is broken.
