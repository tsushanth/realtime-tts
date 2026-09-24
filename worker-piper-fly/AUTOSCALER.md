# Dynamic autoscaling (fly-autoscaler)

## What this solves

Before this, "scaling" meant manually running `fly scale count N` to change
the size of the machine pool - nothing grew the pool automatically in
response to real load. `max_machines_running` in `fly.toml` looked like it
should do this but is inert on this Fly platform version (see `fly.toml`'s
own comment and `LOAD_TEST.md`'s history section).

This adds a real, working piece: [fly-autoscaler](https://github.com/superfly/fly-autoscaler),
deployed as its own tiny Fly app (`piper-tts-sjc-autoscaler`), which polls
Fly's own hosted Prometheus every 15s and starts/stops machines in
`piper-tts-sjc`'s existing pool to match real, live concurrent-connection
load - automatically, with no manual intervention.

## How it actually works (the part that's easy to get wrong)

`fly-autoscaler` does **not** create brand-new Fly machines beyond what
already exists for the app. It can only start or stop machines that are
*already provisioned* - "it will never exceed the number of machines
available for a Fly app" (from its own docs). That means true elastic
range requires two separate things working together:

1. **A large enough pre-provisioned pool.** `fly scale count 10` was run
   against `piper-tts-sjc` to create a pool of 10 machines (most sit
   `stopped` most of the time - Fly does not bill for stopped machines,
   so a bigger pool costs nothing extra unless it's actually running).
2. **This autoscaler**, which decides *how many of those 10* should be
   `started` right now, based on real concurrency.

Fly's own built-in proxy-level `auto_start_machines`/`auto_stop_machines`
(already configured in `../fly.toml`) still separately and immediately
starts/stops machines based on live in-process connection state - that
part reacts instantly. This autoscaler's Prometheus-based view lags by
roughly 1-2 minutes (real, measured ingestion delay on Fly's hosted
Prometheus - see verification below), so it's a complement to the
built-in instant reaction, not a replacement for it: the built-in
mechanism absorbs sudden bursts within the current pool, and this
autoscaler adjusts the pool's own baseline size for sustained load shifts.

## The metric: zero app-side instrumentation needed

Fly automatically publishes `fly_app_concurrency` for every app via its
own hosted Prometheus (`https://api.fly.io/prometheus/<org-slug>/`) - no
`/metrics` endpoint needed in `server.py`. Confirmed live against
`piper-tts-sjc` on 2026-09-23: a real load test produced a real,
queryable `fly_app_concurrency` time series within about a minute.

Query used: `sum(fly_app_concurrency{app="piper-tts-sjc"})`

## Scaling expression

```
FAS_STARTED_MACHINE_COUNT = "max(1, ceil(concurrency / 3))"
```

One machine per ~3 concurrent connections (matches `../fly.toml`'s
`soft_limit = 3`), floored at 1 so the expression never fights
`piper-tts-sjc`'s own `min_machines_running = 1`.

## Real verification (2026-09-23)

Ran `tests/load_test.py --concurrency 15 --requests 45` against production
while tailing `fly logs -a piper-tts-sjc-autoscaler`:

- Baseline: autoscaler correctly scaled a freshly-grown 10-machine pool
  down to 3 started (real demand was low at deploy time) - confirms it
  also saves cost, not just adds capacity.
- Under load: computed target climbed to 5 as real concurrency rose;
  the autoscaler genuinely started 3 additional machines from the
  stopped pool on its own (`"machine started"` log lines with real
  machine IDs, no manual `fly scale count` run during the test) -
  settled at a stable target of 4 as load leveled off.
- This is the first real, automatic pool-size change observed all
  session without a human running a scale command.

## Setup (for a fresh environment)

1. `fly scale count <N>` against the target app first, sized for your
   real expected peak (stopped machines are free, so err generous).
2. `fly apps create <name>-autoscaler`
3. `fly tokens create deploy -a <target-app>` -> `FAS_API_TOKEN`
4. `fly tokens create readonly -o <org>` -> `FAS_PROMETHEUS_TOKEN`
5. `fly secrets set FAS_API_TOKEN="FlyV1 <token>" FAS_PROMETHEUS_TOKEN="FlyV1 <token>" -a <name>-autoscaler`
6. `fly deploy --remote-only -a <name>-autoscaler` from `autoscaler/`

## Cost

The autoscaler app itself is a single small always-on machine (~$3-4/mo
at Fly's shared-cpu-1x rate). The pool-growth itself (10 vs 4 machines)
adds no cost by default since stopped machines aren't billed - only
actual started time is, same as before this change.
