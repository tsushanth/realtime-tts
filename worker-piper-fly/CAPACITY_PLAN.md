# Capacity plan for the Piper service

Status today (updated after Tasks 1-4): Fly `shared-cpu-4x` (2 GB, about $13.27/month/machine, San Jose),
capped at 4 concurrent connections per machine (`MAX_CONNECTIONS`), autoscaled 1-4 machines
(`fly scale count 4`, `min_machines_running = 1`, `auto_start_machines`/`auto_stop_machines` in `fly.toml`) -
see `LOAD_TEST.md` for real, re-verified production evidence of scale-out working. Voice storage is
Tigris-backed and stateless (Tasks 1-3; see `TIGRIS_SETUP.md`), so every machine is interchangeable and
safe to stop/start. Measured: shared CPU collapses at 2-4 realistic calls per machine (burst-credit
throttling); `performance-2x` 4 GB ($64.39/month) held 8-16 calls at ~325-345 ms median first audio and
collapsed near 24. ElevenLabs sells 4-40 concurrent calls per customer plan.

## What has to change, in order
0. **Done.** Tigris-backed voice storage (Tasks 1-3) and Fly autoscaling 1-4 machines (Task 4) are both
   live in production. Remaining from the original "no spend" step: a per-key cap (`MAX_PER_KEY`) so one
   customer cannot take every slot, and a status page; the honest "at capacity" error already exists
   (close code 1013 / HTTP 503 + Retry-After).
1. **First paying customer (+~$51/month per machine).** Move to `performance-2x` 4 GB machines and raise
   `MAX_CONNECTIONS` to about 12. Re-run the load test at 8/16/24 calls per machine before trusting the
   number - current numbers below are for `shared-cpu-4x` only.
2. **Second region.** Machines are already interchangeable (Tigris-backed, stateless) within one region;
   a second Fly region would need its own `primary_region`/app or Fly's multi-region primitives, plus
   confirming Tigris read latency from the new region. Not yet done or measured.
3. **Beyond 4 machines / sell concurrency as a plan feature.** Current ceiling is `fly scale count 4`
   (see `TIGRIS_SETUP.md`); raising it further needs the org's Fly machine quota increased first (see
   `LOAD_TEST.md`'s history section - this org shares one quota across ~30 apps).

## Economics (assumptions, not measurements)
A call minute uses about 450 characters of TTS. At $4 per million characters that is $0.0018 of revenue per
call minute. A slot occupied 24/7 earns about $78/month; a slot costs about $4.40 (shared) to $5.40 (performance)
per month, so break-even is under 10 percent utilisation. Thin margins appear only at very low utilisation.

## Load-test plan
Use `worker-piper-fly/tests/load_test.py` against the live app (see `LOAD_TEST.md` for the exact commands,
auth options, and current pass criteria - a 12-concurrency/40-request in-capacity run and a separate
20-concurrency/60-request overload probe). The superseded `benchmarks/load_test.js` plan (8/16/24/32
concurrent, mixed WebSocket and HTTP, run from a same-region Fly machine) has not been executed against
the current autoscaled/Tigris-backed setup; treat its pass bar (median first audio under 350ms, p95 under
700ms) as aspirational until re-run. Not yet measured: `performance-4x`, `performance-2x` under the current
autoscaling config, and voice-cache misses under sustained multi-machine load.
