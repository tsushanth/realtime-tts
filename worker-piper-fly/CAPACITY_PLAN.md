# Capacity plan for the Piper service

Status today: one Fly `shared-cpu-4x` (2 GB, about $13.27/month, San Jose), capped at 4 concurrent
connections (`MAX_CONNECTIONS`). Measured: shared CPU collapses at 2-4 realistic calls (burst-credit
throttling); `performance-2x` 4 GB ($64.39/month) held 8-16 calls at ~325-345 ms median first audio and
collapsed near 24. ElevenLabs sells 4-40 concurrent calls per customer plan.

## What has to change, in order
0. **Now, no spend.** Add a per-key cap (`MAX_PER_KEY`) so one customer cannot take every slot; add a status
   page; keep the honest "at capacity" error (already returns close code 1013 / HTTP 503 + Retry-After).
1. **First paying customer (+~$51/month).** Move to one `performance-2x` 4 GB machine and raise
   `MAX_CONNECTIONS` to about 12. Re-run the load test at 8/16/24 calls before trusting the number.
2. **Uptime and a second region (~$130-150/month).** Two performance machines in two regions.
   Blocker to design first: custom and house voices live on each machine's own volume, so voices would not
   replicate. Make object storage (for example Tigris on Fly) the source of truth and lazy-download to a local
   cache on a miss; the admin PUT writes to storage. Also needs health checks and rolling deploys.
3. **Autoscale.** Scale machine count on open connections; sell concurrency as part of a plan later.

## Economics (assumptions, not measurements)
A call minute uses about 450 characters of TTS. At $4 per million characters that is $0.0018 of revenue per
call minute. A slot occupied 24/7 earns about $78/month; a slot costs about $4.40 (shared) to $5.40 (performance)
per month, so break-even is under 10 percent utilisation. Thin margins appear only at very low utilisation.

## Load-test plan
Run from a Fly machine in the same region (`benchmarks/load_test.js`): 8, 16, 24, 32 concurrent realistic calls,
mixed WebSocket and HTTP. Pass bar: median first audio under 350 ms and p95 under 700 ms. Not yet measured:
`performance-4x`, multi-machine behaviour, and voice-cache misses under load.
