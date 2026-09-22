# Pricing proposal: audio isolation ("denoise" engine)

Status: **proposed, not decided or wired into billing.** `report_usage` already sends
`{id, audio_seconds, engine: "denoise"}` to the gateway's usage counter (see README.md "Auth
wiring"); this document proposes the *price* per unit, which nothing in the codebase sets yet.
This is a business decision someone needs to sign off on - this doc gives the cost math to make
that decision defensibly instead of guessing, following the same cost-basis method
`worker-piper-fly/CAPACITY_PLAN.md` used for Piper/Kokoro pricing.

## 1. Measured compute cost (this pass, real measurement)

Ran `verification/noisy_mix.wav` (10.389s clip) through the real `Separator.separate` path 3x
warm (model already loaded), instrumenting with `resource.getrusage` for actual CPU time
(user+sys), not just wall clock, on this dev machine (Apple M2 Pro, 10 cores,
`TORCH_THREADS=10`):

| run | wall time | CPU time (user+sys) | CPU-s per audio-s |
|---|---|---|---|
| 1 | 4.465s | 21.882s | 2.106 |
| 2 | 4.757s | 22.371s | 2.153 |
| 3 | 4.799s | 22.403s | 2.156 |

**~2.14 CPU-seconds consumed per second of input audio** (average). This number is (to first
order) a property of the model's total FLOPs for this clip length, not of how many cores happen
to be available - more cores parallelize the *same* total work into less wall-clock time, they
don't reduce total CPU-seconds billed. Caveat: measured on an M2 Pro; per-core throughput on
Fly's `performance-2x` machines (likely a different x86/ARM microarchitecture) will differ
somewhat - this is the best available number without deploying and profiling on the actual Fly
VM class, which wasn't done in this pass (see "Open items" below).

## 2. Machine cost basis

`worker-demucs-fly/fly.toml` specifies `performance-2x`/4GB - the same VM class
`worker-piper-fly/CAPACITY_PLAN.md` already priced at **$64.39/month** as a reference point (that
doc used it for Piper's higher-tier plan; this worker's `fly.toml` comment independently chose it
"bigger than worker-piper-fly's shared-cpu-4x... trades cost for headroom"). `performance-2x` =
2 dedicated vCPUs.

Total compute budget at 100% utilization, 24/7, 30-day month:

    2 vCPU * 2,592,000 s/month = 5,184,000 vCPU-seconds/month

Max theoretical audio throughput at 100% utilization:

    5,184,000 vCPU-s / 2.14 CPU-s-per-audio-s ≈ 2,422,000 audio-seconds/month
                                              ≈ 40,370 audio-minutes/month

**Note on the concurrency lock**: `Separator.separate()` holds a global lock (model is not
verified thread-safe for concurrent forward passes), so in practice the 2 vCPUs are used by *one*
request at a time, not 2 requests in parallel (confirmed in load testing, below - see "Load
testing" for the actual observed behavior). This does **not** change the throughput-per-dollar
math above (total CPU-seconds consumed per audio-second is the same either way), but it does mean
`MAX_CONNECTIONS=2` buys queueing depth, not parallel throughput - relevant to latency/SLA
promises, not to the cost basis.

## 3. Break-even price at different utilization assumptions

    price_per_min(utilization) = $64.39 / (40,370 * utilization)

| utilization | audio-minutes/month | break-even $/min | break-even $/sec |
|---|---|---|---|
| 100% (theoretical max, unrealistic) | 40,370 | $0.0016 | $0.0000266 |
| 25% | 10,093 | $0.0064 | $0.000106 |
| 10% (launch-stage assumption, matches Piper's CAPACITY_PLAN) | 4,037 | $0.0160 | $0.000266 |
| 2% (pessimistic, near-zero early traffic) | 807 | $0.0798 | $0.00133 |

Piper's own capacity plan explicitly assumes thin margins are fine at low utilization and gets
better as traffic grows - same logic applies here, just with a much higher per-unit cost because
Demucs is far more CPU-intensive per second of audio than TTS is per character (2.14 CPU-s per
audio-second here vs. Piper's ~450 chars/call-minute costing a small fraction of a CPU-second).

## 4. Proposed price

Using the **10% utilization** break-even ($0.0160/min) as the launch-stage cost floor, and
applying a margin multiplier consistent with this product's own existing pricing practice
(voice/TTS is priced at ~$0.10/min against a measured ~$0.044/min cost - a ~2.3x cost-to-price
ratio, ~56% gross margin), plus headroom for bandwidth, retries, and unmodeled overhead
(gateway proxy hop, occasional cold starts):

    $0.0160/min * ~3x ≈ $0.05/audio-minute processed  ($0.00083/sec, ~$3.00/audio-hour)

**Proposed: $0.05 per minute of audio processed** (billed on `audio_seconds` already reported by
`report_usage`, rounded up to the nearest second or minute - billing granularity is a separate
decision). This is ~19x the full-utilization marginal compute cost and ~3x the 10%-utilization
break-even, which is deliberately conservative headroom given:
- concurrency is effectively serialized (see above), so this worker cannot be packed as
  efficiently as a naively-parallel service even at "high utilization",
- no load test has run on the actual Fly VM class yet to confirm the CPU-seconds/audio-second
  ratio holds there (see "Open items"),
- clip-length variance (`MAX_DURATION_S=120`) means tail requests cost much more than the 10.4s
  clip measured here - a flat per-minute rate needs to absorb that variance.

**What this proposal is NOT**: a market-benchmarked price. Unlike Piper/Kokoro's pricing (which
cross-referenced ElevenLabs' published list prices), no competitor "denoise as a service" pricing
(Dolby.io Media Enhance, Adobe Podcast Enhance, Krisp, AssemblyAI's audio intelligence add-ons)
was looked up or verified in this pass. This is a cost-plus floor, not a market-clearing price -
whoever finalizes this should check it against actual competitor quotes before shipping it,
and could reasonably land higher if competitors charge more for a comparable feature.

## Open items on pricing
- Re-measure CPU-seconds/audio-second on an actual `performance-2x` Fly machine (this used a
  local M2 Pro proxy).
- Decide billing granularity (per-second vs. per-minute-rounded-up) and whether to charge a
  request-level minimum (a 1s clip still pays full model-load-adjacent overhead in practice).
- Look up real competitor pricing for comparable audio-denoising APIs before finalizing.
- Decide whether the serialization behavior (see "Load testing" in README.md) should be fixed
  (e.g., per-worker model instances to get real parallelism) before or after this price ships -
  it affects how much throughput $64.39/month actually buys.
