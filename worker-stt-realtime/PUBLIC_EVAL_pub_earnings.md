# Public human speech: pub_earnings

Reference transcripts come from the public dataset; paced replay as in bench/rtbench.py; local engines measured on the dev machine's CPU (a proxy, not Fly).

| engine | n utts | WER | keyterm recall | first partial ms | commit final ms med/p90 | native final ms med/p90 | cpu-s per audio-s | $/audio-hr (compute-only, 100% packed, dev-machine CPU proxy, Modal CPU rate; not like-for-like with per-hour cloud pricing) |
|---|---|---|---|---|---|---|---|---|
| dg-flux | 50 | 19.6% | - | 1086 | 183/318 | 826/1260 | - | 0.3900 |
| el-scribe | 50 | 13.7% | - | 2244 | 259/507 | 1386/1561 | - | 0.3900 |
| nemo-480 | 50 | 25.0% | - | 1287 | 67/93 | -/- | 0.084 | 0.0040 |
| zip-en-int8 | 50 | 52.3% | - | 1148 | 56/93 | -/- | 0.083 | 0.0039 |

Keyterm check not applicable (public sets have no keyterms); the WER/latency table is the result.
