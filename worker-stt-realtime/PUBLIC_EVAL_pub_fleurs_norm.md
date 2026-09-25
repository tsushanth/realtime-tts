# Public human speech: pub_fleurs_norm

Reference transcripts come from the public dataset; paced replay as in bench/rtbench.py; local engines measured on the dev machine's CPU (a proxy, not Fly).

| engine | n utts | WER | keyterm recall | first partial ms | commit final ms med/p90 | native final ms med/p90 | cpu-s per audio-s | $/audio-hr (compute-only, 100% packed, dev-machine CPU proxy, Modal CPU rate; not like-for-like with per-hour cloud pricing) |
|---|---|---|---|---|---|---|---|---|
| dg-flux | 50 | 6.9% | - | 1323 | 176/290 | 366/862 | - | 0.3900 |
| el-scribe | 50 | 4.6% | - | 2325 | 101/166 | 1206/1503 | - | 0.3900 |
| nemo-480 | 50 | 10.3% | - | 1844 | 42/71 | -/- | 0.086 | 0.0041 |
| zip-en-int8 | 50 | 18.7% | - | 1473 | 37/77 | -/- | 0.078 | 0.0037 |

Keyterm check not applicable (public sets have no keyterms); the WER/latency table is the result.
