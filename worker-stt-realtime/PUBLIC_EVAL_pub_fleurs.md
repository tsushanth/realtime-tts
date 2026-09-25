# Public human speech: pub_fleurs

Reference transcripts come from the public dataset; paced replay as in bench/rtbench.py; local engines measured on the dev machine's CPU (a proxy, not Fly).

| engine | n utts | WER | keyterm recall | first partial ms | commit final ms med/p90 | native final ms med/p90 | cpu-s per audio-s | $/audio-hr (compute-only, 100% packed, dev-machine CPU proxy, Modal CPU rate; not like-for-like with per-hour cloud pricing) |
|---|---|---|---|---|---|---|---|---|
| dg-flux | 50 | 7.7% | - | 1344 | 160/253 | 666/1220 | - | 0.3900 |
| el-scribe | 50 | 4.5% | - | 2325 | 97/157 | 1166/1497 | - | 0.3900 |
| nemo-480 | 50 | 30.3% | - | 1848 | 39/68 | -/- | 0.079 | 0.0037 |
| zip-en-int8 | 50 | 76.7% | - | 1787 | 31/48 | -/- | 0.084 | 0.0040 |

Keyterm check not applicable (public sets have no keyterms); the WER/latency table is the result.
