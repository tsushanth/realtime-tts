# Public human speech: pub_fleurs_tel

Reference transcripts come from the public dataset; paced replay as in bench/rtbench.py; local engines measured on the dev machine's CPU (a proxy, not Fly).

| engine | n utts | WER | keyterm recall | first partial ms | commit final ms med/p90 | native final ms med/p90 | cpu-s per audio-s | $/audio-hr (compute-only, 100% packed, dev-machine CPU proxy, Modal CPU rate; not like-for-like with per-hour cloud pricing) |
|---|---|---|---|---|---|---|---|---|
| dg-flux | 50 | 12.2% | - | 1524 | 164/328 | 344/821 | - | 0.3900 |
| el-scribe | 50 | 6.6% | - | 2322 | 98/222 | 1146/1386 | - | 0.3900 |
| nemo-480 | 50 | 15.6% | - | 1846 | 47/73 | -/- | 0.089 | 0.0042 |
| zip-en-int8 | 50 | 29.2% | - | 1781 | 40/78 | -/- | 0.089 | 0.0042 |

Keyterm check not applicable (public sets have no keyterms); the WER/latency table is the result.
