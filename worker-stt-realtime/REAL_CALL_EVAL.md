# Real-call STT evaluation (Phase 1)

Verified utterances only; paced replay as in `bench/rtbench.py`; local engines measured on the dev machine's CPU (a proxy, not Fly).

| engine | n utts | WER | keyterm recall | first partial ms | commit final ms med/p90 | native final ms med/p90 | cpu-s per audio-s | $/audio-hr (compute-only, 100% packed, dev-machine CPU proxy, Modal CPU rate; not like-for-like with per-hour cloud pricing) |
|---|---|---|---|---|---|---|---|---|
| dg-flux | 110 | 4.0% | 78.8% | 1045 | 168/265 | 580/1127 | - | 0.3900 |
| el-scribe | 110 | 2.5% | 93.9% | 2245 | 110/274 | 1517/1606 | - | 0.3900 |
| nemo-480 | 110 | 8.3% | 63.6% | 1288 | 52/85 | -/- | 0.089 | 0.0042 |
| zip-en-int8 | 110 | 15.4% | 39.4% | 1139 | 37/67 | -/- | 0.063 | 0.0030 |

## Decision gate

Cloud baselines: WER vs **el-scribe** (lowest WER), keyterm recall vs **el-scribe** (highest keyterm recall).

### nemo-480: FAIL
- FAIL: wer_within_1.5x
- FAIL: keyterm_within_10pts
- PASS: commit_final_le_150ms
- PASS: cpu_le_0.15

### zip-en-int8: FAIL
- FAIL: wer_within_1.5x
- FAIL: keyterm_within_10pts
- PASS: commit_final_le_150ms
- PASS: cpu_le_0.15

**Result: not yet, Phase 2 becomes an accuracy investigation**

Footnotes: a check is INSUFFICIENT (not FAIL) when data is missing or thin: WER needs n >= 30 on both engines and equal n; keyterm needs keyterm_total >= 30 on both; commit latency needs fired_frac >= 0.9. Price is compute-only, 100% packed, dev-machine CPU proxy at the Modal CPU rate; not like-for-like with per-hour cloud pricing.

## Robustness checks (same 110 utterances, 1,340 words)

| engine | WER all | WER on the 51 hand-listened utterances (640 words) | WER all, fillers dropped |
|---|---|---|---|
| el-scribe | 2.5% | 0.9% | 2.2% |
| dg-flux | 4.0% | 3.6% | 3.8% |
| nemo-480 | 8.3% | 8.8% | 8.4% |
| zip-en-int8 | 15.4% | 14.1% | 15.7% |

The ranking is identical on hand-listened references only and with fillers dropped, so the local-vs-cloud gap is not an artefact of Whisper-drafted references or filler handling.

## Read with care

- **Audio:** 8 kHz phone recordings of the owner's own test calls, far-end channel only. Most far-end speech is synthetic (test-framework personas); only ~15 utterances (calls to the owner's cell, voicemail robots) are not. Human callers may behave differently.
- **Small sample:** 110 utterances / 1,340 words / 21 calls, 33 keyterms (just over the gate's 30 minimum). The 8.3% vs 2.5% gap is large; 2.5% vs 4.0% (ElevenLabs vs Deepgram) is not reliable.
- **References:** 51 hand-listened, 59 accepted from Whisper drafts (`ref_source` in the manifest). Non-English calls/utterances (French, Spanish, Italian, Dutch, Polish) and clipped/garbled ones were excluded.
- **Latency is not like-for-like:** local "commit" latency assumes the client tells us speech ended (52 ms nemo); Deepgram's includes the CloseStream flush (168 ms). "Native" is each engine's own endpointing: Deepgram 580 ms; ElevenLabs used its default VAD commit (1,517 ms, untuned); local engines have no server-side endpointing here.
- **CPU:** dev-machine CPU, not Fly; no concurrent-load test yet.
