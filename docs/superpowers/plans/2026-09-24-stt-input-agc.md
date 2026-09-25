# Streaming input level normalisation (AGC) for the local STT engines

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Give the local streaming engines (nemo-480 etc.) automatic gain control so quiet input (like raw FLEURS, median peak 0.007) is not catastrophic (nemo-480: 30.3% WER raw vs 10.3% when peak-normalised), without hurting well-levelled audio (real calls peak ~0.76; earnings peak ~0.44).

**Context:** `worker-stt-realtime/PUBLIC_EVAL.md` (the finding and numbers). Baselines to beat/preserve for `nemo-480`: pub_fleurs 30.3%, pub_fleurs_norm 10.3%, pub_fleurs_tel 15.6%, pub_earnings 25.0%, real-call verified set 8.3%.

## Global Constraints
- Streaming and causal: no lookahead, no added latency, chunk-size agnostic (harness feeds 40 ms = 640 samples; server may use others); float32 mono 16 kHz in [-1,1] in and out; output hard-limited to [-1, 1] (no NaN/inf, no wrap).
- Must not amplify silence/noise floors into speech-like level (noise gate), must not pump audibly (smoothed gain, slew-limited), must leave already-healthy audio close to unchanged (gain within about +/-3 dB of 1 for peaks between 0.3 and 0.9).
- Work in `worker-stt-realtime/bench/`, tests `../.venv/bin/python -m pytest tests -q` (159 pass now). No network, no `data/` access in tests. Do not modify `server.py`, gateway or deploy config (server integration is a later step once the benchmark shows it helps).

### Task 1: `StreamingAGC` and the `+agc` engine wrapper

**Files:** Create `bench/agc.py`, `bench/tests/test_agc.py`; modify `bench/engines.py` (`build()` accepts a `+agc` suffix on any LOCAL engine name, e.g. `nemo-480+agc`; cloud engines with `+agc` must raise ValueError) and add tests in `tests/test_engines_agc.py` (fake engine/stream).

**Interfaces:** `agc.StreamingAGC(target_peak=0.6, max_gain_db=40.0, min_gain_db=-12.0, floor_dbfs=-58.0, attack_ms=20.0, release_ms=1500.0, gain_slew_db_per_s=80.0, sr=16000)` with `process(x: np.ndarray) -> np.ndarray` (stateful, per stream; a new object per stream) and `reset()`. `engines.build("nemo-480+agc", threads)` returns an Engine whose `new_stream()` returns a stream that forwards `push(agc.process(x))` and delegates `text()/final()/native_eos()` (and `close()` if the wrapped stream has one).

Design (implementer may refine, keeping these behaviours): track a running PEAK envelope over the recent past (fast attack, slow release); desired gain = target_peak / max(envelope, floor); if the envelope is below the floor (silence/noise) hold the current gain instead of increasing it; clamp to [min_gain, max_gain]; move the applied gain toward the desired gain with a slew limit (dB/s) and interpolate linearly across each chunk to avoid steps; hard-limit the output.

Required tests (write first, show red, then green; mutation-check each protective behaviour and report):
1. Quiet speech-like signal (bursts of a 200 Hz + 1 kHz + 2.5 kHz mix at peak 0.007 with 30% silence) fed in 640-sample chunks: after the first 1.0 s, output peak of later bursts is within [0.25, 0.85].
2. Healthy signal (same shape at peak 0.7): output peak stays within [0.5, 0.9] (about +/-3 dB) throughout, including the first chunks.
3. Loud signal (peak 1.0 clipped) never produces |y| > 1.0 and no NaN.
4. Silence / low noise floor (Gaussian at -70 dBFS RMS, 5 s): the output RMS never exceeds -50 dBFS (the gate holds gain; noise is not blown up); pure zeros stay zeros.
5. Chunk-size independence: processing the same signal in 320, 640, 1280 and 4000-sample chunks gives output within a small tolerance of each other in the region after settling (RMS within 1.5 dB per second).
6. Statefulness: process(concat(a,b)) equals process(a) then process(b) for the SAME object (streaming equivalence up to floating tolerance); reset() restores the initial state; two objects are independent.
7. Level step: quiet (0.01) for 3 s then loud (0.7) for 3 s: no output sample exceeds 1.0, and the gain drops within about 150 ms of the loud onset (peak of the first 300 ms after onset is <= 1.0 and the later 1 s is within [0.4, 0.85]).
8. Edge cases: empty array returns empty; int16 input raises TypeError or is converted explicitly (state which); NaN/inf input raises ValueError.
9. Wrapper: with a fake engine recording pushed arrays, `build("fake+agc")`-style wrapping pushes processed audio (not the original), delegates text/final/native_eos/close, creates a fresh AGC per new_stream, and `dg-flux+agc` / `el-scribe+agc` raise ValueError.

- [ ] Steps: tests first (red), implement, green (whole suite), commit only the files you changed.

### Task 2 (CONTROLLER steps, not for subagents)
Run nemo-480+agc on: pub_fleurs, pub_fleurs_norm, pub_fleurs_tel, pub_earnings (n=50) and the real-call verified set (`--set real --only-verified`, local engine so no third-party gate; result file must be named `real__nemo-480+agc__real.json`); compare with the baselines above; success = pub_fleurs (raw) comes within ~1.5 points of pub_fleurs_norm and no set regresses by more than ~0.7 points. Write results into `PUBLIC_EVAL.md` (new section) and commit.
