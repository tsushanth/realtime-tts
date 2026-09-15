#!/usr/bin/env python3
# Response-onset latency analysis for a dual-channel phone-call recording.
# Measures the objective, audio-derived counterpart to call-loop-poc's
# server-side [latency] logs, applied identically to BOTH legs of a
# mystery-shopper run (ours vs Retell), because Retell's side has no server
# logs of ours to read — the recording is the only fair common ground.
#
# Twilio dual-channel recordings put the caller (From — our shopper, since
# /place-test-call dials out) on channel 0 and the callee (To — the business/
# agent) on channel 1 consistently for both legs. We VAD each channel, find
# each agent speech segment, and measure its onset minus the preceding
# shopper segment's end = the caller's perceived response latency per turn.
#
# The same numbers only surfaced server-side (llmTtfbMs/ttsLegMs/responseMs)
# exist for OUR engine; this gives a comparable single number for anyone.
#
# Usage:
#   TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=... \
#   python3 analyze-call-ttfb.py --recording <RecordingSid> [--json]
#   python3 analyze-call-ttfb.py --audio local.wav [--json]
#
# Options:
#   --shopper-channel 0|1   channel index for the shopper (default 0)
#   --json                 emit only the JSON summary (for run.sh)
#   --min-segment-ms       min voiced segment to count as a turn (default 150)
#   --fill-gap-ms          collapse intra-turn gaps below this (default 300)
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import urllib.request
import wave
from base64 import b64encode
from pathlib import Path

FRAME_MS = 20
ABSOLUTE_FLOOR_DB = -45.0
THRESHOLD_BELOW_PCTL_DB = 12.0


def fetch_recording(twilio_sid, twilio_token, recording_sid, tmpdir):
    url = (
        f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Recordings/"
        f"{recording_sid}.wav"
    )
    auth = b64encode(f"{twilio_sid}:{twilio_token}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if resp.status != 200 or len(data) == 0:
        raise RuntimeError(f"recording fetch failed HTTP {resp.status}")
    p = Path(tmpdir) / f"{recording_sid}.wav"
    p.write_bytes(data)
    return p


def probe_channels(ffprobe, audio):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels", "-of", "csv=p=0", str(audio)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def split_channels(ffmpeg, audio, tmpdir):
    base = Path(audio).stem
    left = Path(tmpdir) / f"{base}_c0.wav"
    right = Path(tmpdir) / f"{base}_c1.wav"
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(audio),
         "-filter_complex", "channelsplit=channel_layout=stereo[c0][c1]",
         "-map", "[c0]", str(left), "-map", "[c1]", str(right)],
        check=True,
    )
    return left, right


def read_pcm(wav_path):
    with wave.open(str(wav_path), "rb") as w:
        rate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    n = len(raw) // 2
    samples = struct.unpack(f"<{n}h", raw)
    return samples, rate


def frame_rms_db(samples, start, count, bits_per_unit=32768.0):
    if count <= 0:
        return None
    acc = 0.0
    for s in samples[start : start + count]:
        acc += (s / bits_per_unit) ** 2
    m = acc / count
    if m <= 1e-12:
        return None
    return 20.0 * math.log10(math.sqrt(m))


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int((p / 100.0) * (len(s) - 1)))
    return s[idx]


def silence_free_vad(samples, rate):
    """20ms-frame energy VAD. Returns list of voiced (start_ms, end_ms)."""
    frame = int((FRAME_MS / 1000.0) * rate)
    nframes = len(samples) // frame
    dbs = [frame_rms_db(samples, i * frame, frame) for i in range(nframes)]
    voiced = [i for i, d in enumerate(dbs) if d is not None]
    ref = percentile([dbs[i] for i in voiced], 80)
    if ref is None:
        return []
    thr = max(ref - THRESHOLD_BELOW_PCTL_DB, ABSOLUTE_FLOOR_DB)
    segs = []
    cur = None
    for i in range(nframes):
        d = dbs[i]
        if d is not None and d > thr:
            if cur is None:
                cur = [i, i]
            else:
                cur[1] = i
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)
    return [(a * FRAME_MS, b * FRAME_MS + FRAME_MS) for a, b in segs]


def merge_segments(segs, min_ms, fill_ms):
    merged = []
    for s in segs:
        if not merged or s[0] - merged[-1][1] > fill_ms:
            merged.append(list(s))
        else:
            merged[-1][1] = max(merged[-1][1], s[1])
    return [tuple(s) for s in merged if s[1] - s[0] >= min_ms]


def measure_latencies(shopper, agent, max_precede_ms=30000):
    """For each agent segment: onset - end of the nearest prior shopper
    segment, skipping overlaps (barge-in) and the first exchange."""
    latencies = []
    measured = 0
    skipped_overlap = 0
    for a_start, a_end in agent:
        # The caller is still talking when the agent starts (barge-in) — no
        # clean "response latency" exists for this turn. Admissible overlap is
        # only when a segment genuinely spans the agent's onset
        # (s_start < a_start < s_end); a shopper segment that merely ENDS
        # later in the call is irrelevant.
        if any(s_start < a_start < s_end for s_start, s_end in shopper):
            skipped_overlap += 1
            continue
        prior = None
        for s_start, s_end in shopper:
            if s_end <= a_start and (prior is None or s_end > prior[1]):
                prior = (s_start, s_end)
        if prior is None:
            continue
        gap_ms = a_start - prior[1]
        if gap_ms > max_precede_ms:
            continue
        latencies.append(gap_ms)
        measured += 1
    return latencies, measured, skipped_overlap


def generic_gaps(segs, max_gap_ms=30000):
    """Mono fallback: gaps between consecutive voiced segments."""
    gaps = []
    for i in range(1, len(segs)):
        g = segs[i][0] - segs[i - 1][1]
        if 0 < g <= max_gap_ms:
            gaps.append(g)
    return gaps


def summarize(latencies):
    if not latencies:
        return {"n": 0}
    s = sorted(latencies)
    n = len(s)
    return {
        "n": n,
        "median_ms": int(percentile(s, 50)),
        "p90_ms": int(percentile(s, 90)),
        "p95_ms": int(percentile(s, 95)),
        "max_ms": int(s[-1]),
        "mean_ms": int(round(sum(s) / n)),
        "min_ms": int(s[0]),
    }


def probe_duration(ffprobe, audio):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", help="Twilio RecordingSid to fetch + analyze")
    ap.add_argument("--audio", help="local audio file instead of a Twilio recording")
    ap.add_argument("--shopper-channel", type=int, default=0, choices=[0, 1])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-segment-ms", type=int, default=150)
    ap.add_argument("--fill-gap-ms", type=int, default=300)
    args = ap.parse_args()

    if not args.recording and not args.audio:
        ap.error("need --recording <RecordingSid> or --audio <file>")

    twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    twilio_token = os.environ.get("TWILIO_AUTH_TOKEN")
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    ffprobe = os.environ.get("FFPROBE", "ffprobe")

    with tempfile.TemporaryDirectory() as tmp:
        if args.recording:
            if not twilio_sid or not twilio_token:
                ap.error("TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN required for --recording")
            audio = fetch_recording(twilio_sid, twilio_token, args.recording, tmp)
        else:
            audio = Path(args.audio)

        channels = probe_channels(ffprobe, audio)
        duration_s = round(probe_duration(ffprobe, audio))
        if channels == 2:
            left, right = split_channels(ffmpeg, audio, tmp)
            ch0, ch1 = left, right
        else:
            ch0 = ch1 = audio  # mono — same file both channels, generic gap fallback

        def channel_segs(path):
            samples, rate = read_pcm(path)
            segs = merge_segments(silence_free_vad(samples, rate),
                                  args.min_segment_ms, args.fill_gap_ms)
            return segs, rate

        shopper_segs, rate = channel_segs(ch0 if args.shopper_channel == 0 else ch1)
        agent_segs, _ = channel_segs(ch1 if args.shopper_channel == 0 else ch0)

        if channels == 2:
            latencies, measured, overlaps = measure_latencies(shopper_segs, agent_segs)
            result = {
                "channels": 2,
                "duration_s": duration_s,
                "shopper_turns": len(shopper_segs),
                "agent_turns": len(agent_segs),
                "agent_turns_measured": measured,
                "overlaps_skipped": overlaps,
                "response_latency_ms": summarize(latencies),
            }
        else:
            gaps = generic_gaps(shopper_segs)
            result = {
                "channels": 1,
                "note": "mono recording — no channel attribution; inter-segment gaps only",
                "duration_s": duration_s,
                "audio_turns": len(shopper_segs),
                "inter_turn_gap_ms": summarize(gaps),
                "response_latency_ms": summarize(gaps),
            }

    if args.json:
        print(json.dumps(result))
        return

    for k, v in result.items():
        if isinstance(v, dict):
            print(f"{k}: {json.dumps(v)}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()