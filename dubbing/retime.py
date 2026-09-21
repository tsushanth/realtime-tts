"""Silence-aware retiming + bounded time-stretch, to bring dubbed audio close to the source clip's
duration.

Two-stage fit, cheapest/least-audible adjustment first:
  1. Silence-aware retiming: trim or pad inter-sentence pauses in the dubbed audio so total
     duration moves toward the source duration, without touching speech segments at all. This is
     "free" quality-wise - it only changes pause length, not speech.
  2. Bounded time-stretch (+-15%): if retiming alone can't close the gap, apply a single global
     pitch-preserving tempo change as a last resort. Capped at +-15% because beyond that, tempo
     changes become audibly unnatural (this is a widely used rule of thumb for dubbing/timing
     work, not a hard technical limit).

Implementation note: the task suggested librosa or pyrubberband for the time-stretch step. This
environment had ~140MB of free disk space at the time of building this (`df -h /`), not enough to
pip install librosa (numpy+scipy+numba+librosa itself) or pyrubberband (needs the rubberband CLI)
without risking filling the disk further. ffmpeg was already installed system-wide, so this module
uses ffmpeg's `atempo` filter instead - a standard WSOLA-style pitch-preserving tempo filter,
functionally the same category of tool. Flagging this substitution explicitly per the task's ask
to "note which you used". If disk space frees up, swapping in librosa's `time_stretch` behind the
same `time_stretch()` function signature is a small change.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import os
from dataclasses import dataclass

MAX_STRETCH = 0.15  # +-15% bound, per task spec


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr}")
    return r


def get_duration_s(path: str) -> float:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=noprint_wrappers=1:nokey=1", path])
    return float(r.stdout.strip())


@dataclass
class Silence:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_silences(path: str, noise_db: float = -35.0, min_silence_s: float = 0.15) -> list[Silence]:
    """Inter-sentence pauses via ffmpeg's silencedetect filter (stderr-parsed, that's where it logs)."""
    r = subprocess.run(
        ["ffmpeg", "-i", path, "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", r.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", r.stderr)]
    return [Silence(s, e) for s, e in zip(starts, ends)]


def retime_silences(input_path: str, output_path: str, target_duration_s: float, scratch_dir: str | None = None) -> float:
    """Scale ONLY the inter-sentence silence gaps (not speech) so total duration moves toward
    target_duration_s. Returns the resulting duration. If there are no usable silences (e.g. one
    short sentence with no internal pause), this is a no-op and returns the input's duration
    unchanged - the caller falls back to time_stretch() for the remaining gap.
    """
    cur = get_duration_s(input_path)
    gap = target_duration_s - cur
    silences = detect_silences(input_path)
    if not silences or abs(gap) < 0.02:
        if output_path != input_path:
            _run(["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path])
        return cur

    total_silence = sum(s.duration for s in silences)
    # Never shrink a silence to below 30ms (audible splice artifact) or grow it more than 3x.
    scale = 1.0
    if gap > 0 and total_silence > 0:
        scale = 1.0 + gap / total_silence
    elif gap < 0 and total_silence > 0:
        scale = max(0.05, 1.0 + gap / total_silence)  # don't invert/negative-scale
    scale = min(scale, 3.0)

    # Build a filter graph that speeds up/slows down only the silent spans, via atrim/concat of
    # alternating speech/silence segments - each silence segment gets `atempo` scaled so ITS
    # duration changes by `scale`, while speech segments pass through with atempo=1 (unchanged).
    segments = []
    cursor = 0.0
    for s in silences:
        if s.start > cursor:
            segments.append((cursor, s.start, 1.0))       # speech, unscaled
        segments.append((s.start, s.end, scale))            # silence, scaled
        cursor = s.end
    if cursor < cur:
        segments.append((cursor, cur, 1.0))

    filters = []
    labels = []
    for i, (a, b, sc) in enumerate(segments):
        dur = b - a
        if dur <= 0:
            continue
        label = f"seg{i}"
        # atempo changes speed; to change *duration* of a silence by `sc`, tempo = 1/sc.
        tempo = max(0.5, min(2.0, 1.0 / sc)) if sc != 1.0 else 1.0
        filters.append(f"[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS,atempo={tempo}[{label}]")
        labels.append(f"[{label}]")
    concat = "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]"
    filter_complex = ";".join(filters + [concat])

    _run(["ffmpeg", "-y", "-i", input_path, "-filter_complex", filter_complex,
          "-map", "[out]", output_path])
    return get_duration_s(output_path)


def time_stretch(input_path: str, output_path: str, target_duration_s: float) -> tuple[float, float]:
    """Bounded global pitch-preserving tempo change (ffmpeg atempo), last-resort fit. Returns
    (applied_stretch_ratio, resulting_duration_s). Ratio is clamped to +-MAX_STRETCH; if the gap
    needs more than that, the clamp is applied and the result will NOT exactly match
    target_duration_s (documented limitation, not silently hidden)."""
    cur = get_duration_s(input_path)
    if cur <= 0:
        raise RuntimeError(f"zero-duration input: {input_path}")
    desired_ratio = cur / target_duration_s  # atempo>1 = faster/shorter
    ratio = max(1 - MAX_STRETCH, min(1 + MAX_STRETCH, desired_ratio))
    if abs(ratio - 1.0) < 0.005:
        if output_path != input_path:
            _run(["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path])
        return 1.0, cur
    _run(["ffmpeg", "-y", "-i", input_path, "-af", f"atempo={ratio}", output_path])
    return ratio, get_duration_s(output_path)


def fit_duration(input_path: str, output_path: str, target_duration_s: float, scratch_dir: str | None = None) -> dict:
    """Full two-stage fit: silence retiming, then bounded stretch for whatever gap remains."""
    scratch_dir = scratch_dir or tempfile.mkdtemp(prefix="dub_retime_")
    os.makedirs(scratch_dir, exist_ok=True)
    stage1_path = os.path.join(scratch_dir, "stage1_retimed.wav")

    before = get_duration_s(input_path)
    after_retime = retime_silences(input_path, stage1_path, target_duration_s, scratch_dir)
    ratio, final_duration = time_stretch(stage1_path, output_path, target_duration_s)

    return {
        "source_target_s": target_duration_s,
        "before_s": before,
        "after_silence_retime_s": after_retime,
        "stretch_ratio": ratio,
        "final_s": final_duration,
        "within_bound": abs(ratio - 1.0) <= MAX_STRETCH + 1e-6,
        "hit_target": abs(final_duration - target_duration_s) < 0.15,
    }
