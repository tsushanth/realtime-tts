"""Dataset quality gates for the voice pipeline (pure numpy/scipy, no Modal, so they are unit-testable
and shared by train_job.py and the calibration job). All measurements are per clip on the ORIGINAL audio
(before resampling/normalisation) and are aggregated by the caller. Thresholds are in THRESHOLDS and were
calibrated by calibrate_gates2.py on LibriTTS-R speakers (see voice-pipeline/README-gates in the final
report / CALIBRATION notes in that file). These are heuristics, not guarantees: see each function.
"""
import numpy as np

THRESHOLDS = {
    # SNR estimate (p90 - p10 of 25 ms frame energy, dB), median over clips.
    "snr_reject_db": 12.0, "snr_warn_db": 20.0,
    # High/low band energy ratio (4.5-7.5 kHz vs 0.3-3.4 kHz, dB), median over clips. Native 8 kHz files count as telephone.
    "band_warn_db": -35.0,
    # Two-cluster split of per-clip median F0 (semitones between cluster means / minority fraction).
    # Reject only when the two pitch clusters straddle the male/female boundary (lower cluster < 150 Hz,
    # upper > 170 Hz): 0/20 single speakers rejected, ~80% of male+female mixtures caught, ~0% same-gender.
    "spk_gap_reject_st": 6.0, "spk_gap_warn_st": 4.5, "spk_minor_frac": 0.15, "spk_lo_hz": 150.0, "spk_hi_hz": 170.0,
    # Transcript characters per second of audio. Natural read speech is ~10-20.
    "cps_lo": 5.0, "cps_hi": 28.0, "cps_bad_clip_reject_frac": 0.30, "cps_bad_clip_warn_frac": 0.10,
    # Coefficient of variation of chars/sec across clips: 0.13 median (max 0.16) for matched transcripts,
    # >= 0.55 when transcripts were shuffled among a speaker's own clips.
    "cps_cv_reject": 0.40, "cps_cv_warn": 0.28,
}


def frame_db(x, sr, win_s=0.025):
    n = max(1, int(win_s * sr))
    m = len(x) // n
    if m < 1:
        return np.array([])
    fr = x[: m * n].reshape(m, n)
    return 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)


def snr_estimate_db(x, sr):
    """Speech level (p90 frame dB) minus noise floor (p10 frame dB). Underestimates SNR on very tightly
    trimmed clean clips (no pauses -> the floor is soft speech) and overestimates when noise is speech-like
    (babble) or constant-level and the clip has no pauses at all. None if the clip is too short."""
    db = frame_db(x, sr)
    if len(db) < 20:
        return None
    return float(np.percentile(db, 90) - np.percentile(db, 10))


def band_ratio_db(x, sr):
    """10*log10(power in 4.5-7.5 kHz / power in 0.3-3.4 kHz). Real wideband speech is around -25..-40 dB;
    a signal that was telephone-band-limited (even if later stored at 16/22/44 kHz) is far lower.
    Returns -120 for native <=8 kHz audio (no content above 4 kHz can exist)."""
    from scipy.signal import welch
    if sr < 9000:
        return -120.0
    f, p = welch(x, sr, nperseg=1024)
    lo = p[(f >= 300) & (f <= 3400)].mean()
    hi = p[(f >= 4500) & (f <= min(7500, sr / 2 - 300))].mean()
    return float(10 * np.log10((hi + 1e-20) / (lo + 1e-20)))


def clip_f0_median(x, sr, max_seconds=12.0):
    """Median autocorrelation F0 (60-400 Hz) over voiced 40 ms frames of one clip; None if < 8 voiced frames."""
    x = x[: int(max_seconds * sr)]
    n = int(0.04 * sr)
    lo, hi = int(sr / 400), int(sr / 60)
    out = []
    for st in range(0, len(x) - n, n // 2):
        fr = x[st:st + n] - np.mean(x[st:st + n])
        if np.sqrt(np.mean(fr ** 2)) < 0.02:
            continue
        spec = np.fft.rfft(fr, 2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] / (ac[0] + 1e-9) > 0.5:
            out.append(sr / k)
    return float(np.median(out)) if len(out) >= 8 else None


def two_cluster_split(clip_f0s):
    """Optimal 1-D 2-means split of per-clip median F0 in semitones. Returns
    (gap_semitones, minority_fraction, separation) where separation = gap / pooled within-cluster std.
    Embedding-free: it can only see speakers whose pitch differs (e.g. male+female). Two speakers of similar
    pitch, or one speaker changing register, are not reliably distinguished."""
    v = np.sort(np.asarray([f for f in clip_f0s if f]))
    if len(v) < 20:
        return 0.0, 0.0, 0.0
    st = 12 * np.log2(v / np.median(v))
    best = None
    for i in range(3, len(st) - 2):
        a, b = st[:i], st[i:]
        ss = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if best is None or ss < best[0]:
            best = (ss, i)
    _, i = best
    a, b = st[:i], st[i:]
    within = np.sqrt(best[0] / max(1, len(st) - 2)) + 1e-6
    gap = float(b.mean() - a.mean())
    return gap, float(min(len(a), len(b)) / len(st)), float(gap / within)


def split_details(clip_f0s):
    """Like two_cluster_split but also returns the median F0 (Hz) of the lower and upper cluster:
    {gap, minor, sep, lo_hz, hi_hz} or None if there are too few voiced clips."""
    v = np.sort(np.asarray([f for f in clip_f0s if f]))
    gap, minor, sep = two_cluster_split(v)
    if len(v) < 20:
        return None
    st = 12 * np.log2(v / np.median(v))
    best = None
    for i in range(3, len(st) - 2):
        a, b = st[:i], st[i:]
        ss = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if best is None or ss < best[0]:
            best = (ss, i)
    i = best[1]
    return {"gap": gap, "minor": minor, "sep": sep, "lo_hz": float(np.median(v[:i])), "hi_hz": float(np.median(v[i:]))}


def chars_per_second(text, duration_s):
    return len(text.strip()) / max(duration_s, 1e-6)


def evaluate(snrs, bands, clip_f0s, cps_values, T=THRESHOLDS):
    """Turns per-clip measurements into (rejection or None, [warnings]). Each entry is a dict
    {code, message} with a plain-language message for the customer. cps_values: transcript chars/sec of every
    clip considered (before any clip is dropped)."""
    cps_flags = [not T["cps_lo"] <= v <= T["cps_hi"] for v in cps_values]
    cv = float(np.std(cps_values) / np.mean(cps_values)) if len(cps_values) >= 20 and np.mean(cps_values) > 0 else 0.0
    warnings, reject = [], None
    s = [v for v in snrs if v is not None]
    if s:
        med = float(np.median(s))
        if med < T["snr_reject_db"]:
            reject = {"code": "noisy_audio", "message": "The recordings have too much background noise (estimated signal-to-noise ratio %.0f dB). Re-record in a quiet room, close to the microphone." % med}
        elif med < T["snr_warn_db"]:
            warnings.append({"code": "background_noise", "message": "There is noticeable background noise (estimated %.0f dB signal-to-noise). The voice will likely reproduce it; a quieter room gives a cleaner voice." % med})
    if bands:
        med_b = float(np.median(bands))
        if med_b < T["band_warn_db"]:
            warnings.append({"code": "telephone_quality", "message": "These recordings look like telephone-quality audio (little content above 4 kHz). The custom voice will sound just as narrow and muffled; record with a headset or microphone at 16 kHz or higher for a fuller voice."})
    d = split_details(clip_f0s) if clip_f0s else None
    if d and d["minor"] >= T["spk_minor_frac"]:
        crosses = d["lo_hz"] < T["spk_lo_hz"] and d["hi_hz"] > T["spk_hi_hz"]
        if d["gap"] >= T["spk_gap_reject_st"] and crosses:
            reject = reject or {"code": "multiple_speakers", "message": "The recordings appear to contain both a lower (male-range) and a higher (female-range) voice, so more than one speaker. A custom voice must be one person only. Remove the other speaker's clips and upload again."}
        elif d["gap"] >= T["spk_gap_warn_st"] and d["sep"] >= 3.0:
            warnings.append({"code": "possible_multiple_speakers", "message": "The pitch of the recordings splits into two groups, which can mean more than one speaker or a very different speaking style in part of the set. Make sure every clip is the same person, speaking the same way."})
    if cps_flags:
        frac = sum(cps_flags) / len(cps_flags)
        if frac >= T["cps_bad_clip_reject_frac"] or cv >= T["cps_cv_reject"]:
            reject = reject or {"code": "transcript_mismatch", "message": "The transcript lengths do not fit the audio lengths (%d%% of clips clearly off), so the transcripts probably do not match the recordings. Check that each transcript is exactly what is said in its clip." % round(frac * 100)}
        elif frac >= T["cps_bad_clip_warn_frac"] or cv >= T["cps_cv_warn"]:
            warnings.append({"code": "transcript_mismatch_some", "message": "For some clips the transcript length does not fit the audio length (%d%% clearly off); off clips were left out. Check that every transcript is exactly what is said." % round(frac * 100)})
    return reject, warnings
