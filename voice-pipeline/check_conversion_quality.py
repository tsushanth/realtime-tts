"""Evidence-gathering for the VC MVP, since there is no audio playback available here.

Measures (no playback needed):
  - duration match: converted output vs source (timing/prosody preserved -> durations should be close)
  - energy/RMS envelope correlation between source and output (same speech rhythm -> correlated envelope)
  - pitch (median F0) of output vs target reference vs source (voice identity should have moved toward
    the target, away from the source) - reuses the same autocorrelation estimator as voice-pipeline/train_job.py
  - optional: faster-whisper/whisper transcript of source vs output, word-level similarity (content
    fidelity - did the words survive conversion) if `openai-whisper` or `faster-whisper` is installed

The metric implementations and the automated pass/fail heuristics now live in voice-pipeline/quality.py
(shared with convert_job.py, which runs the same gate automatically right after each conversion inside
the Modal container). This script is the human-facing CLI wrapper around that module.

Usage: python3 check_conversion_quality.py --source s.wav --target t.wav --output out.wav
"""
import argparse
import sys

from quality import load_mono, median_f0, envelope_correlation, validate_conversion_quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--transcribe", action="store_true", help="also run whisper on source+output and diff words")
    args = ap.parse_args()

    src, sr_s = load_mono(args.source)
    tgt, sr_t = load_mono(args.target)
    out, sr_o = load_mono(args.output)

    dur_src, dur_out = len(src) / sr_s, len(out) / sr_o
    print(f"duration: source={dur_src:.2f}s output={dur_out:.2f}s ratio={dur_out / dur_src:.3f}")

    f0_src = median_f0(src, sr_s)
    f0_tgt = median_f0(tgt, sr_t)
    f0_out = median_f0(out, sr_o)
    print(f"median F0 (Hz): source={f0_src:.0f} target={f0_tgt:.0f} output={f0_out:.0f}")
    if f0_src and f0_tgt and f0_out:
        d_to_src = abs(f0_out - f0_src)
        d_to_tgt = abs(f0_out - f0_tgt)
        print(f"  output pitch distance: to source={d_to_src:.1f} Hz, to target={d_to_tgt:.1f} Hz "
              f"-> {'moved toward TARGET (expected for VC)' if d_to_tgt < d_to_src else 'closer to SOURCE (unexpected)'}")

    corr = envelope_correlation(src, sr_s, out, sr_o)
    if corr is not None:
        print(f"energy-envelope correlation (source vs output, time-normalized): {corr:.3f} "
              f"({'consistent with preserved rhythm/prosody' if corr > 0.4 else 'weak - check output audibly'})")

    gate = validate_conversion_quality(args.source, args.target, args.output)
    print(f"\nquality gate: {'PASS' if gate['passed'] else 'FAIL'}")
    for r in gate["reasons"]:
        print(f"  FAIL reason: {r}")
    for w in gate["warnings"]:
        print(f"  warning: {w}")

    if args.transcribe:
        try:
            import whisper
            model = whisper.load_model("base")
            t_src = model.transcribe(args.source)["text"].strip()
            t_out = model.transcribe(args.output)["text"].strip()
            print(f"whisper source: {t_src!r}")
            print(f"whisper output: {t_out!r}")
            import difflib
            ratio = difflib.SequenceMatcher(None, t_src.lower(), t_out.lower()).ratio()
            print(f"content similarity (SequenceMatcher ratio): {ratio:.3f}")
        except ImportError:
            print("openai-whisper not installed; skipping transcript check (pip install openai-whisper)", file=sys.stderr)


if __name__ == "__main__":
    main()
