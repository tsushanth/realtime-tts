"""
WER-based evaluation on a GENUINELY noisy real-world recording (this pass).

Every prior verification pass in this repo (make_and_verify.py, expand_coverage.py,
eval_deepfilternet.py) used the same methodology: take clean-or-relatively-clean speech,
DIGITALLY MIX IN synthetic noise after the fact, then measure SNR recovery. Even the
"real_speech_drone" case used a real speech recording, but the noise on top of it was still
synthetic and digitally added - the noise was never actually captured through a real recording
chain together with the speech.

This script tests a genuinely different scenario: a recording where the noise is NOT digitally
added at all. It's baked into the source audio because it was captured live, through a real
1941 broadcast/recording chain (radio pickup, disc-cutting lathe, decades of media transfers).
There's no clean reference to compute SNR against, so this uses a TRANSCRIPT-BASED metric
instead: word error rate (WER) from OpenAI Whisper transcription, against the well-documented
official text of the speech, before and after htdemucs isolation.

**Audio source**: Internet Archive item `FDR_Declares_War_19411208`
(https://archive.org/details/FDR_Declares_War_19411208), a public-domain 1941 radio recording of
President Franklin D. Roosevelt's "Day of Infamy" address to Congress. Downloaded as
`fdr_infamy_full.mp3` (not committed - large, and re-derivable from the archive.org URL below);
trimmed to a 60s excerpt (the famous opening paragraph, timestamp 0:20-1:20 of the source file)
and resampled to 22.05kHz mono 16-bit as `real_noisy_fdr_infamy.wav` (committed).

This recording has real, audible period noise: broadcast/disc surface hiss and crackle,
room/microphone coloration from a 1941 radio PA system, and generational loss from decades of
analog-to-analog transfers before digitization - none of it added by us. This is exactly the
kind of "real background noise baked into a real recording" that the prior synthetic-mixing
methodology never tested.

**Ground truth transcript**: the opening paragraph of the "Day of Infamy" speech is one of the
most widely quoted and independently verified passages in US political history - the official
text is published by the National Archives (Records of the US Senate) and reproduced verbatim
on Wikipedia's "Day of Infamy speech" article. It is NOT derived from a transcription of this
audio file (that would be circular) - it's the independently-published historical record of
what Roosevelt said, used here as ground truth to score transcriptions against.

**Method**:
  1. Transcribe the noisy original (`real_noisy_fdr_infamy.wav`) with Whisper.
  2. Run it through `Separator.separate` (the same code path `/v1/isolate` uses) to get the
     htdemucs-isolated vocals stem.
  3. Transcribe the isolated output with Whisper.
  4. Compute WER for both transcriptions against the ground-truth text (word-level Levenshtein
     edit distance / reference word count - implemented here directly, no extra dependency).
  5. Compare.

    .venv/bin/python verification/eval_real_noise_wer.py
"""
import json
import os
import re
import subprocess
import sys

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from server import Separator  # noqa: E402

AUDIO_IN = os.path.join(HERE, "real_noisy_fdr_infamy.wav")
ISOLATED_OUT = os.path.join(HERE, "real_noisy_fdr_infamy_isolated.wav")

# Official text (National Archives / Wikipedia "Day of Infamy speech"), covering exactly the
# ~60s excerpt used here (the Vice President's introduction plus FDR's opening paragraph, up to
# "at the solicitation of Japan"). Independently sourced, not derived from this audio.
GROUND_TRUTH = """
Mr Vice President Mr Speaker members of the Senate and the House of Representatives
Yesterday December 7 1941 a date which will live in infamy
the United States of America was suddenly and deliberately attacked by naval and air forces
of the Empire of Japan
The United States was at peace with that nation and at the solicitation of Japan
"""


def normalize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return [w for w in text.split() if w]


def word_error_rate(hypothesis: str, reference: str) -> dict:
    """Word-level Levenshtein edit distance / reference word count. No external WER dependency
    (jiwer was not installed in this environment - see README) - this is the standard WER
    definition (substitutions + insertions + deletions) / len(reference words)."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    edits = dp[n][m]
    return {
        "reference_words": n,
        "hypothesis_words": m,
        "edit_distance": edits,
        "wer": round(edits / n, 4) if n else None,
    }


def transcribe(wav_path: str, whisper_model: str = "small") -> str:
    out_dir = os.path.join(HERE, "_whisper_tmp")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["whisper", wav_path, "--model", whisper_model, "--language", "English",
         "--fp16", "False", "--output_dir", out_dir, "--output_format", "txt"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = os.path.splitext(os.path.basename(wav_path))[0]
    with open(os.path.join(out_dir, base + ".txt")) as f:
        return f.read().strip()


def main():
    if not os.path.exists(AUDIO_IN):
        raise SystemExit(
            f"missing {AUDIO_IN} - see this script's docstring for the archive.org source and "
            "how the excerpt was trimmed (ffmpeg -ss 20 -t 60)")

    audio, sr = sf.read(AUDIO_IN, dtype="float32", always_2d=False)
    print(f"loaded {AUDIO_IN}: {len(audio)/sr:.1f}s @ {sr}Hz")

    print("running htdemucs isolation (Separator.separate, same path as /v1/isolate)...")
    sep = Separator(os.environ.get("MODEL_NAME", "htdemucs"))
    isolated, out_sr = sep.separate(audio.reshape(-1, 1), sr, "vocals")
    sf.write(ISOLATED_OUT, isolated, out_sr, subtype="PCM_16")
    print(f"wrote {ISOLATED_OUT}")

    print("transcribing noisy original with whisper (this downloads model weights on first run)...")
    text_before = transcribe(AUDIO_IN)
    print("--- noisy original transcript ---")
    print(text_before)

    print("transcribing htdemucs-isolated output with whisper...")
    text_after = transcribe(ISOLATED_OUT)
    print("--- isolated transcript ---")
    print(text_after)

    wer_before = word_error_rate(text_before, GROUND_TRUTH)
    wer_after = word_error_rate(text_after, GROUND_TRUTH)

    result = {
        "audio_source": "https://archive.org/details/FDR_Declares_War_19411208",
        "excerpt": "0:20-1:20 of source mp3 (60s), resampled 22.05kHz mono 16-bit",
        "ground_truth": " ".join(normalize(GROUND_TRUTH)),
        "transcript_before_isolation": text_before,
        "transcript_after_isolation": text_after,
        "wer_before_isolation": wer_before,
        "wer_after_isolation": wer_after,
    }
    with open(os.path.join(HERE, "wer_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"WER before isolation (noisy original): {wer_before['wer']}  "
          f"({wer_before['edit_distance']}/{wer_before['reference_words']} word edits)")
    print(f"WER after isolation (htdemucs vocals):  {wer_after['wer']}  "
          f"({wer_after['edit_distance']}/{wer_after['reference_words']} word edits)")
    print("written to verification/wer_results.json")


if __name__ == "__main__":
    main()
