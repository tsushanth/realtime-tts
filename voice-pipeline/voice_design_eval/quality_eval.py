"""Voice-design quality/adherence eval (objective proxy, not a listening test).

Generates audio for 8 varied voice descriptions (age/gender/accent/tone combinations, including
some deliberately atypical ones as a stress test for whether the description actually steers
generation vs. the model defaulting to some generic output) and scores mean F0 (fundamental
frequency / pitch) as a proxy for whether the described age/gender lands in a physiologically
plausible pitch band. Writes results to quality_eval_results.json next to this script.

Methodology and what this can/cannot tell you: see the written caveat printed at the end and
duplicated in VOICE_DESIGN_MVP.md - short version, F0 range checks whether a described gender/age
"sounds physically plausible," not whether the voice actually sounds "warm," "gravelly," "noir
detective," etc. - those are not proxy-measurable and require a human listen.
"""
import json
import time
from pathlib import Path

import librosa
import modal
import numpy as np

OUT_DIR = Path(__file__).parent / "quality_audio"
OUT_DIR.mkdir(exist_ok=True)
RESULTS_PATH = Path(__file__).parent / "quality_eval_results.json"

# (id, description, text, expected_f0_band_hz, band_rationale)
CASES = [
    ("young_woman", "a young adult woman, cheerful and energetic, American accent",
     "I just got back from the most amazing trip, you have to hear about it!",
     (165, 300), "typical adult female speaking F0"),
    ("older_british_woman", "a warm, older British woman, gentle and a little frail, refined accent",
     "Would you like a cup of tea, dear? I've just put the kettle on.",
     (140, 240), "adult female; older speakers trend toward the lower end of the female band"),
    ("male_anchor", "a middle-aged man, deep and authoritative, American news anchor delivery",
     "Good evening. Our top story tonight comes from the capital.",
     (85, 160), "typical adult male speaking F0"),
    ("young_boy", "a young boy, high-pitched, excited and a little breathless",
     "Mom, mom, look what I found in the backyard, it's so cool!",
     (250, 400), "pre-pubescent child F0, well above adult male or female"),
    ("elderly_gravelly_man", "an elderly man, gravelly and slow, Southern American drawl",
     "Back in my day, we didn't have any of these newfangled gadgets.",
     (70, 140), "adult male, gravelly/creaky voice quality often lowers effective F0 further"),
    ("husky_woman", "a woman with a deep, husky, sultry voice, slow and deliberate",
     "Take your time. There's no rush at all tonight.",
     (120, 210), "female but deliberately low-register - adversarial case, overlaps male band"),
    ("nervous_high_man", "a man with a high-pitched, nervous, fast-talking voice",
     "Wait, wait, I don't think that's right, let me check again, sorry, sorry.",
     (140, 220), "male but deliberately high-register - adversarial case, overlaps female band"),
    ("neutral_narrator", "a calm, neutral voice, professional documentary narrator, no strong gender cues",
     "The migration begins at dawn, when the first light touches the water.",
     None, "control case - no directional pitch prediction, sanity check only"),
]


def mean_f0(wav_path: Path) -> dict:
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr
    )
    voiced = f0[~np.isnan(f0)]
    if len(voiced) == 0:
        return {"mean_f0_hz": None, "median_f0_hz": None, "voiced_frac": 0.0}
    return {
        "mean_f0_hz": round(float(np.mean(voiced)), 1),
        "median_f0_hz": round(float(np.median(voiced)), 1),
        "voiced_frac": round(float(len(voiced) / len(f0)), 3),
    }


def main():
    Model = modal.Cls.from_name("voice-design-dev", "VoiceDesignModel")
    m = Model()
    results = []
    for idx, (case_id, desc, text, band, rationale) in enumerate(CASES):
        jid = f"d-eval{idx:02d}abcd"  # bypasses api()'s jid_ok() regex since we call generate() directly;
        # fine for this offline eval script, not a real /designs submission. idx keeps every id unique.
        t0 = time.time()
        r = m.generate.remote(jid, desc, text)
        wall = round(time.time() - t0, 2)
        print(f"[{case_id}] generated in {wall}s -> {r}")
        results.append({
            "id": case_id, "description": desc, "text": text,
            "expected_band_hz": band, "rationale": rationale,
            "job_id": jid, "wall_seconds": wall, "gen_meta": r,
        })

    print("\nDownloading audio and computing F0...")
    import subprocess
    for r in results:
        jid = r["job_id"]
        dest = OUT_DIR / f"{r['id']}.wav"
        proc = subprocess.run(
            ["modal", "volume", "get", "voice-design-dev-jobs", f"{jid}/audio.wav", str(dest)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print("STDOUT:", proc.stdout, "STDERR:", proc.stderr)
        proc.check_returncode()
        f0 = mean_f0(dest)
        r["f0"] = f0
        band = r["expected_band_hz"]
        if band is None:
            r["band_verdict"] = "n/a (control)"
        elif f0["mean_f0_hz"] is None:
            r["band_verdict"] = "no voiced pitch detected"
        else:
            in_band = band[0] <= f0["mean_f0_hz"] <= band[1]
            r["band_verdict"] = "in expected band" if in_band else "OUTSIDE expected band"
        print(f"[{r['id']}] mean_f0={f0['mean_f0_hz']}Hz band={band} -> {r['band_verdict']}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
