"""Replays the calibration data (calibration_gates.json, made by calibrate_gates2.py from 20 LibriTTS-R
speakers) through gates.evaluate(), plus synthetic-signal checks.   python3 test_gates.py  (numpy, scipy)"""
import json, random, sys, os, unittest, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import gates

D = json.load(open(os.path.join(os.path.dirname(__file__), "calibration_gates.json")))
BY = {}
for c in D["clean"]:
    BY.setdefault(c["spk"], []).append(c)


def ev(clips, cps=None):
    return gates.evaluate([c["snr"] for c in clips], [c["band"] for c in clips], [c["f0"] for c in clips],
                          cps if cps is not None else [c["chars"] / c["dur"] for c in clips])


def codes(res):
    r, w = res
    return (r["code"] if r else None), sorted(x["code"] for x in w)


class Replay(unittest.TestCase):
    def test_clean_speakers_are_never_rejected(self):
        for spk, cs in BY.items():
            r, _ = ev(cs)
            self.assertIsNone(r, spk)

    def test_shuffled_transcripts_are_rejected(self):
        rng = random.Random(3)
        for spk, cs in BY.items():
            cps = [cs[rng.randrange(len(cs))]["chars"] / c["dur"] for c in cs]
            self.assertEqual(codes(ev(cs, cps))[0], "transcript_mismatch", spk)

    def test_male_plus_female_mixture_is_rejected_most_of_the_time(self):
        hits = n = 0
        M = [s for s, cs in BY.items() if cs[0]["g"] == "M"]; F = [s for s, cs in BY.items() if cs[0]["g"] == "F"]
        for a, b in itertools.product(M, F):
            mix = BY[a] + BY[b][:30]
            n += 1; hits += codes(ev(mix))[0] == "multiple_speakers"
        self.assertGreater(hits / n, 0.7)  # measured 0.81 at a 1/3 minority share; same-gender mixtures are NOT detected

    def test_noise_levels(self):
        rng = np.random.default_rng(0)
        sr = 16000
        t = np.arange(sr * 4) / sr
        speech = (np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 3 * t) > 0)) * 0.3  # bursts with pauses
        clean = gates.snr_estimate_db(speech + rng.standard_normal(len(t)) * 1e-4, sr)
        loud = gates.snr_estimate_db(speech + rng.standard_normal(len(t)) * 0.2, sr)
        self.assertGreater(clean, 30); self.assertLess(loud, gates.THRESHOLDS["snr_reject_db"] + 6)

    def test_bandwidth(self):
        from scipy.signal import butter, sosfilt
        rng = np.random.default_rng(1)
        sr = 16000
        x = rng.standard_normal(sr * 3)
        narrow = sosfilt(butter(6, [300, 3400], "band", fs=sr, output="sos"), x)
        self.assertLess(gates.band_ratio_db(narrow, sr), gates.THRESHOLDS["band_warn_db"])
        self.assertGreater(gates.band_ratio_db(x, sr), gates.THRESHOLDS["band_warn_db"])
        self.assertEqual(gates.band_ratio_db(x[:8000], 8000), -120.0)  # native 8 kHz always flagged


if __name__ == "__main__":
    unittest.main()
