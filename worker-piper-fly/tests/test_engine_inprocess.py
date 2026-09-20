"""In-process tests that need the model (imports server.py). Run inside the image:
  docker run --rm -e SESSION_SECRET=x -e PYTHONPATH=/app -v $PWD/tests:/tests piper-tts-feat \
      python -m unittest discover -s /tests -p 'test_engine_inprocess.py' -v"""
import unittest
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

import audiofmt
import server


def ph(text):
    """Fake phoneme list: one 'phoneme' per char, punctuation kept, like espeak's output shape."""
    return list(text)


class Synth(unittest.TestCase):
    def setUp(self):
        self.eng = server.engine
        rng = np.random.default_rng(1)
        t = np.arange(22050) / 22050
        self.fixed = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.standard_normal(22050)).astype(np.float32)
        self.orig = self.eng.voice.phoneme_ids_to_audio
        self.eng.voice.phoneme_ids_to_audio = lambda ids, cfg: self.fixed.copy()

    def tearDown(self):
        self.eng.voice.phoneme_ids_to_audio = self.orig

    def test_default_output_is_byte_identical_to_legacy_code(self):
        audio = self.fixed / np.max(np.abs(self.fixed))
        r = Fraction(24000, 22050).limit_denominator(1000)
        legacy = resample_poly(audio, r.numerator, r.denominator)
        legacy = (np.clip(legacy, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        data, dur, sr = self.eng.synth([1, 2, 3], 1.0)  # no fmt argument = default
        self.assertEqual(data, legacy)
        self.assertEqual(sr, 24000)
        self.assertAlmostEqual(dur, len(legacy) / 2 / 24000)

    def test_formats_consistent(self):
        pcm8 = np.frombuffer(self.eng.synth([1], 1.0, "pcm_8000")[0], dtype=np.int16)
        for fmt, dec in (("mulaw_8000", audiofmt.ulaw2lin), ("alaw_8000", audiofmt.alaw2lin)):
            data, dur, sr = self.eng.synth([1], 1.0, fmt)
            self.assertEqual(sr, 8000)
            self.assertEqual(len(data), len(pcm8))
            self.assertAlmostEqual(dur, len(pcm8) / 8000)
            err = pcm8.astype(float) - dec(data).astype(float)
            snr = 10 * np.log10(np.sum(pcm8.astype(float) ** 2) / np.sum(err ** 2))
            print(fmt, f"decoded-vs-pcm_8000 SNR {snr:.1f} dB")
            self.assertGreater(snr, 30)
        # the 24k stream and the 8k stream carry the same tone: compare RMS levels
        p24 = np.frombuffer(self.eng.synth([1], 1.0, "pcm_24000")[0], dtype=np.int16).astype(float)
        self.assertLess(abs(20 * np.log10(np.std(p24) / np.std(pcm8.astype(float)))), 0.5)


class Split(unittest.TestCase):
    def test_short_sentence_not_split(self):
        p = ph("Sure, I can help you with that.")  # 7 words
        self.assertEqual(server.split_clause(p), [p])

    def test_long_sentence_split_at_first_valid_clause(self):
        p = ph("Thanks for calling, I can help you with that and pull up your account right now.")
        a, b = server.split_clause(p)
        self.assertEqual("".join(a), "Thanks for calling,")
        self.assertEqual("".join(b), "I can help you with that and pull up your account right now.")
        self.assertEqual(a + [" "] + b, p)

    def test_first_half_too_short_skips_to_next_boundary(self):
        p = ph("Yes, I can help you with that, and then we will confirm the details today.")
        a, b = server.split_clause(p)
        self.assertEqual("".join(a), "Yes, I can help you with that,")

    def test_second_half_too_short_no_split(self):
        p = ph("I can help you with that request for your account right now, okay.")
        self.assertEqual(server.split_clause(p), [p])

    def test_no_boundary_no_split(self):
        p = ph("I can help you with that request for your account right now today.")
        self.assertEqual(server.split_clause(p), [p])

    def test_dash_and_semicolon(self):
        for sep in (";", ":", "—"):
            p = ph(f"One moment while I check that{sep} I will need your account number please.")
            self.assertEqual(len(server.split_clause(p)), 2, sep)

    def test_only_first_sentence_is_split(self):
        text = ("Thanks for calling Acme, I can help you with that and pull up your account right now. "
                "Then, as soon as you confirm your billing address, we will send the replacement out today.")
        off = server.engine.sentences(text, split_first=False)
        on = server.engine.sentences(text, split_first=True)
        self.assertEqual(len(off), 2)
        self.assertEqual(len(on), 3)
        self.assertEqual(on[2], off[1])  # later sentences untouched


if __name__ == "__main__":
    unittest.main()
