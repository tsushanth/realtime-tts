"""Pure-numpy tests for audiofmt (no model needed). Run inside the image:
  docker run --rm -v $PWD/tests:/tests -e PYTHONPATH=/app piper-tts-feat python -m unittest discover -s /tests -p 'test_audiofmt.py' -v
G.711 is compared bit-for-bit against Python 3.11's audioop when it is importable."""
import unittest
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

import audiofmt

try:
    import audioop  # removed in 3.13
except ImportError:
    audioop = None

ALL16 = np.arange(-32768, 32768, dtype=np.int16)


def snr_db(ref, test):
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - test) ** 2))


def speechlike(n=48000, sr=8000):
    t = np.arange(n) / sr
    x = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.12 * np.sin(2 * np.pi * 730 * t) + 0.05 * np.sin(2 * np.pi * 2100 * t)
    return (x * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) * 32767).astype(np.int16)


class G711(unittest.TestCase):
    @unittest.skipIf(audioop is None, "audioop unavailable")
    def test_ulaw_matches_audioop_all_values(self):
        self.assertEqual(audiofmt.lin2ulaw(ALL16), audioop.lin2ulaw(ALL16.tobytes(), 2))

    @unittest.skipIf(audioop is None, "audioop unavailable")
    def test_alaw_matches_audioop_all_values(self):
        self.assertEqual(audiofmt.lin2alaw(ALL16), audioop.lin2alaw(ALL16.tobytes(), 2))

    @unittest.skipIf(audioop is None, "audioop unavailable")
    def test_decoders_match_audioop(self):
        b = bytes(range(256))
        self.assertEqual(audiofmt.ulaw2lin(b).tobytes(), audioop.ulaw2lin(b, 2))
        self.assertEqual(audiofmt.alaw2lin(b).tobytes(), audioop.alaw2lin(b, 2))

    def test_roundtrip_snr(self):
        x = speechlike()
        for enc, dec in ((audiofmt.lin2ulaw, audiofmt.ulaw2lin), (audiofmt.lin2alaw, audiofmt.alaw2lin)):
            snr = snr_db(x, dec(enc(x)))
            print(f"{enc.__name__} round-trip SNR {snr:.1f} dB")
            self.assertGreater(snr, 30)  # G.711 gives ~35-38 dB on speech-level signals

    def test_length_and_extremes(self):
        x = np.array([-32768, 32767, 0, -1, 1], dtype=np.int16)
        self.assertEqual(len(audiofmt.lin2ulaw(x)), 5)
        self.assertEqual(len(audiofmt.lin2alaw(x)), 5)
        self.assertEqual(audiofmt.lin2ulaw(np.array([], dtype=np.int16)), b"")


class Formats(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(audiofmt.parse_format(None), "pcm_24000")
        for f in audiofmt.FORMATS:
            self.assertEqual(audiofmt.parse_format(f), f)
        for bad in ("opus", "", "PCM_24000", 5, ["pcm_8000"]):
            with self.assertRaises(ValueError):
                audiofmt.parse_format(bad)

    def test_default_path_identical_to_legacy(self):
        """pcm_24000 must equal the pre-change formula bit-for-bit (22050 -> 24000)."""
        rng = np.random.default_rng(0)
        audio = rng.uniform(-1, 1, 22050).astype(np.float32)
        ratio = Fraction(24000, 22050).limit_denominator(1000)
        legacy = resample_poly(audio, ratio.numerator, ratio.denominator)
        legacy = (np.clip(legacy, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        data, sr = audiofmt.encode(audio, 22050, "pcm_24000")
        self.assertEqual(sr, 24000)
        self.assertEqual(data, legacy)


def tone(freq, sr=22050, secs=1.0, amp=0.5):
    t = np.arange(int(sr * secs)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms_db(x, trim=400):
    x = x[trim:-trim].astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


class Resample(unittest.TestCase):
    def test_length(self):
        y = audiofmt.resample(tone(1000), 22050, 8000)
        self.assertAlmostEqual(len(y), 8000, delta=2)

    def test_passband_flat(self):
        ref = rms_db(tone(1000) * 1.0)
        for f in (300, 1000, 2000, 3400):
            y = audiofmt.resample(tone(f), 22050, 8000)
            d = rms_db(y, 100) - rms_db(tone(f))
            self.assertLess(abs(d), 0.5, f"{f} Hz gain {d:.2f} dB")

    def test_stopband_alias_rejection(self):
        """Tones above the 4 kHz output Nyquist must be crushed, not folded back in-band."""
        for f in (4600, 5000, 6500, 9000):
            y = audiofmt.resample(tone(f), 22050, 8000)
            att = rms_db(y, 100) - rms_db(tone(f))
            print(f"{f} Hz -> {att:.1f} dB")
            self.assertLess(att, -60)

    def test_encode_all_formats(self):
        a = tone(500)
        p8, sr8 = audiofmt.encode(a, 22050, "pcm_8000")
        self.assertEqual(sr8, 8000)
        pcm8 = np.frombuffer(p8, dtype=np.int16)
        for fmt, dec in (("mulaw_8000", audiofmt.ulaw2lin), ("alaw_8000", audiofmt.alaw2lin)):
            b, sr = audiofmt.encode(a, 22050, fmt)
            self.assertEqual((len(b), sr), (len(pcm8), 8000))
            self.assertGreater(snr_db(pcm8, dec(b)), 30)


if __name__ == "__main__":
    unittest.main()
