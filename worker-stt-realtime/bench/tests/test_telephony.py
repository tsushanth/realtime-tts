import numpy as np

from telephony import mulaw_roundtrip, to_telephone


def _tone(freq, sr=16000, secs=1.0, amp=0.5):
    t = np.arange(int(sr * secs)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _band_energy(x, lo, hi, sr=16000):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return spec[(f >= lo) & (f < hi)].sum()


def test_mulaw_roundtrip_is_close_and_shape_preserving():
    x = _tone(440, sr=8000)
    y = mulaw_roundtrip(x)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.max(np.abs(y - x)) < 0.05          # companding error bound for a 0.5-amplitude tone


def test_mulaw_roundtrip_clips_out_of_range_input():
    y = mulaw_roundtrip(np.array([2.0, -2.0, 0.0], dtype=np.float32))
    assert np.all(np.abs(y) <= 1.0 + 1e-6)


def test_to_telephone_length_and_dtype():
    x = _tone(1000)
    y = to_telephone(x)
    assert y.dtype == np.float32 and abs(len(y) - len(x)) <= 2


def test_to_telephone_removes_high_band_and_keeps_speech_band():
    x = _tone(1000) + _tone(6000)
    y = to_telephone(x, snr_db=60.0)
    high = _band_energy(y, 4200, 8000); mid = _band_energy(y, 500, 3000)
    assert high < 1e-3 * mid                      # 6 kHz component is gone
    assert mid > 0.05 * _band_energy(x, 500, 3000)  # 1 kHz component survives


def test_to_telephone_is_deterministic_per_seed_and_differs_across_seeds():
    x = _tone(800)
    assert np.array_equal(to_telephone(x, seed=1), to_telephone(x, seed=1))
    assert not np.array_equal(to_telephone(x, seed=1), to_telephone(x, seed=2))


def test_to_telephone_silence_stays_silent():
    y = to_telephone(np.zeros(16000, dtype=np.float32))
    assert np.max(np.abs(y)) < 0.02
