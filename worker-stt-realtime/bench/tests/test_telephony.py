import numpy as np
import pytest

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


def test_low_cut_attenuates_sub_300hz():
    # Both tones in one signal so the peak normalisation is shared; same input amplitude.
    y = to_telephone(_tone(100) + _tone(1000), snr_db=60.0)
    lo = _band_energy(y, 50, 200); mid = _band_energy(y, 800, 1200)
    assert lo < 10 ** (-20 / 10) * mid            # >= 20 dB down


def test_mulaw_quantises_to_at_most_256_levels():
    ramp = np.linspace(-1, 1, 20001, dtype=np.float32)
    assert len(np.unique(mulaw_roundtrip(ramp))) <= 256


def test_mulaw_is_not_identity():
    x = _tone(440, sr=8000, amp=0.3)
    assert not np.allclose(mulaw_roundtrip(x), x, atol=1e-4)


def test_to_telephone_empty_input():
    y = to_telephone(np.zeros(0, dtype=np.float32))
    assert y.dtype == np.float32 and len(y) == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_to_telephone_rejects_non_finite(bad):
    x = _tone(1000)
    x[10] = bad
    with pytest.raises(ValueError, match="non-finite audio"):
        to_telephone(x)


def test_to_telephone_int16_matches_float_equivalent():
    xi = (_tone(1000, amp=0.4) * 32768).astype(np.int16)
    assert np.array_equal(to_telephone(xi, seed=3), to_telephone(xi.astype(np.float64) / 32768.0, seed=3))
    # scale is explicit: a quiet int16 signal must not be treated as full-scale float
    quiet = (_tone(1000, amp=0.001) * 32768).astype(np.int16)
    assert np.max(np.abs(to_telephone(quiet))) <= 1.0
