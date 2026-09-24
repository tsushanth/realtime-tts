import pytest
from real_calls.segment import split_speech_ranges

F = 0.1


def test_two_blocks_separated_by_long_silence_stay_separate():
    flags = [False] * 5 + [True] * 20 + [False] * 10 + [True] * 20 + [False] * 5
    got = split_speech_ranges(flags, F)
    assert got == [pytest.approx((0.25, 2.75)), pytest.approx((3.25, 5.75))]


def test_short_gap_is_merged_and_padding_is_clipped():
    flags = [True] * 20 + [False] * 3 + [True] * 20
    assert split_speech_ranges(flags, F) == [pytest.approx((0.0, 4.3))]


def test_blip_shorter_than_min_segment_is_dropped():
    assert split_speech_ranges([False] * 10 + [True] * 5 + [False] * 10, F) == []


def test_long_continuous_speech_is_hard_split():
    got = split_speech_ranges([True] * 400, F)
    assert len(got) == 3
    assert all(e - s <= 15.5 + 1e-9 for s, e in got)


def test_merged_group_over_max_is_split_at_run_boundary():
    flags = [True] * 100 + [False] * 3 + [True] * 100
    assert len(split_speech_ranges(flags, F)) == 2
