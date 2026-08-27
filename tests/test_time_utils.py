import logging

import numpy as np
import pytest

from setisignals.analysis.time_utils import (
    detect_sessions,
    julian_to_relative_seconds,
    restrict_to_epoch,
    stack_combined_on_off,
    stack_sessions,
)


def test_julian_to_relative_seconds():
    jd = np.array([100.0, 100.0, 100.00001157407407])  # +1 second
    secs = julian_to_relative_seconds(jd)
    assert secs[0] == pytest.approx(0.0)
    assert secs[2] == pytest.approx(1.0, abs=1e-3)


def _cluster(start_jd: float, n: int, spacing_days: float) -> np.ndarray:
    return start_jd + np.arange(n) * spacing_days


def test_detect_sessions_finds_gap_separated_clusters():
    spacing = 1e-7  # ~0.0086 sec between points within a cluster
    big_gap = 0.01  # ~864 sec between clusters

    c1 = _cluster(100.0, 3, spacing)
    c2 = _cluster(c1[-1] + big_gap, 4, spacing)
    c3 = _cluster(c2[-1] + big_gap, 5, spacing)
    time_jd = np.concatenate([c1, c2, c3])

    sessions = detect_sessions(time_jd, expected_sessions=3)

    assert len(sessions) == 3
    assert [s.start_idx for s in sessions] == [0, 3, 7]
    assert [s.end_idx for s in sessions] == [3, 7, 12]


def test_detect_sessions_equal_split_fallback(caplog):
    spacing = 1e-7
    time_jd = _cluster(100.0, 10, spacing)  # uniform spacing, no gaps

    with caplog.at_level(logging.WARNING):
        sessions = detect_sessions(time_jd, expected_sessions=2)

    assert len(sessions) == 2
    assert sessions[0].start_idx == 0
    assert sessions[-1].end_idx == 10
    assert any("equal-count split" in rec.message for rec in caplog.records)


def test_restrict_to_epoch_picks_overlapping_cluster():
    spacing = 1e-7
    epoch_a = _cluster(100.0, 5, spacing)  # a short cluster near jd=100
    epoch_b = _cluster(625.0, 5, spacing)  # a much later cluster, ~525 days on
    time_jd = np.concatenate([epoch_a, epoch_b])

    mask = restrict_to_epoch(time_jd, reference_range=(100.0, 100.001))

    assert mask.sum() == 5
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(5))


def test_restrict_to_epoch_no_overlap_picks_closest():
    spacing = 1e-7
    epoch_a = _cluster(100.0, 3, spacing)
    epoch_b = _cluster(200.0, 3, spacing)
    time_jd = np.concatenate([epoch_a, epoch_b])

    # Reference range is closer to epoch_b than epoch_a.
    mask = restrict_to_epoch(time_jd, reference_range=(199.0, 199.5))

    assert mask.sum() == 3
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(3, 6))


def test_stack_combined_on_off_interleaves_bands():
    spacing = 1e-7
    big_gap = 0.01
    # on, off, on dwells interleaved in real observing time.
    on1 = _cluster(100.0, 3, spacing)
    off1 = _cluster(on1[-1] + big_gap, 3, spacing)
    on2 = _cluster(off1[-1] + big_gap, 3, spacing)

    on_time = np.concatenate([on1, on2])
    off_time = off1

    on_y, off_y = stack_combined_on_off(on_time, off_time, dwells_per_source=None)

    # The on-source band that comes after off1 (on2, i.e. on_y[3:]) must be
    # stacked above the off band, which itself is above the first on band.
    assert on_y[:3].max() < off_y.min()
    assert off_y.max() < on_y[3:].min()


def test_stack_sessions_no_overlap():
    spacing = 1e-7
    big_gap = 0.01
    c1 = _cluster(100.0, 3, spacing)
    c2 = _cluster(c1[-1] + big_gap, 4, spacing)
    time_jd = np.concatenate([c1, c2])

    sessions = detect_sessions(time_jd, expected_sessions=2)
    y = stack_sessions(time_jd, sessions)

    assert y[:3].max() < y[3:].min()
