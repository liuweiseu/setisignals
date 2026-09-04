from pathlib import Path

import numpy as np
import pytest

from setisignals.analysis.rfi import _group_coincidence, _solve_bin_size, classify_rfi
from setisignals.io.reader import read_spike_file
from setisignals.ray_utils import ray_session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def _ray():
    with ray_session(workers=2):
        yield


def test_classify_rfi_fixed_width_on_tiny_fixtures():
    """bin_width_hz explicitly set bypasses fft_len grouping entirely."""
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)

    on_is_rfi, off_is_rfi = classify_rfi(
        on["detection_freq"], off["detection_freq"], bin_width_hz=93.0, workers=2
    )

    # Hand-computed: on ids 1 and 4 coincide (within one 93 Hz bin) with off
    # ids 101 and 104 respectively; the rest are isolated -> Clean.
    expected_on = np.array([True, False, False, True, False, False])
    expected_off = np.array([True, False, False, True, False, False])

    np.testing.assert_array_equal(on_is_rfi, expected_on)
    np.testing.assert_array_equal(off_is_rfi, expected_off)


def test_classify_rfi_requires_fft_len_in_adaptive_mode():
    on = np.array([1.0])
    off = np.array([1.0])
    with pytest.raises(ValueError, match="on_fft_len/off_fft_len"):
        classify_rfi(on, off, workers=2)


def test_group_coincidence_marks_close_pairs_only():
    # Unit-level test of the per-fft_len-group coincidence test at a fixed
    # bin_size, decoupled from the adaptive bin-size search below (which is
    # only well-behaved with realistically large, densely-sampled inputs --
    # see test_solve_bin_size_caps_to_max_bins_per_group for why a tiny
    # handful of points isn't a meaningful input to that search).
    on_freq_g = np.array([1_000_000.0, 1_005_000.0])
    off_freq_g = np.array([1_000_050.0, 2_000_000.0])
    lo = min(on_freq_g.min(), off_freq_g.min())
    hi = max(on_freq_g.max(), off_freq_g.max())

    on_is_rfi, off_is_rfi = _group_coincidence(on_freq_g, off_freq_g, lo, hi, bin_size=200.0, workers=2)

    assert on_is_rfi[0] and not on_is_rfi[1]
    assert off_is_rfi[0] and not off_is_rfi[1]


def test_classify_rfi_adaptive_does_not_cross_fft_len_groups():
    # Same numeric frequencies, but on a different fft_len group per side ->
    # groups never overlap, so there's no possible on/off coincidence.
    on_freq = np.array([1_000_000.0])
    off_freq = np.array([1_000_000.5])
    on_fft_len = np.array([128])
    off_fft_len = np.array([64])

    on_is_rfi, off_is_rfi = classify_rfi(
        on_freq, off_freq, on_fft_len, off_fft_len, workers=2
    )

    assert not on_is_rfi[0]
    assert not off_is_rfi[0]


def test_solve_bin_size_caps_to_max_bins_per_group():
    # Dense, evenly-spaced hits (one per Hz) keep the density-based bin_size
    # pinned at bin_floor -- which, over a wide enough range, still implies
    # far more bins than max_bins_per_group allows. Must be capped rather
    # than left to blow up memory (see the rfi_density plotting module's
    # MAX_FREQ_BINS fix for the same class of issue).
    on_freq_g = np.arange(10_000, dtype=float)
    lo, hi = 0.0, 10_000.0
    bin_size, n_bins = _solve_bin_size(
        on_freq_g, off_count=10_000, lo=lo, hi=hi, bin_floor=1.0, rfi_prob=1e-2, max_bins_per_group=10
    )

    assert n_bins == 10
    assert bin_size == pytest.approx((hi - lo) / 10)
