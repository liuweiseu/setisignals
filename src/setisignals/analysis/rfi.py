"""On/off RFI cross-match, ported from the paper's IDL analysis pipeline.

Historically this module used a single fixed-width (~93 Hz) frequency-bin
coincidence test between on-source and off-source hits. The original IDL
implementation (``idl/compare_on_off.pro``, lines 196-288) is more
elaborate, and this module reproduces it:

* Hits are grouped by ``fft_len``, since different FFT lengths have
  different native frequency resolutions (``subband_sample_rate / fft_len``)
  and lumping them into one binning scheme washes out that resolution.
* Within each ``fft_len`` group, the bin width is *not* fixed -- it's
  solved for so that a bin picking up a coincidence purely by chance has
  about ``rfi_prob`` probability (default 1%), given how densely that
  group's hits are packed into its frequency range. Sparse groups get wide
  bins, dense groups get narrow ones (see ``_solve_bin_size``).
* Because a real coincidence can straddle a bin edge, each group is binned
  twice, offset by half a bin width, and a hit counts as RFI if either
  alignment shows a coincidence (see ``_group_coincidence``).
* A group with too few hits on either side can't support the median-based
  calibration above (there aren't enough samples for "the median bin" to
  mean anything, and it tends to solve for an implausibly wide bin). Below
  ``min_group_samples`` hits on either side, the group falls back to the
  bare native-FFT-resolution bin width (``bin_floor``) instead.

Passing an explicit ``bin_width_hz`` bypasses all of the above and
reproduces the old single-fixed-width behavior across the whole dataset
(a quick/manual override, or a fallback when ``fft_len`` isn't available).
"""

from __future__ import annotations

import numpy as np

from setisignals.analysis.hist_utils import parallel_histogram
from setisignals.utils import get_logger

logger = get_logger(__name__)

# Kept as the paper's ~93 Hz manual/plotting reference width; no longer the
# automatic classification default (see module docstring) -- pass
# bin_width_hz explicitly to opt into fixed-width classification.
DEFAULT_BIN_WIDTH_HZ = 93.0

# Target probability of a purely-random on/off coincidence per bin, per
# idl/compare_on_off.pro's `rfi_prob` keyword default.
DEFAULT_RFI_PROB = 1e-2

# GBT breakthrough workunit subband sample rate (Hz), from
# idl/compare_on_off.pro's `subband_sample_rate` constant.
DEFAULT_SUBBAND_SAMPLE_RATE = 11444.091796875

# Minimum bin width, as a multiple of a group's native FFT frequency
# resolution (subband_sample_rate / fft_len). Matches idl/compare_on_off.pro's
# `window_size` constant.
DEFAULT_WINDOW_SIZE = float(np.sqrt(2))

# Safety cap on bins per fft_len group, in case the density-based bin size
# solves to something implausibly fine (e.g. a near-empty group spanning a
# very wide frequency range).
MAX_BINS_PER_GROUP = 2_000_000

# Below this many hits on either side of a group, skip the median-based bin
# search (too few samples to make "the median bin" a meaningful statistic)
# and just use the native FFT-resolution bin width instead.
MIN_GROUP_SAMPLES = 1_000

_BIN_GROWTH_FACTOR = float(np.sqrt(2))
_MAX_GROWTH_STEPS = 60


def classify_rfi(
    on_detection_freq: np.ndarray,
    off_detection_freq: np.ndarray,
    on_fft_len: np.ndarray | None = None,
    off_fft_len: np.ndarray | None = None,
    *,
    rfi_prob: float = DEFAULT_RFI_PROB,
    subband_sample_rate: float = DEFAULT_SUBBAND_SAMPLE_RATE,
    window_size: float = DEFAULT_WINDOW_SIZE,
    bin_width_hz: float | None = None,
    max_bins_per_group: int = MAX_BINS_PER_GROUP,
    min_group_samples: int = MIN_GROUP_SAMPLES,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each on/off hit as RFI or Clean via frequency-bin coincidence.

    Returns ``(on_is_rfi, off_is_rfi)`` boolean arrays, same length/order as
    the inputs.

    Default behavior (``bin_width_hz=None``) reproduces the IDL pipeline's
    multi-resolution approach: hits are grouped by ``fft_len``, and each
    group gets its own coincidence-probability-driven bin width (see module
    docstring). ``on_fft_len``/``off_fft_len`` are required in this mode. A
    group with fewer than ``min_group_samples`` hits on either side skips
    that calibration and uses the native FFT-resolution bin width instead.

    Passing ``bin_width_hz`` skips all of that and classifies every hit
    against one shared, fixed-width binning (the original behavior);
    ``fft_len`` is ignored/unneeded in this mode.
    """
    on_is_rfi = np.zeros(on_detection_freq.size, dtype=bool)
    off_is_rfi = np.zeros(off_detection_freq.size, dtype=bool)
    if on_detection_freq.size == 0 or off_detection_freq.size == 0:
        return on_is_rfi, off_is_rfi

    if bin_width_hz is not None:
        return _classify_fixed_width(on_detection_freq, off_detection_freq, bin_width_hz, workers)

    if on_fft_len is None or off_fft_len is None:
        raise ValueError(
            "on_fft_len/off_fft_len are required for adaptive multi-resolution "
            "classification; pass bin_width_hz for the fixed-width fallback instead."
        )

    fft_lens = np.union1d(np.unique(on_fft_len), np.unique(off_fft_len))
    for fft_len in sorted(fft_lens, reverse=True):
        on_mask = on_fft_len == fft_len
        off_mask = off_fft_len == fft_len
        on_freq_g = on_detection_freq[on_mask]
        off_freq_g = off_detection_freq[off_mask]
        if on_freq_g.size == 0 or off_freq_g.size == 0:
            continue  # no possible on/off coincidence in this fft_len group

        lo = min(on_freq_g.min(), off_freq_g.min())
        hi = max(on_freq_g.max(), off_freq_g.max())
        if hi <= lo:
            # every hit in this group sits at the exact same frequency
            on_is_rfi[on_mask] = True
            off_is_rfi[off_mask] = True
            continue

        bin_step = subband_sample_rate / fft_len
        bin_floor = bin_step * window_size
        if min(on_freq_g.size, off_freq_g.size) < min_group_samples:
            bin_size = bin_floor
            n_bins = max(1, int(np.ceil((hi - lo) / bin_size)))
            if n_bins > max_bins_per_group:
                bin_size = (hi - lo) / max_bins_per_group
                n_bins = max_bins_per_group
            logger.info(
                f"RFI classify: fft_len={fft_len}: only {on_freq_g.size:,}/{off_freq_g.size:,} "
                f"on/off hits (< min_group_samples={min_group_samples:,}); skipping median "
                f"calibration, using native-resolution bin_size={bin_size:.3f} Hz"
            )
        else:
            bin_size, n_bins = _solve_bin_size(
                on_freq_g, off_freq_g.size, lo, hi, bin_floor, rfi_prob, max_bins_per_group
            )
        g_on_rfi, g_off_rfi = _group_coincidence(on_freq_g, off_freq_g, lo, hi, bin_size, workers)
        on_is_rfi[on_mask] = g_on_rfi
        off_is_rfi[off_mask] = g_off_rfi

        tagged = int(g_on_rfi.sum()) + int(g_off_rfi.sum())
        total = on_freq_g.size + off_freq_g.size
        logger.info(
            f"RFI classify: fft_len={fft_len}: bin_size={bin_size:.3f} Hz "
            f"({n_bins:,} bins), tagged {tagged:,}/{total:,} ({100 * tagged / total:.2f}%)"
        )

    return on_is_rfi, off_is_rfi


def _classify_fixed_width(
    on_detection_freq: np.ndarray,
    off_detection_freq: np.ndarray,
    bin_width_hz: float,
    workers: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    lo = min(on_detection_freq.min(), off_detection_freq.min())
    hi = max(on_detection_freq.max(), off_detection_freq.max())
    n_bins = int(np.ceil((hi - lo) / bin_width_hz)) + 1
    edges = lo + np.arange(n_bins + 1) * bin_width_hz

    on_hist = parallel_histogram(on_detection_freq, edges, workers=workers)
    off_hist = parallel_histogram(off_detection_freq, edges, workers=workers)
    rfi_bins = (on_hist > 0) & (off_hist > 0)

    on_bin_idx = np.clip(np.digitize(on_detection_freq, edges) - 1, 0, len(rfi_bins) - 1)
    off_bin_idx = np.clip(np.digitize(off_detection_freq, edges) - 1, 0, len(rfi_bins) - 1)

    return rfi_bins[on_bin_idx], rfi_bins[off_bin_idx]


def _solve_bin_size(
    on_freq_g: np.ndarray,
    off_count: int,
    lo: float,
    hi: float,
    bin_floor: float,
    rfi_prob: float,
    max_bins_per_group: int,
) -> tuple[float, int]:
    """Solve for a bin width giving roughly ``rfi_prob``-probability random
    coincidences, per idl/compare_on_off.pro lines 199-224.

    Start from an analytic estimate (expected count per bin, assuming
    uniform density, equals ``sqrt(rfi_prob)``, using the average of the on
    and off hit counts as the density), then empirically correct it against
    the *median* per-bin count of on-source hits -- since real spectra
    aren't uniformly dense -- by growing the bin geometrically and
    interpolating to a target median count. (The IDL original only handles
    the case where the initial guess undershoots; this also shrinks back
    down when it overshoots, which the fixed growth-only search can't do.)
    """
    freq_range = hi - lo
    n = (on_freq_g.size + off_count) / 2.0

    def median_count(bin_size: float) -> float:
        n_bins = max(1, int(np.ceil(freq_range / bin_size)))
        edges = lo + np.arange(n_bins + 1) * bin_size
        hist, _ = np.histogram(on_freq_g, bins=edges)
        return float(np.median(hist))

    bin_size = max(bin_floor, np.sqrt(rfi_prob) * freq_range / n)
    med = median_count(bin_size)

    samples_b = [0.0]
    samples_m = [0.0]
    steps = 0
    while med < 1.0 and bin_size < freq_range and steps < _MAX_GROWTH_STEPS:
        samples_b.append(bin_size)
        samples_m.append(med)
        bin_size *= _BIN_GROWTH_FACTOR
        med = median_count(bin_size)
        steps += 1
    samples_b.append(bin_size)
    samples_m.append(med)

    target_median = np.sqrt(rfi_prob) * 2.0
    bin_size = max(float(np.interp(target_median, samples_m, samples_b)), bin_floor)

    n_bins = max(1, int(np.ceil(freq_range / bin_size)))
    if n_bins > max_bins_per_group:
        capped = freq_range / max_bins_per_group
        logger.info(
            f"RFI classify: solved bin_size {bin_size:.3f} Hz needs {n_bins:,} bins, "
            f"exceeds MAX_BINS_PER_GROUP ({max_bins_per_group:,}); capping to {capped:.3f} Hz"
        )
        bin_size = capped
        n_bins = max_bins_per_group

    return bin_size, n_bins


def _group_coincidence(
    on_freq_g: np.ndarray,
    off_freq_g: np.ndarray,
    lo: float,
    hi: float,
    bin_size: float,
    workers: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Coincidence test for one fft_len group, binned twice (offset by half a
    bin) to avoid missing real coincidences that straddle a bin edge -- a hit
    counts as RFI if either alignment shows both on and off present in its bin.
    """
    n_bins = max(1, int(np.ceil((hi - lo) / bin_size)))
    edges_a = lo + np.arange(n_bins + 1) * bin_size
    edges_b = (lo + bin_size / 2) + np.arange(n_bins + 2) * bin_size

    def coincidence_mask(edges: np.ndarray) -> np.ndarray:
        on_hist = parallel_histogram(on_freq_g, edges, workers=workers)
        off_hist = parallel_histogram(off_freq_g, edges, workers=workers)
        return (on_hist > 0) & (off_hist > 0)

    def hit_mask(freq: np.ndarray, edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
        idx = np.clip(np.digitize(freq, edges) - 1, 0, len(mask) - 1)
        return mask[idx]

    mask_a = coincidence_mask(edges_a)
    mask_b = coincidence_mask(edges_b)

    on_is_rfi = hit_mask(on_freq_g, edges_a, mask_a) | hit_mask(on_freq_g, edges_b, mask_b)
    off_is_rfi = hit_mask(off_freq_g, edges_a, mask_a) | hit_mask(off_freq_g, edges_b, mask_b)
    return on_is_rfi, off_is_rfi
