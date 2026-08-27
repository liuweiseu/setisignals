"""Approximate reproduction of the paper's on/off RFI cross-match.

The paper (Notes/setiathome_listen_data_analysis.pdf) identifies RFI by
matching topocentric frequencies between on-source and off-source
observations within a ~93 Hz window (chosen so a random coincidence has
about a 1% chance of occurring in any given bin). The exact binning/matching
implementation isn't specified, so this is an approximate reproduction: bin
`detection_freq` into shared bin_width_hz-wide bins across on+off, and mark
a hit as RFI iff its bin contains hits from *both* datasets.
"""

from __future__ import annotations

import numpy as np

from setisignals.analysis.hist_utils import parallel_histogram

DEFAULT_BIN_WIDTH_HZ = 93.0


def classify_rfi(
    on_detection_freq: np.ndarray,
    off_detection_freq: np.ndarray,
    bin_width_hz: float = DEFAULT_BIN_WIDTH_HZ,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each on/off hit as RFI or Clean via frequency-bin coincidence.

    Returns ``(on_is_rfi, off_is_rfi)`` boolean arrays, same length/order as
    the inputs. Bin edges are shared between on and off so a bin index means
    the same frequency range in both.
    """
    if on_detection_freq.size == 0 or off_detection_freq.size == 0:
        return (
            np.zeros(on_detection_freq.size, dtype=bool),
            np.zeros(off_detection_freq.size, dtype=bool),
        )

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
