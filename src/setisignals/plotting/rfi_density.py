"""Reproduction of the paper's RFI-vs-Clean grayscale density pair.

See analysis/rfi.py for the on/off frequency cross-match algorithm used to
classify hits as RFI or Clean (an approximate reproduction of the paper's
method, since it doesn't specify exact binning details).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from setisignals.analysis.hist_utils import parallel_histogram2d
from setisignals.analysis.rfi import DEFAULT_BIN_WIDTH_HZ
from setisignals.analysis.time_utils import stack_combined_on_off
from setisignals.utils import get_logger

logger = get_logger(__name__)

# Above this many bins, a fixed 93 Hz bin width would need more frequency
# bins than the plot (and the memory backing it) can reasonably hold --
# e.g. a merged multi-band file spanning several GHz would blow up to tens
# of millions of bins. Fall back to a fixed bin count instead.
MAX_FREQ_BINS = 16384
FALLBACK_FREQ_BINS = 16384


def _native_freq_edges(freq: np.ndarray) -> np.ndarray:
    """Bin edges giving one bin per distinct ``detection_freq`` value, so the
    density grid reflects the data's actual frequency resolution instead of
    an arbitrary equal-width approximation. Edges fall at the midpoints
    between neighboring unique values."""
    unique_freq = np.unique(freq)
    if unique_freq.size < 2:
        half = 0.5 if unique_freq.size == 0 else abs(unique_freq[0]) * 1e-6 or 0.5
        center = unique_freq[0] if unique_freq.size else 0.0
        return np.array([center - half, center + half])
    mids = (unique_freq[:-1] + unique_freq[1:]) / 2.0
    first_edge = unique_freq[0] - (mids[0] - unique_freq[0])
    last_edge = unique_freq[-1] + (unique_freq[-1] - mids[-1])
    return np.concatenate(([first_edge], mids, [last_edge]))


def compute_rfi_density_grids(
    on: np.ndarray,
    off: np.ndarray,
    on_is_rfi: np.ndarray,
    off_is_rfi: np.ndarray,
    freq_bin_width_hz: float | None = DEFAULT_BIN_WIDTH_HZ,
    time_bins: int = 200,
    workers: int | None = None,
    expected_sessions: int | None = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rfi_grid, clean_grid, freq_edges, time_edges).

    Combines RFI hits from both on+off into one 2D histogram grid, and
    Clean hits from both on+off into another, matching the paper's
    "RFI" vs "Clean" density-pair framing.

    ``freq_bin_width_hz`` defaults to the same window width used for RFI
    classification (``analysis.rfi.DEFAULT_BIN_WIDTH_HZ``, 93 Hz); the
    number of frequency bins is derived from the data's frequency range
    divided by this width. If that would exceed ``MAX_FREQ_BINS`` (e.g. for
    a merged file spanning a wide frequency range), it falls back to
    ``FALLBACK_FREQ_BINS`` equal-width bins instead, to avoid building a
    grid too large to fit in memory. Pass ``None`` instead to use the
    data's native frequency resolution (one bin per unique
    ``detection_freq`` value, see ``_native_freq_edges``).
    """
    on_y, off_y = stack_combined_on_off(on["time"], off["time"], dwells_per_source=expected_sessions)

    freq = np.concatenate([on["detection_freq"], off["detection_freq"]])
    y = np.concatenate([on_y, off_y])
    is_rfi = np.concatenate([on_is_rfi, off_is_rfi])

    if freq_bin_width_hz is None:
        freq_edges = _native_freq_edges(freq)
        logger.info(f"RFI density: using native frequency resolution ({len(freq_edges) - 1:,} bins)")
    else:
        lo, hi = freq.min(), freq.max()
        n_freq_bins = int(np.ceil((hi - lo) / freq_bin_width_hz)) + 1
        if n_freq_bins > MAX_FREQ_BINS:
            freq_edges = np.linspace(lo, hi, FALLBACK_FREQ_BINS + 1)
            logger.info(
                f"RFI density: {n_freq_bins:,} bins at {freq_bin_width_hz} Hz width exceeds "
                f"MAX_FREQ_BINS ({MAX_FREQ_BINS:,}); falling back to {FALLBACK_FREQ_BINS:,} equal-width bins"
            )
        else:
            freq_edges = lo + np.arange(n_freq_bins + 1) * freq_bin_width_hz
            logger.info(f"RFI density: {n_freq_bins:,} frequency bins at {freq_bin_width_hz} Hz width")
    time_edges = np.linspace(y.min(), y.max(), time_bins + 1)

    rfi_grid = parallel_histogram2d(
        freq[is_rfi], y[is_rfi], freq_edges, time_edges, workers=workers
    )
    clean_grid = parallel_histogram2d(
        freq[~is_rfi], y[~is_rfi], freq_edges, time_edges, workers=workers
    )
    return rfi_grid, clean_grid, freq_edges, time_edges


def plot_rfi_density(
    rfi_grid: np.ndarray,
    clean_grid: np.ndarray,
    freq_edges: np.ndarray,
    time_edges: np.ndarray,
    out_path: Path | None,
    source_name: str | None = None,
) -> None:
    """If ``out_path`` is None, the figure is left open for the caller to
    display (e.g. via a single ``plt.show()`` covering several figures)
    instead of being saved to disk."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    extent = (freq_edges[0], freq_edges[-1], time_edges[0], time_edges[-1])
    norm = mcolors.LogNorm(vmin=1, vmax=max(rfi_grid.max(), clean_grid.max(), 1))
    n_freq_bins = len(freq_edges) - 1

    for ax, grid, title in ((axes[0], rfi_grid, "RFI"), (axes[1], clean_grid, "Clean")):
        ax.imshow(
            grid.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="gray_r",
            norm=norm,
        )
        ax.set_title(title, color="black", fontsize=13)
        ax.set_ylabel("Time (sec)")
        ax.text(
            0.98,
            0.98,
            f"{n_freq_bins:,} freq bins",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
        )
    axes[-1].set_xlabel("Frequency (Hz)")
    if source_name:
        fig.suptitle(f"RFI Density of {source_name}", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
