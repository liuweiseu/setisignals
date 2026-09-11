"""Generic frequency-vs-time 2D density plot, one subplot per input table.

Originally built specifically for the paper's RFI-vs-Clean grayscale density
pair; generalized to plot any number of on/off tables side by side (e.g.
`classify-rfi`'s rfi.hdf5/clean.hdf5, or any other on/off split you want to
compare).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from setisignals.analysis.hist_utils import parallel_histogram2d
from setisignals.analysis.rfi import DEFAULT_BIN_WIDTH_HZ
from setisignals.analysis.time_utils import stack_combined_on_off
from setisignals.io.targets import split_on_off
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


def compute_density_grids(
    tables: list[np.ndarray],
    freq_bin_width_hz: float | None = DEFAULT_BIN_WIDTH_HZ,
    time_bins: int = 200,
    workers: int | None = None,
    expected_sessions: int | None = 3,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return (grids, freq_edges, time_edges), one grid per input table.

    Each table must have both on-source and off-source rows (distinguished
    by its `target` column, see `io.targets.split_on_off`) -- e.g. the
    `rfi.hdf5`/`clean.hdf5` outputs of `classify-rfi`.

    `stack_combined_on_off` needs the *full* on-source and off-source time
    series together to detect dwell/session boundaries correctly, so all
    tables' on/off rows are recombined for that stacking step, then
    re-split back into one grid per table.

    ``freq_bin_width_hz`` defaults to the same window width historically used
    for RFI classification (``analysis.rfi.DEFAULT_BIN_WIDTH_HZ``, 93 Hz);
    the number of frequency bins is derived from the data's frequency range
    divided by this width. If that would exceed ``MAX_FREQ_BINS`` (e.g. for
    a merged file spanning a wide frequency range), it falls back to
    ``FALLBACK_FREQ_BINS`` equal-width bins instead, to avoid building a
    grid too large to fit in memory. Pass ``None`` instead to use the
    data's native frequency resolution (one bin per unique
    ``detection_freq`` value, see ``_native_freq_edges``).
    """
    splits = [split_on_off(t) for t in tables]

    on_time = np.concatenate([on["time"] for on, _off in splits])
    off_time = np.concatenate([off["time"] for _on, off in splits])
    on_y_all, off_y_all = stack_combined_on_off(on_time, off_time, dwells_per_source=expected_sessions)

    freq_per_table: list[np.ndarray] = []
    y_per_table: list[np.ndarray] = []
    on_offset = off_offset = 0
    for on, off in splits:
        on_y = on_y_all[on_offset : on_offset + on.size]
        off_y = off_y_all[off_offset : off_offset + off.size]
        on_offset += on.size
        off_offset += off.size
        freq_per_table.append(np.concatenate([on["detection_freq"], off["detection_freq"]]))
        y_per_table.append(np.concatenate([on_y, off_y]))

    all_freq = np.concatenate(freq_per_table)
    all_y = np.concatenate(y_per_table)

    if freq_bin_width_hz is None:
        freq_edges = _native_freq_edges(all_freq)
        logger.info(f"Density: using native frequency resolution ({len(freq_edges) - 1:,} bins)")
    else:
        lo, hi = all_freq.min(), all_freq.max()
        n_freq_bins = int(np.ceil((hi - lo) / freq_bin_width_hz)) + 1
        if n_freq_bins > MAX_FREQ_BINS:
            freq_edges = np.linspace(lo, hi, FALLBACK_FREQ_BINS + 1)
            logger.info(
                f"Density: {n_freq_bins:,} bins at {freq_bin_width_hz} Hz width exceeds "
                f"MAX_FREQ_BINS ({MAX_FREQ_BINS:,}); falling back to {FALLBACK_FREQ_BINS:,} equal-width bins"
            )
        else:
            freq_edges = lo + np.arange(n_freq_bins + 1) * freq_bin_width_hz
            logger.info(f"Density: {n_freq_bins:,} frequency bins at {freq_bin_width_hz} Hz width")
    time_edges = np.linspace(all_y.min(), all_y.max(), time_bins + 1)

    grids = [
        parallel_histogram2d(freq, y, freq_edges, time_edges, workers=workers)
        for freq, y in zip(freq_per_table, y_per_table)
    ]
    return grids, freq_edges, time_edges


def plot_density(
    grids: list[np.ndarray],
    titles: list[str],
    freq_edges: np.ndarray,
    time_edges: np.ndarray,
    out_path: Path | None,
    source_name: str | None = None,
) -> None:
    """If ``out_path`` is None, the figure is left open for the caller to
    display (e.g. via a single ``plt.show()`` covering several figures)
    instead of being saved to disk."""
    fig, axes = plt.subplots(len(grids), 1, figsize=(8, 5 * len(grids)), sharex=True, squeeze=False)
    axes = axes.ravel()
    extent = (freq_edges[0], freq_edges[-1], time_edges[0], time_edges[-1])
    norm = mcolors.LogNorm(vmin=1, vmax=max((g.max() for g in grids), default=1) or 1)
    n_freq_bins = len(freq_edges) - 1
    counts = [int(g.sum()) for g in grids]
    total = sum(counts)

    for ax, grid, title, count in zip(axes, grids, titles, counts):
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
        pct = 100 * count / total if total else 0.0
        ax.text(
            0.02,
            0.98,
            f"{count:,}/{total:,} ({pct:.1f}%)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
        )
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
        fig.suptitle(f"Density of {source_name}", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
