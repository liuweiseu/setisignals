"""Reproduction of the paper's RFI-vs-Clean grayscale density pair.

See analysis/rfi.py for the on/off frequency cross-match algorithm used to
classify hits as RFI or Clean (an approximate reproduction of the paper's
method, since it doesn't specify exact binning details).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from setisignals.analysis.hist_utils import parallel_histogram2d
from setisignals.analysis.time_utils import stack_combined_on_off


def compute_rfi_density_grids(
    on: np.ndarray,
    off: np.ndarray,
    on_is_rfi: np.ndarray,
    off_is_rfi: np.ndarray,
    freq_bins: int = 200,
    time_bins: int = 200,
    workers: int | None = None,
    expected_sessions: int | None = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (rfi_grid, clean_grid, freq_edges, time_edges).

    Combines RFI hits from both on+off into one 2D histogram grid, and
    Clean hits from both on+off into another, matching the paper's
    "RFI" vs "Clean" density-pair framing.
    """
    on_y, off_y = stack_combined_on_off(on["time"], off["time"], dwells_per_source=expected_sessions)

    freq = np.concatenate([on["detection_freq"], off["detection_freq"]])
    y = np.concatenate([on_y, off_y])
    is_rfi = np.concatenate([on_is_rfi, off_is_rfi])

    freq_edges = np.linspace(freq.min(), freq.max(), freq_bins + 1)
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
    out_path: Path,
    source_name: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    extent = (freq_edges[0], freq_edges[-1], time_edges[0], time_edges[-1])
    norm = mcolors.LogNorm(vmin=1, vmax=max(rfi_grid.max(), clean_grid.max(), 1))

    for ax, grid, title in ((axes[0], rfi_grid, "RFI"), (axes[1], clean_grid, "Clean")):
        ax.imshow(
            grid.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="gray_r",
            norm=norm,
        )
        ax.set_title(title, color="red" if title == "RFI" else "black")
        ax.set_ylabel("Time (sec)")
    axes[-1].set_xlabel("Frequency (Hz)")
    if source_name:
        fig.suptitle(f"RFI Density of {source_name}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
