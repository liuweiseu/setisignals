"""Reproduction of the paper's power-distribution histogram (Figure 2 style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from setisignals.analysis.hist_utils import parallel_histogram


def compute_power_hist(
    peak_power: np.ndarray,
    mean_power: np.ndarray,
    n_bins: int = 100,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-binned histogram of peak_power/mean_power. Returns (bin_edges, counts)."""
    ratio = peak_power / mean_power
    ratio = ratio[ratio > 0]
    lo = max(ratio.min(), 1.0)
    hi = ratio.max()
    bin_edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    counts = parallel_histogram(ratio, bin_edges, workers=workers)
    return bin_edges, counts


def plot_power_hist(
    bin_edges: np.ndarray,
    counts: np.ndarray,
    out_path: Path | None,
    source_name: str | None = None,
) -> None:
    """Render the plot. If ``out_path`` is None, the figure is left open for
    the caller to display (e.g. via a single ``plt.show()`` covering several
    figures) instead of being saved to disk."""
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(centers, counts, where="mid", color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Power/Mean Power")
    ax.set_ylabel("Number Found")
    if source_name:
        ax.set_title(f"Power Histogram of {source_name}")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
