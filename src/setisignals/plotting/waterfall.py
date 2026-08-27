"""Reproduction of the paper's on/off frequency-time waterfall scatter (Figure 3 style).

See analysis/time_utils.py for the caveat that dwell segmentation/stacking is
an approximate visual reproduction, not a literal decode of the paper's method.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from setisignals.analysis.time_utils import stack_combined_on_off


def plot_waterfall(
    on: np.ndarray,
    off: np.ndarray,
    out_path: Path,
    expected_sessions: int | None = 3,
) -> None:
    """Plot the on/off waterfall scatter. Callers should pre-restrict ``on``
    to the observing epoch matching ``off`` (see
    ``analysis.time_utils.restrict_to_epoch``) if the on-source file may
    bundle multiple widely-separated epochs. ``expected_sessions`` is the
    expected dwell count *per source* (on and off dwells are assumed
    interleaved on a shared timeline, see ``stack_combined_on_off``)."""
    on_y, off_y = stack_combined_on_off(on["time"], off["time"], dwells_per_source=expected_sessions)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(off["detection_freq"], off_y, s=0.2, c="magenta", alpha=0.5, label="off")
    ax.scatter(on["detection_freq"], on_y, s=0.2, c="cyan", alpha=0.5, label="on")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (sec)")
    ax.legend(markerscale=20, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
