"""Resolve observed target names by time from a `target_time.txt`-style file.

File format (e.g. `HIP63121_data/target_time.txt`): whitespace-separated
columns `target_name start_time end_time start_ra end_ra start_dec end_dec`
(Julian dates, degrees), one row per observation window, no header line.
Target names in this file are truncated to a fixed 10 characters at the
source (e.g. "HIP63121_O" for the full name "HIP63121_OFF" listed in
`targets.txt`) -- this module preserves them exactly as written, truncation
included.

Important data quirk: a window's [start, end] spans the *entire* observing
block for that target, not individual dwell boundaries -- so an on-source
window and its off-source counterpart routinely overlap in time (real
observing alternates on/off/on/off/... within that span). See
`resolve_target_names`'s `is_off` parameter for how this is resolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ray


@dataclass(frozen=True)
class TargetWindow:
    name: str
    start_jd: float
    end_jd: float


def parse_targets_file(path: Path) -> list[TargetWindow]:
    """Parse a `target_time.txt`-style file into a list of observation windows."""
    windows: list[TargetWindow] = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            windows.append(TargetWindow(parts[0], float(parts[1]), float(parts[2])))
    return windows


def is_off_variant(name: str) -> bool:
    """Heuristic: does this name look like an off-source pointing?

    Matches target_time.txt-style truncated suffixes (names are truncated
    to 10 characters at the source, so a full "_OFF" suffix may itself be
    cut down to "_OF" or "_O", e.g. "HIP63121_O") as well as a bare "off"
    label (e.g. a `merge --targets off` literal).
    """
    upper = name.strip().upper()
    return upper == "OFF" or upper.endswith(("_OFF", "_OF", "_O"))


def looks_like_off_source(path: Path) -> bool:
    """Heuristic: does this input file's name look like an off-source file?

    Matches this codebase's own naming convention (`<TARGET>_OFF.spike`).
    """
    return path.stem.upper().endswith("_OFF")


@ray.remote
def _resolve_shard(
    time_shard: np.ndarray, windows: list[tuple[bytes, float, float]], dtype_str: str
) -> np.ndarray:
    out = np.full(time_shard.shape, b"", dtype=dtype_str)
    for name, start, end in windows:
        mask = (time_shard >= start) & (time_shard <= end)
        out[mask] = name
    return out


def _resolve_uniform(
    time_jd: np.ndarray,
    windows_tuples: list[tuple[bytes, float, float]],
    dtype_str: str,
    workers: int | None,
) -> np.ndarray:
    if time_jd.size == 0:
        return np.full(0, b"", dtype=dtype_str)
    n_shards = max(1, workers or 1)
    shards = np.array_split(time_jd, n_shards)
    futures = [_resolve_shard.remote(shard, windows_tuples, dtype_str) for shard in shards]
    parts = ray.get(futures)
    return np.concatenate(parts)


def resolve_target_names(
    time_jd: np.ndarray,
    windows: list[TargetWindow],
    is_off: np.ndarray | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """Resolve each ``time_jd`` value to its containing window's target name.

    Rows whose time falls in no window get an empty string.

    ``is_off`` (optional, boolean, same length as ``time_jd``): when a row's
    on/off origin is already known (e.g. from `merge`'s two input files, or
    inferred from an input filename via `looks_like_off_source`), pass it
    here to resolve the on/off window overlap described in this module's
    docstring -- candidate windows are restricted to off-variant names (see
    `is_off_variant`) for rows where ``is_off`` is True, and to
    non-off-variant names otherwise. Without it, an overlapping time is
    resolved by "later-listed window (file order) wins", which can mislabel
    on-source hits as off (or vice versa) when windows overlap.
    """
    if not windows or time_jd.size == 0:
        return np.full(time_jd.shape, b"", dtype="S1")

    max_len = max(len(w.name) for w in windows)
    dtype_str = f"S{max_len}"

    if is_off is None:
        windows_tuples = [(w.name.encode(), w.start_jd, w.end_jd) for w in windows]
        return _resolve_uniform(time_jd, windows_tuples, dtype_str, workers)

    on_tuples = [
        (w.name.encode(), w.start_jd, w.end_jd) for w in windows if not is_off_variant(w.name)
    ]
    off_tuples = [
        (w.name.encode(), w.start_jd, w.end_jd) for w in windows if is_off_variant(w.name)
    ]

    out = np.full(time_jd.shape, b"", dtype=dtype_str)
    on_idx = np.flatnonzero(~is_off)
    off_idx = np.flatnonzero(is_off)
    if on_idx.size:
        out[on_idx] = _resolve_uniform(time_jd[on_idx], on_tuples, dtype_str, workers)
    if off_idx.size:
        out[off_idx] = _resolve_uniform(time_jd[off_idx], off_tuples, dtype_str, workers)
    return out
