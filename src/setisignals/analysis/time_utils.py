"""Julian-date conversion and dwell/session segmentation for the waterfall plot.

The paper's Figure 3 stacks several observation dwells (3 on-source + 3
off-source for HIP63121) as separate horizontal bands, but gives no exact
recipe for how the stacking was done. This module implements a defensible
approximation: detect dwell boundaries via gaps in the time series, then
stack each dwell's within-dwell elapsed time with a vertical offset so bands
don't overlap. This is flagged as an approximate visual reproduction, not a
literal decode of the original method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from setisignals.utils import get_logger

logger = get_logger(__name__)

SECONDS_PER_DAY = 86400.0


def julian_to_relative_seconds(jd: np.ndarray) -> np.ndarray:
    """Convert Julian-date values to seconds relative to their minimum."""
    if jd.size == 0:
        return np.empty(0, dtype=np.float64)
    return (jd - jd.min()) * SECONDS_PER_DAY


def restrict_to_epoch(
    time_jd: np.ndarray,
    reference_range: tuple[float, float],
    gap_hours: float = 1.0,
) -> np.ndarray:
    """Boolean mask selecting the contiguous observing epoch overlapping ``reference_range``.

    Some raw hit files (e.g. an on-source file re-observed months later at a
    different receiver band) bundle multiple, widely time-separated
    observing epochs into a single file. Splitting one on-source file
    against a single-epoch off-source file for a waterfall/RFI comparison
    requires first restricting the on-source data to the epoch that
    actually corresponds to the off-source observation. This splits
    ``time_jd`` into epochs wherever a consecutive gap exceeds
    ``gap_hours``, then keeps whichever epoch has the greatest overlap with
    ``reference_range`` (falls back to the epoch closest to it, if none
    truly overlap).
    """
    if time_jd.size == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(time_jd, kind="stable")
    sorted_t = time_jd[order]
    gaps = np.diff(sorted_t) * SECONDS_PER_DAY
    boundaries = np.flatnonzero(gaps > gap_hours * 3600.0)
    starts = [0] + [int(i) + 1 for i in boundaries]
    ends = [int(i) + 1 for i in boundaries] + [time_jd.size]

    ref_lo, ref_hi = reference_range
    best_span: tuple[int, int] | None = None
    best_overlap = -np.inf
    for s, e in zip(starts, ends):
        lo, hi = sorted_t[s], sorted_t[e - 1]
        overlap = min(hi, ref_hi) - max(lo, ref_lo)
        if overlap > best_overlap:
            best_overlap = overlap
            best_span = (s, e)

    assert best_span is not None
    s, e = best_span
    if len(starts) > 1:
        logger.info(
            "restrict_to_epoch: split %d hits into %d epoch(s) (gap > %.1fh); "
            "kept epoch with %d hits overlapping the reference time range",
            time_jd.size,
            len(starts),
            gap_hours,
            e - s,
        )

    mask = np.zeros(time_jd.size, dtype=bool)
    mask[order[s:e]] = True
    return mask


def stack_combined_on_off(
    on_time_jd: np.ndarray,
    off_time_jd: np.ndarray,
    dwells_per_source: int | None = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack on-source and off-source hits on one shared, gap-segmented timeline.

    On-source and off-source dwells are typically interleaved in real
    observing time (on, off, on, off, ...), not independent sequences — so
    session/dwell boundaries must be detected across the *combined*
    timeline, not separately per source (detecting them separately makes
    each source's first dwell start at y=0, overlapping dwells that were
    actually observed at different times). Returns ``(on_y, off_y)`` elapsed-
    plus-stacking-offset seconds, aligned to ``on_time_jd``/``off_time_jd``.
    """
    combined = np.concatenate([on_time_jd, off_time_jd])
    expected = None if dwells_per_source is None else 2 * dwells_per_source
    sessions = detect_sessions(combined, expected_sessions=expected)
    y = stack_sessions(combined, sessions)
    return y[: on_time_jd.size], y[on_time_jd.size :]


@dataclass(frozen=True)
class Session:
    """A contiguous dwell within a time-sorted array of hits."""

    start_idx: int
    end_idx: int  # exclusive, indices into the time-sorted array
    start_jd: float


def detect_sessions(
    time_jd: np.ndarray,
    expected_sessions: int | None = 3,
    gap_factor: float = 20.0,
) -> list[Session]:
    """Segment time-sorted hits into dwells/sessions via gap detection.

    Sorts ``time_jd`` ascending, computes consecutive gaps in seconds, and
    treats any gap larger than ``gap_factor * median(gap)`` as a candidate
    dwell boundary. If ``expected_sessions`` is given and the candidate
    count doesn't match it:
      - too many candidates: keep the ``expected_sessions - 1`` largest gaps.
      - too few candidates: fall back to an equal-count split (logs a
        warning), since dwells may run back-to-back with no detectable gap.
    If ``expected_sessions`` is None, uses however many candidates the
    threshold naturally finds.
    """
    n = time_jd.size
    if n == 0:
        return []
    if n == 1:
        return [Session(0, 1, float(time_jd[0]))]

    order = np.argsort(time_jd, kind="stable")
    sorted_jd = time_jd[order]

    gaps_seconds = np.diff(sorted_jd) * SECONDS_PER_DAY
    median_gap = np.median(gaps_seconds)
    if median_gap <= 0:
        median_gap = np.finfo(np.float64).eps

    candidate_mask = gaps_seconds > (gap_factor * median_gap)
    candidate_idx = np.flatnonzero(candidate_mask)  # boundary after index i

    if expected_sessions is not None:
        target_boundaries = expected_sessions - 1
        if candidate_idx.size > target_boundaries:
            # Keep the largest `target_boundaries` gaps.
            largest = np.argsort(gaps_seconds[candidate_idx])[::-1][:target_boundaries]
            candidate_idx = np.sort(candidate_idx[largest])
        elif candidate_idx.size < target_boundaries:
            logger.warning(
                "detect_sessions: found %d gap boundaries but expected %d sessions "
                "(%d boundaries); falling back to an equal-count split",
                candidate_idx.size,
                expected_sessions,
                target_boundaries,
            )
            split_points = np.linspace(0, n, expected_sessions + 1, dtype=int)[1:-1]
            candidate_idx = split_points - 1

    boundaries = [0] + [int(i) + 1 for i in candidate_idx] + [n]
    boundaries = sorted(set(boundaries))

    sessions = [
        Session(start, end, float(sorted_jd[start]))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    return sessions


def stack_sessions(
    time_jd: np.ndarray,
    sessions: list[Session],
    pitch: float | None = None,
) -> np.ndarray:
    """Return a y-array (seconds) stacking each session with a vertical offset.

    Each session's elapsed time is measured relative to that session's own
    start. Sessions are then offset by ``session_index * pitch`` so bands
    never overlap. If ``pitch`` is None, defaults to
    ``1.1 * max(session duration)``.
    """
    order = np.argsort(time_jd, kind="stable")
    sorted_jd = time_jd[order]

    if pitch is None:
        durations = [
            (sorted_jd[s.end_idx - 1] - sorted_jd[s.start_idx]) * SECONDS_PER_DAY
            for s in sessions
        ]
        pitch = 1.1 * max(durations) if durations and max(durations) > 0 else 1.0

    y_sorted = np.empty(time_jd.size, dtype=np.float64)
    for i, session in enumerate(sessions):
        segment = sorted_jd[session.start_idx : session.end_idx]
        elapsed = julian_to_relative_seconds(segment)
        y_sorted[session.start_idx : session.end_idx] = elapsed + i * pitch

    y = np.empty_like(y_sorted)
    y[order] = y_sorted
    return y
