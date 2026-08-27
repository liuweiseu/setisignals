"""Merge an on-source and off-source hit file/array into one, with a `target` column."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ray

from setisignals.ray_utils import chunk_file
from setisignals.schema import SPIKE_DTYPE, SPIKE_WITH_TARGET_DTYPE

ON_LABEL = b"on"
OFF_LABEL = b"off"


def merge_on_off(on: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Concatenate ``on`` and ``off`` into one array with an added `target` field.

    ``target`` is ``b"on"`` for rows from ``on`` and ``b"off"`` for rows from
    ``off``. Row order is preserved: all ``on`` rows first, then all ``off``
    rows. No filtering/deduplication is applied — this is a plain union.
    """
    merged = np.empty(on.size + off.size, dtype=SPIKE_WITH_TARGET_DTYPE)
    for name in SPIKE_DTYPE.names:
        merged[name][: on.size] = on[name]
        merged[name][on.size :] = off[name]
    merged["target"][: on.size] = ON_LABEL
    merged["target"][on.size :] = OFF_LABEL
    return merged


@ray.remote
def _label_chunk(path: str, start: int, end: int, label: bytes) -> bytes:
    """Read byte range [start, end) of `path` and append `label|` to each line.

    Preserves the original field formatting byte-for-byte (no numeric
    reparsing) aside from the appended field and normalizing every line to
    end in exactly one '\\n'.
    """
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    out = bytearray()
    for line in raw.splitlines():
        if not line:
            continue
        out += line
        out += label
        out += b"|\n"
    return bytes(out)


def merge_on_off_text(
    on_path: Path, off_path: Path, out_path: Path, workers: int | None = None
) -> tuple[int, int]:
    """Merge two pipe-delimited hit files into one text file, original format.

    Appends a `target` field (``on``/``off``) to each line; all other fields
    are copied verbatim from the source files (no reparsing/reformatting).
    Row order is a plain union: all on-source lines, then all off-source
    lines. Returns ``(on_line_count, off_line_count)``.
    """
    on_path = Path(on_path).resolve()
    off_path = Path(off_path).resolve()
    n_chunks = max(1, workers or 1)

    on_ranges = chunk_file(on_path, n_chunks)
    off_ranges = chunk_file(off_path, n_chunks)

    on_futures = [_label_chunk.remote(str(on_path), s, e, ON_LABEL) for s, e in on_ranges]
    off_futures = [_label_chunk.remote(str(off_path), s, e, OFF_LABEL) for s, e in off_ranges]

    on_parts = ray.get(on_futures)
    off_parts = ray.get(off_futures)

    on_count = sum(part.count(b"\n") for part in on_parts)
    off_count = sum(part.count(b"\n") for part in off_parts)

    with open(out_path, "wb") as out:
        for part in on_parts:
            out.write(part)
        for part in off_parts:
            out.write(part)

    return on_count, off_count
