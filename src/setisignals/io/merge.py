"""Merge two or more hit files/arrays into one, with a `target` column."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ray

from setisignals.ray_utils import chunk_file
from setisignals.schema import SPIKE_DTYPE, spike_with_target_dtype


def merge_files(arrays: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Concatenate ``arrays`` into one array with an added `target` field.

    ``target`` is set to ``labels[i]`` for every row from ``arrays[i]``.
    Row order is preserved: all of ``arrays[0]``'s rows, then all of
    ``arrays[1]``'s, and so on. No filtering/deduplication is applied —
    this is a plain union. ``len(arrays)`` must equal ``len(labels)``.
    """
    if len(arrays) != len(labels):
        raise ValueError(f"got {len(arrays)} arrays but {len(labels)} labels")

    total = sum(a.size for a in arrays)
    max_len = max((len(label) for label in labels), default=1)
    dtype = spike_with_target_dtype(max(max_len, 1))

    merged = np.empty(total, dtype=dtype)
    offset = 0
    for arr, label in zip(arrays, labels):
        n = arr.size
        for name in SPIKE_DTYPE.names:
            merged[name][offset : offset + n] = arr[name]
        merged["target"][offset : offset + n] = label.encode()
        offset += n
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


def merge_files_text(
    paths: list[Path], labels: list[str], out_path: Path, workers: int | None = None
) -> list[int]:
    """Merge pipe-delimited hit files into one text file, original format.

    Appends a `target` field (``labels[i]`` for lines from ``paths[i]``) to
    each line; all other fields are copied verbatim from the source files
    (no reparsing/reformatting). Row order is a plain union: all of
    ``paths[0]``'s lines, then all of ``paths[1]``'s, and so on. Returns the
    per-file line count, same order as ``paths``. ``len(paths)`` must equal
    ``len(labels)``.
    """
    if len(paths) != len(labels):
        raise ValueError(f"got {len(paths)} paths but {len(labels)} labels")

    n_chunks = max(1, workers or 1)
    per_file_futures = []
    for path, label in zip(paths, labels):
        path = Path(path).resolve()
        ranges = chunk_file(path, n_chunks)
        futures = [_label_chunk.remote(str(path), s, e, label.encode()) for s, e in ranges]
        per_file_futures.append(futures)

    counts: list[int] = []
    with open(out_path, "wb") as out:
        for futures in per_file_futures:
            parts = ray.get(futures)
            counts.append(sum(part.count(b"\n") for part in parts))
            for part in parts:
                out.write(part)

    return counts
