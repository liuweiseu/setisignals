"""Ray-parallel parsing of pipe-delimited SETI@home hit files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ray
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from setisignals.ray_utils import chunk_file
from setisignals.schema import SignalKind, get_dtype


@ray.remote
def _parse_chunk(
    path: str, start: int, end: int, dtype_descr: list[tuple[str, str]]
) -> np.ndarray:
    """Parse byte range ``[start, end)`` of ``path`` into a structured array.

    Each line is pipe-delimited with a trailing ``|`` (so splitting yields
    one extra empty trailing field, which is ignored).
    """
    dtype = np.dtype(dtype_descr)
    names = dtype.names
    assert names is not None

    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    lines = [line for line in raw.splitlines() if line]
    out = np.empty(len(lines), dtype=dtype)
    for i, line in enumerate(lines):
        fields = line.split(b"|")
        for j, name in enumerate(names):
            out[name][i] = fields[j]
    return out


def read_spike_file(
    path: Path,
    kind: SignalKind = SignalKind.SPIKE,
    workers: int | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Read a pipe-delimited hit file into a structured NumPy array.

    Splits ``path`` into line-aligned byte-range chunks (one per worker),
    parses each chunk in a Ray remote task, and concatenates the results.
    """
    path = Path(path).resolve()
    dtype = get_dtype(kind)
    n_chunks = max(1, workers or 1)
    ranges = chunk_file(path, n_chunks)

    if not ranges:
        return np.empty(0, dtype=dtype)

    dtype_descr = [(name, dtype[name].str) for name in dtype.names]  # type: ignore[index]
    path_str = str(path)

    futures = [
        _parse_chunk.remote(path_str, start, end, dtype_descr) for start, end in ranges
    ]

    if progress:
        arrays: list[np.ndarray] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as bar:
            task_id = bar.add_task(f"Parsing {path.name}", total=len(futures))
            pending = list(futures)
            while pending:
                done, pending = ray.wait(pending, num_returns=1)
                arrays.append(ray.get(done[0]))
                bar.advance(task_id)
    else:
        arrays = ray.get(futures)

    return np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)


def read_with_progress(
    path: Path, kind: SignalKind = SignalKind.SPIKE, workers: int | None = None
) -> np.ndarray:
    return read_spike_file(path, kind=kind, workers=workers, progress=True)
