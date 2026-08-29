"""Ray session management and line-aligned file chunking."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import ray

from setisignals.utils import CONSOLE_FORMAT


def chunk_file(path: Path, n_chunks: int) -> list[tuple[int, int]]:
    """Split ``path`` into up to ``n_chunks`` line-aligned byte ranges.

    Returns a list of half-open ``(start, end)`` byte ranges covering the
    whole file. Every range starts immediately after a newline (or at byte
    0) and ends immediately after a newline (or at EOF), so no line is ever
    split across two ranges. The result may contain fewer than ``n_chunks``
    ranges (e.g. if the file has fewer lines than requested chunks, or is
    empty).
    """
    size = path.stat().st_size
    if size == 0:
        return []
    if n_chunks <= 1:
        return [(0, size)]

    raw_offsets = [i * size // n_chunks for i in range(1, n_chunks)]
    boundaries = {0, size}
    with open(path, "rb") as f:
        for off in raw_offsets:
            f.seek(off)
            f.readline()  # consume through the next '\n' (or EOF)
            boundaries.add(f.tell())

    ordered = sorted(boundaries)
    return list(zip(ordered[:-1], ordered[1:]))


@contextmanager
def ray_session(workers: int | None = None, num_gpus: int = 0) -> Iterator[None]:
    """Idempotent Ray init/shutdown context manager.

    Passes `logging_format=CONSOLE_FORMAT` so `ray.init()`'s own internal
    logging setup doesn't overwrite the formatter on the "ray" logger's
    handlers with Ray's own default -- this matters when `utils.mirror_logger`
    has pointed those handlers at ours (shared handler instances), since
    Ray reformatting them would silently reformat our own logger's console
    output too.
    """
    ray_logger_level = logging.getLogger("ray").level or logging.INFO
    ray.init(
        num_cpus=workers,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        logging_level=ray_logger_level,
        logging_format=CONSOLE_FORMAT,
    )
    try:
        yield
    finally:
        ray.shutdown()
