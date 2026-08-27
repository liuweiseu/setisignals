"""Generic Ray map-reduce histogram primitives, shared by power_hist and rfi."""

from __future__ import annotations

import numpy as np
import ray


@ray.remote
def _partial_histogram(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=bin_edges)[0]


@ray.remote
def _partial_histogram_gpu(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    import cupy as cp

    counts = cp.histogram(cp.asarray(values), bins=cp.asarray(bin_edges))[0]
    return cp.asnumpy(counts)


@ray.remote
def _partial_histogram2d(
    x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    return np.histogram2d(x, y, bins=[x_edges, y_edges])[0]


@ray.remote
def _partial_histogram2d_gpu(
    x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    import cupy as cp

    counts = cp.histogram2d(
        cp.asarray(x), cp.asarray(y), bins=[cp.asarray(x_edges), cp.asarray(y_edges)]
    )[0]
    return cp.asnumpy(counts)


def parallel_histogram(
    values: np.ndarray,
    bin_edges: np.ndarray,
    workers: int | None = None,
    use_gpu: bool = False,
) -> np.ndarray:
    """1D histogram of ``values`` over ``bin_edges``, computed via Ray map-reduce."""
    n_shards = max(1, workers or 1)
    shards = np.array_split(values, n_shards) if values.size else [values]
    task = _partial_histogram_gpu if use_gpu else _partial_histogram
    remote_kwargs = {"num_gpus": 1} if use_gpu else {}
    futures = [task.options(**remote_kwargs).remote(shard, bin_edges) for shard in shards]
    partials = ray.get(futures)
    return np.sum(partials, axis=0)


def parallel_histogram2d(
    x: np.ndarray,
    y: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    workers: int | None = None,
    use_gpu: bool = False,
) -> np.ndarray:
    """2D histogram of (x, y) over (x_edges, y_edges), computed via Ray map-reduce."""
    n_shards = max(1, workers or 1)
    if x.size:
        x_shards = np.array_split(x, n_shards)
        y_shards = np.array_split(y, n_shards)
    else:
        x_shards, y_shards = [x], [y]
    task = _partial_histogram2d_gpu if use_gpu else _partial_histogram2d
    remote_kwargs = {"num_gpus": 1} if use_gpu else {}
    futures = [
        task.options(**remote_kwargs).remote(xs, ys, x_edges, y_edges)
        for xs, ys in zip(x_shards, y_shards)
    ]
    partials = ray.get(futures)
    return np.sum(partials, axis=0)
