"""setisignals: CLI for SETI@home Listen `.spike` hit files."""

from __future__ import annotations

import os
import sys
import time
from functools import wraps
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated, Callable, TypeVar

import typer

import matplotlib

# Backend must be picked before the first `import matplotlib.pyplot` (here, or
# in any of the setisignals.plotting modules below) -- matplotlib initializes
# it lazily on that first import, and switching afterwards is unreliable. With
# --save (no interactive display needed), force the lightweight Agg backend;
# it has no GUI-toolkit init cost, unlike the auto-detected interactive one.
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from setisignals.analysis.rfi import DEFAULT_RFI_PROB, MIN_GROUP_SAMPLES, classify_rfi
from setisignals.analysis.time_utils import restrict_to_epoch
from setisignals.io.merge import merge_files
from setisignals.io.reader import read_with_progress
from setisignals.io.table_reader import SUPPORTED_SUFFIXES, read_table_file
from setisignals.io.targets import (
    looks_like_off_source,
    parse_targets_file,
    resolve_target_names,
    split_on_off,
)
from setisignals.io.writer import write_classified_tables, write_table
from setisignals.plotting.power_hist import compute_power_hist, plot_power_hist
from setisignals.plotting.rfi_density import compute_rfi_density_grids, plot_rfi_density
from setisignals.plotting.waterfall import plot_waterfall
from setisignals.ray_utils import ray_session
from setisignals.utils import get_logger, mirror_logger

_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
_DEFAULT_LOG_DIR = Path("logs")

app = typer.Typer(
    name="setisignals",
    help="Process SETI@home Listen .spike hit files.",
    context_settings=_CONTEXT_SETTINGS,
)
plot_app = typer.Typer(
    name="plot",
    help="Reproduce figures from the Listen data analysis notes.",
    context_settings=_CONTEXT_SETTINGS,
)
app.add_typer(plot_app, name="plot")

# Console-only until `main()` below reconfigures it with the resolved
# --log-dir (file logging needs the parsed CLI option, not available yet here).
logger = get_logger(__name__)

# Set by `main()` from --timestamp, before any command body runs.
_TIMESTAMP_ENABLED = False

_F = TypeVar("_F", bound=Callable[..., None])


def _timed(label: str) -> Callable[[_F], _F]:
    """Decorator logging a command's total wall-clock runtime when --timestamp is set."""

    def decorator(func: _F) -> _F:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> None:
            if not _TIMESTAMP_ENABLED:
                func(*args, **kwargs)
                return
            start = time.perf_counter()
            try:
                func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info(f"[timestamp] {label} completed in {elapsed:.3f}s")

        return wrapper  # type: ignore[return-value]

    return decorator


def _default_workers() -> int:
    return os.cpu_count() or 1


def _version_callback(value: bool) -> None:
    if value:
        print(f"setisignals {_pkg_version('setisignals')}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=_version_callback,
            is_eager=True,
            help="Show the setisignals version and exit.",
        ),
    ] = False,
    log_dir: Annotated[
        Path,
        typer.Option("--log-dir", help="Directory for log files (created if it doesn't exist)."),
    ] = _DEFAULT_LOG_DIR,
    timestamp: Annotated[
        bool,
        typer.Option(
            "--timestamp",
            help="Log the command's total wall-clock runtime (high-resolution) on completion.",
        ),
    ] = False,
) -> None:
    global logger, _TIMESTAMP_ENABLED
    logger = get_logger(__name__, log_dir=log_dir)
    mirror_logger(logger, "ray")
    _TIMESTAMP_ENABLED = timestamp


def _restrict_on_to_off_epoch(on_data: np.ndarray, off_data: np.ndarray) -> np.ndarray:
    """Restrict on-source hits to the observing epoch matching off-source.

    Raw on-source files can bundle hits from multiple, widely-separated
    observing epochs (e.g. a later re-observation at a different receiver
    band); the off-source file only covers a single epoch. This keeps only
    the on-source epoch overlapping the off-source time range so waterfall/
    RFI comparisons are apples-to-apples.
    """
    if on_data.size == 0 or off_data.size == 0:
        return on_data
    ref_range = (off_data["time"].min(), off_data["time"].max())
    mask = restrict_to_epoch(on_data["time"], ref_range)
    if mask.sum() < on_data.size:
        logger.warning(
            f"Restricted on-source data to {mask.sum():,}/{on_data.size:,} "
            "rows matching the off-source observing epoch"
        )
    return on_data[mask]


_TARGETS_HELP = (
    "Either a path to an existing target_time.txt-style file (whitespace "
    "columns: target_name start_time end_time start_ra end_ra start_dec "
    "end_dec, Julian dates) -- each row's `time` is looked up against "
    "these windows and the matching target name is written into a "
    "`target` column (empty if no window contains that time) -- or, if "
    "not an existing file, a literal label string written as `target` for "
    "every row. On-source and off-source windows for the same target "
    "routinely overlap in target_time.txt (it records whole observing "
    "blocks, not per-dwell boundaries); the input filename's `_OFF` "
    "suffix is used automatically to resolve that ambiguity."
)


def _resolve_target_column(
    time_jd: np.ndarray,
    targets_path: Path,
    workers: int | None,
    is_off: np.ndarray | None = None,
) -> np.ndarray:
    windows = parse_targets_file(targets_path)
    names = resolve_target_names(time_jd, windows, is_off=is_off, workers=workers)
    matched = int((names != b"").sum())
    if matched < len(names):
        logger.warning(
            f"{len(names) - matched:,}/{len(names):,} rows had no matching "
            f"time window in {targets_path}"
        )
    return names


def _load_table(path: Path) -> np.ndarray:
    """Load a `plot`-command input file, requiring FITS or HDF5."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise typer.BadParameter(
            f"{path} must be a FITS or HDF5 file ({'/'.join(SUPPORTED_SUFFIXES)}) "
            "-- produced by `convert` or `merge`"
        )
    return read_table_file(path)


def _split_on_off(data: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """`io.targets.split_on_off`, translating its ``ValueError`` into a
    `typer.BadParameter` that names the offending ``path``."""
    try:
        return split_on_off(data)
    except ValueError as e:
        raise typer.BadParameter(f"{path}: {e}") from e


@app.command()
@_timed("convert")
def convert(
    input: Annotated[Path, typer.Argument(help="Path to a .spike file")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output file path")],
    format: Annotated[
        str, typer.Option("--format", help="Output format: fits or hdf5")
    ] = "hdf5",
    targets: Annotated[str | None, typer.Option(help=_TARGETS_HELP)] = None,
    workers: Annotated[
        int | None, typer.Option(help="Number of parallel Ray workers")
    ] = None,
) -> None:
    """Convert a .spike file into a standard astronomical table format."""
    if format not in ("fits", "hdf5"):
        raise typer.BadParameter("format must be 'fits' or 'hdf5'")
    workers = workers or _default_workers()

    with ray_session(workers=workers):
        data = read_with_progress(input, workers=workers)
        extra_columns = None
        if targets is not None:
            if Path(targets).is_file():
                is_off = np.full(data.shape, looks_like_off_source(input), dtype=bool)
                extra_columns = {
                    "target": _resolve_target_column(
                        data["time"], Path(targets), workers, is_off=is_off
                    )
                }
            else:
                extra_columns = {
                    "target": np.full(data.shape, targets.encode(), dtype=f"S{len(targets)}")
                }

    write_table(data, output, format, extra_columns=extra_columns)  # type: ignore[arg-type]
    logger.info(f"Wrote {len(data):,} rows to {output} ({format})")


_MERGE_TARGETS_HELP = (
    "Either a single path to an existing target_time.txt-style file "
    "(whitespace columns: target_name start_time end_time start_ra end_ra "
    "start_dec end_dec, Julian dates) -- each row's `time` is looked up "
    "against these windows and the matching target name is written as its "
    "`target` value (empty if no window contains that time) -- or exactly "
    "one label string per FILE (repeat --targets once per label, same "
    "order as FILES), used as that file's literal `target` value. Without "
    "--targets, each file's own name (stem) is used as its label."
)


@app.command()
@_timed("merge")
def merge(
    files: Annotated[
        list[Path], typer.Argument(help="Two or more .spike files to merge")
    ],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output file path")],
    format: Annotated[
        str, typer.Option("--format", help="Output format: fits or hdf5")
    ] = "hdf5",
    targets: Annotated[list[str] | None, typer.Option(help=_MERGE_TARGETS_HELP)] = None,
    workers: Annotated[
        int | None, typer.Option(help="Number of parallel Ray workers")
    ] = None,
) -> None:
    """Merge two or more .spike files into one, with a `target` column.

    Rows are a plain union: all of the first file's rows, then the second
    file's, and so on, unfiltered.
    """
    if len(files) < 2:
        raise typer.BadParameter("merge requires at least 2 files")
    if format not in ("fits", "hdf5"):
        raise typer.BadParameter("format must be 'fits' or 'hdf5'")

    targets_file: Path | None = None
    labels: list[str] | None = None
    if targets:
        if len(targets) == 1 and Path(targets[0]).is_file():
            targets_file = Path(targets[0])
        elif len(targets) == len(files):
            labels = targets
        else:
            raise typer.BadParameter(
                f"--targets got {len(targets)} value(s) for {len(files)} files; pass "
                "either one existing target_time.txt-style file path, or exactly one "
                "label per file"
            )

    workers = workers or _default_workers()
    row_labels = labels if labels is not None else [f.stem for f in files]

    with ray_session(workers=workers):
        arrays = [read_with_progress(f, workers=workers) for f in files]
        merged = merge_files(arrays, row_labels)
        extra_columns = None
        if targets_file is not None:
            is_off = np.concatenate(
                [
                    np.full(arr.size, looks_like_off_source(f), dtype=bool)
                    for arr, f in zip(arrays, files)
                ]
            )
            extra_columns = {
                # Replaces the plain per-file `target` label with the
                # resolved target name (e.g. "HIP63121" / "HIP63121_O").
                "target": _resolve_target_column(
                    merged["time"], targets_file, workers, is_off=is_off
                )
            }

    write_table(merged, output, format, extra_columns=extra_columns)  # type: ignore[arg-type]
    summary = ", ".join(f"{arr.size:,} {lbl}" for arr, lbl in zip(arrays, row_labels))
    logger.info(f"Wrote {len(merged):,} rows ({summary}) to {output} ({format})")


_PLOT_INPUT_HELP = f"Path to a FITS or HDF5 file ({'/'.join(SUPPORTED_SUFFIXES)}), as produced by `convert`/`merge`"
_PLOT_ON_OFF_INPUT_HELP = (
    f"{_PLOT_INPUT_HELP}. Must be a single file with both on-source and "
    "off-source rows distinguished by its `target` column (i.e. `merge` "
    "output) -- not a plain `convert` output of one source."
)


@app.command("classify-rfi")
@_timed("classify-rfi")
def classify_rfi_cmd(
    input: Annotated[Path, typer.Argument(help=_PLOT_ON_OFF_INPUT_HELP)],
    rfi_output: Annotated[Path, typer.Option("--rfi-output", help="Output path for RFI-classified hits")] = Path(
        "rfi.hdf5"
    ),
    clean_output: Annotated[
        Path, typer.Option("--clean-output", help="Output path for Clean-classified hits")
    ] = Path("clean.hdf5"),
    workers: Annotated[int | None, typer.Option()] = None,
    rfi_prob: Annotated[
        float,
        typer.Option(help="Target random-coincidence probability per frequency bin (adaptive mode)"),
    ] = DEFAULT_RFI_PROB,
    bin_width_hz: Annotated[
        float | None,
        typer.Option(help="Force one fixed frequency-bin width (Hz) instead of adaptive per-fft_len binning"),
    ] = None,
    min_group_samples: Annotated[
        int,
        typer.Option(
            help="Below this many on/off hits in an fft_len group, skip adaptive calibration "
            "and use the native FFT-resolution bin width instead"
        ),
    ] = MIN_GROUP_SAMPLES,
    gpu: Annotated[bool, typer.Option()] = False,
) -> None:
    """Classify on/off hits as RFI or Clean; write rfi.hdf5/clean.hdf5.

    Splits into two tables (each combining on+off rows for that class,
    keeping the `target` column) for downstream analysis or `plot rfi`.
    """
    workers = workers or _default_workers()
    on_data, off_data = _split_on_off(_load_table(input), input)
    with ray_session(workers=workers, num_gpus=1 if gpu else 0):
        on_data = _restrict_on_to_off_epoch(on_data, off_data)
        on_is_rfi, off_is_rfi = classify_rfi(
            on_data["detection_freq"],
            off_data["detection_freq"],
            on_data["fft_len"],
            off_data["fft_len"],
            rfi_prob=rfi_prob,
            bin_width_hz=bin_width_hz,
            min_group_samples=min_group_samples,
            workers=workers,
        )
    write_classified_tables(on_data, off_data, on_is_rfi, off_is_rfi, rfi_output, clean_output)
    logger.info(f"Wrote {rfi_output}, {clean_output}")


@plot_app.command("power-hist")
@_timed("plot power-hist")
def power_hist_cmd(
    input: Annotated[Path, typer.Argument(help=_PLOT_INPUT_HELP)],
    save: Annotated[
        bool, typer.Option("--save", help="Save to disk instead of displaying interactively")
    ] = False,
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("power_hist.png"),
    workers: Annotated[int | None, typer.Option()] = None,
    n_bins: Annotated[int, typer.Option()] = 2000,
) -> None:
    """Reproduce the power/mean-power distribution histogram."""
    workers = workers or _default_workers()
    data = _load_table(input)
    with ray_session(workers=workers):
        bin_edges, counts = compute_power_hist(
            data["peak_power"], data["mean_power"], n_bins=n_bins, workers=workers
        )
    plot_power_hist(bin_edges, counts, output if save else None, source_name=input.stem)
    if save:
        logger.info(f"Wrote {output}")
    else:
        plt.show()


@plot_app.command("waterfall")
@_timed("plot waterfall")
def waterfall_cmd(
    input: Annotated[Path, typer.Argument(help=_PLOT_ON_OFF_INPUT_HELP)],
    save: Annotated[
        bool, typer.Option("--save", help="Save to disk instead of displaying interactively")
    ] = False,
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("waterfall.png"),
    workers: Annotated[int | None, typer.Option()] = None,
    expected_sessions: Annotated[int | None, typer.Option()] = 3,
) -> None:
    """Reproduce the on/off frequency-time waterfall scatter plot (approximate)."""
    workers = workers or _default_workers()
    on_data, off_data = _split_on_off(_load_table(input), input)
    with ray_session(workers=workers):
        on_data = _restrict_on_to_off_epoch(on_data, off_data)
    plot_waterfall(
        on_data,
        off_data,
        output if save else None,
        expected_sessions=expected_sessions,
        source_name=input.stem,
    )
    if save:
        logger.info(f"Wrote {output}")
    else:
        plt.show()


@plot_app.command("rfi")
@_timed("plot rfi")
def rfi_cmd(
    rfi_input: Annotated[Path, typer.Argument(help="Path to rfi.hdf5, as produced by `classify-rfi`")],
    clean_input: Annotated[Path, typer.Argument(help="Path to clean.hdf5, as produced by `classify-rfi`")],
    save: Annotated[
        bool, typer.Option("--save", help="Save to disk instead of displaying interactively")
    ] = False,
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("rfi_density.png"),
    workers: Annotated[int | None, typer.Option()] = None,
    source_name: Annotated[
        str | None, typer.Option(help="Plot title source name; defaults to rfi_input's stem")
    ] = None,
) -> None:
    """Reproduce the RFI-vs-Clean grayscale density pair from already-classified data."""
    workers = workers or _default_workers()
    rfi_data = _load_table(rfi_input)
    clean_data = _load_table(clean_input)
    with ray_session(workers=workers):
        rfi_grid, clean_grid, freq_edges, time_edges = compute_rfi_density_grids(
            rfi_data, clean_data, workers=workers
        )
    plot_rfi_density(
        rfi_grid,
        clean_grid,
        freq_edges,
        time_edges,
        output if save else None,
        source_name=source_name or rfi_input.stem,
    )
    if save:
        logger.info(f"Wrote {output}")
    else:
        plt.show()


if __name__ == "__main__":
    app()
