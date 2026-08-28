"""setisignals: CLI for SETI@home Listen `.spike` hit files."""

from __future__ import annotations

import os
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

import numpy as np

from setisignals.analysis.rfi import DEFAULT_BIN_WIDTH_HZ, classify_rfi
from setisignals.analysis.time_utils import restrict_to_epoch
from setisignals.io.merge import merge_files, merge_files_text
from setisignals.io.reader import read_with_progress
from setisignals.io.table_reader import SUPPORTED_SUFFIXES, read_table_file
from setisignals.io.targets import looks_like_off_source, parse_targets_file, resolve_target_names
from setisignals.io.writer import write_table
from setisignals.plotting.power_hist import compute_power_hist, plot_power_hist
from setisignals.plotting.rfi_density import compute_rfi_density_grids, plot_rfi_density
from setisignals.plotting.waterfall import plot_waterfall
from setisignals.ray_utils import ray_session

_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

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

console = Console()


def _default_workers() -> int:
    return os.cpu_count() or 1


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"setisignals {_pkg_version('setisignals')}")
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
) -> None:
    pass


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
        console.print(
            f"[yellow]Restricted on-source data to {mask.sum():,}/{on_data.size:,} "
            "rows matching the off-source observing epoch[/yellow]"
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
        console.print(
            f"[yellow]{len(names) - matched:,}/{len(names):,} rows had no matching "
            f"time window in {targets_path}[/yellow]"
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


@app.command()
def convert(
    input: Annotated[Path, typer.Argument(help="Path to a .spike file")],
    format: Annotated[str, typer.Option("--format", help="Output format: fits or hdf5")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output file path")],
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
    console.print(f"[green]Wrote {len(data):,} rows to {output} ({format})[/green]")


_MERGE_TARGETS_HELP = (
    "Either a single path to an existing target_time.txt-style file "
    "(whitespace columns: target_name start_time end_time start_ra end_ra "
    "start_dec end_dec, Julian dates) -- each row's `time` is looked up "
    "against these windows and the matching target name is written as its "
    "`target` value (empty if no window contains that time; requires "
    "--format) -- or exactly one label string per FILE (repeat --targets "
    "once per label, same order as FILES), used as that file's literal "
    "`target` value. Without --targets, each file's own name (stem) is "
    "used as its label."
)


@app.command()
def merge(
    files: Annotated[
        list[Path], typer.Argument(help="Two or more .spike files to merge")
    ],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output file path")],
    format: Annotated[
        str | None,
        typer.Option(
            "--format",
            help="Output format: fits or hdf5. Omit to write a pipe-delimited "
            ".spike text file (original format) with a `target` field appended.",
        ),
    ] = None,
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
    if format is not None and format not in ("fits", "hdf5"):
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
    if targets_file is not None and format is None:
        raise typer.BadParameter(
            "--targets <target_time.txt> (time-based lookup) requires --format"
        )

    workers = workers or _default_workers()
    row_labels = labels if labels is not None else [f.stem for f in files]

    if format is None:
        with ray_session(workers=workers):
            counts = merge_files_text(files, row_labels, output, workers=workers)
        summary = ", ".join(f"{c:,} {lbl}" for c, lbl in zip(counts, row_labels))
        console.print(
            f"[green]Wrote {sum(counts):,} rows ({summary}) to {output} (.spike text)[/green]"
        )
        return

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
    console.print(f"[green]Wrote {len(merged):,} rows ({summary}) to {output} ({format})[/green]")


_PLOT_INPUT_HELP = f"Path to a FITS or HDF5 file ({'/'.join(SUPPORTED_SUFFIXES)}), as produced by `convert`/`merge`"


@plot_app.command("power-hist")
def power_hist_cmd(
    input: Annotated[Path, typer.Argument(help=_PLOT_INPUT_HELP)],
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("power_hist.png"),
    workers: Annotated[int | None, typer.Option()] = None,
    n_bins: Annotated[int, typer.Option()] = 100,
) -> None:
    """Reproduce the power/mean-power distribution histogram."""
    workers = workers or _default_workers()
    data = _load_table(input)
    with ray_session(workers=workers):
        bin_edges, counts = compute_power_hist(
            data["peak_power"], data["mean_power"], n_bins=n_bins, workers=workers
        )
    plot_power_hist(bin_edges, counts, output)
    console.print(f"[green]Wrote {output}[/green]")


@plot_app.command("waterfall")
def waterfall_cmd(
    on: Annotated[Path, typer.Option(help=f"On-source input. {_PLOT_INPUT_HELP}")],
    off: Annotated[Path, typer.Option(help=f"Off-source input. {_PLOT_INPUT_HELP}")],
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("waterfall.png"),
    workers: Annotated[int | None, typer.Option()] = None,
    expected_sessions: Annotated[int | None, typer.Option()] = 3,
) -> None:
    """Reproduce the on/off frequency-time waterfall scatter plot (approximate)."""
    workers = workers or _default_workers()
    on_data = _load_table(on)
    off_data = _load_table(off)
    with ray_session(workers=workers):
        on_data = _restrict_on_to_off_epoch(on_data, off_data)
    plot_waterfall(on_data, off_data, output, expected_sessions=expected_sessions)
    console.print(f"[green]Wrote {output}[/green]")


@plot_app.command("rfi")
def rfi_cmd(
    on: Annotated[Path, typer.Option(help=f"On-source input. {_PLOT_INPUT_HELP}")],
    off: Annotated[Path, typer.Option(help=f"Off-source input. {_PLOT_INPUT_HELP}")],
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("rfi_density.png"),
    workers: Annotated[int | None, typer.Option()] = None,
    bin_width_hz: Annotated[float, typer.Option()] = DEFAULT_BIN_WIDTH_HZ,
    gpu: Annotated[bool, typer.Option()] = False,
) -> None:
    """Reproduce the RFI-vs-Clean grayscale density pair (approximate)."""
    workers = workers or _default_workers()
    on_data = _load_table(on)
    off_data = _load_table(off)
    with ray_session(workers=workers, num_gpus=1 if gpu else 0):
        on_data = _restrict_on_to_off_epoch(on_data, off_data)
        on_is_rfi, off_is_rfi = classify_rfi(
            on_data["detection_freq"],
            off_data["detection_freq"],
            bin_width_hz=bin_width_hz,
            workers=workers,
        )
        rfi_grid, clean_grid, freq_edges, time_edges = compute_rfi_density_grids(
            on_data, off_data, on_is_rfi, off_is_rfi, workers=workers
        )
    plot_rfi_density(rfi_grid, clean_grid, freq_edges, time_edges, output)
    console.print(f"[green]Wrote {output}[/green]")


@plot_app.command("all")
def plot_all_cmd(
    on: Annotated[Path, typer.Option(help=f"On-source input. {_PLOT_INPUT_HELP}")],
    off: Annotated[Path, typer.Option(help=f"Off-source input. {_PLOT_INPUT_HELP}")],
    outdir: Annotated[Path, typer.Option()] = Path("."),
    workers: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Generate all three figures (power-hist, waterfall, rfi) in one Ray session."""
    workers = workers or _default_workers()
    outdir.mkdir(parents=True, exist_ok=True)
    on_data = _load_table(on)
    off_data = _load_table(off)

    with ray_session(workers=workers):
        bin_edges, counts = compute_power_hist(
            on_data["peak_power"], on_data["mean_power"], workers=workers
        )
        plot_power_hist(bin_edges, counts, outdir / "power_hist.png")

        on_epoch_data = _restrict_on_to_off_epoch(on_data, off_data)

        plot_waterfall(on_epoch_data, off_data, outdir / "waterfall.png")

        on_is_rfi, off_is_rfi = classify_rfi(
            on_epoch_data["detection_freq"], off_data["detection_freq"], workers=workers
        )
        rfi_grid, clean_grid, freq_edges, time_edges = compute_rfi_density_grids(
            on_epoch_data, off_data, on_is_rfi, off_is_rfi, workers=workers
        )
        plot_rfi_density(rfi_grid, clean_grid, freq_edges, time_edges, outdir / "rfi_density.png")

    console.print(f"[green]Wrote power_hist.png, waterfall.png, rfi_density.png to {outdir}[/green]")


if __name__ == "__main__":
    app()
