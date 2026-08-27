"""setisignals: CLI for SETI@home Listen `.spike` hit files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

import numpy as np

from setisignals.analysis.rfi import DEFAULT_BIN_WIDTH_HZ, classify_rfi
from setisignals.analysis.time_utils import restrict_to_epoch
from setisignals.io.reader import read_with_progress
from setisignals.io.writer import write_table
from setisignals.plotting.power_hist import compute_power_hist, plot_power_hist
from setisignals.plotting.rfi_density import compute_rfi_density_grids, plot_rfi_density
from setisignals.plotting.waterfall import plot_waterfall
from setisignals.ray_utils import ray_session

app = typer.Typer(name="setisignals", help="Process SETI@home Listen .spike hit files.")
plot_app = typer.Typer(name="plot", help="Reproduce figures from the Listen data analysis notes.")
app.add_typer(plot_app, name="plot")

console = Console()


def _default_workers() -> int:
    return os.cpu_count() or 1


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


@app.command()
def convert(
    input: Annotated[Path, typer.Argument(help="Path to a .spike file")],
    format: Annotated[str, typer.Option("--format", help="Output format: fits or hdf5")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output file path")],
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

    write_table(data, output, format)  # type: ignore[arg-type]
    console.print(f"[green]Wrote {len(data):,} rows to {output} ({format})[/green]")


@plot_app.command("power-hist")
def power_hist_cmd(
    input: Annotated[Path, typer.Argument(help="Path to a .spike file")],
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("power_hist.png"),
    workers: Annotated[int | None, typer.Option()] = None,
    n_bins: Annotated[int, typer.Option()] = 100,
) -> None:
    """Reproduce the power/mean-power distribution histogram."""
    workers = workers or _default_workers()
    with ray_session(workers=workers):
        data = read_with_progress(input, workers=workers)
        bin_edges, counts = compute_power_hist(
            data["peak_power"], data["mean_power"], n_bins=n_bins, workers=workers
        )
    plot_power_hist(bin_edges, counts, output)
    console.print(f"[green]Wrote {output}[/green]")


@plot_app.command("waterfall")
def waterfall_cmd(
    on: Annotated[Path, typer.Option(help="Path to the on-source .spike file")],
    off: Annotated[Path, typer.Option(help="Path to the off-source .spike file")],
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("waterfall.png"),
    workers: Annotated[int | None, typer.Option()] = None,
    expected_sessions: Annotated[int | None, typer.Option()] = 3,
) -> None:
    """Reproduce the on/off frequency-time waterfall scatter plot (approximate)."""
    workers = workers or _default_workers()
    with ray_session(workers=workers):
        on_data = read_with_progress(on, workers=workers)
        off_data = read_with_progress(off, workers=workers)
        on_data = _restrict_on_to_off_epoch(on_data, off_data)
    plot_waterfall(on_data, off_data, output, expected_sessions=expected_sessions)
    console.print(f"[green]Wrote {output}[/green]")


@plot_app.command("rfi")
def rfi_cmd(
    on: Annotated[Path, typer.Option(help="Path to the on-source .spike file")],
    off: Annotated[Path, typer.Option(help="Path to the off-source .spike file")],
    output: Annotated[Path, typer.Option("-o", "--output")] = Path("rfi_density.png"),
    workers: Annotated[int | None, typer.Option()] = None,
    bin_width_hz: Annotated[float, typer.Option()] = DEFAULT_BIN_WIDTH_HZ,
    gpu: Annotated[bool, typer.Option()] = False,
) -> None:
    """Reproduce the RFI-vs-Clean grayscale density pair (approximate)."""
    workers = workers or _default_workers()
    with ray_session(workers=workers, num_gpus=1 if gpu else 0):
        on_data = read_with_progress(on, workers=workers)
        off_data = read_with_progress(off, workers=workers)
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
    on: Annotated[Path, typer.Option(help="Path to the on-source .spike file")],
    off: Annotated[Path, typer.Option(help="Path to the off-source .spike file")],
    outdir: Annotated[Path, typer.Option()] = Path("."),
    workers: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Generate all three figures (power-hist, waterfall, rfi) in one Ray session."""
    workers = workers or _default_workers()
    outdir.mkdir(parents=True, exist_ok=True)

    with ray_session(workers=workers):
        on_data = read_with_progress(on, workers=workers)
        off_data = read_with_progress(off, workers=workers)

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
