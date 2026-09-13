"""CLI integration tests, invoked in-process via Typer's CliRunner.

Exercises `convert`/`merge`/`classify-rfi`/`plot ...` end-to-end against the
tiny fixture files (previously untested: `cli.py` is the largest module in
the repo and had zero direct coverage). `--log-dir` is redirected into each
test's `tmp_path` so runs don't write into the repo's own `./logs`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from setisignals.cli import app
from setisignals.io.table_reader import read_table_file

FIXTURES = Path(__file__).parent / "fixtures"
ON_SPIKE = FIXTURES / "tiny_on.spike"
OFF_SPIKE = FIXTURES / "tiny_off.spike"
TARGETS_FILE = FIXTURES / "tiny_targets.txt"

runner = CliRunner()


def _invoke(tmp_path: Path, *args: str):
    return runner.invoke(app, ["--log-dir", str(tmp_path / "logs"), *args])


def _merge(tmp_path: Path, out_name: str = "merged.h5") -> Path:
    """Merge the tiny on/off fixtures with explicit HIP63121/HIP63121_OFF
    labels (recognizable as on/off by `is_off_variant`), for tests that need
    a `classify-rfi`/`plot waterfall`/`plot density`-ready input."""
    out = tmp_path / out_name
    result = _invoke(
        tmp_path,
        "merge",
        str(ON_SPIKE),
        str(OFF_SPIKE),
        "-o",
        str(out),
        "--targets",
        "HIP63121",
        "--targets",
        "HIP63121_OFF",
        "--workers",
        "2",
    )
    assert result.exit_code == 0, result.output
    return out


class TestConvert:
    def test_hdf5_default_format(self, tmp_path):
        out = tmp_path / "on.h5"
        result = _invoke(tmp_path, "convert", str(ON_SPIKE), "-o", str(out), "--workers", "2")
        assert result.exit_code == 0, result.output
        assert out.exists()
        data = read_table_file(out)
        assert len(data) == 6
        assert "target" not in (data.dtype.names or ())

    def test_fits_format(self, tmp_path):
        out = tmp_path / "on.fits"
        result = _invoke(
            tmp_path, "convert", str(ON_SPIKE), "-o", str(out), "--format", "fits", "--workers", "2"
        )
        assert result.exit_code == 0, result.output
        data = read_table_file(out)
        assert len(data) == 6

    def test_invalid_format_rejected(self, tmp_path):
        out = tmp_path / "on.h5"
        result = _invoke(tmp_path, "convert", str(ON_SPIKE), "-o", str(out), "--format", "bogus")
        assert result.exit_code != 0
        assert not out.exists()

    def test_targets_literal_label(self, tmp_path):
        out = tmp_path / "on.h5"
        result = _invoke(
            tmp_path,
            "convert",
            str(ON_SPIKE),
            "-o",
            str(out),
            "--targets",
            "HIP63121_ON",
            "--workers",
            "2",
        )
        assert result.exit_code == 0, result.output
        data = read_table_file(out)
        np.testing.assert_array_equal(np.unique(data["target"]), [b"HIP63121_ON"])

    def test_targets_file_resolution(self, tmp_path):
        out = tmp_path / "on.h5"
        result = _invoke(
            tmp_path,
            "convert",
            str(ON_SPIKE),
            "-o",
            str(out),
            "--targets",
            str(TARGETS_FILE),
            "--workers",
            "2",
        )
        assert result.exit_code == 0, result.output
        data = read_table_file(out)
        # tiny_on.spike's times all fall inside tiny_targets.txt's TESTON window.
        np.testing.assert_array_equal(np.unique(data["target"]), [b"TESTON"])


class TestMerge:
    def test_requires_at_least_two_files(self, tmp_path):
        out = tmp_path / "merged.h5"
        result = _invoke(tmp_path, "merge", str(ON_SPIKE), "-o", str(out))
        assert result.exit_code != 0
        assert not out.exists()

    def test_default_labels_are_filename_stems(self, tmp_path):
        out = tmp_path / "merged.h5"
        result = _invoke(
            tmp_path, "merge", str(ON_SPIKE), str(OFF_SPIKE), "-o", str(out), "--workers", "2"
        )
        assert result.exit_code == 0, result.output
        data = read_table_file(out)
        assert len(data) == 12
        np.testing.assert_array_equal(data["target"][:6], b"tiny_on")
        np.testing.assert_array_equal(data["target"][6:], b"tiny_off")

    def test_explicit_labels_per_file(self, tmp_path):
        out = _merge(tmp_path)
        data = read_table_file(out)
        np.testing.assert_array_equal(data["target"][:6], b"HIP63121")
        np.testing.assert_array_equal(data["target"][6:], b"HIP63121_OFF")

    def test_mismatched_label_count_rejected(self, tmp_path):
        out = tmp_path / "merged.h5"
        result = _invoke(
            tmp_path,
            "merge",
            str(ON_SPIKE),
            str(OFF_SPIKE),
            "-o",
            str(out),
            "--targets",
            "only-one",
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_targets_file_resolution(self, tmp_path):
        out = tmp_path / "merged.h5"
        result = _invoke(
            tmp_path,
            "merge",
            str(ON_SPIKE),
            str(OFF_SPIKE),
            "-o",
            str(out),
            "--targets",
            str(TARGETS_FILE),
            "--workers",
            "2",
        )
        assert result.exit_code == 0, result.output
        data = read_table_file(out)
        # tiny_on.spike's times fall inside TESTON's window, and TESTON isn't
        # an off-variant name, so is_off=False rows resolve against it.
        np.testing.assert_array_equal(data["target"][:6], b"TESTON")
        # tiny_off.spike is flagged is_off=True (its filename ends in "_OFF"),
        # which restricts candidate windows to off-variant *names*
        # (`is_off_variant`: a trailing "_OFF"/"_OF"/"_O", per io/targets.py).
        # The fixture's "TESTOFF" window name has no underscore before "OFF",
        # so it doesn't qualify as off-variant and there's no matching
        # candidate window for these rows -- they resolve to "" (unmatched),
        # not "TESTOFF". This documents that on/off-aware `--targets <file>`
        # resolution depends on off windows following the "_OFF"-style
        # naming convention, not just any name that happens to read as "off".
        np.testing.assert_array_equal(data["target"][6:], b"")


class TestClassifyRfi:
    def test_writes_rfi_and_clean_outputs(self, tmp_path):
        merged = _merge(tmp_path)
        rfi_out = tmp_path / "rfi.h5"
        clean_out = tmp_path / "clean.h5"
        result = _invoke(
            tmp_path,
            "classify-rfi",
            str(merged),
            "--rfi-output",
            str(rfi_out),
            "--clean-output",
            str(clean_out),
            "--workers",
            "2",
        )
        assert result.exit_code == 0, result.output
        assert rfi_out.exists()
        assert clean_out.exists()
        rfi_data = read_table_file(rfi_out)
        clean_data = read_table_file(clean_out)
        merged_data = read_table_file(merged)
        assert len(rfi_data) + len(clean_data) == len(merged_data)

    def test_rejects_input_without_on_off_distinction(self, tmp_path):
        # A plain `convert` output has no on/off distinction at all.
        on_only = tmp_path / "on.h5"
        result = _invoke(tmp_path, "convert", str(ON_SPIKE), "-o", str(on_only), "--workers", "2")
        assert result.exit_code == 0, result.output

        rfi_out = tmp_path / "rfi.h5"
        clean_out = tmp_path / "clean.h5"
        result = _invoke(
            tmp_path,
            "classify-rfi",
            str(on_only),
            "--rfi-output",
            str(rfi_out),
            "--clean-output",
            str(clean_out),
        )
        assert result.exit_code != 0
        assert not rfi_out.exists()


class TestPlot:
    def test_power_hist_save(self, tmp_path):
        converted = tmp_path / "on.h5"
        result = _invoke(tmp_path, "convert", str(ON_SPIKE), "-o", str(converted), "--workers", "2")
        assert result.exit_code == 0, result.output

        png = tmp_path / "power_hist.png"
        result = _invoke(
            tmp_path,
            "plot",
            "power-hist",
            str(converted),
            "--save",
            "-o",
            str(png),
            "--workers",
            "2",
        )
        assert result.exit_code == 0, result.output
        assert png.exists()

    def test_power_hist_rejects_raw_spike_input(self, tmp_path):
        png = tmp_path / "power_hist.png"
        result = _invoke(tmp_path, "plot", "power-hist", str(ON_SPIKE), "--save", "-o", str(png))
        assert result.exit_code != 0
        assert not png.exists()

    def test_waterfall_save(self, tmp_path):
        merged = _merge(tmp_path)
        png = tmp_path / "waterfall.png"
        result = _invoke(
            tmp_path,
            "plot",
            "waterfall",
            str(merged),
            "--save",
            "-o",
            str(png),
            "--workers",
            "2",
        )
        assert result.exit_code == 0, result.output
        assert png.exists()

    def test_density_save(self, tmp_path):
        merged = _merge(tmp_path)
        rfi_out = tmp_path / "rfi.h5"
        clean_out = tmp_path / "clean.h5"
        result = _invoke(
            tmp_path,
            "classify-rfi",
            str(merged),
            "--rfi-output",
            str(rfi_out),
            "--clean-output",
            str(clean_out),
            "--workers",
            "2",
        )
        assert result.exit_code == 0, result.output

        png = tmp_path / "density.png"
        result = _invoke(
            tmp_path,
            "plot",
            "density",
            str(rfi_out),
            str(clean_out),
            "--save",
            "-o",
            str(png),
            "--workers",
            "2",
        )
        assert result.exit_code == 0, result.output
        assert png.exists()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "setisignals" in result.output
