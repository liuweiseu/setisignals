from pathlib import Path

import numpy as np
import pytest

from setisignals.io.merge import merge_files, merge_files_text
from setisignals.io.reader import read_spike_file
from setisignals.ray_utils import ray_session
from setisignals.schema import spike_with_target_dtype

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def _ray():
    with ray_session(workers=2):
        yield


def test_merge_files_row_count_and_dtype():
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)

    merged = merge_files([on, off], ["on", "off"])

    assert merged.dtype == spike_with_target_dtype(3)  # len("off") == 3
    assert len(merged) == len(on) + len(off)


def test_merge_files_target_labels_and_field_values():
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)

    merged = merge_files([on, off], ["on", "off"])

    np.testing.assert_array_equal(merged["target"][: len(on)], b"on")
    np.testing.assert_array_equal(merged["target"][len(on) :], b"off")

    np.testing.assert_array_equal(merged["id"][: len(on)], on["id"])
    np.testing.assert_array_equal(merged["id"][len(on) :], off["id"])
    np.testing.assert_array_equal(
        merged["detection_freq"][: len(on)], on["detection_freq"]
    )
    np.testing.assert_array_equal(
        merged["detection_freq"][len(on) :], off["detection_freq"]
    )


def test_merge_files_three_way_with_long_labels():
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)
    labels = ["HIP63121", "HIP63121_OFF", "HIP63121"]

    merged = merge_files([on, off, on], labels)

    assert len(merged) == 2 * len(on) + len(off)
    assert merged.dtype == spike_with_target_dtype(len("HIP63121_OFF"))
    np.testing.assert_array_equal(merged["target"][: len(on)], b"HIP63121")
    np.testing.assert_array_equal(
        merged["target"][len(on) : len(on) + len(off)], b"HIP63121_OFF"
    )
    np.testing.assert_array_equal(merged["target"][len(on) + len(off) :], b"HIP63121")


def test_merge_files_mismatched_lengths_raises():
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)
    with pytest.raises(ValueError):
        merge_files([on, off], ["only-one-label"])


def test_merge_files_text_preserves_original_formatting(tmp_path):
    on_path = FIXTURES / "tiny_on.spike"
    off_path = FIXTURES / "tiny_off.spike"
    out_path = tmp_path / "merged.spike"

    counts = merge_files_text([on_path, off_path], ["on", "off"], out_path, workers=2)

    on_lines = on_path.read_text().splitlines()
    off_lines = off_path.read_text().splitlines()
    assert counts == [len(on_lines), len(off_lines)]

    out_lines = out_path.read_text().splitlines()
    assert len(out_lines) == len(on_lines) + len(off_lines)

    # Original fields preserved byte-for-byte; only "target|" appended.
    assert out_lines[0] == on_lines[0] + "on|"
    assert out_lines[len(on_lines)] == off_lines[0] + "off|"
    assert out_lines[-1] == off_lines[-1] + "off|"


def test_merge_files_text_matches_structured_merge():
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)
    merged = merge_files([on, off], ["on", "off"])

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "merged.spike"
        counts = merge_files_text(
            [FIXTURES / "tiny_on.spike", FIXTURES / "tiny_off.spike"],
            ["on", "off"],
            out_path,
            workers=1,
        )
        reparsed = read_spike_file(out_path, workers=1, progress=False)

    assert sum(counts) == len(merged)
    np.testing.assert_array_equal(reparsed["id"], merged["id"])
    np.testing.assert_array_equal(reparsed["detection_freq"], merged["detection_freq"])
