from pathlib import Path

import numpy as np
import pytest

from setisignals.io.reader import read_spike_file
from setisignals.io.table_reader import read_table_file
from setisignals.io.writer import write_table
from setisignals.ray_utils import ray_session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def _ray():
    with ray_session(workers=1):
        yield


@pytest.mark.parametrize("fmt,suffix", [("fits", ".fits"), ("hdf5", ".h5")])
def test_read_table_file_round_trip(tmp_path, fmt, suffix):
    original = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    out_path = tmp_path / f"tiny{suffix}"
    write_table(original, out_path, fmt)

    read_back = read_table_file(out_path)

    assert read_back.dtype.names == original.dtype.names
    for name in original.dtype.names:
        np.testing.assert_array_equal(read_back[name], original[name])


def test_read_table_file_native_byte_order(tmp_path):
    original = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    out_path = tmp_path / "tiny.fits"
    write_table(original, out_path, "fits")

    read_back = read_table_file(out_path)

    for name in read_back.dtype.names:
        assert read_back.dtype[name].byteorder in ("=", "|")


def test_read_table_file_unsupported_suffix(tmp_path):
    bogus = tmp_path / "data.spike"
    bogus.write_text("not a table\n")
    with pytest.raises(ValueError):
        read_table_file(bogus)
