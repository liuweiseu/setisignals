from pathlib import Path

import numpy as np
import pytest

from setisignals.io.reader import read_spike_file
from setisignals.io.table_reader import read_table_file
from setisignals.io.writer import write_classified_tables, write_table
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


def test_write_classified_tables_splits_by_mask(tmp_path):
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)
    on_is_rfi = np.zeros(on.size, dtype=bool)
    on_is_rfi[0] = True
    off_is_rfi = np.zeros(off.size, dtype=bool)
    off_is_rfi[1] = True

    rfi_path = tmp_path / "rfi.hdf5"
    clean_path = tmp_path / "clean.hdf5"
    write_classified_tables(on, off, on_is_rfi, off_is_rfi, rfi_path, clean_path, fmt="hdf5")

    rfi_data = read_table_file(rfi_path)
    clean_data = read_table_file(clean_path)

    assert len(rfi_data) + len(clean_data) == on.size + off.size
    np.testing.assert_array_equal(rfi_data["id"], np.concatenate([on["id"][on_is_rfi], off["id"][off_is_rfi]]))
    np.testing.assert_array_equal(
        clean_data["id"], np.concatenate([on["id"][~on_is_rfi], off["id"][~off_is_rfi]])
    )
