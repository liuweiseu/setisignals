from pathlib import Path

import pytest

from setisignals.io.reader import read_spike_file
from setisignals.ray_utils import ray_session
from setisignals.schema import SPIKE_DTYPE

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def _ray():
    with ray_session(workers=2):
        yield


def test_read_spike_file_dtype_and_row_count():
    arr = read_spike_file(FIXTURES / "tiny_on.spike", workers=2, progress=False)
    assert arr.dtype == SPIKE_DTYPE
    assert len(arr) == 6


def test_read_spike_file_first_row_values():
    arr = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    row = arr[0]
    assert row["id"] == 1
    assert row["result_id"] == 101
    assert row["peak_power"] == pytest.approx(26.1, rel=1e-5)
    assert row["mean_power"] == pytest.approx(1.0)
    assert row["time"] == pytest.approx(2457451.10000000)
    assert row["q_pix"] == 1001
    assert row["detection_freq"] == pytest.approx(1400000000.0)
    assert row["fft_len"] == 128
    assert row["rfi_checked"] == 0
    assert row["rfi_found"] == 0
