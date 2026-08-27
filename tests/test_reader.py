from pathlib import Path

import numpy as np
import pytest

from setisignals.io.reader import read_spike_file
from setisignals.ray_utils import ray_session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def _ray():
    with ray_session(workers=4):
        yield


def test_serial_and_parallel_parse_agree():
    serial = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    parallel = read_spike_file(FIXTURES / "tiny_on.spike", workers=4, progress=False)

    serial_sorted = np.sort(serial, order="id")
    parallel_sorted = np.sort(parallel, order="id")

    assert len(serial_sorted) == len(parallel_sorted) == 6
    for name in serial.dtype.names:
        np.testing.assert_array_equal(serial_sorted[name], parallel_sorted[name])
