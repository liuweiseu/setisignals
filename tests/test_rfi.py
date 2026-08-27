from pathlib import Path

import numpy as np
import pytest

from setisignals.analysis.rfi import classify_rfi
from setisignals.io.reader import read_spike_file
from setisignals.ray_utils import ray_session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def _ray():
    with ray_session(workers=2):
        yield


def test_classify_rfi_on_tiny_fixtures():
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)

    on_is_rfi, off_is_rfi = classify_rfi(
        on["detection_freq"], off["detection_freq"], bin_width_hz=93.0, workers=2
    )

    # Hand-computed: on ids 1 and 4 coincide (within one 93 Hz bin) with off
    # ids 101 and 104 respectively; the rest are isolated -> Clean.
    expected_on = np.array([True, False, False, True, False, False])
    expected_off = np.array([True, False, False, True, False, False])

    np.testing.assert_array_equal(on_is_rfi, expected_on)
    np.testing.assert_array_equal(off_is_rfi, expected_off)
