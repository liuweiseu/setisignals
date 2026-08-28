from pathlib import Path

import numpy as np
import pytest

from setisignals.io.reader import read_spike_file
from setisignals.io.targets import (
    TargetWindow,
    is_off_variant,
    looks_like_off_source,
    parse_targets_file,
    resolve_target_names,
)
from setisignals.ray_utils import ray_session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def _ray():
    with ray_session(workers=2):
        yield


def test_parse_targets_file():
    windows = parse_targets_file(FIXTURES / "tiny_targets.txt")

    assert len(windows) == 2
    assert windows[0].name == "TESTON"
    assert windows[0].start_jd == pytest.approx(2457451.09)
    assert windows[0].end_jd == pytest.approx(2457451.11)
    assert windows[1].name == "TESTOFF"


def test_resolve_target_names_matches_and_unmatched():
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)
    windows = parse_targets_file(FIXTURES / "tiny_targets.txt")

    time_jd = np.concatenate([on["time"], off["time"], np.array([2457500.0])])
    names = resolve_target_names(time_jd, windows, workers=2)

    assert names.dtype == np.dtype("S7")  # max(len("TESTON"), len("TESTOFF")) == 7
    np.testing.assert_array_equal(names[: on.size], b"TESTON")
    np.testing.assert_array_equal(names[on.size : on.size + off.size], b"TESTOFF")
    assert names[-1] == b""  # outside any window


def test_resolve_target_names_empty_windows():
    time_jd = np.array([2457451.1, 2457451.2])
    names = resolve_target_names(time_jd, [], workers=1)
    np.testing.assert_array_equal(names, b"")


def test_is_off_variant_heuristic():
    assert is_off_variant("HIP63121_O")  # truncated "_OFF" -> "_O"
    assert is_off_variant("And_XI_off")
    assert is_off_variant("HIP11048_OFF")
    assert is_off_variant("off")  # bare label, e.g. `merge --targets off`
    assert is_off_variant("OFF")
    assert not is_off_variant("HIP63121")
    assert not is_off_variant("3C249_1")
    assert not is_off_variant("on")


def test_looks_like_off_source():
    assert looks_like_off_source(Path("HIP63121_OFF.spike"))
    assert looks_like_off_source(Path("/a/b/hip63121_off.spike"))
    assert not looks_like_off_source(Path("HIP63121.spike"))


def test_resolve_target_names_disambiguates_overlapping_windows_with_is_off():
    # Mirrors the real HIP63121 data quirk: the "on" window and its "off"
    # counterpart overlap in time because each records the whole observing
    # block, not per-dwell boundaries.
    windows = [
        TargetWindow("HIP63121", 100.0, 110.0),
        TargetWindow("HIP63121_O", 103.0, 113.0),
    ]
    # Times: two clearly inside the overlap region [103, 110].
    time_jd = np.array([104.0, 108.0])
    is_off = np.array([False, True])  # known: row0 is an on-source hit, row1 off

    names = resolve_target_names(time_jd, windows, is_off=is_off, workers=2)

    np.testing.assert_array_equal(names, [b"HIP63121", b"HIP63121_O"])


def test_resolve_target_names_without_is_off_shows_the_overlap_ambiguity():
    windows = [
        TargetWindow("HIP63121", 100.0, 110.0),
        TargetWindow("HIP63121_O", 103.0, 113.0),
    ]
    time_jd = np.array([104.0])  # in the overlap region

    names = resolve_target_names(time_jd, windows, workers=1)

    # Without is_off, the later-listed window wins -- documented ambiguity.
    assert names[0] == b"HIP63121_O"
