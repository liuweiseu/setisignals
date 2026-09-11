from pathlib import Path

import numpy as np
import pytest

from setisignals.io.merge import merge_files
from setisignals.io.reader import read_spike_file
from setisignals.plotting.density import compute_density_grids
from setisignals.ray_utils import ray_session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def _ray():
    with ray_session(workers=2):
        yield


def _tagged_on_off():
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    off = read_spike_file(FIXTURES / "tiny_off.spike", workers=1, progress=False)
    on_tagged = merge_files([on], ["HIP63121"])
    off_tagged = merge_files([off], ["HIP63121_OFF"])
    return on_tagged, off_tagged


def _rfi_clean_tables():
    on_tagged, off_tagged = _tagged_on_off()
    on_is_rfi = np.zeros(on_tagged.size, dtype=bool)
    on_is_rfi[0] = True
    off_is_rfi = np.zeros(off_tagged.size, dtype=bool)
    off_is_rfi[1] = True

    rfi_data = np.concatenate([on_tagged[on_is_rfi], off_tagged[off_is_rfi]])
    clean_data = np.concatenate([on_tagged[~on_is_rfi], off_tagged[~off_is_rfi]])
    return rfi_data, clean_data, on_tagged.size + off_tagged.size


def test_compute_density_grids_two_tables_accounts_for_every_row():
    rfi_data, clean_data, total_rows = _rfi_clean_tables()

    grids, freq_edges, time_edges = compute_density_grids([rfi_data, clean_data], workers=2)

    assert len(grids) == 2
    for grid in grids:
        assert grid.shape == (len(freq_edges) - 1, len(time_edges) - 1)
    assert sum(int(g.sum()) for g in grids) == total_rows


def test_compute_density_grids_single_table():
    on_tagged, off_tagged = _tagged_on_off()
    table = np.concatenate([on_tagged, off_tagged])

    grids, freq_edges, time_edges = compute_density_grids([table], workers=2)

    assert len(grids) == 1
    assert int(grids[0].sum()) == table.size


def test_compute_density_grids_three_tables():
    rfi_data, clean_data, total_rows = _rfi_clean_tables()
    on_tagged, off_tagged = _tagged_on_off()
    third = np.concatenate([on_tagged, off_tagged])

    grids, freq_edges, time_edges = compute_density_grids([rfi_data, clean_data, third], workers=2)

    assert len(grids) == 3
    assert sum(int(g.sum()) for g in grids) == total_rows + third.size


def test_compute_density_grids_requires_target_column():
    on = read_spike_file(FIXTURES / "tiny_on.spike", workers=1, progress=False)
    with pytest.raises(ValueError, match="target"):
        compute_density_grids([on, on], workers=2)
