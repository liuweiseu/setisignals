"""Read FITS/HDF5 tables (as produced by `convert`/`merge`) for plotting.

The `plot` commands consume the standard astronomical table formats this
tool itself writes, not raw `.spike` text -- this keeps the pipeline as
convert/merge (parse .spike -> fits/hdf5) then plot (fits/hdf5 -> figure),
and lets `plot` skip the text-parsing step entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

from setisignals.io.writer import HDF5_DATASET_PATH

FITS_SUFFIXES = (".fits", ".fit")
HDF5_SUFFIXES = (".h5", ".hdf5")
SUPPORTED_SUFFIXES = FITS_SUFFIXES + HDF5_SUFFIXES


def read_table_file(path: Path) -> np.ndarray:
    """Read a FITS or HDF5 file into a plain, native-byte-order structured array.

    Format is picked from the file extension (case-insensitive):
    `.fits`/`.fit` -> FITS binary table; `.h5`/`.hdf5` -> HDF5 dataset at
    `HDF5_DATASET_PATH`. Column names/dtypes are whatever the file has (the
    `id`..`reserved` spike fields, plus `target` if present from `merge`).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in FITS_SUFFIXES:
        table = Table.read(path, format="fits")
    elif suffix in HDF5_SUFFIXES:
        table = Table.read(path, format="hdf5", path=HDF5_DATASET_PATH)
    else:
        raise ValueError(
            f"unsupported file format {path.suffix!r} for {path} "
            f"(expected one of {SUPPORTED_SUFFIXES})"
        )
    # `.as_array()` can come back as a MaskedArray (e.g. a `target` column
    # with empty/unmatched entries) and in non-native byte order (FITS is
    # always big-endian on disk) -- normalize both so downstream NumPy/Ray
    # code doesn't have to special-case either.
    arr = np.asarray(table.as_array())
    return arr.astype(arr.dtype.newbyteorder("="))
