"""Write parsed hit data to standard astronomical table formats."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from astropy.table import Table

HDF5_DATASET_PATH = "spike"


def write_table(
    data: np.ndarray,
    out_path: Path,
    fmt: Literal["fits", "hdf5"],
    extra_columns: dict[str, np.ndarray] | None = None,
) -> None:
    """Write a structured array to ``out_path`` as FITS or HDF5.

    Column names/order come directly from the structured array's dtype
    fields, so they match the IDL struct field names verbatim. HDF5 output
    stores the table under the fixed dataset name ``HDF5_DATASET_PATH``.
    ``extra_columns`` (name -> array, same length as ``data``) are appended
    to the table after the struct fields, in insertion order.
    """
    table = Table(data)
    for name, values in (extra_columns or {}).items():
        table[name] = values
    if fmt == "fits":
        table.write(out_path, format="fits", overwrite=True)
    elif fmt == "hdf5":
        table.write(
            out_path,
            format="hdf5",
            path=HDF5_DATASET_PATH,
            overwrite=True,
            serialize_meta=False,
        )
    else:
        raise ValueError(f"unsupported format: {fmt!r}")
