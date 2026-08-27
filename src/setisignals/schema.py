"""Field layout for SETI@home Listen hit files.

Field order/types are the single source of truth mirroring
``idl/spike__define.pro``. The registry is keyed by :class:`SignalKind` so
sibling formats (``.pulse``, ``.triplet``, ``.autocorr``), which share
``spike``'s first 16 fields, can be added later without restructuring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class SignalKind(StrEnum):
    SPIKE = "spike"
    # PULSE, TRIPLET, AUTOCORR reserved for future extension.


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str  # numpy dtype string, e.g. "i8", "f4", "f8"
    fits_format: str  # FITS TFORM code, e.g. "K", "E", "D"


SPIKE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("id", "i8", "K"),
    FieldSpec("result_id", "i8", "K"),
    FieldSpec("peak_power", "f4", "E"),
    FieldSpec("mean_power", "f4", "E"),
    FieldSpec("time", "f8", "D"),
    FieldSpec("ra", "f4", "E"),
    FieldSpec("decl", "f4", "E"),
    FieldSpec("q_pix", "i8", "K"),
    FieldSpec("freq", "f8", "D"),
    FieldSpec("detection_freq", "f8", "D"),
    FieldSpec("barycentric_freq", "f8", "D"),
    FieldSpec("fft_len", "i4", "J"),
    FieldSpec("chirp_rate", "f4", "E"),
    FieldSpec("rfi_checked", "i4", "J"),
    FieldSpec("rfi_found", "i4", "J"),
    FieldSpec("reserved", "i4", "J"),
)

SCHEMAS: dict[SignalKind, tuple[FieldSpec, ...]] = {
    SignalKind.SPIKE: SPIKE_FIELDS,
}


def dtype_for(fields: tuple[FieldSpec, ...]) -> np.dtype:
    return np.dtype([(f.name, f.dtype) for f in fields])


def get_dtype(kind: SignalKind) -> np.dtype:
    return dtype_for(SCHEMAS[kind])


SPIKE_DTYPE: np.dtype = dtype_for(SPIKE_FIELDS)
