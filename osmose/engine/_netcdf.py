"""Thread-safe NetCDF opening for engine code.

The netCDF4-python C extension is not thread-safe: concurrent calls to
``xr.open_dataset`` can corrupt HDF5 internal state, surfacing as
``NetCDF: Can't open HDF5 attribute`` or ``double free or corruption``.
Engine code runs under ``ThreadPoolExecutor`` for parallel candidate
evaluation (NSGA-II with ``n_parallel > 1``), so every NetCDF open in the
engine path must be serialized. Data is also eagerly loaded so the
underlying file handle is released before the helper returns — the
returned Dataset is detached from the file and safe to keep for the
lifetime of a simulation.
"""

from __future__ import annotations

import threading
from pathlib import Path

import xarray as xr

_OPEN_LOCK = threading.Lock()


def open_dataset_safe(path: str | Path) -> xr.Dataset:
    """Open a NetCDF file thread-safely and return an in-memory Dataset.

    The file handle is closed before returning; the caller can retain the
    result without holding an HDF5 resource. Intended for the engine's
    small forcing files (grid, biomass forcing, physical fields). For
    multi-GiB forcing a chunked+locked strategy would be needed instead.
    """
    with _OPEN_LOCK, xr.open_dataset(path) as src:
        return src.load()
