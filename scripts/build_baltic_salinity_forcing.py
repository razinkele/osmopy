"""Build a Baltic bottom-salinity climatology NetCDF for the salinity gate.

Streams the full-depth CMEMS `so` year-files (memory-safe), bottom-extracts the
deepest valid salinity per cell, builds a per-month seasonal climatology,
regrids to the Baltic grid, gap-fills ocean NaN, and writes (24, ny, nx)
salinity. See docs/superpowers/specs/2026-07-04-baltic-salinity-forcing-design.md.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bottom_extract(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Deepest valid (non-NaN) salinity per cell. arr: (nt, ndepth, nlat, nlon),
    depth ascending (index 0 shallowest). Returns (nt, nlat, nlon); land (all-NaN
    columns) -> NaN."""
    finite = np.isfinite(arr)
    ndepth = arr.shape[1]
    # first finite scanning from the deepest level upward = deepest valid level
    rev_first = np.argmax(finite[:, ::-1, :, :], axis=1)  # (nt, nlat, nlon)
    bottom_idx = (ndepth - 1) - rev_first
    bottom = np.take_along_axis(arr, bottom_idx[:, None, :, :], axis=1)[:, 0, :, :]
    has_any = finite.any(axis=1)
    return np.where(has_any, bottom, np.nan)


def fill_ocean_nan(
    field: NDArray[np.float64], ocean_mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Fill NaN OCEAN cells with the nearest finite value (per time step). Land
    cells (ocean_mask False) are left untouched (may stay NaN)."""
    from scipy import ndimage

    out = field.copy()
    for t in range(out.shape[0]):
        f = out[t]
        valid = np.isfinite(f)
        nan_ocean = ocean_mask & ~valid
        if not nan_ocean.any() or not valid.any():
            continue
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        nearest = f[tuple(idx)]
        f[nan_ocean] = nearest[nan_ocean]
        out[t] = f
    return out
