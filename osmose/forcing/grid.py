# osmose/forcing/grid.py
"""Grid geometry helpers for CMEMS->OSMOSE forcing conversion.

Pure: numpy/xarray + osmose.maps.builder.GridSpec only. No CMEMS/MCP deps,
so this module (and its tests) run in the clean CI venv.

regrid/resample use O(nlat*nlon) Python loops (verbatim from the MCP source),
intended for OSMOSE-scale config grids (~1e3-1e4 cells, coarse by construction).
A much finer grid would want a vectorized scipy.spatial.cKDTree / np.searchsorted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from osmose.logging import setup_logging
from osmose.maps.builder import GridSpec

_log = setup_logging("osmose.forcing")


def target_coords(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center (lat[nlat], lon[nlon]); latitude descending (north->south)."""
    rows = np.arange(grid.nlat)
    cols = np.arange(grid.nlon)
    lat = grid.upleft_lat - (rows + 0.5) * grid.dy
    lon = grid.upleft_lon + (cols + 0.5) * grid.dx
    return lat, lon


def regrid(
    data_3d: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray, grid: GridSpec
) -> np.ndarray:
    """Nearest-neighbor regrid (time, src_lat, src_lon) -> (time, nlat, nlon).

    Warns once when the target grid extends beyond the source data extent:
    out-of-coverage cells are silently filled by nearest-EDGE extrapolation
    (argmin always returns an in-bounds index). The MCP path was implicitly
    safe (download bbox always covered the grid); the grid-general + BYO-file
    surface this package adds removes that guarantee, so surface it loudly.
    """
    tlat, tlon = target_coords(grid)
    if (
        tlat.min() < src_lat.min()
        or tlat.max() > src_lat.max()
        or tlon.min() < src_lon.min()
        or tlon.max() > src_lon.max()
    ):
        _log.warning(
            "target grid (lat %.2f-%.2f lon %.2f-%.2f) extends beyond source "
            "coverage (lat %.2f-%.2f lon %.2f-%.2f); out-of-coverage cells are "
            "filled by nearest-edge extrapolation",
            tlat.min(),
            tlat.max(),
            tlon.min(),
            tlon.max(),
            src_lat.min(),
            src_lat.max(),
            src_lon.min(),
            src_lon.max(),
        )
    nlat, nlon = len(tlat), len(tlon)
    nt = data_3d.shape[0]
    out = np.zeros((nt, nlat, nlon), dtype=np.float64)
    for j in range(nlat):
        lat_idx = int(np.argmin(np.abs(src_lat - tlat[j])))
        for i in range(nlon):
            lon_idx = int(np.argmin(np.abs(src_lon - tlon[i])))
            out[:, j, i] = data_3d[:, lat_idx, lon_idx]
    return out


def resample_to_24(data: np.ndarray) -> np.ndarray:
    """Linear-interpolate (time, lat, lon) to 24 biweekly steps; identity if already 24."""
    nt, nlat, nlon = data.shape
    if nt == 24:
        return data
    out = np.zeros((24, nlat, nlon), dtype=np.float64)
    xp = np.linspace(0, 1, nt)
    x = np.linspace(0, 1, 24)
    for j in range(nlat):
        for i in range(nlon):
            out[:, j, i] = np.interp(x, xp, data[:, j, i])
    return out


def cell_volume_m3(grid: GridSpec, depth_m: float) -> float:
    """Approximate cell volume (m^3) using the grid's mid-latitude cos factor."""
    mid_lat = (grid.upleft_lat + grid.lowright_lat) / 2.0
    cos_lat = np.cos(np.radians(mid_lat))
    area = (abs(grid.dy) * 111320) * (abs(grid.dx) * 111320 * cos_lat)
    return float(area * depth_m)


def get_coords(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract lat/lon arrays, tolerating 'latitude'/'lat' and 'longitude'/'lon'."""
    lat = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
    lon = ds.longitude.values if "longitude" in ds.coords else ds.lon.values
    return lat, lon


def get_var(ds: xr.Dataset, name: str) -> np.ndarray | None:
    """Variable as 3D (time, lat, lon), NaN->0; None if absent."""
    if name not in ds:
        return None
    arr = np.nan_to_num(ds[name].values, nan=0.0)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return arr


def load_ocean_mask(grid_file: Path | None) -> np.ndarray | None:
    """Load a bool (nlat, nlon) ocean mask (True=ocean) from a grid NetCDF, or None."""
    if grid_file is None or not Path(grid_file).exists():
        return None
    try:
        with xr.open_dataset(grid_file) as gds:
            return gds["mask"].values.astype(bool)
    except (OSError, KeyError):
        return None


def apply_land_mask(groups: dict[str, np.ndarray], ocean_mask: np.ndarray) -> None:
    """Set land cells to NaN in every (time, lat, lon) array, in place.

    Shape mismatch is a logged no-op so conversion still runs with a stale grid file.
    """
    for arr in groups.values():
        if arr.shape[1:] != ocean_mask.shape:
            _log.warning(
                "ocean mask %s does not match data grid %s; skipping land mask",
                ocean_mask.shape,
                arr.shape[1:],
            )
            return
        arr[:, ~ocean_mask] = np.nan
