"""Grid creation and spatial utilities for OSMOSE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from osmose.logging import setup_logging

_log = setup_logging("osmose.grid")


def create_grid_csv(
    nrows: int,
    ncols: int,
    output: Path,
    mask: np.ndarray | None = None,
) -> None:
    """Create a CSV grid mask file.

    Args:
        nrows: Number of latitude rows.
        ncols: Number of longitude columns.
        output: Output file path.
        mask: Optional 2D array. Positive = ocean, negative = land. Default: all ocean.
    """
    if mask is None:
        mask = np.ones((nrows, ncols))
    pd.DataFrame(mask).to_csv(output, header=False, index=False)


def create_grid_netcdf(
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    nlat: int,
    nlon: int,
    output: Path,
    mask: np.ndarray | None = None,
) -> None:
    """Create an OSMOSE NetCDF grid file.

    Args:
        lat_bounds: (south, north) latitude limits.
        lon_bounds: (west, east) longitude limits.
        nlat: Number of latitude cells.
        nlon: Number of longitude cells.
        output: Output file path.
        mask: Optional 2D mask array. Default: all ocean (1).
    """
    lat = np.linspace(lat_bounds[0], lat_bounds[1], nlat)
    lon = np.linspace(lon_bounds[0], lon_bounds[1], nlon)

    if mask is None:
        mask = np.ones((nlat, nlon), dtype=np.float32)

    ds = xr.Dataset(
        {"mask": xr.DataArray(mask, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})},
        attrs={"description": "OSMOSE grid", "grid_type": "regular"},
    )
    ds.to_netcdf(output)
    ds.close()


def csv_maps_to_netcdf(
    csv_dir: Path,
    output: Path,
    nlat: int,
    nlon: int,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> None:
    """Convert CSV distribution maps to a single NetCDF file.

    Reads all *.csv files in csv_dir, each assumed to be a (nlat x nlon) grid.
    Creates a NetCDF with one variable per CSV file.
    """
    csv_dir = Path(csv_dir)
    lat = np.linspace(lat_bounds[0], lat_bounds[1], nlat)
    lon = np.linspace(lon_bounds[0], lon_bounds[1], nlon)

    data_vars = {}
    for csv_file in sorted(csv_dir.glob("*.csv")):
        arr = np.loadtxt(csv_file, delimiter=",")
        if arr.shape != (nlat, nlon):
            _log.warning("Skipping %s: shape %s != (%d, %d)", csv_file.name, arr.shape, nlat, nlon)
            continue
        var_name = csv_file.stem
        data_vars[var_name] = xr.DataArray(
            arr.astype(np.float32),
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )

    if data_vars:
        ds = xr.Dataset(data_vars)
        ds.to_netcdf(output)
        ds.close()
