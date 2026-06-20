# osmose/forcing/physics.py
"""PHY NetCDF -> OSMOSE temperature/salinity forcing. Pure port of the MCP logic."""

from __future__ import annotations

import numpy as np
import xarray as xr

from osmose.forcing.grid import get_coords, get_var, regrid, resample_to_24, target_coords
from osmose.logging import setup_logging
from osmose.maps.builder import GridSpec

_log = setup_logging("osmose.forcing")

_VARS = [("thetao", "temperature", "degC"), ("so", "salinity", "PSU")]


def phy_to_physics(
    ds: xr.Dataset,
    grid: GridSpec,
    *,
    year: int = 0,
    depth_surface_m: float = 10.0,
) -> dict[str, xr.Dataset]:
    """Convert CMEMS physics into OSMOSE temperature/salinity forcing datasets.

    Returns a dict keyed by 'temperature'/'salinity' for whichever source
    variables are present; an empty dict if neither thetao nor so exists.
    """
    tlat, tlon = target_coords(grid)

    work = ds
    if year > 0 and "time" in work.dims:
        work = work.sel(time=work.time.dt.year == year)
    if "depth" in work.dims:
        work = work.sel(depth=depth_surface_m, method="nearest")

    src_lat, src_lon = get_coords(work)
    out: dict[str, xr.Dataset] = {}
    for src_name, osmose_name, units in _VARS:
        data = get_var(work, src_name)
        if data is None:
            _log.info("physics: %s not found in source, skipped", src_name)
            continue
        regridded = resample_to_24(regrid(data, src_lat, src_lon, grid))
        out[osmose_name] = xr.Dataset(
            {osmose_name: (["time", "latitude", "longitude"], regridded)},
            coords={"time": np.arange(24), "latitude": tlat, "longitude": tlon},
            attrs={
                "title": f"OSMOSE {osmose_name.title()} Forcing (from CMEMS)",
                "units": units,
                "depth_m": depth_surface_m,
                "conventions": "Latitude descending (north to south) to match grid.nc",
            },
        )
    return out
