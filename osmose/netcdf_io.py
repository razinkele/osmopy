"""Opening NetCDF without xarray's backend-plugin scan (depends on nothing else in osmose).

Imported by the engine (``osmose.engine._netcdf``) and by readers such as ``osmose.results``;
kept outside ``osmose.engine`` so a reader importing it does not initialise the engine package
(which imports ``osmose.results``: a cycle).
"""

from __future__ import annotations

import xarray as xr


def _netcdf_backend():
    """The netCDF4 backend CLASS, or None (xarray then guesses). Passing the class, not the name,
    skips xarray's scan of every installed backend plugin: ``engine="netcdf4"`` still lists them
    all, and in the production env that imported rioxarray -> rasterio/GDAL on the first open,
    about 0.85 s and 200 MB per process (measured 2026-11-26)."""
    try:
        import netCDF4  # noqa: F401  (the backend needs it)
        from xarray.backends import NetCDF4BackendEntrypoint
    except ImportError:  # pragma: no cover
        return None
    return NetCDF4BackendEntrypoint


NC_BACKEND = _netcdf_backend()


def open_dataset(path, **kwargs) -> xr.Dataset:
    """``xr.open_dataset`` with the netCDF4 backend named by class (see ``_netcdf_backend``)."""
    if NC_BACKEND is not None and "engine" not in kwargs:
        kwargs["engine"] = NC_BACKEND
    return xr.open_dataset(path, **kwargs)
