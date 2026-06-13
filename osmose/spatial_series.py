"""Extract a single grid cell's value over time from a spatial NetCDF output.

The Python engine writes spatial outputs as ``(time, species, lat, lon)`` arrays
(``{prefix}_spatial_biomass_Simu{i}.nc`` etc., see
``osmose.engine.output.write_outputs_netcdf_spatial``) when
``output.spatial.enabled=true``. ``cell_timeseries`` reads one cell's trajectory
out of such a file without materialising the whole cube.

NaN marks land cells (per the writer's ``nan_semantics`` attribute); ocean cells
with no schools in the averaging window hold ``0.0``. Land cells are returned as
NaN so callers can show an empty state rather than a flat-zero line.

This module is part of the core library and deliberately does not import from
``ui/`` — the dimension-name detection sets are defined locally (the UI's
``grid_helpers`` has parallel constants for its own use).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

# Canonical dim names the engine emits, plus common aliases for robustness.
_TIME_DIM_NAMES = frozenset({"time", "t"})
_LAT_DIM_NAMES = frozenset({"lat", "latitude", "y"})
_LON_DIM_NAMES = frozenset({"lon", "longitude", "x"})
_SPECIES_DIM_NAMES = frozenset({"species", "sp"})


def _find_dim(dims, names):
    """Return the first dim in ``dims`` whose lowercased name is in ``names``."""
    for dim in dims:
        if str(dim).lower() in names:
            return dim
    return None


def cell_timeseries(
    nc_path,
    variable,
    *,
    lat_index,
    lon_index,
    species=None,
    reduce="sum",
):
    """Time series of one grid cell from a spatial NetCDF variable.

    Parameters
    ----------
    nc_path : path-like
        A spatial NetCDF file (e.g. ``osm_spatial_biomass_Simu0.nc``).
    variable : str
        A data variable with time, lat, and lon dimensions (e.g.
        ``"spatial_biomass"``).
    lat_index, lon_index : int
        Zero-based indices into the lat/lon dimensions (the caller maps a
        clicked grid cell's row/col to these).
    species : str | int | None
        If the variable has a species dimension: select a single species by
        name or integer index. When ``None``, the species dimension is collapsed
        with ``reduce``.
    reduce : {"sum", "mean"}
        How to collapse any non-time dimension (e.g. species) that remains after
        cell selection. Ignored when a single ``species`` is chosen and no other
        extra dimension remains.

    Returns
    -------
    (times, values) : tuple[np.ndarray, np.ndarray]
        ``times`` is the time coordinate (fractional years from simulation
        start); ``values`` is the 1-D per-cell series. A land cell (all-NaN)
        yields all-NaN ``values``.

    Raises
    ------
    KeyError
        ``variable`` is not in the file.
    ValueError
        ``variable`` lacks time/lat/lon dims, or ``reduce`` is invalid.
    IndexError
        ``lat_index``/``lon_index`` is out of range.
    """
    if reduce not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")

    path = Path(nc_path)
    # open_dataset is lazy; the .values call below materialises only the
    # selected cell's vectors, never the full (time, species, lat, lon) cube.
    with xr.open_dataset(path) as ds:
        if variable not in ds.data_vars:
            raise KeyError(
                f"variable {variable!r} not in {path.name}; available: {sorted(map(str, ds.data_vars))}"
            )
        da = ds[variable]
        time_dim = _find_dim(da.dims, _TIME_DIM_NAMES)
        lat_dim = _find_dim(da.dims, _LAT_DIM_NAMES)
        lon_dim = _find_dim(da.dims, _LON_DIM_NAMES)
        if time_dim is None or lat_dim is None or lon_dim is None:
            raise ValueError(
                f"variable {variable!r} is not a spatial time series "
                f"(need time+lat+lon dims, have {tuple(da.dims)})"
            )

        ny, nx = int(da.sizes[lat_dim]), int(da.sizes[lon_dim])
        if not (0 <= lat_index < ny and 0 <= lon_index < nx):
            raise IndexError(
                f"cell (lat={lat_index}, lon={lon_index}) out of range for grid {ny}x{nx}"
            )

        cell = da.isel({lat_dim: lat_index, lon_dim: lon_index})

        species_dim = _find_dim(cell.dims, _SPECIES_DIM_NAMES)
        if species is not None and species_dim is not None:
            if isinstance(species, str):
                cell = cell.sel({species_dim: species})
            else:
                cell = cell.isel({species_dim: int(species)})

        extra = [d for d in cell.dims if d != time_dim]
        if extra:
            # skipna=False so land (uniformly NaN across species) stays NaN; ocean
            # cells carry 0.0/positive values, so the reduction is well-defined.
            if reduce == "sum":
                cell = cell.sum(dim=extra, skipna=False)
            else:
                cell = cell.mean(dim=extra, skipna=False)

        times = (
            ds[time_dim].values if time_dim in ds.coords else np.arange(int(cell.sizes[time_dim]))
        )
        values = cell.values

    return np.asarray(times), np.asarray(values)
