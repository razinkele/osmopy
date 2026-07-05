"""Sub-grid bathymetry statistics for the finer-grid percid experiment. Pure,
grid/data-source-agnostic. shallow_fraction bins high-res EMODnet elevation
(negative=below sea level) into the target grid and returns per cell the fraction
of WET sub-pixels shallower than depth_max_m."""

from __future__ import annotations

import numpy as np

from osmose.maps.builder import GridSpec


def shallow_fraction(elev_hi, lat_hi, lon_hi, grid: GridSpec, depth_max_m: float):
    lat_edges = np.linspace(grid.upleft_lat, grid.lowright_lat, grid.nlat + 1)  # descending
    lon_edges = np.linspace(grid.upleft_lon, grid.lowright_lon, grid.nlon + 1)  # ascending
    row = np.clip(
        np.searchsorted(-lat_edges, -np.asarray(lat_hi), side="right") - 1, 0, grid.nlat - 1
    )
    col = np.clip(
        np.searchsorted(lon_edges, np.asarray(lon_hi), side="right") - 1, 0, grid.nlon - 1
    )
    depth = -np.asarray(elev_hi, dtype=np.float64)
    wet = depth > 0.0
    shallow = wet & (depth <= depth_max_m)
    n_wet = np.zeros((grid.nlat, grid.nlon))
    n_sh = np.zeros((grid.nlat, grid.nlon))
    R = np.broadcast_to(row[:, None], depth.shape)
    C = np.broadcast_to(col[None, :], depth.shape)
    np.add.at(n_wet, (R, C), wet.astype(float))
    np.add.at(n_sh, (R, C), shallow.astype(float))
    ocean = n_wet > 0
    frac = np.zeros_like(n_wet)
    np.divide(n_sh, n_wet, out=frac, where=ocean)
    return frac, ocean
