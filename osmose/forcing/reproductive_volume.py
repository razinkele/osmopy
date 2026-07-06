"""Reproductive-volume field generator (Baltic cod egg survival).

Turns depth-resolved CMEMS salinity (`so`) + oxygen (`o2`) into a per-cell field
= summed thickness of the water column where salinity >= sal_thresh AND
oxygen >= o2_thresh co-occur (the classic Baltic-cod reproductive-volume). Cod
eggs float mid-column in the saline layer, NOT on the anoxic seafloor, so the
whole column is scanned rather than the bottom slice.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from osmose.forcing.grid import get_coords, regrid, resample_to_24, target_coords


def _layer_thickness(depths: NDArray[np.float64]) -> NDArray[np.float64]:
    """Thickness (m) attributed to each depth level = span between mid-points."""
    d = np.asarray(depths, dtype=np.float64)
    edges = np.empty(d.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (d[:-1] + d[1:])
    edges[0] = d[0] - 0.5 * (d[1] - d[0]) if d.size > 1 else d[0]
    edges[-1] = d[-1] + 0.5 * (d[-1] - d[-2]) if d.size > 1 else d[0]
    return np.clip(np.diff(edges), 0.0, None)


def viable_thickness(
    so: NDArray[np.float64],
    o2: NDArray[np.float64],
    depths: NDArray[np.float64],
    sal_thresh: float,
    o2_thresh: float,
) -> float:
    """Summed thickness (m) of viable water in one column (any NaN level is not viable)."""
    thick = _layer_thickness(depths)
    viable = (np.nan_to_num(so, nan=-1.0) >= sal_thresh) & (
        np.nan_to_num(o2, nan=-1.0) >= o2_thresh
    )
    return float(thick[viable].sum())


def _rv_year(
    phy_ds: xr.Dataset, bgc_ds: xr.Dataset, sal_thresh: float, o2_thresh: float
) -> NDArray:
    """Per-(time,lat,lon) viable thickness on the SOURCE grid for one year's data."""
    so = phy_ds["so"]  # (time, depth, lat, lon)
    o2 = bgc_ds["o2"]
    depths = so["depth"].values.astype(np.float64)
    thick = _layer_thickness(depths)  # (depth,)
    so_v = so.values.astype(np.float64)
    o2_v = o2.values.astype(np.float64)
    viable = (np.nan_to_num(so_v, nan=-1.0) >= sal_thresh) & (
        np.nan_to_num(o2_v, nan=-1.0) >= o2_thresh
    )  # (time, depth, lat, lon)
    return np.einsum("tdyx,d->tyx", viable.astype(np.float64), thick)  # (time, lat, lon)


def build_rv_field(
    phy_years: list[xr.Dataset],
    bgc_years: list[xr.Dataset],
    grid,
    *,
    sal_thresh: float = 11.0,
    o2_thresh: float = 89.3,
    ocean_mask: NDArray[np.bool_],
    spawning_mask: NDArray[np.bool_],
) -> xr.Dataset:
    """Build the 24-step climatology RV field (mean of per-year RV) + RV_ref attr.

    phy_years[i]/bgc_years[i] are one year's depth-resolved `so`/`o2` datasets.
    ocean_mask / spawning_mask are (nlat, nlon) bool on the TARGET (engine) grid,
    north-first. Returns a Dataset with var `reproductive_volume` (24, nlat, nlon).
    """
    per_year_24 = []
    for phy, bgc in zip(phy_years, bgc_years):
        rv_src = _rv_year(phy, bgc, sal_thresh, o2_thresh)  # (t, srclat, srclon)
        src_lat, src_lon = get_coords(phy)
        rv_grid = regrid(rv_src, src_lat, src_lon, grid)  # (t, nlat, nlon)
        per_year_24.append(resample_to_24(rv_grid))  # (24, nlat, nlon)
    rv = np.mean(np.stack(per_year_24, axis=0), axis=0)  # mean-of-RV climatology
    rv[:, ~ocean_mask] = (
        np.nan
    )  # land -> NaN (spec §4; consumer's finite-guard + cell guard skip it)

    # RV_ref = mean over RV>0 spawning cells across all 24 steps
    sp_vals = rv[:, spawning_mask]
    nonzero = sp_vals[sp_vals > 0]
    rv_ref = float(nonzero.mean()) if nonzero.size else 1.0

    lat, lon = target_coords(grid)
    ds = xr.Dataset(
        {"reproductive_volume": (("time", "latitude", "longitude"), rv)},
        coords={"time": np.arange(24), "latitude": lat, "longitude": lon},
    )
    ds["reproductive_volume"].attrs["RV_ref"] = rv_ref
    ds["reproductive_volume"].attrs["units"] = "m"
    return ds
