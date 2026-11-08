#!/usr/bin/env python
"""Measure how much the Baltic LTL pipeline overstates biomass through its column assumption.

osmose.forcing.bgc_to_ltl averages a native pixel's concentration over its depth LEVELS in
0-50 m and multiplies by a full 50 m column. Two biases stack: levels crowd near the surface
(about 1 m thick at the top and 5.8 m at depth), so an unweighted mean over-weights surface
plankton; and a shallow pixel is credited with a 50 m column it does not have.

For phyc and zooc this compares the thickness-weighted 0-50 m integral with the pipeline's
mean x 50 m, regrids both with the production masked nearest-neighbour, and reports:
  * the domain-total ratio (correct / current);
  * per-cell ratio quantiles over the ocean;
  * the same split by cells that had forcing before the 2026-11-08 hole fill vs the 65 filled
    cells (needs the pre-fill file from git: 682a68a9);
  * what the hole fill adds to the domain total under each integration.
Engine-free. Source: the cached CMEMS ERGOM 2024 monthly file (gitignored).

Usage: PYTHONPATH=. .venv/bin/python scripts/baltic_ltl_depth_bias.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from osmose.forcing import load_ocean_mask
from osmose.forcing.grid import regrid
from osmose.maps.builder import GridSpec

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "cmems_cache" / "cmems_downloads" / "baltic_bgc_forecast_2024.nc"
GRID_NC = ROOT / "data" / "baltic" / "baltic_grid.nc"
PRE_FILL = "682a68a9:data/baltic/baltic_ltl_biomass.nc"
GRID = GridSpec(nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)


def _pre_fill_holes(ocean: np.ndarray) -> np.ndarray:
    raw = subprocess.run(
        ["git", "-C", str(ROOT), "show", PRE_FILL], capture_output=True, check=True
    ).stdout
    with xr.open_dataset(io.BytesIO(raw)) as old:
        arr = np.stack([old[g].values for g in old.data_vars])
    return ocean & (arr == 0).all(axis=(0, 1))


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"source not found: {SOURCE}")
    ocean = load_ocean_mask(GRID_NC)
    if ocean is None:
        raise SystemExit(f"no ocean mask in {GRID_NC}")
    holes = _pre_fill_holes(ocean)
    had = ocean & ~holes
    with xr.open_dataset(SOURCE) as src_all:
        src = src_all.sortby("depth").sel(depth=slice(0, 50)).load()
    z = src.depth.values
    edges = np.concatenate([[0.0], (z[:-1] + z[1:]) / 2, [min(50.0, z[-1] + (z[-1] - z[-2]) / 2)]])
    dz = np.diff(edges)
    print(
        f"{len(z)} levels in 0-50 m; layer thickness {dz[0]:.2f} m at the top, {dz[-1]:.2f} m at depth"
    )
    print(
        f"cells: {int(ocean.sum())} ocean = {int(had.sum())} had forcing + {int(holes.sum())} filled"
    )
    lat, lon = src.latitude.values, src.longitude.values
    for var in ("phyc", "zooc"):
        c = src[var].values  # (t, z, y, x)
        valid = np.isfinite(c)
        integral = np.sum(np.where(valid, c, 0.0) * dz[None, :, None, None], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN land columns
            current = np.nanmean(np.where(valid, c, np.nan), axis=1) * 50.0
        px_valid = np.isfinite(current).all(axis=0)
        kw = {"src_valid": px_valid, "target_mask": ocean}
        a = regrid(np.nan_to_num(integral), lat, lon, GRID, **kw).sum(axis=0)
        b = regrid(np.nan_to_num(current), lat, lon, GRID, **kw).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(b > 0, a / b, np.nan)
        q = np.nanquantile(ratio[ocean], [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"\n{var}: correct / current, domain total {a[ocean].sum() / b[ocean].sum():.3f}")
        print(f"  per-cell quantiles (5/25/50/75/95%): {np.round(q, 3).tolist()}")
        print(
            f"  median ratio: cells that had forcing {np.nanmedian(ratio[had]):.3f}, "
            f"filled cells {np.nanmedian(ratio[holes]):.3f}"
        )
        print(
            f"  the hole fill adds {b[holes].sum() / b[had].sum():+.1%} with the current "
            f"integration, {a[holes].sum() / a[had].sum():+.1%} with the correct one"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
