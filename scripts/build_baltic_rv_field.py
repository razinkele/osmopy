#!/usr/bin/env python
"""Build data/baltic/forcing/baltic_rv_field.nc from the on-disk CMEMS reanalysis.

Usage: PYTHONPATH=. .venv/bin/python scripts/build_baltic_rv_field.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from osmose.forcing.grid import load_ocean_mask
from osmose.forcing.reproductive_volume import build_rv_field
from osmose.maps.builder import GridSpec

ROOT = Path(__file__).resolve().parent.parent
CMEMS = ROOT / "data" / "cmems_cache" / "cmems_downloads"
OUT = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"


def main() -> int:
    grid = GridSpec(
        nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30
    )
    ocean = load_ocean_mask(ROOT / "data" / "baltic" / "baltic_grid.nc")
    if ocean is None:
        print("could not load ocean mask from data/baltic/baltic_grid.nc", file=sys.stderr)
        return 1
    spawning = (
        np.flipud(
            np.genfromtxt(ROOT / "data" / "baltic" / "maps" / "cod_spawning.csv", delimiter=";")
        )
        > 0
    )
    phy = sorted(CMEMS.glob("*phy_monthly_reanalysis_so_*.nc"))
    bgc = sorted(CMEMS.glob("*bgc_monthly_reanalysis_o2_*.nc"))
    if not phy or not bgc or len(phy) != len(bgc):
        print(f"missing/mismatched CMEMS files: phy={len(phy)} bgc={len(bgc)}", file=sys.stderr)
        return 1
    phy_years = [xr.open_dataset(p) for p in phy]
    bgc_years = [xr.open_dataset(b) for b in bgc]
    ds = build_rv_field(phy_years, bgc_years, grid, ocean_mask=ocean, spawning_mask=spawning)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    print(
        f"wrote {OUT}: RV_ref={ds['reproductive_volume'].attrs['RV_ref']:.2f} m, "
        f"mean over spawning cells="
        f"{float(ds['reproductive_volume'].values[:, spawning].mean()):.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
