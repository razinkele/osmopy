#!/usr/bin/env python
"""Rebuild data/baltic/maps/cod_east_spawning.csv as the SD 25 spawning ground.

Eastern Baltic cod spawn almost only in SD 25: "SD 25 has been basically the only area supporting
successful reproduction of the eastern Baltic cod since the mid-1980s" (Eero et al., 2012; also
Bartolino et al., 2017; Orio et al., 2019). The previous map put 7 of its 144 cells in SD 25, all
in one row at 55.95 N on the box's northern edge, none in the Bornholm Basin itself, and 89 at
or above 57 N (review C2, docs/reviews/2026-11-07-temporal-spatial-dynamics-review.md).

A cell is a spawning cell when all three hold:
  * it is in cod_east's adult map (the stock stays on its side of the E/W split);
  * its centre is in the review's SD 25 box, 54.5-56.2 N x 14-18 E;
  * its annual-mean bottom salinity is >= 11 PSU, the egg threshold the model's reproductive
    volume already uses (Koster et al., 2005), from baltic_salinity_bottom_climatology.nc.
On the current inputs that is 14 cells: the Bornholm Basin (54.75-55.65 N x 15.4-16.6 E, 12) and
the Slupsk Furrow (55.35 N, 17.4-17.8 E, 2).

Map CSVs are stored SOUTH-first: flipped, their land pattern matches baltic_grid.nc's
north-first ocean mask on every cell. The salinity file is north-first. The land (-99) pattern
of the existing file is kept.

Usage: PYTHONPATH=. .venv/bin/python scripts/build_cod_east_sd25_spawning.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
MAPS = ROOT / "data" / "baltic" / "maps"
SPAWN = MAPS / "cod_east_spawning.csv"
ADULT = MAPS / "cod_east_adult.csv"
SALINITY = ROOT / "data" / "baltic" / "baltic_salinity_bottom_climatology.nc"
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 54.5, 56.2, 14.0, 18.0
S_MIN = 11.0
NLAT, NLON, TOP, LEFT, DLAT, DLON = 40, 50, 66.0, 10.0, 0.3, 0.4


def read_map(path: Path) -> np.ndarray:
    """North-first (row 0 = northernmost) grid from a south-first map CSV."""
    rows = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return np.flipud(np.array([[float(v) for v in r.split(";")] for r in rows]))


def write_map(path: Path, grid_nf: np.ndarray) -> None:
    rows = np.flipud(grid_nf).astype(int)
    path.write_text("\n".join(";".join(str(v) for v in r) for r in rows) + "\n")


def spawning_cells() -> np.ndarray:
    lat = TOP - (np.arange(NLAT) + 0.5) * DLAT
    lon = LEFT + (np.arange(NLON) + 0.5) * DLON
    box = (
        (lat[:, None] >= LAT_MIN)
        & (lat[:, None] <= LAT_MAX)
        & (lon[None, :] >= LON_MIN)
        & (lon[None, :] <= LON_MAX)
    )
    with xr.open_dataset(SALINITY) as ds:
        s = ds["salinity"].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # land is NaN in every frame
        s_mean = np.nanmean(s, axis=0)
    adult = read_map(ADULT) > 0
    return adult & box & (np.nan_to_num(s_mean, nan=-1.0) >= S_MIN)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()
    old = read_map(SPAWN)
    cells = spawning_cells()
    new = np.where(old == -99, -99, np.where(cells, 1, 0))
    print(f"cod_east spawning cells: {int((old > 0).sum())} -> {int((new > 0).sum())}")
    if (new[old == -99] != -99).any() or (cells & (old == -99)).any():
        raise SystemExit("a spawning cell falls on land")
    if args.dry_run:
        return 0
    write_map(SPAWN, new)
    print(f"wrote {SPAWN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
