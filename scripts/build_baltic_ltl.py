#!/usr/bin/env python
"""Regenerate data/baltic/baltic_ltl_biomass.nc from its CMEMS source, with no coastal holes.

The 2026-04 grid-mask rebuild (912 -> 616 ocean cells, d61ea1c1) added coastal cells without
regenerating this file. Plain nearest-neighbour regridding snapped them to land pixels of the
native ~1 nm ERGOM grid, so 65 ocean cells were exactly 0 in all six groups and all 24 frames,
and fish mapped there found no plankton or benthos (review S3,
docs/reviews/2026-11-07-temporal-spatial-dynamics-review.md).

This rebuilds the file the way mcp_servers/copernicus/server.py:generate_osmose_ltl does
(mode A, 0-50 m mean, all twelve months, C:Chl 50), with the ocean mask passed. With the mask,
``regrid``'s masked fallback fills a wet cell whose nearest pixel is land from the nearest
valid pixel, the same rule and metric as the oxygen forcing (scripts/make_baltic_oxygen_forcing.py).
Land is written as 0.0, the file's convention, not NaN.

Source: CMEMS BAL MFC ERGOM monthly fields for 2024 (SMHI; "CMEMS ERGOM monthly integrated model
fields"), cached at data/cmems_cache/cmems_downloads/baltic_bgc_forecast_2024.nc (gitignored).

--check compares a fresh build against the committed file and writes nothing. Every ocean cell
that had forcing must be bit-identical, and only the previously empty ones may change.

Usage: PYTHONPATH=. .venv/bin/python scripts/build_baltic_ltl.py [--check] [--source PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from osmose.forcing import LtlParams, bgc_to_ltl, load_ocean_mask, write_ltl
from osmose.maps.builder import GridSpec

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "cmems_cache" / "cmems_downloads" / "baltic_bgc_forecast_2024.nc"
GRID_NC = ROOT / "data" / "baltic" / "baltic_grid.nc"
OUT = ROOT / "data" / "baltic" / "baltic_ltl_biomass.nc"
GRID = GridSpec(nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)


def build(source: Path) -> xr.Dataset:
    ocean = load_ocean_mask(GRID_NC)
    if ocean is None:
        raise SystemExit(f"no ocean mask in {GRID_NC}")
    with xr.open_dataset(source) as ds:
        out = bgc_to_ltl(
            ds,
            GRID,
            year=0,
            depth_integrate_m=50.0,
            params=LtlParams(chl_to_biomass_factor=50.0),
            ocean_mask=ocean,
        ).load()
    for name in out.data_vars:
        out[name] = out[name].fillna(0.0)  # land is 0.0 in this file, never NaN
    out.attrs.update(
        title="Baltic Sea OSMOSE LTL Forcing (from CMEMS)",
        source="CMEMS BAL MFC ERGOM monthly fields 2024 (data/cmems_cache/cmems_downloads/"
        "baltic_bgc_forecast_2024.nc)",
        regrid=(
            "nearest neighbour; a wet cell whose nearest native pixel is land takes the nearest "
            "valid pixel (plain lat/lon degrees), so no ocean cell is an all-zero regrid hole"
        ),
        land_encoding="0.0 (not NaN)",
        conventions="Latitude descending (north to south) to match grid.nc",
        built_by="scripts/build_baltic_ltl.py",
    )
    return out


def check(new: xr.Dataset) -> int:
    ocean = load_ocean_mask(GRID_NC)
    assert ocean is not None
    with xr.open_dataset(OUT) as old_ds:
        groups = list(old_ds.data_vars)
        old = np.stack([old_ds[g].values for g in groups])
    fresh = np.stack([new[g].values for g in groups])
    was_empty = ocean & (old == 0).all(axis=(0, 1))
    had = ocean & ~was_empty
    same = np.array_equal(fresh[..., had], old[..., had])
    still_empty = ocean & (fresh == 0).all(axis=(0, 1))
    print(
        f"ocean cells {int(ocean.sum())}: had forcing {int(had.sum())}, empty {int(was_empty.sum())}"
    )
    print(f"  cells that had forcing bit-identical: {same}")
    print(
        f"  empty cells now filled: {int((was_empty & ~still_empty).sum())}/{int(was_empty.sum())}"
    )
    print(f"  ocean cells still empty: {int(still_empty.sum())}")
    land_old = int((old[..., ~ocean] != 0).any(axis=(0, 1)).sum())
    print(
        f"  land cells with forcing: before {land_old}, after "
        f"{int((fresh[..., ~ocean] != 0).any(axis=(0, 1)).sum())}"
    )
    for g, name in enumerate(groups):
        print(
            f"  {name:17s} ocean total, mean over frames: {old[g][:, ocean].sum(axis=1).mean():.6g}"
            f" -> {fresh[g][:, ocean].sum(axis=1).mean():.6g} t"
        )
    return 0 if same and not still_empty.any() else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed file")
    ap.add_argument("--source", type=Path, default=SOURCE)
    args = ap.parse_args()
    if not args.source.exists():
        raise SystemExit(f"source not found: {args.source} (download it via the Copernicus MCP)")
    new = build(args.source)
    if args.check:
        return check(new)
    write_ltl(new, OUT, overwrite=True)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
