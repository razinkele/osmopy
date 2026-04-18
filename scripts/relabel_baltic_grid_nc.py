"""Relabel baltic_grid.nc lat/lon to LTL cell-center convention.

The original file placed cell centers at bounding-box edges (lat[0]=66, step=12/39),
which overhangs the config bbox by half a cell and misaligns with baltic_ltl_biomass.nc
(lat[0]=65.85, step=12/40). The mask content is unchanged — only the coordinate labels.

Run from repo root:
    .venv/bin/python scripts/relabel_baltic_grid_nc.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

GRID_PATH = Path("data/baltic/baltic_grid.nc")

# Config bbox from data/baltic/baltic_param-grid.csv
UL_LAT = 66.0
UL_LON = 10.0
LR_LAT = 54.0
LR_LON = 30.0


def main() -> None:
    ds = xr.open_dataset(GRID_PATH)
    try:
        mask = ds["mask"].values.copy()
        ny, nx = mask.shape
        existing_attrs = dict(ds.attrs)
    finally:
        ds.close()

    # Cell-center convention: N cells span the bbox exactly, centers at edge+step/2.
    # Matches baltic_ltl_biomass.nc (step=12/40=0.30) and config bounding box.
    lat_step = (UL_LAT - LR_LAT) / ny  # descending → positive step magnitude
    lon_step = (LR_LON - UL_LON) / nx
    latitude = UL_LAT - (np.arange(ny) + 0.5) * lat_step
    longitude = UL_LON + (np.arange(nx) + 0.5) * lon_step

    new_ds = xr.Dataset(
        data_vars={"mask": (("latitude", "longitude"), mask)},
        coords={"latitude": latitude, "longitude": longitude},
        attrs={
            **existing_attrs,
            "history": (
                existing_attrs.get("history", "")
                + " | 2026-04-17: lat/lon relabeled to cell-center convention"
                f" (step={lat_step:.4f}°lat, {lon_step:.4f}°lon) to match"
                " baltic_ltl_biomass.nc and config bounding box."
            ).strip(" |"),
            "conventions": (
                "Cell centers. lat descending (north to south); N*step spans bbox."
            ),
        },
    )

    # Match original dtypes/encoding where it matters
    new_ds["mask"].encoding = {"dtype": "int32"}
    new_ds["latitude"].encoding = {"dtype": "float64"}
    new_ds["longitude"].encoding = {"dtype": "float64"}

    new_ds.to_netcdf(GRID_PATH)
    print(f"Wrote {GRID_PATH}")
    print(f"  latitude: {latitude[0]:.4f} .. {latitude[-1]:.4f}  step={lat_step:.4f}")
    print(f"  longitude: {longitude[0]:.4f} .. {longitude[-1]:.4f}  step={lon_step:.4f}")
    print(f"  mask ocean cells: {int((mask > 0).sum())} (unchanged)")


if __name__ == "__main__":
    main()
