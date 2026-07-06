"""Regrid BoB's LTL forcing onto the exact Osmose model grid (lat x lon shape).

The synthetic ROMS-N2P2Z2D2 forcing (data/examples/ltl/roms_n2p2z2d2_biscay.nc)
was authored on a finer longitude axis (30 points) than the Osmose model grid
(data/examples/grid/bay_of_biscay_grid.nc, 20 lat x 20 lon; see
osm_param-grid.csv grid.nlon=20/grid.nlat=20). The legacy ltl.* forcing scheme
(fr.ird.osmose.ltl.LTLFastForcing) tolerated this by regridding internally.
The native 4.4.1 per-species species.file.* forcing path does NOT:
fr.ird.osmose.util.io.ForcingFile.init() hard-requires the NetCDF variable's
(ny, nx) shape to equal the Osmose grid's, and readVariable() indexes the
array directly by cell (jgrid, igrid) with no coordinate lookup. Without this
fix, the 4.4.1 jar aborts at Simulation.initResourceForcing() with
"NetCDF grid dimensions of variable <X> does not match Osmose grid dimensions".

This script linearly interpolates every data var from the source lat/lon axes
onto the grid's lat/lon axes (lat already matches 1:1; lon needs 30 -> 20).
Both axes span the same physical bounds (-6..-1 lon, 43..48 lat) so this is
interpolation, not extrapolation. Overwrites the 365-day source in place;
re-run scripts/resample_bob_forcing.py afterwards to regenerate the derived
24-step file consumed by species.file.sp8-13.

  PYTHONPATH=. .venv/bin/python scripts/regrid_bob_forcing_to_grid.py
  PYTHONPATH=. .venv/bin/python scripts/resample_bob_forcing.py
"""

from __future__ import annotations
import os
from pathlib import Path
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
GRID_NC = ROOT / "data" / "examples" / "grid" / "bay_of_biscay_grid.nc"
SRC = ROOT / "data" / "examples" / "ltl" / "roms_n2p2z2d2_biscay.nc"


def main() -> None:
    grid = xr.open_dataset(GRID_NC)
    target_lat = grid["lat"].values
    target_lon = grid["lon"].values
    grid.close()

    ds = xr.open_dataset(SRC, decode_times=False).load()
    src_vars = list(ds.data_vars)
    out = ds.interp(lat=target_lat, lon=target_lon, method="linear")
    # Snap coords to the grid's exact values (avoid float round-trip drift)
    # and restore each var's original dtype (interp promotes to float64).
    out = out.assign_coords(lat=target_lat, lon=target_lon)
    for v in src_vars:
        out[v] = out[v].astype(ds[v].dtype)
        out[v].attrs = ds[v].attrs
    out.attrs = ds.attrs
    ds.close()

    tmp = SRC.with_suffix(".nc.tmp")
    out.to_netcdf(tmp)
    os.replace(tmp, SRC)
    print(f"regridded {SRC} to lat={out.sizes['lat']} lon={out.sizes['lon']} (vars={src_vars})")


if __name__ == "__main__":
    main()
