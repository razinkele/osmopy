"""Merge Benguela's 4 single-variable ROMS forcing NetCDFs into one multi-variable file.

osmopy's ResourceState loads a SINGLE resource NetCDF and looks up each resource BY NAME in it
(resources.py:216). Benguela ships one file per resource, so only the first would load. Merge them
so all 4 (sphy/lphy/szoo/lzoo) resolve. Pass the external source clone dir as argv[1].
"""
from __future__ import annotations
import sys
from pathlib import Path
import xarray as xr

RES = {
    "sphy": "roms_climatological-sphy_benguela_15days_2000_2009.nc",
    "lphy": "roms_climatological-lphy_benguela_15days_2000_2009.nc",
    "szoo": "roms_climatological-szoo_benguela_15days_2000_2009.nc",
    "lzoo": "roms_climatological-lzoo_benguela_15days_2000_2009.nc",
}


def merge_forcing(src_dir: Path, out_path: Path) -> None:
    data_vars = {}
    for name, fname in RES.items():
        # Close each source handle before building the merged file; .load() materializes the
        # variable into memory so the merged Dataset holds real data, not a lazy file reference.
        with xr.open_dataset(src_dir / "input" / fname) as ds:
            var = list(ds.data_vars)[0]  # each file holds exactly one data variable
            data_vars[name] = ds[var].rename(name).load()
    merged = xr.Dataset(data_vars)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(out_path)
    merged.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    src = Path(sys.argv[1])
    merge_forcing(src, root / "data" / "benguela" / "input" / "roms_climatological_merged.nc")
    print("wrote merged forcing")
