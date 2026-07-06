"""Bin-average BoB's 365-day resource forcing to a 24-step/year axis.

The 4.4.1 engine requires the forcing's steps/year to divide ndt=24; 365 does not.
Input day d -> output step floor(d*24/365); mean each bin (window mean conserved).
Writes roms_n2p2z2d2_biscay_24step.nc next to the original (original kept).

  PYTHONPATH=. .venv/bin/python scripts/resample_bob_forcing.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "examples" / "ltl" / "roms_n2p2z2d2_biscay.nc"
DST = ROOT / "data" / "examples" / "ltl" / "roms_n2p2z2d2_biscay_24step.nc"
NSTEPS = 24

def resample_to_24_steps(ds: xr.Dataset) -> xr.Dataset:
    n_in = ds.sizes["time"]
    step_of_day = (np.arange(n_in) * NSTEPS) // n_in  # day -> 0..23
    groups = xr.DataArray(step_of_day, dims="time", name="step")
    out = ds.groupby(groups).mean("time").rename({"step": "time"})
    out = out.assign_coords(time=np.arange(NSTEPS))
    return out[list(ds.data_vars)]  # preserve var order

def main() -> None:
    ds = xr.open_dataset(SRC, decode_times=False)
    out = resample_to_24_steps(ds)
    for v in out.data_vars:              # carry attrs
        out[v].attrs = ds[v].attrs
    out.attrs = ds.attrs
    out.to_netcdf(DST)
    print(f"wrote {DST} (time={out.sizes['time']}, vars={list(out.data_vars)})")

if __name__ == "__main__":
    main()
