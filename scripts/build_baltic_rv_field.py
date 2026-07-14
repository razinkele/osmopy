#!/usr/bin/env python
"""Build data/baltic/forcing/baltic_rv_field.nc from the on-disk CMEMS reanalysis.

Usage: PYTHONPATH=. .venv/bin/python scripts/build_baltic_rv_field.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from osmose.forcing.grid import load_ocean_mask
from osmose.forcing.reproductive_volume import build_rv_field, build_rv_field_interannual
from osmose.maps.builder import GridSpec

ROOT = Path(__file__).resolve().parent.parent
CMEMS = ROOT / "data" / "cmems_cache" / "cmems_downloads"
OUT = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"
OUT_INTER = ROOT / "data" / "baltic_rv" / "baltic_rv_field_interannual.nc"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interannual", action="store_true")
    args = parser.parse_args()

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
    if args.interannual:
        years = sorted({int(p.name.split("_so_")[1][:4]) for p in phy})
        start_year = years[0]
        # Stream one year at a time. The full 744x746x36 `so` grid is ~1.9 GB/year, so
        # eagerly opening all 29 years (and letting xarray cache each .values back into the
        # retained Dataset) grows RSS ~0.4 GB/year and OOM-kills a 32 GB box mid-build --
        # silently, because an OOM kill leaves no Python traceback. Feed the builder's zip()
        # two LAZY generators instead: each year's datasets are released (the with-block
        # closes and the reference drops) before the next year opens, bounding RSS flat at
        # ~4 GB regardless of the number of years. Do NOT refactor these back into eager
        # lists. (.close() alone does not help -- the cached arrays persist while the Dataset
        # object is referenced; the fix is to not retain the reference.)
        #
        # The CMEMS Baltic BGC (o2) reanalysis reports only the deepest 16 of the physics
        # (so) product's 36 standard depth levels (>= ~41 m; near-surface oxygen isn't
        # modeled). Restrict `so` to o2's depth levels before computing viable thickness,
        # since the RV metric is only defined where salinity AND oxygen co-occur. All o2 (and
        # all so) depth axes are identical across years (verified), so read o2's depths once
        # and align every so year to them; the per-year guard still fires if a future file
        # deviates. This is a genuine data-grid restriction, not a tuning fudge.
        o2_depths = xr.open_dataset(bgc[0])["depth"].values

        def _phy_aligned():
            for p in phy:
                with xr.open_dataset(p) as phy_ds:
                    aligned = phy_ds.sel(depth=o2_depths, method="nearest")
                    mismatch = float(np.abs(aligned["depth"].values - o2_depths).max())
                    if mismatch > 1e-3:
                        raise ValueError(
                            f"depth alignment mismatch {mismatch:.4f} m between so/o2 grids"
                        )
                    yield aligned

        def _bgc():
            for b in bgc:
                with xr.open_dataset(b) as bgc_ds:
                    yield bgc_ds

        ds = build_rv_field_interannual(
            _phy_aligned(),
            _bgc(),
            grid,
            ocean_mask=ocean,
            spawning_mask=spawning,
            start_year=start_year,
        )
        rv = ds["reproductive_volume"].values  # (nyear*24, ny, nx)
        n_year = rv.shape[0] // 24
        field_annual = np.nan_to_num(rv[:, spawning]).reshape(n_year, 24, -1).mean(axis=(1, 2))
        import pandas as pd

        off = pd.read_csv(ROOT / "docs" / "diagnostics" / "baltic_rv_fraction.csv")
        off["yr"] = pd.to_datetime(off["time"]).dt.year
        off_annual = (
            off.groupby("yr")["rv_fraction"]
            .mean()
            .reindex(range(start_year, start_year + n_year))
            .values
        )
        r = float(np.corrcoef(field_annual, off_annual)[0, 1])
        # SOFT check only. The offline series is a bottom-slice areal FRACTION; the engine
        # field is a full-column viable THICKNESS (different metrics — correlation is
        # scale-invariant but not guaranteed high even for a correct field). The real
        # correctness gate is the shape + finite-fraction sanity check (see Step 3 in
        # docs). A genuine orientation/np.flipud error shows up there as a
        # degenerate/NaN-heavy field, NOT as a marginal correlation — do not "fix"
        # orientation to force this number up.
        if r < 0.3:
            print(
                f"WARNING: field vs offline annual corr={r:.3f} (<0.3) — "
                "inspect the field before use."
            )
        else:
            print(
                f"tie-back OK-ish: field vs offline annual corr={r:.3f} "
                "(metrics differ; soft check)."
            )
        OUT_INTER.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(
            OUT_INTER,
            encoding={"reproductive_volume": {"dtype": "float32", "zlib": True, "complevel": 4}},
        )
        print(
            f"wrote {OUT_INTER}: {rv.shape}, RV_ref={ds['reproductive_volume'].attrs['RV_ref']:.2f}"
        )
        return 0

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
