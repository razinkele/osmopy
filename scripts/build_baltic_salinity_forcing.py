"""Build a Baltic bottom-salinity climatology NetCDF for the salinity gate.

Streams the full-depth CMEMS `so` year-files (memory-safe), bottom-extracts the
deepest valid salinity per cell, builds a per-month seasonal climatology,
regrids to the Baltic grid, gap-fills ocean NaN, and writes (24, ny, nx)
salinity. See docs/superpowers/specs/2026-07-04-baltic-salinity-forcing-design.md.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def bottom_extract(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Deepest valid (non-NaN) salinity per cell. arr: (nt, ndepth, nlat, nlon),
    depth ascending (index 0 shallowest). Returns (nt, nlat, nlon); land (all-NaN
    columns) -> NaN."""
    finite = np.isfinite(arr)
    ndepth = arr.shape[1]
    # first finite scanning from the deepest level upward = deepest valid level
    rev_first = np.argmax(finite[:, ::-1, :, :], axis=1)  # (nt, nlat, nlon)
    bottom_idx = (ndepth - 1) - rev_first
    bottom = np.take_along_axis(arr, bottom_idx[:, None, :, :], axis=1)[:, 0, :, :]
    has_any = finite.any(axis=1)
    return np.where(has_any, bottom, np.nan)


def fill_ocean_nan(
    field: NDArray[np.float64], ocean_mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Fill NaN OCEAN cells with the nearest finite value (per time step). Land
    cells (ocean_mask False) are left untouched (may stay NaN)."""
    from scipy import ndimage

    out = field.copy()
    for t in range(out.shape[0]):
        f = out[t]
        valid = np.isfinite(f)
        nan_ocean = ocean_mask & ~valid
        if not nan_ocean.any() or not valid.any():
            continue
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        nearest = f[tuple(idx)]
        f[nan_ocean] = nearest[nan_ocean]
        out[t] = f
    return out


def accumulate_climatology(so_files):
    """Stream year-files, return (clim (12, nlat, nlon), src_lat, src_lon).
    Per-month mean of bottom salinity across years; NaN where no month had data."""
    import xarray as xr

    sum_ = cnt = None
    src_lat = src_lon = None
    for f in so_files:
        ds = xr.open_dataset(f)
        try:
            bottom = bottom_extract(ds["so"].values)  # (12, nlat, nlon)
            months = ds["time"].dt.month.values
            if sum_ is None:
                nlat, nlon = bottom.shape[1:]
                sum_ = np.zeros((12, nlat, nlon), dtype=np.float64)
                cnt = np.zeros((12, nlat, nlon), dtype=np.float64)
                src_lat = ds["latitude"].values
                src_lon = ds["longitude"].values
            else:
                if bottom.shape[1:] != sum_.shape[1:]:
                    raise ValueError(
                        f"source grid mismatch in {f}: {bottom.shape[1:]} != {sum_.shape[1:]}"
                    )
            for k in range(len(months)):
                m = int(months[k]) - 1
                b = bottom[k]
                fin = np.isfinite(b)
                sum_[m][fin] += b[fin]
                cnt[m][fin] += 1.0
        finally:
            ds.close()
    clim = np.where(cnt > 0, sum_ / np.maximum(cnt, 1.0), np.nan)
    return clim, src_lat, src_lon


def build(config_dir: str, out_path: str) -> Path:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.forcing.grid import load_ocean_mask, regrid, resample_to_24, target_coords
    from osmose.maps.builder import GridSpec
    import xarray as xr

    cfg = OsmoseConfigReader().read(sorted(Path(config_dir).glob("*all-parameters*.csv"))[0])
    grid = GridSpec.from_config(cfg)
    so_files = sorted(
        glob.glob(
            str(
                Path(config_dir).parent
                / "cmems_cache"
                / "cmems_downloads"
                / "baltic_phy_monthly_reanalysis_so_*.nc"
            )
        )
    )
    if not so_files:
        raise FileNotFoundError(
            "no full-depth so files found under data/cmems_cache/cmems_downloads"
        )

    clim, src_lat, src_lon = accumulate_climatology(so_files)  # (12, src_lat, src_lon)
    regridded = regrid(clim, src_lat, src_lon, grid)  # (12, ny, nx)
    field24 = resample_to_24(regridded)  # (24, ny, nx)

    grid_nc = Path(config_dir) / "baltic_grid.nc"
    ocean_mask = load_ocean_mask(grid_nc)
    if ocean_mask is None:
        raise FileNotFoundError(
            f"ocean mask not loadable from {grid_nc} (need a 'mask' var); refusing to write a "
            f"salinity field with unfilled ocean NaN that the gate would misread as exclusions."
        )
    field24 = fill_ocean_nan(field24, ocean_mask)
    if bool(np.isnan(field24[np.broadcast_to(ocean_mask, field24.shape)]).any()):
        raise ValueError("salinity field has NaN in ocean cells after gap-fill; refusing to write.")

    tlat, tlon = target_coords(grid)
    out = xr.Dataset(
        {"salinity": (["time", "latitude", "longitude"], field24)},
        coords={"time": np.arange(24), "latitude": tlat, "longitude": tlon},
        attrs={
            "title": "OSMOSE Baltic bottom-salinity climatology (from CMEMS so)",
            "units": "PSU",
            "source": "CMEMS cmems_mod_bal_phy_my_P1M-m, deepest-valid level",
            "conventions": "Latitude descending (north to south) to match grid.nc",
        },
    )
    outp = Path(out_path)
    out.to_netcdf(outp)
    return outp


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build the Baltic bottom-salinity climatology.")
    ap.add_argument("--config-dir", default="data/baltic")
    ap.add_argument("--out", default="data/baltic/baltic_salinity_bottom_climatology.nc")
    args = ap.parse_args(argv)
    p = build(args.config_dir, args.out)
    import xarray as xr

    with xr.open_dataset(p) as ds:
        s = ds["salinity"].values
        print(f"wrote {p} shape={s.shape} salinity range {np.nanmin(s):.2f}-{np.nanmax(s):.2f} PSU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
