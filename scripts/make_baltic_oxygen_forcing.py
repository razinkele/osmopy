#!/usr/bin/env python3
"""Build the Baltic bottom dissolved-oxygen (o2b) forcing NetCDF.

Phase 2a (hypoxia-benthos coupling), Task 1. Downloads the 2024 monthly `o2b`
field (bottom dissolved oxygen, mmol m-3) from the CMEMS Baltic Sea
Biogeochemistry Analysis/Forecast product (ERGOM model, dataset
`cmems_mod_bal_bgc_anfc_P1M-m` -- see mcp_servers/copernicus/server.py's
DATASETS catalog, key "bgc_monthly_forecast"), regrids it to the OSMOSE
Baltic model grid (50x40), clips negative values (ERGOM's H2S proxy for
anoxic bottom water) to 0, and duplicates each of the 12 monthly frames x2 to
produce a 24-frame year.

The x2 duplication (NOT `osmose.forcing.grid.resample_to_24`'s linear
interpolation) is load-bearing: OSMOSE's `PhysicalData.get_value` indexes
`step % nframes` over the 24-step simulation year, so 12 monthly frames would
silently misalign the calendar from step 13 onward. Duplicating instead of
interpolating also means the raw monthly extrema are preserved exactly
(frames 0-1 = Jan, ..., 22-23 = Dec) rather than smoothed.

Regridding uses a MASKED nearest-neighbour (`_masked_regrid`, via
`scipy.spatial.cKDTree`), not the plain nearest-neighbour in
`osmose.forcing.grid.regrid`. `generate_osmose_ltl`'s pipeline (and a first
cut of this script) used plain nearest-neighbour, which snaps to the
geometrically nearest native ~1nm ERGOM pixel regardless of whether that
pixel has data; near the coast, that pixel is sometimes dry (NaN in the
native product), and the value collapsed to an artifact exact-0.0 (measured:
65 of 616 OSMOSE ocean cells, 10.55% of ocean cell-months). That is a real
defect here, not a downstream one: Task 2's O2 response curve satisfies
f(0) = 0 exactly, so an artifact zero would silently zero out benthos K in
real coastal habitat, and -- because genuine near-anoxia in this product
registers as a tiny positive value (~1e-18), never an exact 0.0 -- the
artifact is indistinguishable downstream from a legitimate low-O2 reading.
`_masked_regrid` restricts the nearest-neighbour search to VALID (non-NaN)
native pixels for every model wet cell (per `osmose.engine.grid.Grid`, the
same mask class the simulation engine itself loads at run time -- not the
forcing-side `load_ocean_mask` heuristic), so every wet cell always draws
from a real water pixel, however far. Land cells are untouched by this and
stay at 0.0 unconditionally.

Land cells are encoded as 0.0, matching data/baltic/baltic_ltl_biomass.nc's
convention -- NOT NaN, which the sibling
data/baltic/baltic_salinity_bottom_climatology.nc uses instead. The two
existing Baltic forcing files disagree on this; the task brief directs us to
match baltic_ltl_biomass.nc specifically. Land carries no benthos K to scale,
so 0.0 there is unambiguous (unlike a 0.0 at a wet cell, which is exactly the
ambiguity `_masked_regrid` eliminates).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/make_baltic_oxygen_forcing.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np
import xarray as xr
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
# Load .env from the project root, exactly like mcp_servers/copernicus/server.py does,
# so the script can be run without the parent shell having the vars exported.
load_dotenv(REPO_ROOT / ".env")

DATASET_ID = "cmems_mod_bal_bgc_anfc_P1M-m"
PRODUCT_ID = "BALTICSEA_ANALYSISFORECAST_BGC_003_007"
VARIABLE = "o2b"
YEAR = 2024
# Baltic bbox per the task brief (10-30E, 54-66N) -- narrower than server.py's
# BALTIC_BBOX (9.5-30.5, 53.5-66.5); both cover the OSMOSE grid's 10-30E/54-66N extent.
BBOX = {
    "minimum_longitude": 10.0,
    "maximum_longitude": 30.0,
    "minimum_latitude": 54.0,
    "maximum_latitude": 66.0,
}
CACHE_DIR = REPO_ROOT / "data" / "cmems_cache" / "cmems_downloads"
GRID_NC = REPO_ROOT / "data" / "baltic" / "baltic_grid.nc"
OUT_PATH = REPO_ROOT / "data" / "baltic" / "baltic_oxygen_bottom.nc"


def _require_creds() -> tuple[str, str]:
    """Return non-null (username, password) or raise with operator guidance.

    Same pattern as mcp_servers/copernicus/server.py:_require_creds.
    """
    user = os.environ.get("CMEMS_USERNAME")
    password = os.environ.get("CMEMS_PASSWORD")
    if not user or not password:
        raise RuntimeError(
            "CMEMS_USERNAME and CMEMS_PASSWORD environment variables must be set. "
            "See mcp_servers/copernicus/README.md."
        )
    return user, password


def download_o2b(cache_dir: Path = CACHE_DIR, *, year: int = YEAR) -> Path:
    """Download the monthly o2b field for `year`, subset to the Baltic bbox.

    Mirrors mcp_servers/copernicus/server.py:download_field's cm.subset call
    (dataset id + bbox from that server's catalog). Cached: skips re-download
    if the target file already exists. Note: the installed copernicusmarine
    (2.4.1) subset() takes `overwrite=`, not server.py's `overwrite_output_data=`
    -- that kwarg is stale against the currently pinned client version.
    """
    import copernicusmarine as cm

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"baltic_bgc_forecast_o2b_{year}.nc"
    out_file = cache_dir / filename
    if out_file.exists():
        return out_file

    user, password = _require_creds()
    cm.login(username=user, password=password, force_overwrite=True)
    cm.subset(
        dataset_id=DATASET_ID,
        variables=[VARIABLE],
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T23:59:59",
        output_directory=str(cache_dir),
        output_filename=filename,
        overwrite=True,
        disable_progress_bar=False,
        **BBOX,
    )
    if not out_file.exists():
        raise RuntimeError(f"download completed but file not found at {out_file}")
    return out_file


def _masked_regrid(
    raw: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tlat: np.ndarray,
    tlon: np.ndarray,
    wet_mask: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour regrid restricted to VALID (non-NaN) native pixels.

    Unlike osmose.forcing.grid.regrid (which snaps to the geometrically nearest
    native pixel regardless of validity), this only ever assigns a model wet cell
    (per `wet_mask`, (nlat, nlon) bool, True=ocean) the value of the nearest native
    pixel that actually has data -- per month, since which native pixels are valid
    can in principle vary by frame. Land cells (wet_mask False) are left at 0.0; the
    caller does not need to mask them again.

    raw: (nt, src_nlat, src_nlon), NaN where the native product has no data.
    Returns (nt, nlat, nlon).
    """
    from scipy.spatial import cKDTree

    nt = raw.shape[0]
    nlat, nlon = len(tlat), len(tlon)
    out = np.zeros((nt, nlat, nlon), dtype=np.float64)

    wet_rows, wet_cols = np.nonzero(wet_mask)
    if wet_rows.size == 0:
        return out
    wet_points = np.column_stack([tlat[wet_rows], tlon[wet_cols]])

    lon_grid, lat_grid = np.meshgrid(src_lon, src_lat)  # each (src_nlat, src_nlon)
    for t in range(nt):
        valid = ~np.isnan(raw[t])
        if not valid.any():
            raise ValueError(f"frame {t}: no valid (non-NaN) native pixels anywhere")
        valid_points = np.column_stack([lat_grid[valid], lon_grid[valid]])
        valid_values = raw[t][valid]
        tree = cKDTree(valid_points)
        _, nearest_idx = tree.query(wet_points)
        out[t, wet_rows, wet_cols] = valid_values[nearest_idx]
    return out


def build(source_file: Path, grid_nc: Path = GRID_NC, *, year: int = YEAR) -> xr.Dataset:
    """Regrid the downloaded o2b field to the OSMOSE Baltic grid; 24-frame o2b forcing."""
    from osmose.engine.grid import Grid
    from osmose.forcing.grid import get_coords, target_coords
    from osmose.maps.builder import GridSpec

    grid = GridSpec(
        nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30
    )
    tlat, tlon = target_coords(grid)

    ds = xr.open_dataset(source_file)
    try:
        src_lat, src_lon = get_coords(ds)
        # Preserve NaN (source-invalid) rather than get_var's nan_to_num(nan=0.0): the
        # masked regrid needs to tell "no data here" apart from "data here, value 0".
        data = ds[VARIABLE].values
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        if data.shape[0] != 12:
            raise ValueError(f"expected 12 monthly frames, got {data.shape[0]} in {source_file}")
    finally:
        ds.close()

    # Model wet-cell mask from the same authority the simulation engine loads at run
    # time (osmose.engine.grid.Grid), not the forcing-side load_ocean_mask heuristic.
    engine_grid = Grid.from_netcdf(grid_nc)
    ocean_mask = engine_grid.ocean_mask  # (40, 50) bool, True = ocean
    if ocean_mask.shape != (len(tlat), len(tlon)):
        raise ValueError(
            f"grid mismatch: mask {ocean_mask.shape} vs target ({len(tlat)}, {len(tlon)})"
        )

    regridded = _masked_regrid(data, src_lat, src_lon, tlat, tlon, ocean_mask)  # (12, 40, 50)
    np.clip(regridded, 0.0, None, out=regridded)  # ERGOM H2S proxy: negative O2 -> 0
    # Land cells are already 0.0 (out was zero-initialized and _masked_regrid only ever
    # writes wet cells); this matches baltic_ltl_biomass.nc's land-encoding convention.

    # Exact month duplication (NOT resample_to_24's linear interpolation) -- see module
    # docstring: required so PhysicalData.get_value's step % nframes stays calendar-aligned.
    field24 = np.repeat(regridded, 2, axis=0)

    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    history = (
        f"Generated by scripts/make_baltic_oxygen_forcing.py on {generated} from CMEMS "
        f"product {PRODUCT_ID} (dataset {DATASET_ID}), variable {VARIABLE}, year {year}. "
        "Regrid: nearest-neighbour restricted to valid (non-NaN) native pixels per wet "
        "model cell (osmose.engine.grid.Grid mask), eliminating coastal regrid-hole "
        "artifact zeros that a plain nearest-neighbour regrid produced (65/616 ocean "
        "cells, 10.55% of ocean cell-months, all coastal/archipelago)."
    )

    o2b = xr.DataArray(
        field24,
        dims=["time", "latitude", "longitude"],
        coords={"time": np.arange(24), "latitude": tlat, "longitude": tlon},
        attrs={"units": "mmol m-3", "long_name": "Bottom dissolved oxygen concentration"},
    )
    return xr.Dataset(
        {"o2b": o2b},
        attrs={
            "title": "OSMOSE Baltic bottom dissolved-oxygen forcing (from CMEMS o2b)",
            "units": "mmol m-3",
            "source": f"CMEMS {DATASET_ID} ({PRODUCT_ID}), {year} monthly means",
            "history": history,
            "conventions": (
                "Latitude descending (north to south) to match grid.nc; land cells 0.0 "
                "(matches baltic_ltl_biomass.nc); wet cells never 0.0 by regrid artifact "
                "(masked nearest-neighbour draws only from valid native pixels -- see "
                "history); each calendar month duplicated x2 to 24 frames (frames 0-1 = "
                "Jan, ..., 22-23 = Dec)"
            ),
        },
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Baltic bottom-O2 forcing NetCDF.")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--grid", type=Path, default=GRID_NC)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--year", type=int, default=YEAR)
    args = ap.parse_args(argv)

    source_file = download_o2b(args.cache_dir, year=args.year)
    ds = build(source_file, args.grid, year=args.year)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out)

    v = ds["o2b"].values
    print(f"wrote {args.out} shape={v.shape} o2b range {v.min():.2f}-{v.max():.2f} mmol/m3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
