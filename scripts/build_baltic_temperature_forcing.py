#!/usr/bin/env python3
"""Build the Baltic two-layer temperature climatology NetCDF (C3 bioen Stage-1 Task 10).

Produces ``data/baltic/forcing/baltic_temperature_2layer_climatology.nc``: a
``(24, 2, 40, 50)`` monthly climatology giving bioenergetics its ``species.zlayer``
depth choice (spec 2026-08-30-baltic-c3-bioen-stage1-design.md, decision 4 / section
3.3) -- layer 0 = surface (nan-aware mean of the five cached CMEMS ``thetao`` depth
levels, 0.50-4.68 m), layer 1 = CMEMS ``bottomT``. A second ``bottom_depth`` (40, 50)
static field carries the per-cell water-column depth (deepest finite level of one
full-depth ``so`` year-file), which the builder's own validation uses to check the two
layers are not swapped (bottom colder than surface on deep cells).

No download here: all inputs are already cached under
``data/cmems_cache/cmems_downloads/`` from prior C4/salinity work --

* 29 ``baltic_phy_monthly_reanalysis_thetao_YYYY-01_YYYY-12.nc`` (1993-2021), each
  ``(time=12, depth=5, latitude, longitude)``. The five depth levels do NOT share a
  wet mask: coastal pixels shallower than ~4.7 m are NaN at the deeper levels, so the
  depth-mean must be nan-aware per pixel, not a plain mean.
* 29 ``baltic_phy_monthly_reanalysis_bottomT_YYYY-01_YYYY-12.nc`` (1993-2021), each
  ``(time=12, latitude, longitude)`` -- no depth axis, already "the bottom value".
* One full-depth ``baltic_phy_monthly_reanalysis_so_YYYY-01_YYYY-12.nc`` (any single
  year; defaults to the earliest, 1993), used only for its 36-level depth axis to
  derive a static ``bottom_depth`` field (mirrors
  ``scripts/build_baltic_salinity_forcing.py``'s ``bottom_extract``).

All three products share the product id ``cmems_mod_bal_phy_my_P1M-m`` and the native
~1.85 km analysed grid (744 x 746, latitude ascending, land = NaN).

Pipeline per layer: per-year monthly frame (thetao -> nan-aware depth-mean; bottomT
as-is) -> nan-aware climatology across the 29 years -> wet-aware masked regrid to the
50x40 OSMOSE Baltic grid (``make_baltic_oxygen_forcing._masked_regrid``, reused via
importlib rather than the plain nearest-neighbour ``osmose.forcing.grid.regrid`` --
the design review measured 66/616 wet cells snapping to a dry native pixel with the
latter) -> land cells set to NaN -> each of the 12 monthly frames duplicated x2 to 24
frames (frames 2m, 2m+1 = month m; NOT ``resample_to_24``'s linear interpolation --
same reasoning as ``make_baltic_oxygen_forcing``: ``PhysicalData`` indexes
``step % frame_count``, so 12 frames would silently misalign the month-to-step mapping
from step 13 onward). The two duplicated (24, 40, 50) layers are stacked on a new axis
1 to give ``temperature`` (24, 2, 40, 50) float32.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/build_baltic_temperature_forcing.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import subprocess
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cmems_cache" / "cmems_downloads"
DEFAULT_GRID_NC = REPO_ROOT / "data" / "baltic" / "baltic_grid.nc"
DEFAULT_OUT_PATH = (
    REPO_ROOT / "data" / "baltic" / "forcing" / "baltic_temperature_2layer_climatology.nc"
)

PRODUCT_ID = "cmems_mod_bal_phy_my_P1M-m"
SOURCE_YEARS = "1993-2021"

# Reuse the O2 builder's wet-aware masked-nearest-neighbour regrid (via the established
# scripts/ importlib-from-path idiom, see scripts/baltic_c4_salinity_ab.py) rather than
# reimplementing it or falling back to osmose.forcing.grid.regrid's unmasked
# argmin-nearest, which snaps land-adjacent wet cells to dry native pixels (design
# review: 66/616 wet cells for this exact product).
_oxygen_spec = importlib.util.spec_from_file_location(
    "make_baltic_oxygen_forcing", REPO_ROOT / "scripts" / "make_baltic_oxygen_forcing.py"
)
_oxygen_mod = importlib.util.module_from_spec(_oxygen_spec)
_oxygen_spec.loader.exec_module(_oxygen_mod)

masked_regrid = _oxygen_mod._masked_regrid


def layer0_from_thetao(a: NDArray[np.floating]) -> NDArray[np.floating]:
    """Nan-aware depth-mean of 4-D ``thetao`` (t, z, y, x) -> (t, y, x).

    The five cached depth levels (0.50-4.68 m) do not share a wet mask -- coastal
    pixels shallower than the deepest cached level are NaN there -- so this must
    average over whatever levels are finite per pixel, not blindly over all five.
    A pixel with every level NaN (land, or a native cell with no valid data at all)
    produces NaN and triggers numpy's "Mean of empty slice" RuntimeWarning; that
    warning is expected here (land is handled downstream by the wet-mask regrid and
    the explicit land=NaN step, not by this reduction) and is suppressed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=1)


def monthly_climatology(frames_by_year: list[NDArray[np.floating]]) -> NDArray[np.floating]:
    """Nan-aware mean of per-year (12, y, x) frames across years -> (12, y, x).

    Nan-aware for the same reason as ``layer0_from_thetao``: a native pixel can be
    valid in some years and not others (e.g. QC differences), and a genuinely-dry
    pixel is NaN in every year, which nanmean correctly propagates to NaN rather than
    treating as a 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(frames_by_year, axis=0), axis=0)


def duplicate_months(clim12: NDArray[np.floating]) -> NDArray[np.floating]:
    """Duplicate each of 12 monthly frames x2 -> 24 (frames 2m, 2m+1 = month m).

    Matches ``baltic_oxygen_bottom.nc``'s convention, NOT
    ``osmose.forcing.grid.resample_to_24``'s linear interpolation: ``PhysicalData``
    indexes ``step % frame_count`` over the 24-step simulation year, so 12 frames
    would silently misalign the month-to-step mapping from step 13 onward, and
    duplicating (vs. interpolating) preserves the raw monthly extrema exactly.
    """
    return np.repeat(clim12, 2, axis=0)


def bottom_depth_from_so(
    so_tzyx: NDArray[np.floating], depth: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Deepest finite level's depth (y, x) from one full-depth ``so`` frame; NaN on land.

    Uses only the first time frame (``so_tzyx[0]``) -- bathymetry is treated as static
    for this purpose, mirroring ``scripts/build_baltic_salinity_forcing.py``'s
    ``bottom_extract``, which does the equivalent per-frame (this only needs one).
    """
    finite = np.isfinite(so_tzyx[0])  # (ndepth, y, x)
    ndepth = so_tzyx.shape[1]
    # First finite level found scanning from the deepest level upward = deepest valid level.
    rev_first = np.argmax(finite[::-1], axis=0)
    idx = (ndepth - 1) - rev_first
    return np.where(finite.any(axis=0), depth[idx], np.nan)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def build(
    thetao_files: list[Path],
    bottomt_files: list[Path],
    so_file: Path,
    grid_nc: Path,
) -> xr.Dataset:
    """Build the two-layer temperature climatology + bottom_depth on the Baltic grid.

    Streams year-files one at a time (each ~130 MB for thetao) rather than loading all
    29+29 into memory at once. Same GridSpec/target_coords/ocean_mask calls as
    ``make_baltic_oxygen_forcing.build`` (copied verbatim per the task brief) so the
    two forcing files agree on grid geometry byte-for-byte.
    """
    from osmose.engine.grid import Grid
    from osmose.forcing.grid import get_coords, target_coords
    from osmose.maps.builder import GridSpec

    grid = GridSpec(
        nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30
    )
    tlat, tlon = target_coords(grid)

    engine_grid = Grid.from_netcdf(Path(grid_nc))
    wet = engine_grid.ocean_mask  # (40, 50) bool, True = ocean
    if wet.shape != (len(tlat), len(tlon)):
        raise ValueError(f"grid mismatch: mask {wet.shape} vs target ({len(tlat)}, {len(tlon)})")

    surface_frames: list[NDArray[np.floating]] = []
    bottom_frames: list[NDArray[np.floating]] = []
    src_lat: NDArray[np.floating] | None = None
    src_lon: NDArray[np.floating] | None = None

    for f in sorted(thetao_files):
        ds = xr.open_dataset(f)
        try:
            raw = ds["thetao"].values  # (12, 5, y, x)
            if raw.shape[0] != 12:
                raise ValueError(f"expected 12 monthly frames, got {raw.shape[0]} in {f}")
            surface_frames.append(layer0_from_thetao(raw).astype(np.float32))
            if src_lat is None:
                src_lat, src_lon = get_coords(ds)
        finally:
            ds.close()

    for f in sorted(bottomt_files):
        ds = xr.open_dataset(f)
        try:
            raw = ds["bottomT"].values  # (12, y, x)
            if raw.shape[0] != 12:
                raise ValueError(f"expected 12 monthly frames, got {raw.shape[0]} in {f}")
            bottom_frames.append(raw.astype(np.float32))
        finally:
            ds.close()

    if not surface_frames:
        raise FileNotFoundError("no thetao year-files given")
    if len(surface_frames) != len(bottom_frames):
        raise ValueError(
            f"thetao year-file count ({len(surface_frames)}) != bottomT year-file count "
            f"({len(bottom_frames)}); the two products must cover the same years."
        )
    assert src_lat is not None and src_lon is not None

    clim_surface = monthly_climatology(surface_frames)  # (12, y, x)
    clim_bottom = monthly_climatology(bottom_frames)  # (12, y, x)

    surface_regridded = masked_regrid(clim_surface, src_lat, src_lon, tlat, tlon, wet)
    bottom_regridded = masked_regrid(clim_bottom, src_lat, src_lon, tlat, tlon, wet)
    surface_regridded[:, ~wet] = np.nan
    bottom_regridded[:, ~wet] = np.nan

    surface24 = duplicate_months(surface_regridded)  # (24, y, x)
    bottom24 = duplicate_months(bottom_regridded)  # (24, y, x)
    temperature = np.stack([surface24, bottom24], axis=1).astype(np.float32)  # (24, 2, y, x)

    # bottom_depth: deepest-finite level per native pixel from the one full-depth so
    # year-file, regridded the same masked way, land NaN.
    so_ds = xr.open_dataset(Path(so_file))
    try:
        so_raw = so_ds["so"].values  # (12, ndepth, y, x)
        so_depth = so_ds["depth"].values
        so_src_lat, so_src_lon = get_coords(so_ds)
    finally:
        so_ds.close()
    bd_native = bottom_depth_from_so(so_raw, so_depth)  # (y, x)
    bd_regridded = masked_regrid(
        bd_native[np.newaxis, :, :], so_src_lat, so_src_lon, tlat, tlon, wet
    )[0].copy()  # (40, 50)
    bd_regridded[~wet] = np.nan

    commit = _git_commit()
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    da_temp = xr.DataArray(
        temperature,
        dims=["time", "layer", "latitude", "longitude"],
        coords={"time": np.arange(24), "layer": [0, 1], "latitude": tlat, "longitude": tlon},
        attrs={"units": "degC", "long_name": "Sea water temperature"},
    )
    da_bd = xr.DataArray(
        bd_regridded,
        dims=["latitude", "longitude"],
        coords={"latitude": tlat, "longitude": tlon},
        attrs={"units": "m", "long_name": "Water column depth (deepest finite CMEMS so level)"},
    )
    history = (
        f"Generated by scripts/build_baltic_temperature_forcing.py on {generated} from CMEMS "
        f"product {PRODUCT_ID}, {SOURCE_YEARS} monthly means. thetao year-files: "
        f"{len(surface_frames)}; bottomT year-files: {len(bottom_frames)}; bathymetry from "
        f"{Path(so_file).name}. Regrid: nearest-neighbour restricted to valid (non-NaN) "
        "native pixels per wet model cell (osmose.engine.grid.Grid mask), same as "
        "make_baltic_oxygen_forcing._masked_regrid."
    )
    return xr.Dataset(
        {"temperature": da_temp, "bottom_depth": da_bd},
        attrs={
            "title": "OSMOSE Baltic two-layer temperature climatology (CMEMS thetao/bottomT)",
            "source": f"CMEMS {PRODUCT_ID}, {SOURCE_YEARS} monthly means",
            "source_years": SOURCE_YEARS,
            "history": history,
            "frame_convention": (
                "month duplicated x2 (frames 2m,2m+1 = month m), matches "
                "baltic_oxygen_bottom.nc; NOT resample_to_24 interpolation"
            ),
            "land": "NaN",
            "layers": "0: surface nan-mean thetao 0.50-4.68 m; 1: CMEMS bottomT",
            "generator": f"scripts/build_baltic_temperature_forcing.py@{commit}",
            "conventions": "Latitude descending (north to south) to match grid.nc",
        },
    )


def validate(ds: xr.Dataset, wet: NDArray[np.bool_]) -> None:
    """Fail fast on any spec 3.3 pin: shape, wet-cell finiteness, physical range, and
    layer order (August bottom temperature must not exceed August surface temperature
    on cells deeper than 40 m -- catches a swapped layer axis)."""
    t = ds["temperature"].values
    assert t.shape[0] == 24 and t.shape[1] == 2, f"shape {t.shape}"
    wet_vals = t[:, :, wet]
    assert np.isfinite(wet_vals).all(), "finite: NaN on a wet cell"
    assert (wet_vals >= -2.0).all() and (wet_vals <= 30.0).all(), (
        "range: wet-cell temperature outside [-2, 30] C"
    )
    deep = wet & (ds["bottom_depth"].values > 40.0)
    aug = t[16:18]
    assert np.all(aug[:, 1][:, deep] <= aug[:, 0][:, deep] + 1e-6), (
        "layer-order: August bottom > surface on deep cells"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the Baltic two-layer temperature climatology NetCDF."
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--so-file", type=Path, default=None, help="defaults to the earliest cached so year-file"
    )
    ap.add_argument("--grid", type=Path, default=DEFAULT_GRID_NC)
    args = ap.parse_args(argv)

    thetao_files = sorted(args.cache.glob("baltic_phy_monthly_reanalysis_thetao_*.nc"))
    bottomt_files = sorted(args.cache.glob("baltic_phy_monthly_reanalysis_bottomT_*.nc"))
    if not thetao_files:
        raise FileNotFoundError(f"no thetao year-files found under {args.cache}")
    if not bottomt_files:
        raise FileNotFoundError(f"no bottomT year-files found under {args.cache}")

    so_file = args.so_file
    if so_file is None:
        so_candidates = sorted(args.cache.glob("baltic_phy_monthly_reanalysis_so_*.nc"))
        if not so_candidates:
            raise FileNotFoundError(f"no so year-files found under {args.cache}")
        so_file = so_candidates[0]

    ds = build(thetao_files, bottomt_files, so_file, args.grid)

    from osmose.engine.grid import Grid

    wet = Grid.from_netcdf(args.grid).ocean_mask
    validate(ds, wet)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out, encoding={"temperature": {"dtype": "float32"}})

    t = ds["temperature"].values
    for m in range(12):
        for layer_idx, label in ((0, "surface"), (1, "bottom")):
            vals = t[2 * m, layer_idx][wet]
            print(
                f"month={m + 1:02d} layer={label:7s} "
                f"min={np.nanmin(vals):.2f} mean={np.nanmean(vals):.2f} "
                f"max={np.nanmax(vals):.2f} C"
            )
    deep_count = int((ds["bottom_depth"].values > 40.0).sum())
    print(f"deep cells (bottom_depth > 40 m): {deep_count}")
    print(f"wrote {args.out} shape={t.shape} dtype={t.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
