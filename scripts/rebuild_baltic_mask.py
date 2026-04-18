"""Rebuild baltic_grid.nc mask + CSV land pattern from fish-distribution data.

The mask previously had 912 ocean cells, but 300 of them were "ocean-with-zero"
in every single movement/fishing CSV — no species ever used them, and they
visually corresponded to northern Norway/Sweden mountains (lon 10-15°E at
62-66°N) that were mirrored from the southern Baltic pattern.

Fix: shrink the mask to the union of cells where at least one species or
fishing map has a positive value (612 cells), and rewrite all CSVs so the
300 removed cells become -99 (land) instead of 0 (ocean-with-zero).

Run from repo root:
    .venv/bin/python scripts/rebuild_baltic_mask.py
"""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = Path("data/baltic")
GRID_NC = DATA_DIR / "baltic_grid.nc"
MASK_CSV = DATA_DIR / "grid" / "baltic_mask.csv"
MAPS_DIR = DATA_DIR / "maps"
FISHING_DIR = DATA_DIR / "fishing"
BACKUP_SUFFIX = ".pre-mask-rebuild.bak"

# Config bbox
UL_LAT = 66.0
UL_LON = 10.0
LR_LAT = 54.0
LR_LON = 30.0


def _engine_view(csv_path: Path) -> np.ndarray:
    """Load CSV in engine orientation (row 0 = north)."""
    arr = pd.read_csv(csv_path, sep=";", header=None).values
    return np.flipud(arr)


def _storage_view(engine_arr: np.ndarray) -> np.ndarray:
    """Flip engine orientation back to CSV storage orientation (row 0 = south)."""
    return np.flipud(engine_arr)


def compute_new_mask() -> np.ndarray:
    """Union of cells with value > 0 across every CSV (engine orientation)."""
    mask = np.zeros((40, 50), dtype=bool)
    for d in (MAPS_DIR, FISHING_DIR):
        for f in sorted(d.glob("*.csv")):
            mask |= _engine_view(f) > 0
    return mask


def _backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + BACKUP_SUFFIX)
    if not bak.exists():
        shutil.copy2(p, bak)


def rewrite_csv_with_new_land(path: Path, new_ocean_storage: np.ndarray) -> int:
    """Set value = -99 for any cell outside new_ocean_storage (CSV orientation)."""
    raw = pd.read_csv(path, sep=";", header=None).values.astype(float)
    new_land = ~new_ocean_storage
    changed = int(((raw != -99) & new_land).sum())
    raw[new_land] = -99
    _backup(path)
    pd.DataFrame(raw.astype(int) if np.all(raw == raw.astype(int)) else raw).to_csv(
        path, sep=";", header=False, index=False
    )
    return changed


def rewrite_mask_csv(new_ocean_storage: np.ndarray) -> None:
    """Rewrite grid/baltic_mask.csv: 0 for ocean, -99 for land."""
    arr = np.where(new_ocean_storage, 0, -99).astype(int)
    _backup(MASK_CSV)
    pd.DataFrame(arr).to_csv(MASK_CSV, sep=";", header=False, index=False)


def rewrite_nc_mask(new_ocean_engine: np.ndarray) -> None:
    """Rewrite baltic_grid.nc mask variable (keeps current lat/lon labels)."""
    ds = xr.open_dataset(GRID_NC)
    try:
        lat = ds.latitude.values.copy()
        lon = ds.longitude.values.copy()
        attrs = dict(ds.attrs)
    finally:
        ds.close()

    mask_int = new_ocean_engine.astype(np.int32)
    new_ds = xr.Dataset(
        data_vars={"mask": (("latitude", "longitude"), mask_int)},
        coords={"latitude": lat, "longitude": lon},
        attrs={
            **attrs,
            "history": (
                attrs.get("history", "")
                + " | 2026-04-17: mask rebuilt from union of 26 movement/fishing"
                " CSVs (active cells > 0), removing 300 unused cells that"
                " corresponded to non-Baltic locations."
            ).strip(" |"),
        },
    )
    new_ds["mask"].encoding = {"dtype": "int32"}
    new_ds["latitude"].encoding = {"dtype": "float64"}
    new_ds["longitude"].encoding = {"dtype": "float64"}
    _backup(GRID_NC)
    new_ds.to_netcdf(GRID_NC)


def main() -> None:
    new_ocean_eng = compute_new_mask()
    new_ocean_storage = _storage_view(new_ocean_eng)
    print(f"New mask ocean cells: {int(new_ocean_eng.sum())}")

    rewrite_nc_mask(new_ocean_eng)
    print(f"  wrote {GRID_NC}")

    rewrite_mask_csv(new_ocean_storage)
    print(f"  wrote {MASK_CSV}")

    total_changed = 0
    for d in (MAPS_DIR, FISHING_DIR):
        for f in sorted(d.glob("*.csv")):
            changed = rewrite_csv_with_new_land(f, new_ocean_storage)
            total_changed += changed
            print(f"  wrote {f}  ({changed} cells set to -99)")

    print(f"\nTotal cells changed 0→-99 across CSVs: {total_changed}")
    print(f"Backups created with suffix {BACKUP_SUFFIX}")


if __name__ == "__main__":
    main()
