"""Apply Baltic fix derived from ICES BITS validation.

Two changes, both traceable to ICES DATRAS survey hauls:

(A) Open the mask at 4 cells where BITS hauls consistently land but the mask
    marks them as land. Updates baltic_grid.nc + baltic_mask.csv + all 26
    species/fishing CSVs (land -99 → ocean 0 at the 4 cells).

(B) Extend cod distribution maps to the 15 cells where BITS recorded positive
    cod CPUE but cod_*.csv had zero. Adult and juvenile maps get all 15 cells;
    spawning map gets only the 4 cells inside the Eastern Baltic spawning
    zone (lon ≥ 18°E), to avoid merging Kattegat/Øresund stocks into the
    Eastern Baltic cod spawning footprint.

Run from repo root:
    .venv/bin/python scripts/apply_ices_validation_fixes.py
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
BACKUP_SUFFIX = ".pre-ices-fix.bak"

# (row, col) in engine orientation (row 0 = north, row 39 = south)
NEW_OCEAN_CELLS = [(35, 0), (35, 1), (35, 5), (37, 20)]

COD_BITS_ONLY = [
    (26, 27),  # Eastern Baltic SE edge, lat 58.05 lon 21.00 — spawning-zone adjacent
    (27, 2),   # Kattegat
    (28, 1),   # Kattegat
    (28, 2),   # Kattegat
    (28, 3),   # Kattegat
    (29, 3),   # Kattegat
    (32, 6),   # Arkona/Skagerrak
    (33, 1),   # Kattegat
    (33, 6),   # Arkona NW
    (35, 0),   # Øresund west (mask-land, also in NEW_OCEAN_CELLS)
    (35, 1),   # Øresund central (mask-land)
    (35, 5),   # Arkona NW (mask-land)
    (37, 20),  # Gdańsk Bay (mask-land)
    (38, 22),  # Gdańsk Bay, deeper — Gdańsk Deep vicinity
    (38, 23),  # Gdańsk Bay — Gdańsk Deep vicinity
]

# Spawning extension limited to Eastern Baltic (lon ≥ 18°E ≈ col index 20)
COD_SPAWNING_ADD = [(r, c) for (r, c) in COD_BITS_ONLY if c >= 20]


def _backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + BACKUP_SUFFIX)
    if not bak.exists():
        shutil.copy2(p, bak)


def _engine_view(csv: Path) -> np.ndarray:
    arr = pd.read_csv(csv, sep=";", header=None).values.astype(float)
    return np.flipud(arr)


def _storage_view(arr: np.ndarray) -> np.ndarray:
    return np.flipud(arr)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    _backup(path)
    is_int = np.all(arr == arr.astype(int))
    df = pd.DataFrame(arr.astype(int) if is_int else arr)
    df.to_csv(path, sep=";", header=False, index=False)


def open_mask_and_csvs() -> None:
    """Step A: convert the 4 cells from land to ocean in every spatial file."""
    storage_rows = [(39 - r, c) for (r, c) in NEW_OCEAN_CELLS]

    # NC grid
    ds = xr.open_dataset(GRID_NC)
    try:
        mask_eng = ds.mask.values.astype(np.int32)
        lat = ds.latitude.values.copy()
        lon = ds.longitude.values.copy()
        attrs = dict(ds.attrs)
    finally:
        ds.close()
    for r, c in NEW_OCEAN_CELLS:
        mask_eng[r, c] = 1
    _backup(GRID_NC)
    new_ds = xr.Dataset(
        data_vars={"mask": (("latitude", "longitude"), mask_eng)},
        coords={"latitude": lat, "longitude": lon},
        attrs={
            **attrs,
            "history": (
                attrs.get("history", "")
                + " | 2026-04-17: added 4 cells where ICES BITS surveys land"
                " hauls but post-rebuild mask had marked them as land"
                " (Öresund, Arkona NW, Gdańsk Bay)."
            ).strip(" |"),
        },
    )
    new_ds["mask"].encoding = {"dtype": "int32"}
    new_ds["latitude"].encoding = {"dtype": "float64"}
    new_ds["longitude"].encoding = {"dtype": "float64"}
    new_ds.to_netcdf(GRID_NC)
    print(f"  wrote {GRID_NC}  (ocean cells: {int((mask_eng>0).sum())})")

    # CSV mask (stored south-first — flip row indices)
    mask_csv = pd.read_csv(MASK_CSV, sep=";", header=None).values.astype(float)
    for r_st, c in storage_rows:
        mask_csv[r_st, c] = 0  # 0 = ocean
    _write_csv(MASK_CSV, mask_csv)
    print(f"  wrote {MASK_CSV}")

    # All spatial CSVs: -99 → 0 at the 4 cells (ocean with no effort/no presence).
    changed = 0
    for d in (MAPS_DIR, FISHING_DIR):
        for f in sorted(d.glob("*.csv")):
            arr = pd.read_csv(f, sep=";", header=None).values.astype(float)
            for r_st, c in storage_rows:
                if arr[r_st, c] == -99:
                    arr[r_st, c] = 0
                    changed += 1
            _write_csv(f, arr)
    print(f"  opened {changed} CSV cells (-99 → 0)")


def extend_cod_maps() -> None:
    """Step B: set cod distribution maps to 1 at BITS-only cells."""
    actions = [
        ("cod_adult.csv", COD_BITS_ONLY, "all 15 BITS-only cells"),
        ("cod_juvenile.csv", COD_BITS_ONLY, "all 15 BITS-only cells"),
        ("cod_spawning.csv", COD_SPAWNING_ADD, "4 Eastern-Baltic cells only"),
    ]
    for name, cells, label in actions:
        path = MAPS_DIR / name
        arr = pd.read_csv(path, sep=";", header=None).values.astype(float)
        storage_rows = [(39 - r, c) for (r, c) in cells]
        changed = 0
        for r_st, c in storage_rows:
            if arr[r_st, c] != 1:
                arr[r_st, c] = 1
                changed += 1
        _write_csv(path, arr)
        print(f"  {name}: set {changed} cells to 1 ({label})")


def extend_other_species_maps() -> None:
    """Step C: extend herring/sprat/flounder adult+juvenile to the same 15
    BITS-documented cells. Spawning maps are left alone because they reflect
    species-specific spawning biology that BITS presence does not imply."""
    targets = [
        "herring_adult.csv", "herring_juvenile.csv",
        "sprat_adult.csv", "sprat_juvenile.csv",
        "flounder_adult.csv", "flounder_juvenile.csv",
    ]
    storage_rows = [(39 - r, c) for (r, c) in COD_BITS_ONLY]
    for name in targets:
        path = MAPS_DIR / name
        arr = pd.read_csv(path, sep=";", header=None).values.astype(float)
        changed = 0
        for r_st, c in storage_rows:
            if arr[r_st, c] != 1:
                arr[r_st, c] = 1
                changed += 1
        _write_csv(path, arr)
        print(f"  {name}: set {changed} cells to 1 (15 BITS-only cells)")


def main() -> None:
    print("(A) Opening mask at 4 BITS-documented cells…")
    open_mask_and_csvs()
    print("\n(B) Extending cod distribution maps…")
    extend_cod_maps()
    print("\n(C) Extending herring/sprat/flounder adult+juvenile maps…")
    extend_other_species_maps()
    print(f"\nBackups created with suffix {BACKUP_SUFFIX}")


if __name__ == "__main__":
    main()
