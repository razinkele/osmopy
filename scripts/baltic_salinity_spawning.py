"""Refine Baltic spawning-area maps by salinity (SP-B habitat follow-up, first attempt).

Baltic cod recruitment is governed by the "reproductive volume": deep-basin water saline enough
(>= ~11 PSU, for egg buoyancy) and oxygenated enough, fed by North Sea salt-water inflows through the
Danish straits — reliably the SW/S deep basins (Bornholm/Gdansk), marginally the Gotland Deep, absent
farther north (docs/baltic-fish-lifecycle.md:386-406). Perch/pikeperch spawn in FRESHWATER/low-salinity
(< ~7 PSU) coastal & lagoon water — the northern gulfs and eastern inner coast, NOT the saline SW.

The engine has no salinity state variable (a known gap); spawning is a static presence map. This script
makes those static AREAS salinity-correct using a geographic salinity proxy (the well-established Baltic
gradient: saline SW/deep -> fresh N/coastal). A CMEMS `so`-derived per-cell field would refine the
thresholds — see docs follow-up.

Restricts (never expands) the existing footprints:
  - cod_spawning      -> current footprint intersect the saline reproductive volume (lat <= 57.5)
  - perch_spawning    -> current footprint intersect the freshwater zone (lat >= 58 OR lon >= 21.5)
  - pikeperch_spawning-> current footprint intersect the freshwater zone (drops the saline SW cells)

Maps: ';'-separated 40x50, -99=land / 0=ocean-absent / 1=present, stored SOUTH-FIRST (row 39 = north).
Follows scripts/apply_ices_validation_fixes.py: backup, np.flipud for engine orientation, ';' write.

    PYTHONPATH=. .venv/bin/python scripts/baltic_salinity_spawning.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "data" / "baltic" / "maps"
GRID = ROOT / "data" / "baltic" / "baltic_grid.nc"

# Salinity-proxy thresholds (Baltic oceanography; geographic gradient, saline SW -> fresh N)
COD_SALINE_MAX_LAT = 57.5   # reliable deep reproductive volume (Bornholm/Gdansk, S. Gotland)
FRESH_MIN_LAT = 58.0        # northern gulfs (Bothnian, N. Baltic proper) are fresh
FRESH_MIN_LON = 21.5        # eastern gulfs (Riga, Finland) are fresh even at lower latitude


def _grid_latlon() -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(GRID) as g:
        return g["latitude"].values, g["longitude"].values  # lat DESC (row 0 = north)


def _engine_view(storage: np.ndarray) -> np.ndarray:
    """South-first storage -> engine orientation (row 0 = north), matching the DESC grid latitude."""
    return np.flipud(storage)


def _storage_view(engine: np.ndarray) -> np.ndarray:
    return np.flipud(engine)


def refine(name: str, keep_mask_engine: np.ndarray) -> tuple[int, int]:
    """Intersect a spawning map's present(1) cells with keep_mask_engine; land/absent untouched."""
    src = MAPS / f"{name}.csv"
    shutil.copy2(src, src.with_suffix(".csv.pre-salinity.bak"))
    storage = np.genfromtxt(src, delimiter=";")
    eng = _engine_view(storage)
    before = int((eng == 1).sum())
    present = eng == 1
    drop = present & ~keep_mask_engine
    eng = eng.copy()
    eng[drop] = 0.0  # demote out-of-zone spawning cells to ocean-absent (never touch land -99)
    after = int((eng == 1).sum())
    np.savetxt(src, _storage_view(eng), delimiter=";", fmt="%g")
    return before, after


def main() -> int:
    lat, lon = _grid_latlon()  # (40,), (50,)
    LAT = lat[:, None] * np.ones((1, len(lon)))  # (40,50) engine orientation
    LON = np.ones((len(lat), 1)) * lon[None, :]

    cod_saline = LAT <= COD_SALINE_MAX_LAT
    fresh = (LAT >= FRESH_MIN_LAT) | (LON >= FRESH_MIN_LON)

    print("=== Salinity-refined spawning areas ===")
    for name, keep in (("cod_spawning", cod_saline),
                       ("perch_spawning", fresh),
                       ("pikeperch_spawning", fresh)):
        b, a = refine(name, keep)
        print(f"  {name:20s} {b:3d} -> {a:3d} cells ({b - a} dropped out-of-zone)")
    print("\nBackups written as *.csv.pre-salinity.bak. data/baltic maps refined in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
