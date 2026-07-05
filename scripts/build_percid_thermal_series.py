"""Build the percid thermal sidecar from CMEMS thetao (surface).

Reuses the salinity pipeline (scripts/build_baltic_salinity_forcing.py): thetao
lives in the SAME product as so (cmems_mod_bal_phy_my_P1M-m), so download it with
scripts/download_baltic_rv_forcing.py --vars thetao --depth-min 0 --depth-max 5,
producing data/cmems_cache/cmems_downloads/baltic_phy_monthly_reanalysis_thetao_*.nc.

Per-species summer window: perch (sp4) = (6, 7); pikeperch (sp5) = (7, 8).
Habitat mask per species = cells with nonzero occupancy in its movement maps.
Fails loudly if thetao files are absent (Risk R1) — never substitutes a field.

──────────────────────────────────────────────────────────────────────────────
VERIFY-BEFORE-FIRST-REAL-RUN (review findings 3 & 5 — this driver is NOT unit
tested; only the pure core osmose/forcing/percid_thermal.summer_sst_by_year is):

  1. FILE GRANULARITY (finding 5): _thetao_surface_tyx assumes ONE file per year
     with EXACTLY 12 monthly slices (year parsed from the filename, month = slice
     index + 1). Confirm the actual thetao download matches this against the
     cached `so` files (`ncdump -h` one file): if a file spans multiple years or
     != 12 months, the (year, month) tags are WRONG. Fix the tagging before use.
  2. ORIENTATION: _load_spatial_csv np.flipud's the movement-map CSV, whereas
     regrid() output follows target_coords orientation. Confirm the habitat mask
     and the regridded temp_tyx share row orientation (one-cell sanity check: a
     known warm coastal cell should read a plausibly warm value) — mirror how
     build_baltic_salinity_forcing.py reconciles mask vs regridded field.
  3. SST DISTRIBUTION (finding 3): print the resulting per-year index range. If
     the regridded coastal index runs cool, thermal_cap (t50=18.5/tref=20) bites
     hard and can drive already-fragile perch to ~0 — sanity-check before the A/B.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from osmose.config import OsmoseConfigReader
from osmose.engine.config import _load_spatial_csv
from osmose.forcing.grid import get_coords, regrid
from osmose.forcing.percid_thermal import summer_sst_by_year
from osmose.maps.builder import GridSpec

SPECIES = {4: (6, 7), 5: (7, 8)}  # species index -> summer months
ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "data" / "cmems_cache" / "cmems_downloads"
OUT = ROOT / "data" / "baltic" / "forcing" / "baltic_percid_thermal_series.csv"


def _habitat_mask(cfg: dict[str, str], sp: int, cfg_dir: Path) -> np.ndarray:
    """Union of cells with nonzero occupancy across species sp's movement maps.

    Movement-map keys (data/baltic/baltic_param-movement.csv) are
    `movement.species.map{n}` (value = species NAME) and `movement.file.map{n}`,
    so resolve sp -> name first.
    """
    sp_name = cfg[f"species.name.sp{sp}"].strip()
    mask = None
    n = 0
    while True:
        name = cfg.get(f"movement.species.map{n}", None)
        map_key = cfg.get(f"movement.file.map{n}", "")
        if map_key == "":
            break
        if name is not None and name.strip() == sp_name:
            arr = _load_spatial_csv(cfg_dir / map_key)  # NB: _load_spatial_csv np.flipud's the CSV
            m = arr > 0
            mask = m if mask is None else (mask | m)
        n += 1
    if mask is None:
        raise ValueError(f"no movement map found for {sp_name} (sp{sp}); check movement.species.map* keys")
    return mask


def _thetao_surface_tyx(grid: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (temp_tyx on Baltic grid, times_year, times_month) from surface thetao.

    See the VERIFY-BEFORE-FIRST-REAL-RUN note (finding 5): assumes one file/year,
    12 monthly slices.
    """
    files = sorted(DL.glob("baltic_phy_monthly_reanalysis_thetao_*.nc"))
    if not files:
        raise FileNotFoundError(
            f"no thetao files under {DL}. Download with "
            "scripts/download_baltic_rv_forcing.py --vars thetao --depth-min 0 --depth-max 5 "
            "(needs CMEMS credentials — Risk R1)."
        )
    slices, yrs, mons = [], [], []
    for f in files:
        ds = xr.open_dataset(f)
        theta = ds["thetao"].values  # (12, nlev, nlat, nlon) or (12, nlat, nlon)
        surf = theta[:, 0] if theta.ndim == 4 else theta  # surface level
        src_lat, src_lon = get_coords(ds)
        if surf.shape[0] != 12:
            raise ValueError(
                f"{f.name}: expected 12 monthly slices, got {surf.shape[0]} — the "
                "one-file-per-year assumption is violated (review finding 5); fix the tagging."
            )
        for m in range(surf.shape[0]):
            slices.append(regrid(surf[m][None], src_lat, src_lon, grid)[0])  # (ny, nx)
            yrs.append(int(str(f.stem).split("_")[-1][:4]))
            mons.append(m + 1)
        ds.close()
    return np.stack(slices), np.array(yrs), np.array(mons)


def main() -> int:
    cfg_path = sorted((ROOT / "data" / "baltic").glob("*all-parameters*.csv"))[0]
    cfg = OsmoseConfigReader().read(str(cfg_path))
    cfg_dir = cfg_path.parent
    grid = GridSpec.from_config(cfg)
    temp_tyx, ty, tm = _thetao_surface_tyx(grid)

    per_sp = {}
    for sp, months in SPECIES.items():
        mask = _habitat_mask(cfg, sp, cfg_dir)
        years, means = summer_sst_by_year(temp_tyx, ty, tm, mask, months)
        per_sp[sp] = pd.Series(means, index=years)

    common = sorted(set.intersection(*[set(s.index) for s in per_sp.values()]))
    if list(common) != list(range(common[0], common[0] + len(common))):
        raise ValueError(f"thermal series years not contiguous: {common}")
    df = pd.DataFrame({"year": common})
    for sp in SPECIES:
        df[f"temp_sp{sp}"] = [float(per_sp[sp][y]) for y in common]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} years, cols {list(df.columns)})")
    print("per-year index range:",
          {f"sp{sp}": (round(df[f'temp_sp{sp}'].min(), 2), round(df[f'temp_sp{sp}'].max(), 2))
           for sp in SPECIES})  # finding 3 sanity check
    return 0


if __name__ == "__main__":
    sys.exit(main())
