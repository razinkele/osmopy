# Baltic Finer-Grid Percid-Habitat Deciding Experiment — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an alternative 4× Baltic config whose perch/pikeperch habitat is a real EMODnet-bathymetry-derived **thin littoral set** (cells whose shallow-fraction ≥ τ) with a spawning-stage salinity gate, and a 3-rung attribution ladder (coarse → 4×-upsampled → 4×-real) that decides whether resolving real coastal percid habitat cures the structural percid overshoot.

**Architecture:** Pure, independently-tested builders (grid+bathymetry, percid habitat, occupancy upsampler, conserving forcing regridder, ladder runner) write a new `data/baltic-fine/` config tree; `data/baltic/` is never touched. Each builder has a pure core (unit-tested on synthetic arrays) and a thin IO script. The ladder assembles TWO config variants that differ ONLY in which movement sub-config they include.

**Tech Stack:** Python 3.12, NumPy, xarray/netCDF4, pandas, **rasterio** (NEW — reads EMODnet GeoTIFF; GDAL-based, offline-builder-only), requests (EMODnet WCS), pytest, ruff. Run with `/home/razinka/osmopy/.venv/bin/python`.

> **v2 changelog (folded in from the adversarial plan-review workflow — 3 critical, 14 important):** (C1) variants now differ by including two distinct **movement sub-files**, not by inline entrypoint overrides (the reader's `flat.update` makes an included sub-file WIN over the master, so inline overrides are silently clobbered → rung2==rung3). (C2) the **fishing distribution maps** (`data/baltic/fishing/*.csv`) + the **grid mask** are regridded too (else 40×50 vs 200×160 shape crash). (C3) real habitat is a **binary threshold** on shallow-fraction, not a fractional weight (the engine's `max_proba`/`>0` map semantics make a graded weight's effect on carrying capacity ambiguous; a binary set unambiguously shrinks habitat AREA). (I) `fill_ocean_nan` + NaN guard on the fine salinity climatology (else NaN cells bypass the spawning gate → false-NO-GO bias); vacuity guard measures AREA against the **upsampled percid footprint** (not whole ocean); GO verdict requires the overshoot drop to **exceed the area cut**; runtime **shape guard** on every fine forcing; correct target loader (`load_targets`, not `load_biomass_targets`); preserve master-only keys. **REJECTED false positive:** `.biomass()` returns a WIDE frame with species columns — `bio["perch"]` is correct (verified empirically); do NOT "fix" it to long-form.

## Global Constraints

- Repo root: `/home/razinka/osmopy`. **Work on a NEW branch/worktree `feat/baltic-fine-grid-percid`, never on master.**
- Run tests: `PYTHONPATH=. .venv/bin/python -m pytest`. Lint: `.venv/bin/ruff check` AND `.venv/bin/ruff format --check`.
- `data/baltic/` is READ-ONLY. All outputs under `data/baltic-fine/`.
- **Grid:** `GridSpec` (`osmose/maps/builder.py`) fields `nlon,nlat,upleft_lat,upleft_lon,lowright_lat,lowright_lon`; `.from_config(cfg)`. Fine = `nlon=200,nlat=160` (4×), same extent (upleft 66N/10E, lowright 54N/30E).
- **Spatial CSV:** semicolon-sep, **rows south→north on disk** (`_load_spatial_csv` in `osmose/engine/config.py` does `np.flipud`); internal `(nlat,nlon)` north→south. Land `-99`, ocean-non-habitat `0`, habitat positive. CSV = `nlat` rows × `nlon` cols.
- **Config precedence (verified `osmose/config/reader.py:_read_recursive`):** the master's keys are read, then each `osmose.configuration.*` include recurses and `flat.update()`s — **the included sub-file OVERWRITES the master.** Therefore variants MUST differ by pointing to different sub-files, never by inline master overrides.
- **Forcing aligns by CELL INDEX** — matching shapes suffice; all fine forcing arrays `(…,160,200)`. Every fine forcing must be shape-asserted `(…,160,200)` at build AND load.
- **CRITICAL:** absolute-biomass forcings (`baltic_ltl_biomass.nc` 6 vars; `baltic_predator_biomass.nc` GreySeal/Cormorant; `(24,40,50)` tonnes/cell) MUST be **conserved** on regrid (÷ factor²), never replicated. Intensive fields (salinity) regrid as-is AND get `fill_ocean_nan` + a no-ocean-NaN assert (mirroring `scripts/build_baltic_salinity_forcing.py:build`).
- **EMODnet (spike-confirmed):** WCS `GetCoverage`, `coverageId=emodnet__mean`, `version=2.0.1`, `subset=Lat(a,b)&subset=Long(c,d)`, `format=image/tiff` → GeoTIFF via rasterio; EPSG:4326, ~115 m, 32-bit; **negative = below sea level** (`depth = -elev`, land `elev>=0`). Fetch **tiled** in lat strips AND lon blocks (a full 10–30°E × strip can exceed WCS pixel limits).
- Percid map stages (`data/baltic/baltic_param-movement.csv`): perch `maps/perch_{juvenile,adult,spawning}.csv` (map13/14/15), pikeperch `maps/pikeperch_{...}` (map16/17/18).
- Overshoot: `load_targets(Path)` in `osmose/calibration/targets.py` returns `(list[BiomassTarget], dict)`; build `{t.species: t.target for t in list}`. Overshoot ratio = late-window-mean biomass / target.
- **`.biomass()` returns a WIDE frame** (columns `Time,cod,herring,...,perch,pikeperch,...`); index as `bio["perch"]`.
- Ladder: 3 rungs × **≥5 seeds × nyear=30**.

---

### Task 1: Sub-grid shallow-fraction pure core + EMODnet tiled fetch

**Files:** Create `osmose/forcing/bathymetry.py`, `scripts/build_baltic_fine_grid.py`; Test `tests/test_bathymetry.py`.

**Interfaces:**
- Produces: `shallow_fraction(elev_hi, lat_hi, lon_hi, grid, depth_max_m) -> (frac (nlat,nlon), ocean (nlat,nlon) bool)` — `frac[r,c]` = fraction of that cell's WET hi-res sub-pixels with `0 < depth <= depth_max_m` (`depth=-elev`); `ocean` = any wet sub-pixel.
- Produces (script): `build_shallow_fraction(depth_max_m) -> (frac, ocean)` (whole extent, tiled), `FINE` GridSpec, `OUT` path; writes `data/baltic-fine/baltic_fine_grid.nc` + `grid/baltic_fine_mask.csv`.

- [ ] **Step 1: Write the failing test** (same as before — the pure core is unchanged and correct):

```python
# tests/test_bathymetry.py
import numpy as np
from osmose.forcing.bathymetry import shallow_fraction
from osmose.maps.builder import GridSpec


def _grid():
    return GridSpec(nlon=1, nlat=1, upleft_lat=54.2, upleft_lon=10.0, lowright_lat=54.0, lowright_lon=10.2)


def test_shallow_fraction_and_ocean():
    lat_hi = np.array([54.15, 54.05]); lon_hi = np.array([10.05, 10.15])
    elev = np.array([[-5.0, -50.0], [3.0, -10.0]])   # (lat, lon)
    frac, ocean = shallow_fraction(elev, lat_hi, lon_hi, _grid(), depth_max_m=15.0)
    assert ocean[0, 0]
    assert frac[0, 0] == np.float64(2) / 3           # wet: -5,-50,-10; <=15m: -5,-10


def test_all_land_is_not_ocean_and_zero_fraction():
    frac, ocean = shallow_fraction(np.array([[5.0]]), np.array([54.1]), np.array([10.1]), _grid(), 15.0)
    assert not ocean[0, 0] and frac[0, 0] == 0.0
```

- [ ] **Step 2: Run to verify fail** — `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bathymetry.py -q` → `ModuleNotFoundError`.

- [ ] **Step 3: Write the pure core** (`osmose/forcing/bathymetry.py`):

```python
"""Sub-grid bathymetry statistics for the finer-grid percid experiment. Pure,
grid/data-source-agnostic. shallow_fraction bins high-res EMODnet elevation
(negative=below sea level) into the target grid and returns per cell the fraction
of WET sub-pixels shallower than depth_max_m."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.maps.builder import GridSpec


def shallow_fraction(elev_hi, lat_hi, lon_hi, grid: GridSpec, depth_max_m: float):
    lat_edges = np.linspace(grid.upleft_lat, grid.lowright_lat, grid.nlat + 1)  # descending
    lon_edges = np.linspace(grid.upleft_lon, grid.lowright_lon, grid.nlon + 1)  # ascending
    row = np.clip(np.searchsorted(-lat_edges, -np.asarray(lat_hi), side="right") - 1, 0, grid.nlat - 1)
    col = np.clip(np.searchsorted(lon_edges, np.asarray(lon_hi), side="right") - 1, 0, grid.nlon - 1)
    depth = -np.asarray(elev_hi, dtype=np.float64)
    wet = depth > 0.0
    shallow = wet & (depth <= depth_max_m)
    n_wet = np.zeros((grid.nlat, grid.nlon)); n_sh = np.zeros((grid.nlat, grid.nlon))
    R = np.broadcast_to(row[:, None], depth.shape)
    C = np.broadcast_to(col[None, :], depth.shape)
    np.add.at(n_wet, (R, C), wet.astype(float))
    np.add.at(n_sh, (R, C), shallow.astype(float))
    ocean = n_wet > 0
    frac = np.zeros_like(n_wet)
    np.divide(n_sh, n_wet, out=frac, where=ocean)
    return frac, ocean
```

- [ ] **Step 4: Run to verify pass** — PASS (2 passed).

- [ ] **Step 5: Write the EMODnet tiled-fetch + grid script** (`scripts/build_baltic_fine_grid.py`): fetch EMODnet in **lat strips × lon blocks** (both tiled — a full 20°-wide strip risks WCS pixel caps), accumulate ocean; write mask + grid.nc. Includes an **orientation sanity check** (spec §8): a known shallow bay (e.g. Curonian Lagoon ~55.3N/21.1E) must land in an ocean cell.

```python
# scripts/build_baltic_fine_grid.py
from __future__ import annotations
import io
from pathlib import Path
import numpy as np, rasterio, requests, xarray as xr
from osmose.forcing.bathymetry import shallow_fraction
from osmose.forcing.grid import target_coords
from osmose.maps.builder import GridSpec

WCS = "https://ows.emodnet-bathymetry.eu/wcs"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "baltic-fine"
FINE = GridSpec(nlon=200, nlat=160, upleft_lat=66.0, upleft_lon=10.0, lowright_lat=54.0, lowright_lon=30.0)


def fetch_emodnet(lat0, lat1, lon0, lon1):
    """Return (elev, lat_hi desc, lon_hi asc) for a bbox (lat0<lat1, lon0<lon1)."""
    params = {"service": "WCS", "version": "2.0.1", "request": "GetCoverage",
              "coverageId": "emodnet__mean", "subset": [f"Lat({lat0},{lat1})", f"Long({lon0},{lon1})"],
              "format": "image/tiff"}
    r = requests.get(WCS, params=params, timeout=180); r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as ds:
        elev = ds.read(1).astype(np.float64); b = ds.bounds
        return elev, np.linspace(b.top, b.bottom, ds.height), np.linspace(b.left, b.right, ds.width)


def build_shallow_fraction(depth_max_m: float):
    """Per-4x-cell shallow-fraction + ocean over the whole extent. Tiled in
    row-disjoint lat strips x lon blocks; per-tile cells are disjoint, so ratios
    are exact per tile — assign directly, no cross-tile accumulation."""
    frac = np.zeros((FINE.nlat, FINE.nlon)); ocean = np.zeros((FINE.nlat, FINE.nlon), bool)
    lat_edges = np.linspace(FINE.upleft_lat, FINE.lowright_lat, FINE.nlat + 1)  # desc
    lon_edges = np.linspace(FINE.upleft_lon, FINE.lowright_lon, FINE.nlon + 1)  # asc
    RSTRIP, CSTRIP = 8, 50  # 4x-cells per fetch (keep GeoTIFF < ~40M px)
    for r0 in range(0, FINE.nlat, RSTRIP):
        r1 = min(r0 + RSTRIP, FINE.nlat)
        for c0 in range(0, FINE.nlon, CSTRIP):
            c1 = min(c0 + CSTRIP, FINE.nlon)
            elev, lat_hi, lon_hi = fetch_emodnet(lat_edges[r1], lat_edges[r0], lon_edges[c0], lon_edges[c1])
            f, oc = shallow_fraction(elev, lat_hi, lon_hi, FINE, depth_max_m)
            frac[r0:r1, c0:c1] = f[r0:r1, c0:c1]
            ocean[r0:r1, c0:c1] = oc[r0:r1, c0:c1]
    return frac, ocean


def _cell_of(lat, lon):
    r = int(np.clip(np.searchsorted(-np.linspace(FINE.upleft_lat, FINE.lowright_lat, FINE.nlat + 1),
                                    -lat, "right") - 1, 0, FINE.nlat - 1))
    c = int(np.clip(np.searchsorted(np.linspace(FINE.upleft_lon, FINE.lowright_lon, FINE.nlon + 1),
                                    lon, "right") - 1, 0, FINE.nlon - 1))
    return r, c


def main() -> int:
    _, ocean = build_shallow_fraction(depth_max_m=1e9)
    r, c = _cell_of(55.3, 21.1)  # Curonian Lagoon — must be ocean (orientation sanity, spec §8)
    assert ocean[r, c], "orientation check failed: Curonian Lagoon not ocean"
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "grid").mkdir(exist_ok=True)
    mask = np.where(ocean, 0, -99).astype(int)
    np.savetxt(OUT / "grid" / "baltic_fine_mask.csv", np.flipud(mask), fmt="%d", delimiter=";")
    tlat, tlon = target_coords(FINE)
    xr.Dataset({"mask": (["latitude", "longitude"], ocean.astype("int8"))},
               coords={"latitude": tlat, "longitude": tlon}).to_netcdf(OUT / "baltic_fine_grid.nc")
    print(f"fine grid: ocean cells = {int(ocean.sum())} of {ocean.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE: not unit-tested (real network); the pure `shallow_fraction` is. Manual smoke: run `main()`, expect a few thousand ocean cells + the orientation assert to pass.

- [ ] **Step 6: Add rasterio to deps** — `pyproject.toml` dev/forcing extra `"rasterio>=1.3"` (already in `.venv`). Verify `import rasterio`.

- [ ] **Step 7: Commit** — `git add osmose/forcing/bathymetry.py scripts/build_baltic_fine_grid.py tests/test_bathymetry.py pyproject.toml && git commit -m "feat: shallow-fraction core + EMODnet tiled grid/mask builder (+rasterio)"`

---

### Task 2: Percid habitat builder — binary thin-littoral maps

**Files:** Create `osmose/forcing/percid_habitat.py`, `scripts/build_baltic_fine_percid_maps.py`; Test `tests/test_percid_habitat.py`.

**Interfaces:**
- Consumes: `(frac, ocean)` (Task 1); a fine annual-mean bottom-salinity `(nlat,nlon)` (Task 4, gap-filled).
- Produces: `percid_stage_map(frac, ocean, salinity, tau, land_value=-99, sal_ceiling=None, sal_gate=None) -> NDArray` — **binary** map: `-99` on land; `1.0` where `ocean AND frac>=tau AND salinity<threshold`; `0.0` elsewhere. `sal_ceiling` (adult/juvenile marine exclusion) OR `sal_gate` (spawning egg gate) — exactly one per stage.
- Produces: `vacuity_ok(real_map, upsampled_percid_footprint, max_ratio=0.4) -> bool` — `sum(real_map==1)/sum(upsampled_footprint>0) <= max_ratio` and non-empty.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_percid_habitat.py
import numpy as np, pytest
from osmose.forcing.percid_habitat import percid_stage_map, vacuity_ok


def test_binary_threshold_and_land():
    frac = np.array([[0.5, 0.1], [0.9, 0.0]]); ocean = np.array([[True, True], [True, False]])
    sal = np.full((2, 2), 4.0)
    m = percid_stage_map(frac, ocean, sal, tau=0.25, sal_ceiling=12.0)
    assert m[1, 1] == -99.0                 # land
    assert m[0, 0] == 1.0                   # frac 0.5 >= 0.25
    assert m[0, 1] == 0.0                   # frac 0.1 < 0.25
    assert set(np.unique(m)) <= {-99.0, 0.0, 1.0}  # strictly binary (+ land)


def test_adult_salinity_ceiling():
    m = percid_stage_map(np.array([[0.8, 0.8]]), np.array([[True, True]]),
                         np.array([[5.0, 20.0]]), tau=0.25, sal_ceiling=12.0)
    assert m[0, 0] == 1.0 and m[0, 1] == 0.0   # 20 PSU excluded


def test_spawning_gate_tighter():
    m = percid_stage_map(np.array([[0.8, 0.8]]), np.array([[True, True]]),
                         np.array([[4.0, 5.5]]), tau=0.25, sal_gate=5.0)
    assert m[0, 0] == 1.0 and m[0, 1] == 0.0   # 5.5 >= 5.0 gate


def test_nan_salinity_is_excluded_not_kept():
    # gap-fill should remove NaN, but guard anyway: NaN must NOT pass as habitat
    m = percid_stage_map(np.array([[0.8]]), np.array([[True]]), np.array([[np.nan]]), tau=0.25, sal_ceiling=12.0)
    assert m[0, 0] == 0.0


def test_vacuity_guard_area_vs_upsampled_percid_footprint():
    real = np.array([[0.0, 1.0, 0.0, 0.0]]); up = np.array([[1.0, 1.0, 1.0, 0.0]])  # 3-cell percid footprint
    assert vacuity_ok(real, up, max_ratio=0.4)          # 1/3 = 0.33
    assert not vacuity_ok(up, up, max_ratio=0.4)         # 3/3
    assert not vacuity_ok(np.zeros((1, 4)), up)          # empty
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

```python
# osmose/forcing/percid_habitat.py
"""Real perch/pikeperch habitat as a BINARY thin-littoral set: a cell is habitat
iff a meaningful fraction (>= tau) of it is shallow littoral AND salinity permits.
Binary (not fractional) so the habitat SET/AREA genuinely shrinks and is read
unambiguously by the engine's map semantics. Salinity gate on spawning; relaxed
ceiling on adult/juvenile. Land -> -99."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def percid_stage_map(frac, ocean, salinity, tau, land_value=-99.0, sal_ceiling=None, sal_gate=None):
    frac = np.asarray(frac, float); ocean = np.asarray(ocean, bool); salinity = np.asarray(salinity, float)
    thr = sal_ceiling if sal_ceiling is not None else sal_gate
    sal_ok = np.isfinite(salinity) & (salinity < thr)   # NaN -> False (excluded)
    habitat = ocean & (frac >= tau) & sal_ok
    out = np.where(ocean, 0.0, land_value)
    out[habitat] = 1.0
    return out


def vacuity_ok(real_map, upsampled_percid_footprint, max_ratio=0.4):
    real = int(np.sum(np.asarray(real_map) == 1.0))
    up = int(np.sum(np.asarray(upsampled_percid_footprint) > 0))
    if real == 0 or up == 0:
        return False
    return real / up <= max_ratio
```

- [ ] **Step 4: Run to verify pass** — PASS (5 passed).

- [ ] **Step 5: Write the driver** (`scripts/build_baltic_fine_percid_maps.py`): per stage, threshold with per-species `tau`/`depth`/salinity (CLI-overridable), vacuity-check against the **block-upsampled coarse percid map for that stage** (Task 3 `*_upsampled.csv`, cells>0), write binary map.

```python
# scripts/build_baltic_fine_percid_maps.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
from osmose.forcing.percid_habitat import percid_stage_map, vacuity_ok
from scripts.build_baltic_fine_grid import build_shallow_fraction, OUT

# (stage_file, depth_max_m, sal_ceiling, sal_gate)
STAGES = {
    "perch_adult.csv": (12.0, 12.0, None), "perch_juvenile.csv": (8.0, 12.0, None),
    "perch_spawning.csv": (6.0, None, 6.0),
    "pikeperch_adult.csv": (18.0, 14.0, None), "pikeperch_juvenile.csv": (12.0, 14.0, None),
    "pikeperch_spawning.csv": (8.0, None, 5.0),
}


def _annual_mean_salinity():
    ds = xr.open_dataset(OUT / "baltic_salinity_bottom_climatology.nc")
    v = "salinity" if "salinity" in ds else list(ds.data_vars)[0]
    return np.asarray(ds[v].values).mean(axis=0)   # (160,200); gap-filled in Task 4


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--max-ratio", type=float, default=0.4); args = ap.parse_args()
    maps_dir = OUT / "maps"; maps_dir.mkdir(parents=True, exist_ok=True)
    sal = _annual_mean_salinity()
    for fname, (dmax, ceil, gate) in STAGES.items():
        frac, ocean = build_shallow_fraction(depth_max_m=dmax)
        m = percid_stage_map(frac, ocean, sal, tau=args.tau, sal_ceiling=ceil, sal_gate=gate)
        up = pd.read_csv(maps_dir / (fname[:-4] + "_upsampled.csv"), sep=";", header=None).values.astype(float)
        if not vacuity_ok(m, up, max_ratio=args.max_ratio):
            raise ValueError(f"{fname}: vacuity guard failed (empty or > {args.max_ratio} of upsampled footprint)")
        np.savetxt(maps_dir / fname, np.flipud(m), fmt="%d", delimiter=";")
        print(f"{fname}: {int(np.sum(m == 1))} habitat cells (upsampled footprint {int(np.sum(up > 0))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE: runs AFTER Task 3 (needs `*_upsampled.csv`) and Task 4 (needs the gap-filled salinity climatology). The ladder-assembly ordering (Task 5) enforces this.

- [ ] **Step 6: Commit** — `git add osmose/forcing/percid_habitat.py scripts/build_baltic_fine_percid_maps.py tests/test_percid_habitat.py && git commit -m "feat: binary thin-littoral percid maps (threshold + spawning salinity gate, area vacuity guard)"`

---

### Task 3: Occupancy upsampler — maps, grid mask, AND fishing maps

**Files:** Create `osmose/forcing/grid_upsample.py`, `scripts/baltic_grid_upsample.py`; Test `tests/test_grid_upsample.py`.

**Interfaces:** `block_replicate(arr, factor) -> NDArray` — each cell → `factor×factor` block, preserving values (incl. `-99`).

- [ ] **Step 1: Failing test**

```python
# tests/test_grid_upsample.py
import numpy as np
from osmose.forcing.grid_upsample import block_replicate


def test_block_replicate_preserves_values_and_shape():
    a = np.array([[-99.0, 1.0], [0.0, 0.5]]); out = block_replicate(a, 2)
    assert out.shape == (4, 4)
    assert np.array_equal(out[0:2, 0:2], np.full((2, 2), -99.0))
    assert np.array_equal(out[2:4, 2:4], np.full((2, 2), 0.5))
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Module**

```python
# osmose/forcing/grid_upsample.py
"""Block-replicate an (nlat,nlon) OCCUPANCY array to (nlat*f,nlon*f). For movement
maps / masks / fishing-distribution maps ({-99,0,weight}) — NOT absolute biomass
(use osmose/forcing/conserve_regrid for those)."""
from __future__ import annotations
import numpy as np


def block_replicate(arr, factor: int):
    a = np.asarray(arr, dtype=np.float64)
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)
```

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Driver** (`scripts/baltic_grid_upsample.py`): upsample (a) every `data/baltic/maps/*.csv` → `data/baltic-fine/maps/` (non-percid canonical name; percid → `_upsampled.csv` control copies), (b) `data/baltic/grid/baltic_mask.csv` (kept for reference; the authoritative mask is Task 1's EMODnet one), (c) **every `data/baltic/fishing/*.csv`** → `data/baltic-fine/fishing/` (canonical name — CRITICAL: these are spatial 40×50 maps referenced by `fisheries.movement.fishery.map*` and crash the run if left 40×50).

```python
# scripts/baltic_grid_upsample.py
from pathlib import Path
import numpy as np, pandas as pd
from osmose.forcing.grid_upsample import block_replicate

PERCID = {"perch", "pikeperch"}
JOBS = [(Path("data/baltic/maps"), Path("data/baltic-fine/maps"), True),
        (Path("data/baltic/fishing"), Path("data/baltic-fine/fishing"), False)]


def main() -> int:
    for src, dst, percid_control in JOBS:
        dst.mkdir(parents=True, exist_ok=True)
        for csv in sorted(src.glob("*.csv")):   # *.csv skips *.pre-mask-rebuild.bak siblings
            arr = pd.read_csv(csv, sep=";", header=None).values.astype(float)  # south->north on disk
            up = block_replicate(arr, 4)
            sp = csv.stem.split("_")[0]
            name = csv.stem + "_upsampled.csv" if (percid_control and sp in PERCID) else csv.name
            np.savetxt(dst / name, up, fmt="%.4f", delimiter=";")
            print(f"upsampled {csv.name} -> {dst / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Commit** — `git add osmose/forcing/grid_upsample.py scripts/baltic_grid_upsample.py tests/test_grid_upsample.py && git commit -m "feat: occupancy upsampler for maps + fishing distribution maps"`

---

### Task 4: Conserving forcing regridder + gap-filled salinity + shape guards

**Files:** Create `osmose/forcing/conserve_regrid.py`, `scripts/build_baltic_fine_forcing.py`; Test `tests/test_conserve_regrid.py`.

**Interfaces:** `split_conserve(field, factor) -> NDArray` — regrid `(…,nlat,nlon)` absolute-biomass so total is preserved (each cell ÷ factor² into its sub-cells).

- [ ] **Step 1: Failing test**

```python
# tests/test_conserve_regrid.py
import numpy as np
from osmose.forcing.conserve_regrid import split_conserve


def test_total_conserved():
    f = np.array([[[4.0, 8.0], [0.0, 16.0]]])
    out = split_conserve(f, 2)
    assert out.shape == (1, 4, 4) and np.isclose(out.sum(), f.sum())
    assert np.allclose(out[0, 0:2, 0:2], 1.0)


def test_not_x16_regression():
    f = np.ones((1, 5, 5)); assert np.isclose(split_conserve(f, 4).sum(), f.sum())
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Module**

```python
# osmose/forcing/conserve_regrid.py
"""Conservative regrid for ABSOLUTE-biomass forcing (tonnes/cell): split each
coarse cell's mass equally across its factor**2 sub-cells so the global total is
preserved. block_replicate here would inflate total system biomass factor**2x."""
from __future__ import annotations
import numpy as np


def split_conserve(field, factor: int):
    up = np.repeat(np.repeat(np.asarray(field, dtype=np.float64), factor, axis=-2), factor, axis=-1)
    return up / (factor * factor)
```

- [ ] **Step 4: Run to verify pass** — PASS (2 passed).

- [ ] **Step 5: Forcing builder** (`scripts/build_baltic_fine_forcing.py`):
  - Absolute-biomass NetCDFs → `split_conserve(var,4)` → `(24,160,200)`, **assert `np.isclose(fine.sum(), coarse.sum())` per var AND `fine.shape[-2:]==(160,200)`**.
  - **Salinity climatology at 4×**: `accumulate_climatology` → `regrid(clim, src_lat, src_lon, FINE)` → `resample_to_24` → **`fill_ocean_nan(field24, load_ocean_mask(OUT/'baltic_fine_grid.nc'))`** → **assert no ocean NaN** (mirror `build_baltic_salinity_forcing.build`) → write. Salinity is INTENSIVE (regrid, never split_conserve).

```python
# scripts/build_baltic_fine_forcing.py
from pathlib import Path
import numpy as np, xarray as xr
from osmose.forcing.conserve_regrid import split_conserve
from osmose.forcing.grid import target_coords, regrid, resample_to_24, load_ocean_mask
from scripts.build_baltic_fine_grid import FINE, OUT

SRC = Path("data/baltic"); ABS_BIOMASS = ["baltic_ltl_biomass.nc", "baltic_predator_biomass.nc"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); tlat, tlon = target_coords(FINE)
    for fname in ABS_BIOMASS:
        ds = xr.open_dataset(SRC / fname); out = {}; nt = 24
        for name, da in ds.data_vars.items():
            coarse = np.asarray(da.values); fine = split_conserve(coarse, 4); nt = fine.shape[0]
            assert np.isclose(fine.sum(), coarse.sum()), f"{fname}:{name} biomass not conserved"
            assert fine.shape[-2:] == (160, 200), f"{fname}:{name} shape {fine.shape}"
            out[name] = (["time", "latitude", "longitude"], fine)
        xr.Dataset(out, coords={"time": np.arange(nt), "latitude": tlat, "longitude": tlon}).to_netcdf(OUT / fname)
        print(f"conserved-regrid {fname}")
    # salinity: import accumulate_climatology from the salinity builder, regrid to FINE, gap-fill, guard
    from scripts.build_baltic_salinity_forcing import accumulate_climatology, fill_ocean_nan
    so_files = sorted((Path("data") / "cmems_cache" / "cmems_downloads").glob("baltic_phy_monthly_reanalysis_so_*.nc"))
    clim, src_lat, src_lon = accumulate_climatology([str(p) for p in so_files])  # (12, src_lat, src_lon)
    field24 = resample_to_24(regrid(clim, src_lat, src_lon, FINE))               # (24,160,200), NaN in gaps
    mask = load_ocean_mask(OUT / "baltic_fine_grid.nc")
    assert mask is not None, "fine ocean mask missing — run build_baltic_fine_grid first"
    field24 = fill_ocean_nan(field24, mask)
    assert not bool(np.isnan(field24[np.broadcast_to(mask, field24.shape)]).any()), "ocean NaN after fill"
    xr.Dataset({"salinity": (["time", "latitude", "longitude"], field24)},
               coords={"time": np.arange(24), "latitude": tlat, "longitude": tlon}
               ).to_netcdf(OUT / "baltic_salinity_bottom_climatology.nc")
    print("salinity climatology (gap-filled) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE: confirm `fill_ocean_nan`, `load_ocean_mask`, `resample_to_24`, `regrid`, `target_coords` are all exported by `osmose/forcing/grid.py` (they are) and `accumulate_climatology` by `scripts/build_baltic_salinity_forcing.py` (it is). Salinity climatology MUST be built before Task 2's percid maps.

- [ ] **Step 6: Commit** — `git add osmose/forcing/conserve_regrid.py scripts/build_baltic_fine_forcing.py tests/test_conserve_regrid.py && git commit -m "feat: conserving biomass regrid + gap-filled fine salinity + shape guards"`

---

### Task 5: Assemble the two 4× config variants (distinct movement sub-files)

**Files:** Create `scripts/build_baltic_fine_config.py`, generated `data/baltic-fine/*.csv`; Test `tests/test_baltic_fine_config.py`.

**Interfaces:** Consumes the fine grid/mask/maps/fishing/forcing. Produces two entrypoints `data/baltic-fine/baltic_fine_{upsampled,real}_all-parameters.csv` differing ONLY in which movement sub-file they `osmose.configuration.movement`-include.

- [ ] **Step 1: Failing test**

```python
# tests/test_baltic_fine_config.py
from pathlib import Path
from osmose.config import OsmoseConfigReader
from osmose.engine.config import EngineConfig


def _cfg(variant):
    p = sorted(Path("data/baltic-fine").glob(f"*{variant}_all-parameters*.csv"))[0]
    return dict(OsmoseConfigReader().read(str(p)))


def test_both_variants_load_4x_and_construct():
    for v in ["upsampled", "real"]:
        cfg = _cfg(v)
        assert cfg["grid.nlon"] == "200" and cfg["grid.nlat"] == "160"
        assert EngineConfig.from_dict(cfg).n_species >= 8


def test_variants_differ_only_in_percid_maps_after_include_resolution():
    up, real = _cfg("upsampled"), _cfg("real")
    diffs = {k for k in set(up) | set(real) if up.get(k) != real.get(k)}
    # only the 6 percid movement.file.map values (map13..18) may differ
    assert diffs, "variants identical -> rung2==rung3 (the C1 bug)"
    percid_map_keys = {f"movement.file.map{n}" for n in range(13, 19)}
    assert all(k in percid_map_keys for k in diffs), f"unexpected non-percid diffs: {diffs}"
    assert any("upsampled" in up[k] for k in diffs)      # up points at *_upsampled.csv
    assert all("upsampled" not in real[k] for k in diffs)  # real points at the binary maps
```

- [ ] **Step 2: Run to verify fail** — no config yet.

- [ ] **Step 3: Write the assembler** (`scripts/build_baltic_fine_config.py`):
  1. Read `data/baltic/baltic_all-parameters.csv` (the master aggregator). Copy every `osmose.configuration.*`-referenced `baltic_param-*.csv` into `data/baltic-fine/`, EDITING: grid keys → 200/160 + `grid/baltic_fine_mask.csv` + `baltic_fine_grid.nc`; every `movement.file.map*`, `fisheries.movement.fishery.map*`/fishing-distrib path, and forcing file key (`*ltl*`, `*predator*`, salinity `movement.salinity.field.file`) → the `data/baltic-fine/` version.
  2. **Preserve master-only keys** (e.g. `simulation.nschool.sp*`, `mortality.subdt`, `osmose.version`, any key in the master that is NOT an `osmose.configuration.*` include) by copying them into the two entrypoints verbatim.
  3. Write `baltic_param-movement-upsampled.csv` (percid map13..18 → `maps/{perch,pikeperch}_*_upsampled.csv`) and `baltic_param-movement-real.csv` (→ the binary `maps/{perch,pikeperch}_*.csv`); both identical for the 6 non-percid species' map keys.
  4. Write two entrypoints identical except `osmose.configuration.movement;baltic_param-movement-upsampled.csv` vs `...-real.csv`.

- [ ] **Step 4: Run to verify pass** — both variants load; the include-resolved diff is exactly the 6 percid map values.

- [ ] **Step 5: Commit** — `git add scripts/build_baltic_fine_config.py data/baltic-fine tests/test_baltic_fine_config.py && git commit -m "feat: assemble 4x baltic-fine config; variants differ only by movement sub-file"`

---

### Task 6: Attribution-ladder runner + GO/NO-GO report

**Files:** Create `scripts/baltic_fine_grid_ladder.py`; Test `tests/test_baltic_fine_ladder.py` (smoke).

**Interfaces:** Consumes coarse `data/baltic/` + the two fine entrypoints; `osmose.calibration.targets.load_targets`. Produces per-species overshoot-ratio table (mean ± seed spread) across 3 rungs + area ratio + GO/NO-GO.

- [ ] **Step 1: Smoke test**

```python
# tests/test_baltic_fine_ladder.py
import subprocess, sys


def test_ladder_runs_three_rungs_and_verdict():
    out = subprocess.run([sys.executable, "scripts/baltic_fine_grid_ladder.py", "--nyear", "2", "--seeds", "1"],
                         capture_output=True, text=True, timeout=5400)
    assert out.returncode == 0, out.stderr
    lo = out.stdout.lower()
    assert "coarse" in lo and "4x-upsampled" in lo and "4x-real" in lo
    assert "perch" in lo and "pikeperch" in lo and "area" in lo
    assert "go" in lo or "no-go" in lo
```

- [ ] **Step 2: Run to verify fail** — script missing.

- [ ] **Step 3: Write the runner** — CORRECT target loader; **`.biomass()` is a WIDE frame → `bio["perch"]`** (do NOT long-filter); area ratio in the GO gate; forcing shape guard.

```python
# scripts/baltic_fine_grid_ladder.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.calibration.targets import load_targets

DET = {"movement.randomseed.fixed": "true", "stochastic.mortality.randomseed.fixed": "true"}
PERCIDS, HIGHW = ["perch", "pikeperch"], ["cod", "herring", "sprat"]
RUNGS = {"coarse": "data/baltic/baltic_all-parameters.csv",
         "4x-upsampled": "data/baltic-fine/baltic_fine_upsampled_all-parameters.csv",
         "4x-real": "data/baltic-fine/baltic_fine_real_all-parameters.csv"}


def late_mean(series, frac=1 / 3):
    b = np.asarray(series, float); return float(np.mean(b[int(len(b) * (1 - frac)):]))


def percid_area_ratio():
    """real habitat cells / upsampled footprint cells, averaged over the 6 percid maps."""
    m = Path("data/baltic-fine/maps"); rs, us = 0, 0
    for f in ["perch_adult", "perch_juvenile", "perch_spawning", "pikeperch_adult", "pikeperch_juvenile", "pikeperch_spawning"]:
        real = pd.read_csv(m / f"{f}.csv", sep=";", header=None).values
        up = pd.read_csv(m / f"{f}_upsampled.csv", sep=";", header=None).values
        rs += int(np.sum(real == 1)); us += int(np.sum(up > 0))
    return rs / us if us else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--nyear", type=int, default=30); ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    tlist, _ = load_targets(Path("data/baltic/reference/biomass_targets.csv"))
    targets = {t.species: t.target for t in tlist}
    results = {}
    for rung, path in RUNGS.items():
        base = dict(OsmoseConfigReader().read(path)); base.update(DET); base["simulation.time.nyear"] = str(args.nyear)
        acc = {sp: [] for sp in PERCIDS + HIGHW}
        for s in range(args.seeds):
            bio = PythonEngine().run_in_memory(dict(base), seed=s).biomass()  # WIDE frame
            for sp in acc:
                acc[sp].append(late_mean(bio[sp]) / targets[sp])
        results[rung] = {sp: (float(np.mean(v)), float(np.std(v))) for sp, v in acc.items()}
    area = percid_area_ratio()
    print("species     " + "  ".join(f"{r:>16}" for r in RUNGS) + "   role")
    for sp in PERCIDS + HIGHW:
        row = "  ".join(f"{results[r][sp][0]:8.1f}±{results[r][sp][1]:4.1f}" for r in RUNGS)
        print(f"{sp:11} {row}   {'PERCID' if sp in PERCIDS else 'high-weight'}")
    print(f"real/upsampled percid area ratio = {area:.3f}")
    up, real = results["4x-upsampled"], results["4x-real"]
    # GO: real percids drop toward single digits, STABLY, AND by MORE than the pure area cut
    def dropped(sp):
        rel = (up[sp][0] - real[sp][0]) / up[sp][0] if up[sp][0] else 0.0
        return rel > (1 - area) and real[sp][0] < 10 and real[sp][1] < 0.5 * max(real[sp][0], 1e-9)
    go = all(dropped(sp) for sp in PERCIDS)
    print("VERDICT:", "GO — real habitat damps percids beyond the area cut, stably"
          if go else "NO-GO — real ~= upsampled / only-area / unstable (structural)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE: confirm `BiomassTarget` has a `.species` and a `.target` (grep `osmose/calibration/targets.py`; `losses.py` uses `{t.species: t for t in targets}`). `.biomass()` IS wide (verified: columns include `perch`,`pikeperch`,`cod`). Smoke uses `--nyear 2 --seeds 1`; the real A/B runs at defaults (3×5 runs at 4× ≈ hours — see §10). Record the table + verdict in the branch and memory.

- [ ] **Step 4: Run to verify pass** — script runs, prints 3 rungs + area ratio + verdict.

- [ ] **Step 5: Commit** — `git add scripts/baltic_fine_grid_ladder.py tests/test_baltic_fine_ladder.py && git commit -m "feat: attribution-ladder runner (correct target loader, wide biomass, area-guarded GO/NO-GO)"`

---

## Post-plan verification (run before declaring done)

- [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q -k "bathymetry or percid_habitat or grid_upsample or conserve_regrid or baltic_fine"` — all green.
- [ ] `.venv/bin/ruff check osmose/ scripts/ tests/` and `ruff format --check` — clean.
- [ ] `git status data/baltic/` shows nothing (coarse config untouched).
- [ ] Build end-to-end (real network + engine): grid → forcing (biomass+salinity) → upsample (maps+fishing) → percid maps → config → ladder at defaults. Verify: (a) the two entrypoints' include-resolved dicts differ ONLY in the 6 percid map values; (b) every fine forcing/map is `(…,160,200)` / 160×200; (c) the ladder table shows all 3 rungs.
- [ ] Record the GO/NO-GO outcome (this is the deciding experiment for the full high-res build) in memory.

## §10 compute note
3 rungs × ≥5 seeds × nyear=30; the two 4× rungs run on ~6k ocean cells of 32k, ~16–30× slower than coarse (predation cell-loop dominates) → the full A/B is **hours**, run by hand at defaults. The smoke test (`--nyear 2 --seeds 1`) keeps CI cheap.
