# Baltic Finer-Grid Percid-Habitat Deciding Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an alternative 4× Baltic config whose perch/pikeperch habitat is derived from real EMODnet bathymetry (sub-grid shallow-fraction) + a spawning-stage salinity gate, and a 3-rung attribution ladder (coarse → 4×-upsampled → 4×-real) that decides whether resolving real coastal percid habitat cures the structural percid overshoot.

**Architecture:** Five pure, independently-tested builders (grid+bathymetry, percid habitat, occupancy upsampler, conserving forcing regridder, ladder runner) write a new `data/baltic-fine/` config tree; `data/baltic/` is never touched. Each builder has a pure core (unit-tested on synthetic arrays) and a thin IO script. The ladder runner assembles two 4× config variants and runs the three rungs.

**Tech Stack:** Python 3.12, NumPy, xarray/netCDF4, pandas, **rasterio** (NEW — reads EMODnet GeoTIFF; GDAL-based, offline-builder-only), requests (EMODnet WCS), pytest, ruff. Run everything with `/home/razinka/osmopy/.venv/bin/python`.

## Global Constraints

- Repo root: `/home/razinka/osmopy`. **Work on a NEW branch/worktree `feat/baltic-fine-grid-percid`, never on master.**
- Run tests: `PYTHONPATH=. .venv/bin/python -m pytest`. Lint: `.venv/bin/ruff check` AND `.venv/bin/ruff format --check`.
- `data/baltic/` is READ-ONLY for this work. All new outputs go under `data/baltic-fine/`.
- **Grid:** `GridSpec` (`osmose/maps/builder.py`) fields `nlon,nlat,upleft_lat,upleft_lon,lowright_lat,lowright_lon`; `.from_config(cfg)`. Fine grid = `nlon=200, nlat=160` (4×), same extent (upleft 66N/10E, lowright 54N/30E).
- **Spatial CSV convention:** semicolon-separated, **rows south→north on disk** (loader `_load_spatial_csv` in `osmose/engine/config.py` does `np.flipud`), internal arrays are `(nlat, nlon)` north→south. Land sentinel **`-99`**, ocean-non-habitat `0`, habitat = positive weight. A map CSV is `nlat` rows × `nlon` cols.
- **Forcing alignment is by CELL INDEX, not coordinate** — matching shapes suffice; all fine forcing arrays are `(…, 160, 200)`.
- **CRITICAL:** absolute-biomass forcings (`baltic_ltl_biomass.nc` 6 vars; `baltic_predator_biomass.nc` GreySeal/Cormorant; all `(24,40,50)` tonnes/cell) MUST be **conserved** on regrid (split across sub-cells), never replicated (replication → ×16 system-biomass inflation). Intensive fields (salinity, thetao) regrid/replicate as-is.
- **EMODnet (spike-confirmed):** WCS `GetCoverage`, coverage `emodnet__mean`, `version=2.0.1`, `subset=Lat(a,b)&subset=Long(c,d)`, `format=image/tiff` → GeoTIFF read by rasterio; EPSG:4326, ~115 m (0.0010416667°), 32-bit; **negative = below sea level**, so `depth_m = -elev`, land = `elev >= 0`. Fetch **tiled** (never one 221 M-cell download).
- Percid map stages (existing keys in `data/baltic/baltic_param-movement.csv`): perch `maps/perch_{juvenile,adult,spawning}.csv` (map13/14/15), pikeperch `maps/pikeperch_{juvenile,adult,spawning}.csv` (map16/17/18).
- Overshoot metric: reuse `osmose/calibration/targets.py` + `data/baltic/reference/biomass_targets.csv` for per-species targets; overshoot ratio = late-window-mean biomass / target.
- Ladder: 3 rungs × **≥5 seeds × nyear=30**.

---

### Task 1: Sub-grid shallow-fraction pure core + EMODnet tiled fetch

**Files:**
- Create: `osmose/forcing/bathymetry.py`
- Create: `scripts/build_baltic_fine_grid.py`
- Test: `tests/test_bathymetry.py`

**Interfaces:**
- Produces: `shallow_fraction(elev_hi, lat_hi, lon_hi, grid, depth_max_m) -> tuple[NDArray, NDArray]` returning `(frac (nlat,nlon), ocean (nlat,nlon) bool)`; `frac[r,c]` = fraction of that 4× cell's WET high-res sub-pixels with `0 < depth <= depth_max_m` (depth = -elev). `ocean[r,c]` = any wet sub-pixel.
- Produces (script): `data/baltic-fine/baltic_fine_grid.nc` + `data/baltic-fine/grid/baltic_fine_mask.csv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bathymetry.py
import numpy as np
from osmose.forcing.bathymetry import shallow_fraction
from osmose.maps.builder import GridSpec


def _grid():
    # 1x1 target cell over lon[10,10.2], lat[54,54.2] just to exercise binning
    return GridSpec(nlon=1, nlat=1, upleft_lat=54.2, upleft_lon=10.0,
                    lowright_lat=54.0, lowright_lon=10.2)


def test_shallow_fraction_and_ocean():
    # 4 hi-res pixels in the single cell: elevations -5 (shallow), -50 (deep), +3 (land), -10 (shallow)
    lat_hi = np.array([54.15, 54.05])
    lon_hi = np.array([10.05, 10.15])
    elev = np.array([[-5.0, -50.0], [3.0, -10.0]])  # (lat, lon)
    frac, ocean = shallow_fraction(elev, lat_hi, lon_hi, _grid(), depth_max_m=15.0)
    assert ocean[0, 0]  # has wet pixels
    # wet pixels: -5,-50,-10 (3 of them); <=15 m deep: -5,-10 (2) -> 2/3
    assert frac[0, 0] == np.float64(2) / 3


def test_all_land_is_not_ocean_and_zero_fraction():
    lat_hi = np.array([54.1]); lon_hi = np.array([10.1])
    elev = np.array([[5.0]])  # land
    frac, ocean = shallow_fraction(elev, lat_hi, lon_hi, _grid(), depth_max_m=15.0)
    assert not ocean[0, 0]
    assert frac[0, 0] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bathymetry.py -q`
Expected: FAIL `ModuleNotFoundError: osmose.forcing.bathymetry`.

- [ ] **Step 3: Write the pure core**

```python
# osmose/forcing/bathymetry.py
"""Sub-grid bathymetry statistics for the finer-grid percid experiment.

Pure, grid/data-source-agnostic. shallow_fraction bins high-resolution EMODnet
elevation (negative = below sea level) into the target grid and returns, per
target cell, the fraction of WET sub-pixels shallower than depth_max_m — the
'real habitat detail' that a coarse binary mask cannot express.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.maps.builder import GridSpec


def shallow_fraction(
    elev_hi: NDArray[np.float64],
    lat_hi: NDArray[np.float64],
    lon_hi: NDArray[np.float64],
    grid: GridSpec,
    depth_max_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Fraction of each grid cell that is wet AND shallower than depth_max_m.

    depth = -elev (EMODnet convention). Cells with no wet sub-pixel -> ocean False,
    frac 0.0. Returns (frac (nlat,nlon), ocean (nlat,nlon) bool), lat north->south.
    """
    # target cell-edge bins; latitude descending (north->south) to match target_coords
    lat_edges = np.linspace(grid.upleft_lat, grid.lowright_lat, grid.nlat + 1)  # descending
    lon_edges = np.linspace(grid.upleft_lon, grid.lowright_lon, grid.nlon + 1)  # ascending
    # row index into (nlat) north->south: use descending edges via searchsorted on reversed
    row = np.clip(np.searchsorted(-lat_edges, -lat_hi, side="right") - 1, 0, grid.nlat - 1)
    col = np.clip(np.searchsorted(lon_edges, lon_hi, side="right") - 1, 0, grid.nlon - 1)

    depth = -elev_hi  # (nlat_hi, nlon_hi)
    wet = depth > 0.0
    shallow = wet & (depth <= depth_max_m)

    n_wet = np.zeros((grid.nlat, grid.nlon), dtype=np.float64)
    n_shallow = np.zeros((grid.nlat, grid.nlon), dtype=np.float64)
    R = row[:, None] * np.ones((1, lon_hi.size), dtype=int)
    C = np.ones((lat_hi.size, 1), dtype=int) * col[None, :]
    np.add.at(n_wet, (R, C), wet.astype(np.float64))
    np.add.at(n_shallow, (R, C), shallow.astype(np.float64))

    ocean = n_wet > 0
    frac = np.zeros_like(n_wet)
    np.divide(n_shallow, n_wet, out=frac, where=ocean)
    return frac, ocean
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_bathymetry.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the EMODnet tiled-fetch + grid script**

```python
# scripts/build_baltic_fine_grid.py
"""Build the 4x Baltic grid + mask from EMODnet bathymetry (spike-confirmed WCS).

Fetches EMODnet 'emodnet__mean' (~115 m) in lat/lon strips via WCS GetCoverage
(GeoTIFF, read by rasterio), and computes per-4x-cell ocean mask. Writes
data/baltic-fine/baltic_fine_grid.nc + grid/baltic_fine_mask.csv. Per-species
shallow_fraction is produced on demand by build_baltic_fine_percid_maps (Task 2)
using the same fetch helper exported here.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import rasterio
import requests
import xarray as xr

from osmose.forcing.bathymetry import shallow_fraction
from osmose.maps.builder import GridSpec

WCS = "https://ows.emodnet-bathymetry.eu/wcs"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "baltic-fine"
FINE = GridSpec(nlon=200, nlat=160, upleft_lat=66.0, upleft_lon=10.0,
                lowright_lat=54.0, lowright_lon=30.0)


def fetch_emodnet_strip(lat0: float, lat1: float, lon0: float, lon1: float):
    """Return (elev (nlat_hi,nlon_hi), lat_hi desc, lon_hi asc) for a bbox via WCS."""
    params = {
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
        "coverageId": "emodnet__mean",
        "subset": [f"Lat({lat0},{lat1})", f"Long({lon0},{lon1})"],
        "format": "image/tiff",
    }
    r = requests.get(WCS, params=params, timeout=180)
    r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as ds:
        elev = ds.read(1).astype(np.float64)
        b = ds.bounds
        lat_hi = np.linspace(b.top, b.bottom, ds.height)  # descending
        lon_hi = np.linspace(b.left, b.right, ds.width)    # ascending
    return elev, lat_hi, lon_hi


def build_shallow_fraction(depth_max_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-4x-cell shallow-fraction + ocean over the whole extent, strip by strip.

    Fetches EMODnet in latitude strips (a few 4x-rows tall) to bound memory. Each
    strip is ROW-DISJOINT — a 4x-cell's sub-pixels all come from one strip — so the
    per-strip ratio for its rows is already exact; assign directly, no cross-strip
    accumulation. Returns (frac (160,200), ocean (160,200)), lat north->south.
    """
    frac = np.zeros((FINE.nlat, FINE.nlon))
    ocean = np.zeros((FINE.nlat, FINE.nlon), dtype=bool)
    lat_edges = np.linspace(FINE.upleft_lat, FINE.lowright_lat, FINE.nlat + 1)  # descending
    STRIP = 8  # 4x-rows per fetch
    for r0 in range(0, FINE.nlat, STRIP):
        r1 = min(r0 + STRIP, FINE.nlat)
        # rows r0:r1 span latitudes [lat_edges[r1] (south) .. lat_edges[r0] (north)]
        elev, lat_hi, lon_hi = fetch_emodnet_strip(
            lat_edges[r1], lat_edges[r0], FINE.upleft_lon, FINE.lowright_lon)
        f, oc = shallow_fraction(elev, lat_hi, lon_hi, FINE, depth_max_m)
        frac[r0:r1] = f[r0:r1]
        ocean[r0:r1] = oc[r0:r1]
    return frac, ocean


def main() -> int:
    # mask: ocean iff any wet sub-pixel (use a large depth_max so 'shallow' == wet for the mask)
    _, ocean = build_shallow_fraction(depth_max_m=1e9)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grid").mkdir(exist_ok=True)
    mask = np.where(ocean, 0, -99).astype(int)  # 0 ocean, -99 land (map convention)
    # write south->north (flipud of internal north->south)
    np.savetxt(OUT / "grid" / "baltic_fine_mask.csv", np.flipud(mask), fmt="%d", delimiter=";")
    from osmose.forcing.grid import target_coords
    tlat, tlon = target_coords(FINE)
    xr.Dataset({"mask": (["latitude", "longitude"], ocean.astype("int8"))},
               coords={"latitude": tlat, "longitude": tlon}).to_netcdf(OUT / "baltic_fine_grid.nc")
    print(f"wrote fine grid/mask: ocean cells = {int(ocean.sum())} of {ocean.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE to implementer: `build_shallow_fraction` accumulates by STRIP where strips are row-disjoint, so per-strip ratios are exact — no cross-strip cell is split. Confirm the WCS honours the requested bbox to the pixel; if EMODnet snaps bounds, the returned `b.top/b.bottom` (used for `lat_hi`) keep binning correct regardless. This script does a real network fetch — it is NOT unit-tested; the pure `shallow_fraction` (Steps 1-4) is. A cheap manual smoke: run `main()` once and eyeball `ocean cells` (~a few thousand of 32000).

- [ ] **Step 6: Add rasterio to the dev dependencies**

Edit `pyproject.toml` `[project.optional-dependencies].dev` (or the forcing extra): add `"rasterio>=1.3"`. It is already installed in `.venv`. Run `.venv/bin/python -c "import rasterio; print(rasterio.__version__)"` — expect a version string.

- [ ] **Step 7: Commit**

```bash
git add osmose/forcing/bathymetry.py scripts/build_baltic_fine_grid.py tests/test_bathymetry.py pyproject.toml
git commit -m "feat: sub-grid shallow-fraction core + EMODnet tiled grid/mask builder"
```

---

### Task 2: Percid habitat builder (real adult/juvenile/spawning maps)

**Files:**
- Create: `osmose/forcing/percid_habitat.py`
- Create: `scripts/build_baltic_fine_percid_maps.py`
- Test: `tests/test_percid_habitat.py`

**Interfaces:**
- Consumes: `shallow_fraction` output (frac, ocean) from Task 1; a fine bottom-salinity `(nlat,nlon)` annual-mean array.
- Produces: `percid_stage_map(frac, ocean, salinity, land_value=-99, sal_ceiling=None, sal_gate=None) -> NDArray` returning a `(nlat,nlon)` map: `-99` on land (`~ocean`), else the occupancy weight = `frac` with optional salinity masking (`weight=0` where `salinity >= sal_ceiling` for adult/juvenile, or where `salinity >= sal_gate` for spawning), `0` where `frac==0`.
- Produces: `vacuity_ok(real_map, upsampled_footprint, max_ratio=0.4) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_percid_habitat.py
import numpy as np
import pytest
from osmose.forcing.percid_habitat import percid_stage_map, vacuity_ok


def test_land_is_minus99_and_weight_is_fraction():
    frac = np.array([[0.5, 0.0], [0.9, 0.2]])
    ocean = np.array([[True, True], [True, False]])
    sal = np.array([[4.0, 4.0], [4.0, 4.0]])
    m = percid_stage_map(frac, ocean, sal, sal_ceiling=12.0)
    assert m[1, 1] == -99.0            # land
    assert m[0, 0] == pytest.approx(0.5)
    assert m[0, 1] == 0.0              # ocean, no shallow habitat


def test_adult_salinity_ceiling_excludes_marine_cells():
    frac = np.array([[0.8, 0.8]])
    ocean = np.array([[True, True]])
    sal = np.array([[5.0, 20.0]])      # second cell too saline for perch adult (<12)
    m = percid_stage_map(frac, ocean, sal, sal_ceiling=12.0)
    assert m[0, 0] == pytest.approx(0.8)
    assert m[0, 1] == 0.0


def test_spawning_gate_tighter_than_adult():
    frac = np.array([[0.8, 0.8]])
    ocean = np.array([[True, True]])
    sal = np.array([[4.0, 5.5]])       # perch eggs fail >6; 5.5 ok, but pikeperch eggs <5 would fail
    m = percid_stage_map(frac, ocean, sal, sal_gate=5.0)
    assert m[0, 0] == pytest.approx(0.8)
    assert m[0, 1] == 0.0              # 5.5 >= 5.0 gate


def test_vacuity_guard():
    real = np.array([[0.0, 0.5, 0.0, 0.0]])   # 1 habitat cell
    up = np.array([[1.0, 1.0, 1.0, 1.0]])     # 4-cell footprint
    assert vacuity_ok(real, up, max_ratio=0.4)     # 1/4 = 0.25 <= 0.4
    assert not vacuity_ok(up, up, max_ratio=0.4)   # 4/4 = 1.0 (too fat)
    assert not vacuity_ok(np.zeros((1, 4)), up)    # empty -> False
```

- [ ] **Step 2: Run to verify fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_percid_habitat.py -q`
Expected: FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

```python
# osmose/forcing/percid_habitat.py
"""Real perch/pikeperch habitat maps from sub-grid shallow-fraction + salinity.

Adult/juvenile maps are depth-dominant (occupancy = shallow_fraction) with a
relaxed adult salinity ceiling (adults are euryhaline). The salinity gate proper
lives on the spawning maps (eggs are the salinity-sensitive stage). Land -> -99.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def percid_stage_map(
    frac: NDArray[np.float64],
    ocean: NDArray[np.bool_],
    salinity: NDArray[np.float64],
    land_value: float = -99.0,
    sal_ceiling: float | None = None,
    sal_gate: float | None = None,
) -> NDArray[np.float64]:
    """One life-stage occupancy map. sal_ceiling: adult/juvenile marine-exclusion
    (weight 0 where salinity >= ceiling). sal_gate: spawning egg gate (weight 0
    where salinity >= gate). Exactly one of sal_ceiling/sal_gate is set per stage.
    """
    out = np.where(ocean, frac.astype(np.float64), land_value)
    thr = sal_ceiling if sal_ceiling is not None else sal_gate
    if thr is not None:
        too_saline = ocean & (salinity >= thr)
        out[too_saline] = 0.0
    return out


def vacuity_ok(
    real_map: NDArray[np.float64],
    upsampled_footprint: NDArray[np.float64],
    max_ratio: float = 0.4,
) -> bool:
    """The real habitat must be non-empty AND its habitat-cell count <= max_ratio
    of the upsampled footprint's habitat-cell count (else the test is vacuous)."""
    real_cells = int(np.sum(real_map > 0))
    up_cells = int(np.sum(upsampled_footprint > 0))
    if real_cells == 0 or up_cells == 0:
        return False
    return real_cells / up_cells <= max_ratio
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_percid_habitat.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Write the driver script**

```python
# scripts/build_baltic_fine_percid_maps.py
"""Write the 6 real percid maps (perch/pikeperch x juvenile/adult/spawning) to
data/baltic-fine/maps/, from EMODnet shallow-fraction (Task 1) + fine bottom
salinity (annual mean of the 4x salinity climatology built in Task 4/D).

Per-species thresholds (spec v2 §7): CLI-overridable defaults below. Salinity
gate on spawning only; relaxed ceiling on adult/juvenile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from osmose.forcing.percid_habitat import percid_stage_map, vacuity_ok
from scripts.build_baltic_fine_grid import build_shallow_fraction, FINE, OUT

# (species, stage, depth_max_m, sal_ceiling, sal_gate, filename)
STAGES = [
    ("perch", "adult", 12.0, 12.0, None, "perch_adult.csv"),
    ("perch", "juvenile", 8.0, 12.0, None, "perch_juvenile.csv"),
    ("perch", "spawning", 6.0, None, 6.0, "perch_spawning.csv"),
    ("pikeperch", "adult", 18.0, 14.0, None, "pikeperch_adult.csv"),
    ("pikeperch", "juvenile", 12.0, 14.0, None, "pikeperch_juvenile.csv"),
    ("pikeperch", "spawning", 8.0, None, 5.0, "pikeperch_spawning.csv"),
]


def _annual_mean_salinity() -> np.ndarray:
    ds = xr.open_dataset(OUT / "baltic_salinity_bottom_climatology.nc")
    var = "salinity" if "salinity" in ds else list(ds.data_vars)[0]
    sal = np.asarray(ds[var].values)  # (24,160,200)
    return sal.mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ratio", type=float, default=0.4)
    args = ap.parse_args()
    maps_dir = OUT / "maps"; maps_dir.mkdir(parents=True, exist_ok=True)
    sal = _annual_mean_salinity()
    # upsampled footprint for the vacuity check: any-shallow ocean at a generous depth
    up_frac, up_ocean = build_shallow_fraction(depth_max_m=1e9)  # ~ full coastal ocean
    for _sp, _stage, dmax, ceil, gate, fname in STAGES:
        frac, ocean = build_shallow_fraction(depth_max_m=dmax)
        m = percid_stage_map(frac, ocean, sal, sal_ceiling=ceil, sal_gate=gate)
        if not vacuity_ok(m, np.where(up_ocean, 1.0, 0.0), max_ratio=args.max_ratio):
            raise ValueError(f"{fname}: vacuity guard failed (empty or > {args.max_ratio} of footprint)")
        np.savetxt(maps_dir / fname, np.flipud(m), fmt="%.4f", delimiter=";")
        print(f"wrote {fname}: {int(np.sum(m > 0))} habitat cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE to implementer: `build_shallow_fraction` is called once per stage (6 EMODnet passes). If fetch latency dominates, cache the raw `(elev,lat_hi,lon_hi)` strips in Task 1 and pass a per-depth threshold — optional optimisation, not required for correctness. The script depends on the fine salinity climatology (Task 4); if run before it, it fails loudly on the missing file — that ordering is enforced by the ladder assembly (Task 5).

- [ ] **Step 6: Commit**

```bash
git add osmose/forcing/percid_habitat.py scripts/build_baltic_fine_percid_maps.py tests/test_percid_habitat.py
git commit -m "feat: real percid habitat maps (shallow-fraction + spawning salinity gate)"
```

---

### Task 3: Occupancy-map upsampler (block-replicate, built from scratch)

**Files:**
- Create: `osmose/forcing/grid_upsample.py`
- Create: `scripts/baltic_grid_upsample.py`
- Test: `tests/test_grid_upsample.py`

**Interfaces:**
- Produces: `block_replicate(arr, factor) -> NDArray` — each cell of `(nlat,nlon)` → `factor×factor` block, preserving values (incl. `-99`, `0`, weights). Output `(nlat*factor, nlon*factor)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grid_upsample.py
import numpy as np
from osmose.forcing.grid_upsample import block_replicate


def test_block_replicate_preserves_values_and_shape():
    a = np.array([[-99.0, 1.0], [0.0, 0.5]])
    out = block_replicate(a, 2)
    assert out.shape == (4, 4)
    assert np.array_equal(out[0:2, 0:2], np.full((2, 2), -99.0))
    assert np.array_equal(out[0:2, 2:4], np.ones((2, 2)))
    assert np.array_equal(out[2:4, 2:4], np.full((2, 2), 0.5))
```

- [ ] **Step 2: Run to verify fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_grid_upsample.py -q` — FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

```python
# osmose/forcing/grid_upsample.py
"""Block-replicate an (nlat,nlon) spatial array to (nlat*f, nlon*f). For OCCUPANCY
maps only (movement maps / masks: {-99, 0, weight}); NOT for absolute-biomass
fields (those must be conserved — see osmose/forcing/grid conservation regrid)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def block_replicate(arr: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    return np.repeat(np.repeat(np.asarray(arr, dtype=np.float64), factor, axis=0), factor, axis=1)
```

- [ ] **Step 4: Run to verify pass** — `... -q` → PASS (1 passed).

- [ ] **Step 5: Write the driver** (`scripts/baltic_grid_upsample.py`): reads every `data/baltic/maps/*.csv` via `pandas.read_csv(sep=';', header=None)` (do NOT flip — operate on-disk south→north), `block_replicate(.., 4)`, writes to `data/baltic-fine/maps/` with the SAME filename; also upsamples `data/baltic/grid/baltic_mask.csv`. Produces the 6 non-percid species' maps + a `*_upsampled.csv` copy of the 6 percid maps for the control rung.

```python
# scripts/baltic_grid_upsample.py
from pathlib import Path
import numpy as np, pandas as pd
from osmose.forcing.grid_upsample import block_replicate

SRC = Path("data/baltic/maps"); DST = Path("data/baltic-fine/maps"); DST.mkdir(parents=True, exist_ok=True)
PERCID = {"perch", "pikeperch"}


def main() -> int:
    for csv in sorted(SRC.glob("*.csv")):
        arr = pd.read_csv(csv, sep=";", header=None).values.astype(float)  # south->north on disk
        up = block_replicate(arr, 4)
        species = csv.stem.split("_")[0]
        # non-percid: canonical name; percid: control copy with _upsampled suffix
        name = csv.name if species not in PERCID else csv.stem + "_upsampled.csv"
        np.savetxt(DST / name, up, fmt="%.4f", delimiter=";")
    print(f"upsampled {len(list(SRC.glob('*.csv')))} maps to {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Commit**

```bash
git add osmose/forcing/grid_upsample.py scripts/baltic_grid_upsample.py tests/test_grid_upsample.py
git commit -m "feat: occupancy-map block-replicate upsampler (built from scratch)"
```

---

### Task 4: Conserving forcing regridder (CRITICAL — biomass conservation)

**Files:**
- Create: `osmose/forcing/conserve_regrid.py`
- Create: `scripts/build_baltic_fine_forcing.py`
- Test: `tests/test_conserve_regrid.py`

**Interfaces:**
- Produces: `split_conserve(field, factor) -> NDArray` — regrid an absolute-biomass `(…,nlat,nlon)` field to `(…,nlat*factor,nlon*factor)` such that **total sums are preserved** (each cell's value ÷ factor² into its sub-cells).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conserve_regrid.py
import numpy as np
from osmose.forcing.conserve_regrid import split_conserve


def test_total_biomass_conserved():
    f = np.array([[[4.0, 8.0], [0.0, 16.0]]])  # (1,2,2) tonnes/cell
    out = split_conserve(f, 2)
    assert out.shape == (1, 4, 4)
    assert np.isclose(out.sum(), f.sum())        # conserved
    assert np.allclose(out[0, 0:2, 0:2], 1.0)    # 4 / 4 sub-cells


def test_replication_would_not_conserve_regression():
    f = np.ones((1, 5, 5))
    assert np.isclose(split_conserve(f, 4).sum(), f.sum())  # NOT x16
```

- [ ] **Step 2: Run to verify fail** — FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

```python
# osmose/forcing/conserve_regrid.py
"""Conservative regrid for ABSOLUTE-biomass forcing fields (tonnes/cell). Each
coarse cell's mass is split equally across its factor**2 sub-cells so the global
total is preserved. Using block_replicate here instead would inflate total system
biomass by factor**2 (the ×16 bug the review caught)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def split_conserve(field: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    up = np.repeat(np.repeat(np.asarray(field, dtype=np.float64), factor, axis=-2), factor, axis=-1)
    return up / (factor * factor)
```

- [ ] **Step 4: Run to verify pass** — PASS (2 passed).

- [ ] **Step 5: Write the forcing builder** (`scripts/build_baltic_fine_forcing.py`):
  - `baltic_ltl_biomass.nc` (6 vars) and `baltic_predator_biomass.nc` (2 vars): load each `(24,40,50)` var, `split_conserve(var, 4)` → `(24,160,200)`, write to `data/baltic-fine/` with a per-var **total-conservation assert** (`np.isclose(fine.sum(), coarse.sum())`).
  - Bottom-salinity climatology at 4×: reuse `scripts/build_baltic_salinity_forcing.py`'s pipeline but with `GridSpec` = FINE (intensive field → the existing `regrid` from CMEMS ~2 km handles it; NOT split_conserve). Write `data/baltic-fine/baltic_salinity_bottom_climatology.nc` `(24,160,200)`.

```python
# scripts/build_baltic_fine_forcing.py
from pathlib import Path
import numpy as np, xarray as xr
from osmose.forcing.conserve_regrid import split_conserve
from osmose.forcing.grid import target_coords
from scripts.build_baltic_fine_grid import FINE

SRC = Path("data/baltic"); DST = Path("data/baltic-fine"); DST.mkdir(parents=True, exist_ok=True)
ABS_BIOMASS = ["baltic_ltl_biomass.nc", "baltic_predator_biomass.nc"]


def main() -> int:
    tlat, tlon = target_coords(FINE)
    for fname in ABS_BIOMASS:
        ds = xr.open_dataset(SRC / fname)
        out = {}
        for name, da in ds.data_vars.items():
            coarse = np.asarray(da.values)            # (24,40,50)
            fine = split_conserve(coarse, 4)          # (24,160,200)
            assert np.isclose(fine.sum(), coarse.sum()), f"{fname}:{name} biomass not conserved"
            out[name] = (["time", "latitude", "longitude"], fine)
        xr.Dataset(out, coords={"time": np.arange(coarse.shape[0]), "latitude": tlat,
                                "longitude": tlon}).to_netcdf(DST / fname)
        print(f"conserved-regrid {fname}: {list(ds.data_vars)}")
    # salinity climatology at 4x: delegate to the (intensive) salinity builder with FINE grid
    # (see scripts/build_baltic_salinity_forcing.py::build — call with config_dir pointing at a
    #  FINE grid config, or import its accumulate_climatology + regrid(clim, src_lat, src_lon, FINE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE to implementer: for the salinity climatology at 4×, import `accumulate_climatology` from `scripts/build_baltic_salinity_forcing.py`, then `regrid(clim, src_lat, src_lon, FINE)` + `resample_to_24` + `target_coords(FINE)` and write `baltic_salinity_bottom_climatology.nc`. Salinity is INTENSIVE → plain `regrid`, never `split_conserve`.

- [ ] **Step 6: Commit**

```bash
git add osmose/forcing/conserve_regrid.py scripts/build_baltic_fine_forcing.py tests/test_conserve_regrid.py
git commit -m "feat: conserving forcing regrid (absolute biomass preserved, no x16 inflation)"
```

---

### Task 5: Assemble the two 4× config variants

**Files:**
- Create: `scripts/build_baltic_fine_config.py`
- Create: `data/baltic-fine/` param CSVs (generated)
- Test: `tests/test_baltic_fine_config.py`

**Interfaces:**
- Consumes: the fine grid/mask (T1), maps (T2 real + T3 upsampled), forcing (T4).
- Produces: two loadable config variants — `data/baltic-fine/baltic_fine_upsampled_all-parameters.csv` and `..._real_all-parameters.csv` — identical except percid `movement.file.map*` targets.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baltic_fine_config.py
from pathlib import Path
import numpy as np
from osmose.config import OsmoseConfigReader
from osmose.engine.config import EngineConfig


def test_both_fine_variants_load_and_are_4x():
    for variant in ["upsampled", "real"]:
        p = sorted(Path("data/baltic-fine").glob(f"*{variant}_all-parameters*.csv"))[0]
        cfg = dict(OsmoseConfigReader().read(str(p)))
        assert cfg["grid.nlon"] == "200" and cfg["grid.nlat"] == "160"
        ec = EngineConfig.from_dict(cfg)      # constructs without error
        assert ec.n_species >= 8


def test_variants_differ_only_in_percid_maps():
    up = dict(OsmoseConfigReader().read(str(sorted(Path("data/baltic-fine").glob("*upsampled_all-parameters*.csv"))[0])))
    real = dict(OsmoseConfigReader().read(str(sorted(Path("data/baltic-fine").glob("*real_all-parameters*.csv"))[0])))
    diffs = {k for k in up if up.get(k) != real.get(k)}
    assert diffs, "variants must differ"
    assert all("perch" in up[k] or "pikeperch" in up[k] or "perch" in real.get(k, "") or "pikeperch" in real.get(k, "")
               for k in diffs), f"unexpected non-percid diffs: {diffs}"
```

- [ ] **Step 2: Run to verify fail** — FAIL (no config yet).

- [ ] **Step 3: Write the assembler** (`scripts/build_baltic_fine_config.py`): copy every `data/baltic/baltic_param-*.csv` to `data/baltic-fine/`, then rewrite: `grid.nlon=200`, `grid.nlat=160`, `grid.mask.file=grid/baltic_fine_mask.csv`, `grid.netcdf.file=baltic_fine_grid.nc`; repoint all `movement.file.map*`, LTL/predator/salinity forcing file keys to the `data/baltic-fine/` versions (paths stay relative). Emit two `*_all-parameters.csv` entrypoints that `#include` the shared params but set the percid `movement.file.map{13..18}` to `_upsampled.csv` (variant 1) vs the real `.csv` (variant 2). Match how `data/baltic/baltic_all-parameters.csv` aggregates the param files (open it and mirror the structure).

- [ ] **Step 4: Run to verify pass** — both variants load; diff only percid maps.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_baltic_fine_config.py data/baltic-fine tests/test_baltic_fine_config.py
git commit -m "feat: assemble 4x baltic-fine config (upsampled + real percid variants)"
```

---

### Task 6: Attribution-ladder runner + GO/NO-GO report

**Files:**
- Create: `scripts/baltic_fine_grid_ladder.py`
- Test: `tests/test_baltic_fine_ladder.py` (smoke)

**Interfaces:**
- Consumes: the coarse config (`data/baltic/`), the two fine variants (T5); `osmose.calibration.targets` for per-species targets.
- Produces: a table of per-species overshoot ratio (mean ± seed spread) across the 3 rungs + area ratio + high-weight guardrail, and a printed GO/NO-GO line.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_baltic_fine_ladder.py
import subprocess, sys


def test_ladder_runs_and_reports_three_rungs():
    out = subprocess.run([sys.executable, "scripts/baltic_fine_grid_ladder.py", "--nyear", "2", "--seeds", "1"],
                         capture_output=True, text=True, timeout=3600)
    assert out.returncode == 0, out.stderr
    lo = out.stdout.lower()
    assert "coarse" in lo and "4x-upsampled" in lo and "4x-real" in lo
    assert "perch" in lo and "pikeperch" in lo
    assert "go" in lo or "no-go" in lo
```

- [ ] **Step 2: Run to verify fail** — FAIL (script missing).

- [ ] **Step 3: Write the runner** (`scripts/baltic_fine_grid_ladder.py`):
  - `--nyear` (default 30), `--seeds` (default 5). For each rung config, run `PythonEngine().run_in_memory(cfg, seed=s).biomass()` over `range(seeds)` with `movement.randomseed.fixed=true`, `stochastic.mortality.randomseed.fixed=true`, `simulation.time.nyear=<nyear>`.
  - Overshoot ratio per species = late-window-mean biomass / target (`from osmose.calibration.targets import ...` — read `data/baltic/reference/biomass_targets.csv`; mirror how `osmose/calibration/losses.py` loads targets). Report mean ± std across seeds.
  - Print a table: species × {coarse, 4×-upsampled, 4×-real} overshoot ratios; the perch/pikeperch real-habitat **area ratio** (real habitat cells / upsampled footprint cells); and cod/herring/sprat as the high-weight guardrail.
  - GO/NO-GO line per §5: GO if rung-3 percid overshoot is materially below rung-2 AND toward single digits AND seed-std small (stable, not collapse); else NO-GO (report which).

```python
# scripts/baltic_fine_grid_ladder.py  (skeleton; fill per notes above)
import argparse
from pathlib import Path
import numpy as np
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

DET = {"movement.randomseed.fixed": "true", "stochastic.mortality.randomseed.fixed": "true"}
PERCIDS, HIGHW = ["perch", "pikeperch"], ["cod", "herring", "sprat"]
RUNGS = {
    "coarse": "data/baltic/baltic_all-parameters.csv",
    "4x-upsampled": "data/baltic-fine/baltic_fine_upsampled_all-parameters.csv",
    "4x-real": "data/baltic-fine/baltic_fine_real_all-parameters.csv",
}


def late_mean(series, frac=1 / 3):
    b = np.asarray(series, float); return float(np.mean(b[int(len(b) * (1 - frac)):]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nyear", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    from osmose.calibration.targets import load_biomass_targets  # confirm exact name in targets.py
    targets = load_biomass_targets("data/baltic/reference/biomass_targets.csv")
    results = {}
    for rung, path in RUNGS.items():
        base = dict(OsmoseConfigReader().read(path)); base.update(DET)
        base["simulation.time.nyear"] = str(args.nyear)
        per_sp = {sp: [] for sp in PERCIDS + HIGHW}
        for s in range(args.seeds):
            bio = PythonEngine().run_in_memory(dict(base), seed=s).biomass()
            for sp in per_sp:
                per_sp[sp].append(late_mean(bio[sp]) / targets[sp])
        results[rung] = {sp: (float(np.mean(v)), float(np.std(v))) for sp, v in per_sp.items()}
    print("species     " + "  ".join(f"{r:>16}" for r in RUNGS))
    for sp in PERCIDS + HIGHW:
        cells = "  ".join(f"{results[r][sp][0]:7.1f}±{results[r][sp][1]:4.1f}" for r in RUNGS)
        print(f"{sp:11} {cells}   {'PERCID' if sp in PERCIDS else 'high-weight'}")
    up, real = results["4x-upsampled"], results["4x-real"]
    cured = all(real[sp][0] < up[sp][0] * 0.5 and real[sp][0] < 10 and real[sp][1] < real[sp][0] * 0.5 for sp in PERCIDS)
    print("VERDICT:", "GO — real habitat materially damps percids, stably" if cured
          else "NO-GO — real habitat ~= upsampled (structural), or unstable/collapse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE to implementer: confirm the exact target-loader name/signature in `osmose/calibration/targets.py` (grep for the function `osmose/calibration/losses.py` calls) and the biomass-frame species-name keys (`.biomass()` returns a wide frame keyed by species NAME). The smoke test runs `--nyear 2 --seeds 1` to stay cheap; the real A/B is run by hand at the defaults. Record the ladder table + verdict in the branch and in memory.

- [ ] **Step 4: Run to verify pass** — PASS (script runs, prints 3 rungs + verdict).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_fine_grid_ladder.py tests/test_baltic_fine_ladder.py
git commit -m "feat: attribution-ladder runner + GO/NO-GO report"
```

---

## Post-plan verification (run before declaring done)

- [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q -k "bathymetry or percid_habitat or grid_upsample or conserve_regrid or baltic_fine"` — all green.
- [ ] `.venv/bin/ruff check osmose/ scripts/ tests/` and `.venv/bin/ruff format --check osmose/ scripts/ tests/` — clean.
- [ ] Confirm `data/baltic/` is untouched: `git status data/baltic/` shows nothing.
- [ ] Build the artefacts end-to-end (real network + engine): grid → forcing → percid maps → upsample → config → ladder at defaults. Record the ladder verdict.
- [ ] Update memory with the GO/NO-GO outcome (this is the deciding experiment for the full high-res build).
