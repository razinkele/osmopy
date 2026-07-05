# Baltic Bottom-Salinity Forcing + A/B — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real Baltic-grid bottom-salinity climatology NetCDF from the local full-depth CMEMS `so`, wire it into the Baltic config for the salinity gate, and run the Baltic A/B (gate off vs on) measuring the cod–percid effect.

**Architecture:** A streaming builder reads the 29 full-depth `so` year-files one at a time (memory-safe), bottom-extracts the deepest valid salinity per cell, accumulates a per-month seasonal climatology, regrids to the Baltic grid via the existing `osmose/forcing` primitives, gap-fills ocean NaN, and writes `(24, ny, nx)` salinity. The gate loads it via `movement.salinity.field.file`; a diagnostic runs Baltic gate-off vs gate-on.

**Tech Stack:** Python 3.12, NumPy, xarray, scipy.ndimage (nearest-fill), pytest. Reuses `osmose/forcing/grid.py` (`regrid`, `resample_to_24`, `target_coords`, `get_coords`, `load_ocean_mask`) + `osmose.maps.builder.GridSpec`.

## Global Constraints

- Branch: `baltic-salinity-forcing` (already created).
- Run everything with `.venv/bin/python` (system `python` may not exist).
- Line length 100; lint `.venv/bin/ruff check osmose/ tests/ scripts/` + `format --check` on touched files.
- **Memory-safe:** never load all 29 year-files at once (~55 GB). Stream one year-file at a time and accumulate per-month sum + count.
- **Depth: bottom salinity** = deepest valid (non-NaN) level per cell. CMEMS `so` depth is ascending (index 0 = 0.5 m shallowest, index -1 = deepest); land/below-seafloor = NaN.
- **Time: seasonal climatology** = per-month mean across years → 12 months → `resample_to_24` → 24 steps (cycled by `PhysicalData.get_grid` via `step % 24`).
- **Orientation (load-bearing):** `regrid` outputs latitude-descending `(nlat, nlon)` matching `grid.nc`; the salinity `[cell_y, cell_x]` must align with the engine's movement-map indexing. Baltic grid: `nlat=40` (cell_y=0 ≈ 65.85 N = north, cell_y=39 ≈ 54.15 N = south), `nlon=50`. **Real Baltic salinity is LOW in the north (Bothnian Bay ~2–3 psu) and HIGH in the south-west (Arkona/Kattegat ~15–25 psu)** — so the produced field must increase from low cell_y (north) to high cell_y (south). Test this.
- **Gap-fill:** every OSMOSE **ocean** cell must have finite salinity (nearest-valid fill); land cells (cod map = 0) may stay NaN.
- **Inert-by-default:** wiring adds `movement.salinity.field.file`/`varname` to the Baltic config but keeps `movement.salinity.gate.enabled=false`. The A/B enables it.
- **Honest framing:** spatial-realism correction, NOT a percid-overshoot fix (it raises percid biomass, if anything worsening the ×38–96 overshoot). The A/B reports the spatial effect with this framing.
- Data: `data/cmems_cache/cmems_downloads/baltic_phy_monthly_reanalysis_so_*.nc` (29 files, 1993–2021, dims time=12, depth=36, latitude=744, longitude=746). Cod=sp0, perch=sp4, pikeperch=sp5.

---

## File Structure

- `scripts/build_baltic_salinity_forcing.py` — the climatology builder (pure helpers + streaming `main`).
- `data/baltic/baltic_salinity_bottom_climatology.nc` — generated artifact (varname `salinity`, shape (24, 40, 50)).
- `data/baltic/baltic_param-movement.csv` — add the two `movement.salinity.field.*` keys.
- `scripts/baltic_salinity_gate_diagnostic.py` — the Baltic A/B.
- `tests/test_baltic_salinity_forcing.py` — all tests.

---

## Task 1: Pure helpers (bottom-extract + ocean-NaN fill)

**Files:**
- Create: `scripts/build_baltic_salinity_forcing.py` (helpers only this task)
- Test: `tests/test_baltic_salinity_forcing.py`

**Interfaces:**
- Produces:
  - `bottom_extract(arr)` — `arr` shape `(nt, ndepth, nlat, nlon)`, depth ascending → returns `(nt, nlat, nlon)` deepest-valid salinity per cell; all-NaN columns (land) → NaN.
  - `fill_ocean_nan(field, ocean_mask)` — `field` `(nt, ny, nx)`, `ocean_mask` `(ny, nx)` bool (True=ocean) → copy with NaN *ocean* cells filled by the nearest finite value (per time step); land cells untouched.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_baltic_salinity_forcing.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_baltic_salinity_forcing as bld  # noqa: E402


def test_bottom_extract_deepest_valid():
    # 1 time, 3 depths, 2x2. depth ascending. NaN = below seafloor / land.
    nan = np.nan
    arr = np.array([[  # time 0
        [[10.0, 20.0], [nan, 5.0]],   # depth 0 (shallow)
        [[11.0, 21.0], [nan, 6.0]],   # depth 1
        [[12.0, nan ], [nan, 7.0]],   # depth 2 (deep)
    ]])  # shape (1,3,2,2)
    out = bld.bottom_extract(arr)      # (1,2,2)
    assert out.shape == (1, 2, 2)
    assert out[0, 0, 0] == 12.0        # deepest valid = depth2
    assert out[0, 0, 1] == 21.0        # depth2 is NaN -> deepest valid = depth1
    assert np.isnan(out[0, 1, 0])      # all-NaN column -> NaN
    assert out[0, 1, 1] == 7.0


def test_fill_ocean_nan_nearest():
    field = np.array([[[1.0, np.nan, 3.0]]])   # (1,1,3), middle ocean cell NaN
    ocean = np.array([[True, True, True]])
    out = bld.fill_ocean_nan(field, ocean)
    assert np.isfinite(out[0, 0, 1])           # filled
    assert out[0, 0, 1] in (1.0, 3.0)          # nearest finite neighbor
    assert out[0, 0, 0] == 1.0 and out[0, 0, 2] == 3.0  # existing values untouched


def test_fill_ocean_nan_leaves_land():
    field = np.array([[[np.nan, 2.0]]])        # (1,1,2)
    ocean = np.array([[False, True]])          # cell 0 is land
    out = bld.fill_ocean_nan(field, ocean)
    assert np.isnan(out[0, 0, 0])              # land NaN untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py -k "bottom_extract or fill_ocean" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_baltic_salinity_forcing'`.

- [ ] **Step 3: Write the helpers**

Create `scripts/build_baltic_salinity_forcing.py`:

```python
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
    rev_first = np.argmax(finite[:, ::-1, :, :], axis=1)   # (nt, nlat, nlon)
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
        idx = ndimage.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        nearest = f[tuple(idx)]
        f[nan_ocean] = nearest[nan_ocean]
        out[t] = f
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py -k "bottom_extract or fill_ocean" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_baltic_salinity_forcing.py tests/test_baltic_salinity_forcing.py
git commit -m "feat: bottom-extract + ocean-NaN-fill helpers for salinity forcing"
```

---

## Task 2: Streaming builder + generate the artifact

**Files:**
- Modify: `scripts/build_baltic_salinity_forcing.py` (add the climatology accumulator + `main`)
- Create (generated): `data/baltic/baltic_salinity_bottom_climatology.nc`
- Test: `tests/test_baltic_salinity_forcing.py`

**Interfaces:**
- Consumes: `bottom_extract` (Task 1); `regrid`, `resample_to_24`, `get_coords`, `load_ocean_mask` (osmose.forcing.grid); `target_coords` (osmose.forcing.grid); `GridSpec.from_config` (osmose.maps.builder); `OsmoseConfigReader` (osmose.config.reader).
- Produces:
  - `accumulate_climatology(so_files) -> tuple[NDArray, NDArray, NDArray]` — streams the files, returns `(clim (12, src_lat, src_lon), src_lat, src_lon)` per-month bottom-salinity mean across years (NaN where no data).
  - `build(config_dir, out_path) -> Path` — full pipeline → writes the NetCDF, returns its path.
  - `main(argv=None) -> int` — CLI.

- [ ] **Step 1: Write the failing test (streaming climatology on a synthetic 2-file set)**

Append to `tests/test_baltic_salinity_forcing.py`:

```python
import xarray as xr


def _month_times(path, months):
    # distinct year per file inferred from the filename's 4-digit year
    import re

    yr = int(re.search(r"(\d{4})", Path(path).name).group(1))
    return [np.datetime64(f"{yr}-{m:02d}-15") for m in months]


def _write_so_file(path, months, depth, lat, lon, fill):
    # so shape (time, depth, lat, lon); `fill(m)` returns a (depth,lat,lon) array
    data = np.stack([fill(m) for m in months], axis=0)
    ds = xr.Dataset(
        {"so": (["time", "depth", "latitude", "longitude"], data)},
        coords={
            "time": _month_times(path, months),
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds.to_netcdf(path)


def test_accumulate_climatology_two_years(tmp_path):
    depth = np.array([0.5, 5.0])           # ascending
    lat = np.array([60.0, 59.0])           # descending
    lon = np.array([15.0, 16.0])
    # year A: bottom (depth1) salinity = 10 in Jan; year B: 20 in Jan -> clim Jan = 15
    fA = tmp_path / "so_2001.nc"
    fB = tmp_path / "so_2002.nc"
    _write_so_file(fA, [1], depth, lat, lon, lambda m: np.full((2, 2, 2), 10.0))
    _write_so_file(fB, [1], depth, lat, lon, lambda m: np.full((2, 2, 2), 20.0))
    clim, slat, slon = bld.accumulate_climatology([str(fA), str(fB)])
    assert clim.shape == (12, 2, 2)
    assert np.allclose(clim[0], 15.0)       # Jan mean across the two years
    assert np.all(np.isnan(clim[1]))        # Feb had no data
    np.testing.assert_array_equal(slat, lat)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py::test_accumulate_climatology_two_years -v`
Expected: FAIL — `accumulate_climatology` not defined.

- [ ] **Step 3: Implement the accumulator + build + main**

Append to `scripts/build_baltic_salinity_forcing.py`:

```python
def accumulate_climatology(so_files):
    """Stream year-files, return (clim (12, nlat, nlon), src_lat, src_lon).
    Per-month mean of bottom salinity across years; NaN where no month had data."""
    import xarray as xr

    sum_ = cnt = None
    src_lat = src_lon = None
    for f in so_files:
        ds = xr.open_dataset(f)
        try:
            bottom = bottom_extract(ds["so"].values)            # (12, nlat, nlon)
            months = ds["time"].dt.month.values
            if sum_ is None:
                nlat, nlon = bottom.shape[1:]
                sum_ = np.zeros((12, nlat, nlon), dtype=np.float64)
                cnt = np.zeros((12, nlat, nlon), dtype=np.float64)
                src_lat = ds["latitude"].values
                src_lon = ds["longitude"].values
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
    from osmose.forcing.grid import get_coords, load_ocean_mask, regrid, resample_to_24, target_coords
    from osmose.maps.builder import GridSpec
    import xarray as xr

    cfg = OsmoseConfigReader().read(sorted(Path(config_dir).glob("*all-parameters*.csv"))[0])
    grid = GridSpec.from_config(cfg)
    so_files = sorted(glob.glob(str(Path(config_dir).parent / "cmems_cache" / "cmems_downloads"
                                     / "baltic_phy_monthly_reanalysis_so_*.nc")))
    if not so_files:
        raise FileNotFoundError("no full-depth so files found under data/cmems_cache/cmems_downloads")

    clim, src_lat, src_lon = accumulate_climatology(so_files)         # (12, src_lat, src_lon)
    regridded = regrid(clim, src_lat, src_lon, grid)                  # (12, ny, nx)
    field24 = resample_to_24(regridded)                              # (24, ny, nx)

    grid_nc = Path(config_dir) / "baltic_grid.nc"
    ocean_mask = load_ocean_mask(grid_nc)
    if ocean_mask is not None:
        field24 = fill_ocean_nan(field24, ocean_mask)

    tlat, tlon = target_coords(grid)
    out = xr.Dataset(
        {"salinity": (["time", "latitude", "longitude"], field24)},
        coords={"time": np.arange(24), "latitude": tlat, "longitude": tlon},
        attrs={"title": "OSMOSE Baltic bottom-salinity climatology (from CMEMS so)",
               "units": "PSU", "source": "CMEMS cmems_mod_bal_phy_my_P1M-m, deepest-valid level",
               "conventions": "Latitude descending (north to south) to match grid.nc"},
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
```

- [ ] **Step 4: Run the accumulator test**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py::test_accumulate_climatology_two_years -v`
Expected: PASS.

- [ ] **Step 5: Generate the real artifact**

Run: `.venv/bin/python scripts/build_baltic_salinity_forcing.py`
Expected: prints `wrote data/baltic/baltic_salinity_bottom_climatology.nc shape=(24, 40, 50) salinity range X-Y PSU` with a realistic range (roughly 2–25+ PSU). This streams all 29 files (a few minutes, memory-safe). `data/baltic/baltic_salinity_bottom_climatology.nc` now exists.

- [ ] **Step 6: Orientation + shape test against the real artifact**

Append to `tests/test_baltic_salinity_forcing.py`:

```python
def test_artifact_shape_and_orientation():
    p = Path("data/baltic/baltic_salinity_bottom_climatology.nc")
    if not p.exists():
        pytest.skip("run scripts/build_baltic_salinity_forcing.py first")
    with xr.open_dataset(p) as ds:
        s = ds["salinity"].values                 # (24, 40, 50)
    assert s.shape == (24, 40, 50)
    # every finite value is a plausible Baltic salinity
    fin = s[np.isfinite(s)]
    assert fin.min() >= 0.0 and fin.max() <= 40.0
    # ORIENTATION: mean salinity of the northern rows (low cell_y) must be LOWER
    # than the southern rows (high cell_y). North Baltic (Bothnian Bay) is ~2-3
    # psu; south-west (Arkona/Kattegat) is ~15-25 psu. If flipped, this fails.
    t0 = s[0]
    north = np.nanmean(t0[:10, :])                 # cell_y 0-9 (≈ 66-63 N)
    south = np.nanmean(t0[30:, :])                 # cell_y 30-39 (≈ 57-54 N)
    assert south > north, f"orientation wrong: north={north:.2f} !< south={south:.2f}"
```

- [ ] **Step 7: Run it**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py::test_artifact_shape_and_orientation -v`
Expected: PASS (south saltier than north → orientation correct).

- [ ] **Step 8: Commit**

```bash
git add scripts/build_baltic_salinity_forcing.py tests/test_baltic_salinity_forcing.py data/baltic/baltic_salinity_bottom_climatology.nc
git commit -m "feat: streaming Baltic bottom-salinity climatology builder + generated artifact"
```

---

## Task 3: Wire the Baltic config + smoke test

**Files:**
- Modify: `data/baltic/baltic_param-movement.csv`
- Test: `tests/test_baltic_salinity_forcing.py`

**Interfaces:**
- Consumes: the generated NetCDF; `_load_salinity_gate` / `_movement_salinity_weight` (already in the engine).

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_baltic_salinity_forcing.py`:

```python
from types import SimpleNamespace


def test_gate_loads_real_field_and_grades():
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import _load_salinity_gate
    from osmose.engine.processes.movement import _movement_salinity_weight

    cfg = OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv")
    if "movement.salinity.field.file" not in cfg:
        pytest.skip("config not wired yet")
    cfg["movement.salinity.gate.enabled"] = "true"
    cfg["movement.salinity.gate.species.enabled.sp0"] = "true"
    n_sp = int(float(cfg["simulation.nspecies"]))
    enabled, mask, lo, hi, field = _load_salinity_gate(cfg, n_sp)
    assert enabled and field is not None and not field.is_constant
    assert field.get_grid(0).shape == (40, 50)          # (ny, nx)
    ecfg = SimpleNamespace(
        salinity_gate_enabled=True, salinity_field=field,
        salinity_gate_s_low=lo, salinity_gate_s_high=hi,
    )
    w = _movement_salinity_weight(ecfg, SimpleNamespace(ny=40, nx=50), 0)
    finite = w[np.isfinite(w)]
    # GRADED: not all-0, not all-1 — a real spread of weights across [0,1]
    assert 0.0 < finite.mean() < 1.0
    assert (finite > 0).any() and (finite < 1).any()
```

- [ ] **Step 2: Run to verify it skips/fails (config not wired)**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py::test_gate_loads_real_field_and_grades -v`
Expected: SKIP (`config not wired yet`) — the key isn't in the config until Step 3.

- [ ] **Step 3: Wire the config**

Add to `data/baltic/baltic_param-movement.csv` (the file uses `;`-separated `key;value` lines — match that format):

```
movement.salinity.field.file;baltic_salinity_bottom_climatology.nc
movement.salinity.field.varname;salinity
```

Do NOT add `movement.salinity.gate.enabled` (it stays `false` by schema default → inert). The A/B enables it at runtime.

Verify the file path resolves: `_load_salinity_gate` resolves `movement.salinity.field.file` via `_require_file(..., _cfg_dir(cfg))`, and `_cfg_dir` is the config directory (`data/baltic`), so `baltic_salinity_bottom_climatology.nc` (written there in Task 2) resolves.

- [ ] **Step 4: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py::test_gate_loads_real_field_and_grades -v`
Expected: PASS — the gate loads the real (24,40,50) field and `_movement_salinity_weight` returns a graded weight grid (mean strictly between 0 and 1).

- [ ] **Step 5: Confirm inert-by-default still holds**

Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -q`
Expected: PASS — the two new `movement.salinity.field.*` keys are schema-registered (from the gate feature) and the gate is still off by default, so Baltic loads warning-free.

- [ ] **Step 6: Commit**

```bash
git add data/baltic/baltic_param-movement.csv tests/test_baltic_salinity_forcing.py
git commit -m "feat: wire real bottom-salinity field into Baltic config (gate inert by default)"
```

---

## Task 4: Baltic A/B diagnostic + run

**Files:**
- Create: `scripts/baltic_salinity_gate_diagnostic.py`
- Test: `tests/test_baltic_salinity_forcing.py`

**Interfaces:**
- Consumes: the wired Baltic config; `PythonEngine().run_in_memory(...).biomass()` (species-name-keyed wide frame, as in `tests/test_recruitment_ceiling.py`).

- [ ] **Step 1: Write the failing unit test for the metric**

Append to `tests/test_baltic_salinity_forcing.py`:

```python
import baltic_salinity_gate_diagnostic as abdiag  # noqa: E402


def test_late_mean_basic():
    series = np.array([100.0, 200.0, 300.0, 400.0])
    # late third = last ~1 element (400) ; helper uses last third by default
    assert abdiag.late_mean(series, frac=0.5) == pytest.approx(350.0)  # mean of last 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py::test_late_mean_basic -v`
Expected: FAIL — `ModuleNotFoundError: baltic_salinity_gate_diagnostic`.

- [ ] **Step 3: Write the diagnostic**

Create `scripts/baltic_salinity_gate_diagnostic.py`:

```python
"""A/B diagnostic: Baltic salinity gate OFF vs ON (real bottom-salinity field).

Reports percid (perch, pikeperch) late-window biomass off vs on, and cod's
biomass, so the spatial-realism effect (cod excluded from low-salinity coastal
cells -> less cod predation on percids there -> percid biomass UP) is visible.
NOT an overshoot fix: raising percids would if anything worsen the overshoot.
See docs/superpowers/specs/2026-07-04-baltic-salinity-forcing-design.md.
"""

from __future__ import annotations

import argparse

import numpy as np


def late_mean(series, frac: float = 1.0 / 3.0) -> float:
    b = np.asarray(series, dtype=np.float64)
    n = len(b)
    return float(np.mean(b[int(n * (1.0 - frac)):]))


def run_ab(config_path: str, cod_index: int = 0) -> dict:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine

    base = OsmoseConfigReader().read(config_path)

    def _late(cfg, sp):
        return late_mean(PythonEngine().run_in_memory(dict(cfg), seed=0).biomass()[sp].to_numpy())

    off = {sp: _late(base, sp) for sp in ("cod", "perch", "pikeperch")}
    on_cfg = dict(base)
    on_cfg["movement.salinity.gate.enabled"] = "true"
    on_cfg[f"movement.salinity.gate.species.enabled.sp{cod_index}"] = "true"
    on = {sp: _late(on_cfg, sp) for sp in ("cod", "perch", "pikeperch")}
    return {"off": off, "on": on}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Baltic salinity-gate A/B (real field).")
    ap.add_argument("--config", default="data/baltic/baltic_all-parameters.csv")
    ap.add_argument("--cod-index", type=int, default=0)
    args = ap.parse_args(argv)
    r = run_ab(args.config, args.cod_index)
    print("late-window biomass (t)   OFF        ON        delta%")
    for sp in ("cod", "perch", "pikeperch"):
        o, n = r["off"][sp], r["on"][sp]
        d = (n - o) / o * 100.0 if o else float("nan")
        print(f"  {sp:10s} {o:12.1f} {n:12.1f} {d:+7.1f}%")
    print("\nNOTE: gating cod out of low-salinity coastal cells is a SPATIAL-REALISM")
    print("correction. Higher percid biomass here means less cod predation in the")
    print("refuge — it is NOT an overshoot fix (raising percids worsens overshoot).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the unit test**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py::test_late_mean_basic -v`
Expected: PASS.

- [ ] **Step 5: Run the real A/B**

Run: `.venv/bin/python scripts/baltic_salinity_gate_diagnostic.py`
Expected: prints the OFF vs ON late-window biomass table for cod/perch/pikeperch with delta%. This runs Baltic twice (~a few minutes each). **Record the numbers** — the expected direction is percid biomass UP with the gate on (cod sheltered out of the coastal refuge). Report faithfully whatever it shows; a null or unexpected result is a valid finding, not a failure.

- [ ] **Step 6: Commit**

```bash
git add scripts/baltic_salinity_gate_diagnostic.py tests/test_baltic_salinity_forcing.py
git commit -m "feat: Baltic salinity-gate A/B diagnostic (real bottom-salinity field)"
```

---

## Task 5: Full-suite gate + lint

**Files:** none new (verification).

- [ ] **Step 1: Lint + format**

Run: `.venv/bin/ruff check osmose/ tests/ scripts/` and `.venv/bin/ruff format --check osmose/ tests/ scripts/`
Expected: clean on touched files (fix only touched files; note any pre-existing unformatted ones).

- [ ] **Step 2: Run the feature + related suites**

Run: `.venv/bin/python -m pytest tests/test_baltic_salinity_forcing.py tests/test_salinity_gate.py tests/test_engine_config_validation.py -q`
Expected: all PASS.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint + final verification for Baltic salinity forcing"
```

---

## Self-Review Notes (author, against the spec)

- **Spec §3 bottom salinity / climatology / regrid / gap-fill:** Task 1 (bottom-extract, fill), Task 2 (streaming climatology + regrid + resample + fill + write). Memory-safety (§Global) is honored by the per-file streaming accumulator (never loads all 29 files).
- **Spec §3 orientation (load-bearing):** Task 2 Step 6 tests south-saltier-than-north against the real artifact.
- **Spec §4.2 config wiring, inert-by-default:** Task 3 (field keys added, gate stays off).
- **Spec §4.3 smoke test:** Task 3 (loads real (24,40,50) field, graded weight).
- **Spec §4.4 A/B:** Task 4 (cod/perch/pikeperch off vs on).
- **Spec §5 honest framing:** baked into the diagnostic's printed note and Task 4 Step 5.
- **Known integration point named, not a placeholder:** the `.biomass()[sp]` accessor (Task 4) is copied from the proven harness in `tests/test_recruitment_ceiling.py`; species names cod/perch/pikeperch verified in `data/baltic/baltic_param-species.csv`.
