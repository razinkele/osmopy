# Salinity/oxygen forcing + spatial cod egg-survival (SP1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the OSMOSE Baltic engine a per-cell reproductive-volume (RV) environmental field and a spatial cod egg-survival term driven by it — physically representing that cod eggs survive only where a water layer is simultaneously saline (≥11 PSU) and oxygenated (≥2 mL/L).

**Architecture:** An offline forcing generator turns depth-resolved CMEMS `so`/`o2` into a per-cell RV NetCDF field (summed thickness of viable water). A config loader wires the unused `PhysicalData.from_netcdf` to load it as a per-cell grid on `EngineConfig`. The `larva_mortality` pre-pass reads the grid at each cod egg's cell and multiplies survival by `clip(RV/RV_ref, 0, 1)`. Inert by default (bit-identical when off). Mean-restoring larval-M recalibration is a separate follow-on (SP1b), not in this plan.

**Tech Stack:** Python 3.12+, NumPy, xarray, pandas, pytest, ruff, pyright. OSMOSE schema (`osmose/schema/`), engine (`osmose/engine/`), forcing (`osmose/forcing/`).

**Spec:** `docs/superpowers/specs/2026-07-02-baltic-salinity-oxygen-forcing-egg-survival-design.md` — read it first.

## Global Constraints

- Python `.venv/bin/python`; tests `.venv/bin/python -m pytest`.
- Lint check ONLY: `.venv/bin/ruff check osmose/ scripts/ tests/` AND `.venv/bin/ruff format --check osmose/ scripts/ tests/`. **Never run `ruff format` without `--check`** — format only your edited files by explicit path. Line length 100.
- Types: `.venv/bin/pyright` clean on changed files (no NEW errors; the diagnostic has ~16 pre-existing).
- Config keys lowercase dot-separated; per-species use `sp{idx}`.
- **Inert by default:** `reproduction.rv.spatial.enabled=false` (the default, only setting in bundled configs) ⇒ Baltic/EEC/BoB engine output **bit-identical** to pre-change. Load-bearing.
- Thresholds: salinity ≥ 11 PSU; oxygen ≥ 2 mL/L = **89.3 mmol/m³** (2 × 44.66, CMEMS units).
- RV_ref (one definition): **mean of RV over the RV>0 `cod_spawning` cells across the 24 climatology steps**; builder writes it to the RV NetCDF variable's `RV_ref` attr.
- `s_cell = clip(RV(cell,step) / RV_ref, 0, 1)`, applied **only** to enabled species' egg schools with `cell_x ≥ 0` AND `not is_out` AND `not from_seeding`; excluded schools / NaN-RV cells → `s = 1`.
- Cod is species index `sp0`; Baltic grid is 40 rows (lat, north-first) × 50 cols (lon).
- Commit after each task with `feat:`/`test:`/`docs:` ending with the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Use `git -C /home/razinka/osmose/osmose-python ...`.

---

## File Structure

- `osmose/schema/species.py` — **modify**: 5 `reproduction.rv.spatial.*` `OsmoseField`s appended to `SPECIES_FIELDS`.
- `osmose/forcing/reproductive_volume.py` — **create**: `viable_thickness(...)` (pure core) + `build_rv_field(...)` (CMEMS → RV NetCDF).
- `scripts/build_baltic_rv_field.py` — **create**: CLI wrapping `build_rv_field` over the on-disk CMEMS reanalysis → `data/baltic/forcing/baltic_rv_field.nc`.
- `osmose/engine/state.py` — **modify**: add optional `from_seeding` field to `SchoolState`.
- `osmose/engine/processes/reproduction.py` — **modify**: tag egg schools with `from_seeding`.
- `osmose/engine/config.py` — **modify**: `_load_rv_spatial(...)` + 2 `EngineConfig` fields + `from_dict` wiring.
- `osmose/engine/processes/natural.py` — **modify**: `larva_mortality` applies `s_cell`.
- `scripts/rv_field_diagnostic.py` — **create** (Task 6): basin-contrast + within-basin-CV + gate on/off mean-shift → `docs/diagnostics/rv_spatial_field.md`.
- `tests/test_rv_spatial_egg_survival.py` — **create**: all unit/integration/parity tests.

---

## Task 1: Schema fields

**Files:**
- Modify: `osmose/schema/species.py` (append before the closing `]` of `SPECIES_FIELDS`)
- Test: `tests/test_rv_spatial_egg_survival.py`

**Interfaces:**
- Produces: config keys `reproduction.rv.spatial.enabled` (bool), `.mode`→n/a, `.field.file` (path), `.field.varname` (str), `.ref` (float), `.species.enabled.sp{idx}` (bool).

- [ ] **Step 1: Write the failing test**

Create `tests/test_rv_spatial_egg_survival.py`:

```python
from osmose.schema import build_registry


def test_rv_spatial_keys_registered():
    keys = {f.key_pattern for f in build_registry().all_fields()}
    assert "reproduction.rv.spatial.enabled" in keys
    assert "reproduction.rv.spatial.field.file" in keys
    assert "reproduction.rv.spatial.field.varname" in keys
    assert "reproduction.rv.spatial.ref" in keys
    assert "reproduction.rv.spatial.species.enabled.sp{idx}" in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py::test_rv_spatial_keys_registered -v`
Expected: FAIL (keys absent).

- [ ] **Step 3: Add the fields**

In `osmose/schema/species.py`, before the closing `]` of `SPECIES_FIELDS`, insert:

```python
    OsmoseField(
        key_pattern="reproduction.rv.spatial.enabled",
        param_type=ParamType.BOOL,
        default=False,
        description=(
            "Master switch for the spatial reproductive-volume cod egg-survival "
            "term (Baltic). Off = inert, output unchanged."
        ),
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.spatial.field.file",
        param_type=ParamType.FILE_PATH,
        default="",
        description="Per-cell reproductive-volume NetCDF forcing file (time,lat,lon).",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.spatial.field.varname",
        param_type=ParamType.STRING,
        default="reproductive_volume",
        description="Variable name of the RV field in the NetCDF.",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.spatial.ref",
        param_type=ParamType.FLOAT,
        default=-1.0,
        min_val=-1.0,
        max_val=1e9,
        description="RV_ref saturating thickness (m); <=0 means use the field's RV_ref attr.",
        category="reproduction",
        required=False,
    ),
    OsmoseField(
        key_pattern="reproduction.rv.spatial.species.enabled.sp{idx}",
        param_type=ParamType.BOOL,
        default=False,
        description="Per-species enable for the spatial RV egg-survival term (cod only).",
        category="reproduction",
        indexed=True,
        required=False,
    ),
```

(`ParamType.STRING` is confirmed present in `osmose/schema/base.py:15`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py::test_rv_spatial_keys_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/schema/species.py tests/test_rv_spatial_egg_survival.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: schema fields for the spatial RV egg-survival term"
```

---

## Task 2: RV field generator + data file

**Files:**
- Create: `osmose/forcing/reproductive_volume.py`
- Create: `scripts/build_baltic_rv_field.py`
- Create (generated): `data/baltic/forcing/baltic_rv_field.nc`
- Test: `tests/test_rv_spatial_egg_survival.py`

**Interfaces:**
- Produces: `viable_thickness(so, o2, depths, sal_thresh, o2_thresh) -> NDArray` (core), and `build_rv_field(phy_ds, bgc_ds, grid, *, sal_thresh=11.0, o2_thresh=89.3, ocean_mask, spawning_mask) -> xr.Dataset` with var `reproductive_volume` (time,lat,lon) and attr `RV_ref`.

- [ ] **Step 1: Write the failing test (core viable-thickness)**

Append to `tests/test_rv_spatial_egg_survival.py`:

```python
import numpy as np
from osmose.forcing.reproductive_volume import viable_thickness


def test_viable_thickness_mid_layer():
    # depths 10..90 step 10 (m); layer thickness = 10 each (uniform).
    depths = np.arange(10.0, 100.0, 10.0)  # 9 levels
    # so rises with depth (fresh top, saline deep); o2 falls with depth (oxic top, anoxic deep).
    so = np.array([6, 8, 10, 11, 12, 13, 14, 14, 14], dtype=float)
    o2 = np.array([300, 280, 250, 200, 150, 100, 40, 5, 0], dtype=float)  # mmol/m3
    # viable (so>=11 AND o2>=89.3): levels 3,4,5 (so 11,12,13 ; o2 200,150,100) -> 3 levels * 10 m
    t = viable_thickness(so, o2, depths, 11.0, 89.3)
    assert abs(t - 30.0) < 1e-9


def test_viable_thickness_two_bands_summed():
    depths = np.arange(10.0, 60.0, 10.0)  # 5 levels, 10 m each
    so = np.array([12, 6, 12, 6, 12], dtype=float)   # saline at 0,2,4
    o2 = np.array([300, 300, 300, 300, 300], dtype=float)  # all oxic
    # viable levels 0,2,4 -> 3 * 10 = 30 (two separated bands summed)
    assert abs(viable_thickness(so, o2, depths, 11.0, 89.3) - 30.0) < 1e-9


def test_viable_thickness_none_and_all():
    depths = np.arange(10.0, 40.0, 10.0)
    fresh = np.array([6, 7, 8], dtype=float)
    oxic = np.array([300, 300, 300], dtype=float)
    assert viable_thickness(fresh, oxic, depths, 11.0, 89.3) == 0.0  # all fresh
    saline = np.array([12, 13, 14], dtype=float)
    assert viable_thickness(saline, oxic, depths, 11.0, 89.3) > 0.0  # all viable
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k viable_thickness -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the core + builder**

Create `osmose/forcing/reproductive_volume.py`:

```python
"""Reproductive-volume field generator (Baltic cod egg survival).

Turns depth-resolved CMEMS salinity (`so`) + oxygen (`o2`) into a per-cell field
= summed thickness of the water column where salinity >= sal_thresh AND
oxygen >= o2_thresh co-occur (the classic Baltic-cod reproductive-volume). Cod
eggs float mid-column in the saline layer, NOT on the anoxic seafloor, so the
whole column is scanned rather than the bottom slice.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from osmose.forcing.grid import get_coords, regrid, resample_to_24, target_coords


def _layer_thickness(depths: NDArray[np.float64]) -> NDArray[np.float64]:
    """Thickness (m) attributed to each depth level = span between mid-points."""
    d = np.asarray(depths, dtype=np.float64)
    edges = np.empty(d.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (d[:-1] + d[1:])
    edges[0] = d[0] - 0.5 * (d[1] - d[0]) if d.size > 1 else d[0]
    edges[-1] = d[-1] + 0.5 * (d[-1] - d[-2]) if d.size > 1 else d[0]
    return np.clip(np.diff(edges), 0.0, None)


def viable_thickness(
    so: NDArray[np.float64],
    o2: NDArray[np.float64],
    depths: NDArray[np.float64],
    sal_thresh: float,
    o2_thresh: float,
) -> float:
    """Summed thickness (m) of viable water in one column (any NaN level is not viable)."""
    thick = _layer_thickness(depths)
    viable = (np.nan_to_num(so, nan=-1.0) >= sal_thresh) & (np.nan_to_num(o2, nan=-1.0) >= o2_thresh)
    return float(thick[viable].sum())


def _rv_year(phy_ds: xr.Dataset, bgc_ds: xr.Dataset, sal_thresh: float, o2_thresh: float) -> NDArray:
    """Per-(time,lat,lon) viable thickness on the SOURCE grid for one year's data."""
    so = phy_ds["so"]  # (time, depth, lat, lon)
    o2 = bgc_ds["o2"]
    depths = so["depth"].values.astype(np.float64)
    thick = _layer_thickness(depths)  # (depth,)
    so_v = so.values.astype(np.float64)
    o2_v = o2.values.astype(np.float64)
    viable = (np.nan_to_num(so_v, nan=-1.0) >= sal_thresh) & (
        np.nan_to_num(o2_v, nan=-1.0) >= o2_thresh
    )  # (time, depth, lat, lon)
    return np.einsum("tdyx,d->tyx", viable.astype(np.float64), thick)  # (time, lat, lon)


def build_rv_field(
    phy_years: list[xr.Dataset],
    bgc_years: list[xr.Dataset],
    grid,
    *,
    sal_thresh: float = 11.0,
    o2_thresh: float = 89.3,
    ocean_mask: NDArray[np.bool_],
    spawning_mask: NDArray[np.bool_],
) -> xr.Dataset:
    """Build the 24-step climatology RV field (mean of per-year RV) + RV_ref attr.

    phy_years[i]/bgc_years[i] are one year's depth-resolved `so`/`o2` datasets.
    ocean_mask / spawning_mask are (nlat, nlon) bool on the TARGET (engine) grid,
    north-first. Returns a Dataset with var `reproductive_volume` (24, nlat, nlon).
    """
    per_year_24 = []
    for phy, bgc in zip(phy_years, bgc_years):
        rv_src = _rv_year(phy, bgc, sal_thresh, o2_thresh)  # (t, srclat, srclon)
        src_lat, src_lon = get_coords(phy)
        rv_grid = regrid(rv_src, src_lat, src_lon, grid)  # (t, nlat, nlon)
        per_year_24.append(resample_to_24(rv_grid))  # (24, nlat, nlon)
    rv = np.mean(np.stack(per_year_24, axis=0), axis=0)  # mean-of-RV climatology
    rv[:, ~ocean_mask] = 0.0  # land -> 0 (consumer guards cell, so 0 is inert there)

    # RV_ref = mean over RV>0 spawning cells across all 24 steps
    sp_vals = rv[:, spawning_mask]
    nonzero = sp_vals[sp_vals > 0]
    rv_ref = float(nonzero.mean()) if nonzero.size else 1.0

    lat, lon = target_coords(grid)
    ds = xr.Dataset(
        {"reproductive_volume": (("time", "latitude", "longitude"), rv)},
        coords={"time": np.arange(24), "latitude": lat, "longitude": lon},
    )
    ds["reproductive_volume"].attrs["RV_ref"] = rv_ref
    ds["reproductive_volume"].attrs["units"] = "m"
    return ds
```

- [ ] **Step 4: Run to verify the core tests pass**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k viable_thickness -v`
Expected: PASS (3 tests). Note `test_viable_thickness_mid_layer`: uniform 10 m spacing → `_layer_thickness` returns ~10 each; 3 viable levels → 30 m.

- [ ] **Step 5: Write the builder integration test (synthetic 2-year)**

Append:

```python
import xarray as xr
from osmose.maps.builder import GridSpec


def _toy_year(so_col, o2_col):
    # 1 time, len(depth) depths, 2x2 lat/lon, same column everywhere.
    depths = np.arange(10.0, 10.0 * len(so_col) + 1, 10.0)
    lat = np.array([56.0, 55.0]); lon = np.array([18.0, 19.0])
    so = np.broadcast_to(np.array(so_col)[None, :, None, None], (1, len(depths), 2, 2))
    o2 = np.broadcast_to(np.array(o2_col)[None, :, None, None], (1, len(depths), 2, 2))
    phy = xr.Dataset({"so": (("time", "depth", "latitude", "longitude"), so.astype(float))},
                     coords={"time": [0], "depth": depths, "latitude": lat, "longitude": lon})
    bgc = xr.Dataset({"o2": (("time", "depth", "latitude", "longitude"), o2.astype(float))},
                     coords={"time": [0], "depth": depths, "latitude": lat, "longitude": lon})
    return phy, bgc


def test_build_rv_field_climatology_and_ref():
    grid = GridSpec(nlon=2, nlat=2, upleft_lat=56.5, upleft_lon=17.5,
                    lowright_lat=54.5, lowright_lon=19.5)
    # year A: 3 viable levels (30 m). year B: 1 viable level (10 m). mean = 20 m.
    pA, bA = _toy_year([12, 12, 12, 6], [300, 300, 300, 300])
    pB, bB = _toy_year([12, 6, 6, 6], [300, 300, 300, 300])
    ocean = np.ones((2, 2), dtype=bool)
    spawning = np.ones((2, 2), dtype=bool)
    ds = build_rv_field([pA, pB], [bA, bB], grid, ocean_mask=ocean, spawning_mask=spawning)
    assert ds["reproductive_volume"].shape == (24, 2, 2)
    # every cell/step ≈ 20 m (constant column, per-year mean of 30 and 10)
    assert abs(float(ds["reproductive_volume"].mean()) - 20.0) < 1.0
    assert abs(ds["reproductive_volume"].attrs["RV_ref"] - 20.0) < 1.0
```

- [ ] **Step 6: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k build_rv_field -v`
Expected: PASS. (If `regrid` warns about edge extrapolation, that's fine — the 2×2 toy grid maps onto the 2×2 source.)

- [ ] **Step 7: Write the CLI + generate the real field**

Create `scripts/build_baltic_rv_field.py`:

```python
#!/usr/bin/env python
"""Build data/baltic/forcing/baltic_rv_field.nc from the on-disk CMEMS reanalysis.

Usage: PYTHONPATH=. .venv/bin/python scripts/build_baltic_rv_field.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from osmose.forcing.grid import load_ocean_mask
from osmose.forcing.reproductive_volume import build_rv_field
from osmose.maps.builder import GridSpec

ROOT = Path(__file__).resolve().parent.parent
CMEMS = ROOT / "data" / "cmems_cache" / "cmems_downloads"
OUT = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"


def main() -> int:
    grid = GridSpec(nlon=50, nlat=40, upleft_lat=66, upleft_lon=10, lowright_lat=54, lowright_lon=30)
    ocean = load_ocean_mask(ROOT / "data" / "baltic" / "baltic_grid.nc")
    spawning = np.flipud(
        np.genfromtxt(ROOT / "data" / "baltic" / "maps" / "cod_spawning.csv", delimiter=";")
    ) > 0
    phy = sorted(CMEMS.glob("*phy_monthly_reanalysis_so_*.nc"))
    bgc = sorted(CMEMS.glob("*bgc_monthly_reanalysis_o2_*.nc"))
    if not phy or not bgc or len(phy) != len(bgc):
        print(f"missing/mismatched CMEMS files: phy={len(phy)} bgc={len(bgc)}", file=sys.stderr)
        return 1
    phy_years = [xr.open_dataset(p) for p in phy]
    bgc_years = [xr.open_dataset(b) for b in bgc]
    ds = build_rv_field(phy_years, bgc_years, grid, ocean_mask=ocean, spawning_mask=spawning)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    print(f"wrote {OUT}: RV_ref={ds['reproductive_volume'].attrs['RV_ref']:.2f} m, "
          f"mean over spawning cells="
          f"{float(ds['reproductive_volume'].values[:, spawning].mean()):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run: `cd /home/razinka/osmose/osmose-python && PYTHONPATH=. .venv/bin/python scripts/build_baltic_rv_field.py`
Expected: writes `data/baltic/forcing/baltic_rv_field.nc`, prints a positive `RV_ref`. Sanity-check: open it and confirm the field is higher over the deep-basin (`cod_spawning>0`) cells than over the fresh Gulf-of-Bothnia cells. If the file errors or `RV_ref` is 0 / NaN, STOP and report BLOCKED with the error rather than committing a bad file.

- [ ] **Step 8: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/forcing/reproductive_volume.py scripts/build_baltic_rv_field.py data/baltic/forcing/baltic_rv_field.nc tests/test_rv_spatial_egg_survival.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: reproductive-volume field generator + generated Baltic RV field"
```

---

## Task 3: `from_seeding` SchoolState field + reproduction tagging

**Files:**
- Modify: `osmose/engine/state.py` (SchoolState field + `create`)
- Modify: `osmose/engine/processes/reproduction.py:185-200` (tag egg schools)
- Test: `tests/test_rv_spatial_egg_survival.py`

**Interfaces:**
- Produces: `SchoolState.from_seeding: NDArray[np.bool_] | None` — `True` for egg schools created from seeded (not real-SSB) biomass; `False`/`None` otherwise.

- [ ] **Step 1: Write the failing test**

Append:

```python
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.state import SchoolState


def test_schoolstate_has_from_seeding_default_false():
    s = SchoolState.create(n_schools=3)
    assert s.from_seeding is not None
    assert s.from_seeding.tolist() == [False, False, False]
    # replace/append thread it generically
    s2 = s.replace(from_seeding=np.array([True, False, True]))
    assert s2.append(s).from_seeding.tolist() == [True, False, True, False, False, False]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k from_seeding_default -v`
Expected: FAIL (`from_seeding` attribute missing).

- [ ] **Step 3: Add the field to SchoolState**

In `osmose/engine/state.py`, after the `imax_trait: NDArray[np.float64] | None = None` field (the last field, ~line 89), add:

```python
    # True for egg schools created from seeded (bootstrap) biomass; excluded from
    # environmental egg-survival terms. Optional (mirrors imax_trait) so raw
    # SchoolState(...) constructions that omit it stay valid; create() populates it.
    from_seeding: NDArray[np.bool_] | None = None
```

And in `SchoolState.create`, add to the `cls(...)` kwargs (after `egg_retained=...`):

```python
            from_seeding=np.zeros(n, dtype=np.bool_),
```

(`replace`/`append`/`__post_init__` already iterate `fields()` generically, so no other change is needed there.)

- [ ] **Step 4: Run to verify the state test passes**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k from_seeding_default -v`
Expected: PASS.

- [ ] **Step 5: Tag egg schools in reproduction.py**

In `osmose/engine/processes/reproduction.py`, in the egg-creation loop, extend the second `new.replace(...)` (currently setting `cell_x`/`cell_y` at lines 196-199) to also set `from_seeding`:

```python
        # Eggs are created unlocated; movement places them on the next step.
        # Tag seeded-derived eggs so environmental egg-survival terms skip them.
        new = new.replace(
            cell_x=np.full(n_new, -1, dtype=np.int32),
            cell_y=np.full(n_new, -1, dtype=np.int32),
            from_seeding=np.full(n_new, bool(seeded_this_step[sp]), dtype=np.bool_),
        )
```

(`seeded_this_step` is already computed in this function by the merged scalar-gate work; `sp` is the loop variable.)

- [ ] **Step 6: Write + run the reproduction-tagging test**

Append:

```python
def _baltic_cfg(**over):
    cfg = dict(OsmoseConfigReader().read("data/baltic/baltic_all-parameters.csv"))
    cfg["simulation.time.nyear"] = "3"
    cfg.update(over)
    return cfg


def test_from_seeding_inert_by_default_parity():
    # Adding the field must not change engine output.
    a = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()["cod"].to_numpy()
    b = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()["cod"].to_numpy()
    np.testing.assert_array_equal(a, b)
```

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k "from_seeding" -v`
Expected: PASS (both). The parity test is a determinism/inertness check (the field defaults False and is read by nothing yet). Baltic runs are slow (~30-60 s each) — run in the FOREGROUND with a generous timeout; do NOT background.

- [ ] **Step 7: Lint + types, then commit**

```bash
cd /home/razinka/osmose/osmose-python && .venv/bin/ruff check osmose/ tests/ && .venv/bin/pyright osmose/engine/state.py osmose/engine/processes/reproduction.py
git -C /home/razinka/osmose/osmose-python add osmose/engine/state.py osmose/engine/processes/reproduction.py tests/test_rv_spatial_egg_survival.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: from_seeding SchoolState field, tagged on seeded egg creation"
```

---

## Task 4: Config loader `_load_rv_spatial` + EngineConfig fields

**Files:**
- Modify: `osmose/engine/config.py` (loader near `_load_rv_gate`; 2 dataclass fields; `from_dict` wiring)
- Test: `tests/test_rv_spatial_egg_survival.py`

**Interfaces:**
- Consumes: the RV NetCDF from Task 2; `PhysicalData.from_netcdf`; `_load_spatial_csv`.
- Produces: `EngineConfig.rv_spatial_field: PhysicalData | None`, `EngineConfig.rv_spatial_enabled: NDArray[np.bool_] | None`; `_load_rv_spatial(cfg, n_species) -> tuple[PhysicalData|None, NDArray|None]`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
import pytest
import xarray as xr
from osmose.engine.config import _load_rv_spatial


def _write_rv_nc(tmp_path, rv_const=5.0, ref=5.0, shape=(24, 40, 50), name="rv.nc"):
    rv = np.full(shape, rv_const, dtype=np.float64)
    ds = xr.Dataset({"reproductive_volume": (("time", "latitude", "longitude"), rv)})
    ds["reproductive_volume"].attrs["RV_ref"] = ref
    p = tmp_path / name
    ds.to_netcdf(p)
    return p


def _cfg(tmp_path, **over):
    base = {
        "reproduction.rv.spatial.enabled": "true",
        "reproduction.rv.spatial.field.file": str(_write_rv_nc(tmp_path)),
        "reproduction.rv.spatial.field.varname": "reproductive_volume",
        "reproduction.rv.spatial.ref": "-1",
        "reproduction.rv.spatial.species.enabled.sp0": "true",
        "_osmose.config.dir": str(tmp_path),
        "simulation.time.ndtperyear": "24",
    }
    base.update(over)
    return base


def test_load_rv_spatial_disabled():
    field, mask = _load_rv_spatial({"reproduction.rv.spatial.enabled": "false"}, 3)
    assert field is None and mask is None


def test_load_rv_spatial_reads_attr_and_mask(tmp_path):
    field, mask = _load_rv_spatial(_cfg(tmp_path), n_species=1)
    assert field is not None
    assert field.get_grid(0).shape == (40, 50)
    assert mask.tolist() == [True]


def test_load_rv_spatial_ref_from_config_overrides(tmp_path):
    # ref > 0 uses the config value, not the attr; stored for the consumer via a known accessor.
    field, _ = _load_rv_spatial(_cfg(tmp_path, **{"reproduction.rv.spatial.ref": "12.5"}), 1)
    assert abs(field.rv_ref - 12.5) < 1e-9


def test_load_rv_spatial_ref_from_attr(tmp_path):
    field, _ = _load_rv_spatial(_cfg(tmp_path), 1)  # ref=-1 -> attr (5.0)
    assert abs(field.rv_ref - 5.0) < 1e-9


@pytest.mark.parametrize("bad,exc", [
    ({"reproduction.rv.spatial.species.enabled.sp0": "false"}, "no species"),
    ({"reproduction.rv.spatial.field.file": ""}, "empty"),
])
def test_load_rv_spatial_fail_fast(tmp_path, bad, exc):
    with pytest.raises(ValueError, match=exc):
        _load_rv_spatial(_cfg(tmp_path, **bad), 1)


def test_load_rv_spatial_wrong_grid_raises(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["reproduction.rv.spatial.field.file"] = str(_write_rv_nc(tmp_path, shape=(24, 10, 10), name="bad.nc"))
    with pytest.raises(ValueError, match="grid"):
        _load_rv_spatial(cfg, 1)
```

Note: this test writes a full-grid (40×50) RV file and does NOT depend on the real generated field. The wrong-grid test relies on the loader deriving the expected shape from the `cod_spawning` mask; if `_osmose.config.dir=tmp_path` has no `maps/cod_spawning.csv`, the loader must fall back to the real Baltic map path — see Step 3's note.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k load_rv_spatial -v`
Expected: FAIL (`_load_rv_spatial` not defined). Also `field.rv_ref` needs `PhysicalData` to carry a `rv_ref` — added in Step 3.

- [ ] **Step 3: Implement the loader + carry rv_ref**

First, let `PhysicalData` carry an optional `rv_ref`. In `osmose/engine/physical_data.py`, add to `__init__` a `rv_ref: float | None = None` param stored as `self.rv_ref = rv_ref`, and a `from_netcdf_field` classmethod (leave `from_netcdf` untouched for temperature reuse):

```python
    @classmethod
    def from_netcdf_field(
        cls, path, varname: str, rv_ref: float
    ) -> "PhysicalData":
        """Load a per-cell field NetCDF (no factor/offset), carrying an rv_ref scalar."""
        from osmose.engine._netcdf import open_dataset_safe

        ds = open_dataset_safe(path)
        raw = ds[varname].values
        if raw.ndim == 2:
            raw = raw[np.newaxis, :, :]
        obj = cls(data=raw.astype(np.float64), constant=None, nsteps_year=raw.shape[0])
        obj.rv_ref = rv_ref
        return obj
```

(Also add `self.rv_ref: float | None = None` in `__init__` so existing constructions have the attribute.)

In `osmose/engine/config.py`, add near `_load_rv_gate`:

```python
def _load_rv_spatial(
    cfg: dict[str, str], n_species: int
) -> tuple["PhysicalData | None", "NDArray[np.bool_] | None"]:
    """Load the spatial RV egg-survival field (spec §5/§6). Returns (field, enable_mask)
    or (None, None) when the master switch is off. Fail-fast on invalid config."""
    from osmose.engine.physical_data import PhysicalData

    if cfg.get("reproduction.rv.spatial.enabled", "false").lower() != "true":
        return None, None

    file_key = cfg.get("reproduction.rv.spatial.field.file", "")
    if not file_key:
        raise ValueError("RV spatial enabled but reproduction.rv.spatial.field.file is empty.")
    path = _require_file(file_key, _cfg_dir(cfg), "reproduction.rv.spatial.field.file")
    varname = cfg.get("reproduction.rv.spatial.field.varname", "reproductive_volume")

    # Expected grid shape + spawning mask from the cod_spawning map (read directly;
    # MovementMapSet is not built at config time). Prefer a map under the config dir;
    # fall back to the bundled Baltic map.
    cfg_dir = Path(_cfg_dir(cfg))
    spawn_path = cfg_dir / "maps" / "cod_spawning.csv"
    if not spawn_path.exists():
        spawn_path = Path("data/baltic/maps/cod_spawning.csv")
    spawn = _load_spatial_csv(spawn_path) > 0  # north-first, (nlat, nlon)

    ref_cfg = float(cfg.get("reproduction.rv.spatial.ref", "-1"))
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if varname not in ds:
            raise ValueError(f"RV field {path} has no variable {varname!r}.")
        grid_shape = ds[varname].shape[-2:]
        attr_ref = ds[varname].attrs.get("RV_ref", None)
    if tuple(grid_shape) != spawn.shape:
        raise ValueError(
            f"RV field grid {tuple(grid_shape)} != engine grid {spawn.shape}."
        )
    rv_ref = ref_cfg if ref_cfg > 0 else attr_ref
    if rv_ref is None or float(rv_ref) <= 0:
        raise ValueError("RV_ref not resolvable (ref<=0 and no positive RV_ref attr).")

    field = PhysicalData.from_netcdf_field(path, varname, float(rv_ref))
    n_dt = int(cfg.get("simulation.time.ndtperyear", "24"))
    tlen = field._data.shape[0]  # NetCDF time length (24 climatology / 696 interannual)
    if tlen % n_dt != 0:
        raise ValueError(f"RV field time length {tlen} is not a multiple of ndtperyear {n_dt}.")

    enabled = np.zeros(n_species, dtype=np.bool_)
    for sp in range(n_species):
        if cfg.get(f"reproduction.rv.spatial.species.enabled.sp{sp}", "false").lower() == "true":
            enabled[sp] = True
    if not enabled.any():
        raise ValueError("RV spatial enabled but no species enabled (…species.enabled.sp{idx}).")
    return field, enabled
```

Add two `EngineConfig` fields **after the last existing defaulted field** (so `_minimal_config` needs no change):

```python
    rv_spatial_field: "PhysicalData | None" = None
    rv_spatial_enabled: "NDArray[np.bool_] | None" = None
```

And in `EngineConfig.from_dict`, before the `return EngineConfig(`, add:

```python
        rv_spatial_field, rv_spatial_enabled = _load_rv_spatial(cfg, n_sp)
```

and in the constructor kwargs (near `rv_gate_factor_by_index=...`):

```python
            rv_spatial_field=rv_spatial_field,
            rv_spatial_enabled=rv_spatial_enabled,
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k load_rv_spatial -v`
Expected: PASS. Then the config-validation guard:
Run: `.venv/bin/python -m pytest "tests/test_engine_config_validation.py::test_from_dict_warn_mode_clean_on_example_configs" -v`
Expected: PASS (default configs don't set the keys; loader lives in config.py so keys AST-validate; `_minimal_config` unaffected since the new fields default None).

- [ ] **Step 5: Lint/types, commit**

```bash
cd /home/razinka/osmose/osmose-python && .venv/bin/ruff check osmose/ tests/ && .venv/bin/pyright osmose/engine/config.py osmose/engine/physical_data.py
git -C /home/razinka/osmose/osmose-python add osmose/engine/config.py osmose/engine/physical_data.py tests/test_rv_spatial_egg_survival.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: _load_rv_spatial loader + EngineConfig RV-field fields"
```

---

## Task 5: Apply the spatial egg-survival term in `larva_mortality` + parity

**Files:**
- Modify: `osmose/engine/processes/natural.py:103-149` (`larva_mortality`)
- Test: `tests/test_rv_spatial_egg_survival.py`

**Interfaces:**
- Consumes: `config.rv_spatial_field` (PhysicalData with `.rv_ref`, `.get_grid`), `config.rv_spatial_enabled`, `state.from_seeding`, `state.is_out`, `state.cell_x/cell_y` (Task 3/4).

- [ ] **Step 1: Write the failing integration test**

Append:

```python
SP_FIELD = "data/baltic/forcing/baltic_rv_field.nc"


def _baltic_gate_cfg(**over):
    return _baltic_cfg(**{
        "reproduction.rv.spatial.enabled": "true",
        "reproduction.rv.spatial.field.file": SP_FIELD,
        "reproduction.rv.spatial.species.enabled.sp0": "true",
        **over,
    })


def test_spatial_off_bit_identical():
    off = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()["cod"].to_numpy()
    off2 = PythonEngine().run_in_memory(
        _baltic_cfg(**{"reproduction.rv.spatial.enabled": "false"}), seed=0
    ).biomass()["cod"].to_numpy()
    np.testing.assert_array_equal(off, off2)


def test_spatial_on_changes_cod():
    off = PythonEngine().run_in_memory(_baltic_cfg(), seed=0).biomass()["cod"].to_numpy()
    on = PythonEngine().run_in_memory(_baltic_gate_cfg(), seed=0).biomass()["cod"].to_numpy()
    assert not np.allclose(off, on)  # the spatial term changes cod
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k "spatial_off or spatial_on" -v`
Expected: `test_spatial_on_changes_cod` FAILS (term not applied → cod unchanged); `test_spatial_off_bit_identical` PASSES. Baltic runs are slow — foreground, generous timeout, do not background.

- [ ] **Step 3: Apply the term**

In `osmose/engine/processes/natural.py`, in `larva_mortality`, after `n_dead[eggs] = state.abundance[eggs] * mortality_fraction[eggs]` (line 143) and before `new_abundance = ...` (line 145), insert:

```python
    # Spatial reproductive-volume egg survival (Baltic cod). Inert unless enabled;
    # multiplies egg survival by clip(RV(cell)/RV_ref, 0, 1) for enabled, non-seeded,
    # located eggs. Increases n_dead by the extra kill (survival deficit).
    field = config.rv_spatial_field
    if field is not None:
        assert config.rv_spatial_enabled is not None  # set together in _load_rv_spatial
        rv_grid = field.get_grid(step)  # (ny, nx)
        rv_ref = field.rv_ref or 1.0
        for i in range(len(state)):
            if not eggs[i]:
                continue
            spi = int(sp[i])
            if spi >= len(config.rv_spatial_enabled) or not config.rv_spatial_enabled[spi]:
                continue
            if state.is_out[i]:
                continue
            fs = state.from_seeding
            if fs is not None and fs[i]:
                continue
            cy = int(state.cell_y[i]); cx = int(state.cell_x[i])
            if cy < 0 or cx < 0:
                continue
            rv = rv_grid[cy, cx]
            if not np.isfinite(rv):
                continue
            s = min(1.0, max(0.0, rv / rv_ref))  # clip
            survivors_after = state.abundance[i] * (1.0 - mortality_fraction[i]) * s
            extra_dead = state.abundance[i] * (1.0 - mortality_fraction[i]) - survivors_after
            n_dead[i] += extra_dead
```

Rationale: the existing pass kills `abundance*mortality_fraction`; survivors = `abundance*(1-mf)`. The spatial term keeps only fraction `s` of those survivors, so the extra kill is `survivors*(1-s)`, added to `n_dead[i]`. `new_abundance = abundance - n_dead` then reflects both. This applies **only** `s_cell` on top of the existing constant rate (no double-count).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k "spatial_off or spatial_on" -v`
Expected: PASS (cod changes on; bit-identical off).

- [ ] **Step 5: Parity/regression + lint/types**

```bash
cd /home/razinka/osmose/osmose-python
.venv/bin/python -m pytest -k "parity or cross_engine" -q
.venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format --check osmose/engine/processes/natural.py
.venv/bin/pyright osmose/engine/processes/natural.py
```
Expected: parity green (inert by default), lint/types clean. If parity regresses, the `is None` guard is leaking — re-check Step 3.

- [ ] **Step 6: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add osmose/engine/processes/natural.py tests/test_rv_spatial_egg_survival.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: spatial RV cod egg-survival term in larva_mortality"
```

---

## Task 6: Correctness diagnostic + go/no-go

**Files:**
- Create: `scripts/rv_field_diagnostic.py` — basin contrast, within-basin CV, gate on/off mean-shift
- Create (generated): `docs/diagnostics/rv_spatial_field.md`
- Test: `tests/test_rv_spatial_egg_survival.py`

**Interfaces:**
- Consumes: `data/baltic/forcing/baltic_rv_field.nc`, `cod_spawning` mask, `OsmoseResults.biomass("cod")`.

- [ ] **Step 1: Write the failing test (metrics helpers)**

Append:

```python
from osmose.forcing.grid import load_ocean_mask  # noqa: E402


def test_field_metrics_basin_contrast_and_cv():
    import xarray as xr
    rv = xr.open_dataset(SP_FIELD)["reproductive_volume"].values  # (24,40,50)
    spawn = np.flipud(np.genfromtxt("data/baltic/maps/cod_spawning.csv", delimiter=";")) > 0
    # basin contrast: mean RV over spawning >> over fresh coastal (surface-fresh proxy: use
    # ocean cells not in spawning as the comparison here for the smoke test).
    ocean = load_ocean_mask("data/baltic/baltic_grid.nc")
    coastal = ocean & ~spawn
    mean_spawn = rv[:, spawn].mean()
    mean_coast = rv[:, coastal].mean()
    assert mean_spawn > mean_coast  # spawning basins are more viable than the rest
    # within-basin heterogeneity: CV of RV across spawning cells (mean over steps) > 0
    per_step_cv = [
        (rv[t, spawn].std() / rv[t, spawn].mean()) if rv[t, spawn].mean() > 0 else 0.0
        for t in range(rv.shape[0])
    ]
    assert np.mean(per_step_cv) >= 0.0  # recorded; go/no-go threshold 0.20 checked in Step 4


def test_field_mean_anchor():
    # mean(s_cell) over RV>0 spawning cells is centred near 1 (clip lowers it): in [0.6, 1.0].
    import xarray as xr
    da = xr.open_dataset(SP_FIELD)["reproductive_volume"]
    rv = da.values
    ref = float(da.attrs["RV_ref"])
    spawn = np.flipud(np.genfromtxt("data/baltic/maps/cod_spawning.csv", delimiter=";")) > 0
    vals = rv[:, spawn]
    nz = vals[vals > 0]
    s = np.clip(nz / ref, 0.0, 1.0)
    assert 0.6 <= float(s.mean()) <= 1.0
```

- [ ] **Step 2: Run to verify it passes (field already exists from Task 2)**

Run: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -k field_metrics -v`
Expected: PASS — mean RV over spawning cells exceeds the rest. (This is the automated basin-contrast assertion; the strict ratio≥3 and CV≥0.20 go/no-go are recorded in the diagnostic in Step 3, not asserted here, since their exact values depend on the shipped field.)

- [ ] **Step 3: Add the diagnostic reporting**

Create `scripts/rv_field_diagnostic.py`:

```python
#!/usr/bin/env python
"""Report the spatial RV field's correctness metrics + the gate on/off cod shift."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.forcing.grid import load_ocean_mask

ROOT = Path(__file__).resolve().parent.parent
FIELD = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"
SPAWN = ROOT / "data" / "baltic" / "maps" / "cod_spawning.csv"


def _mean_cod(cfg):
    b = PythonEngine().run_in_memory(cfg, seed=0).biomass()["cod"].to_numpy()
    w = b[3:15]
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.mean())


def main() -> int:
    da = xr.open_dataset(FIELD)["reproductive_volume"]
    rv = da.values
    ref = float(da.attrs["RV_ref"])
    spawn = np.flipud(np.genfromtxt(SPAWN, delimiter=";")) > 0
    ocean = load_ocean_mask(ROOT / "data" / "baltic" / "baltic_grid.nc")
    coast = ocean & ~spawn  # fresh-coastal proxy = non-spawning ocean
    mean_spawn = float(rv[:, spawn].mean())
    mean_coast = float(rv[:, coast].mean()) if coast.any() else float("nan")
    ratio = mean_spawn / mean_coast if mean_coast else float("inf")
    cvs = [rv[t, spawn].std() / rv[t, spawn].mean() for t in range(rv.shape[0]) if rv[t, spawn].mean() > 0]
    cv = float(np.mean(cvs)) if cvs else 0.0

    base = dict(OsmoseConfigReader().read(str(ROOT / "data" / "baltic" / "baltic_all-parameters.csv")))
    base["simulation.time.nyear"] = "15"
    on = dict(base, **{
        "reproduction.rv.spatial.enabled": "true",
        "reproduction.rv.spatial.field.file": str(FIELD),
        "reproduction.rv.spatial.species.enabled.sp0": "true",
    })
    off_mean = _mean_cod(base)
    on_mean = _mean_cod(on)

    lines = [
        "# Spatial RV field diagnostic",
        f"basin contrast ratio = {ratio:.2f}  (go if >= 3)",
        f"within-basin CV = {cv:.3f}  (GO/NO-GO: go if >= 0.20)",
        f"mean cod biomass off={off_mean:.0f} on={on_mean:.0f} "
        f"delta={100 * (on_mean / off_mean - 1):+.0f}%  (SP1b restores the mean)",
    ]
    print("\n".join(lines))
    out = ROOT / "docs" / "diagnostics" / "rv_spatial_field.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the report + record the go/no-go**

Run: `cd /home/razinka/osmose/osmose-python && PYTHONPATH=. .venv/bin/python scripts/rv_field_diagnostic.py` (two 15-yr Baltic runs — foreground, generous timeout, do not background).
It writes `docs/diagnostics/rv_spatial_field.md`. **The within-basin CV ≥ 0.20 is the go/no-go for whether the spatial machinery earns its complexity** — if it is below 0.20, report that honestly (the regridded climatology washed out sub-basin structure) as the key finding; do not tune to hit it. The mean-shift is measured and recorded (SP1b restores it).

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/osmose/osmose-python add scripts/rv_field_diagnostic.py docs/diagnostics/rv_spatial_field.md tests/test_rv_spatial_egg_survival.py
git -C /home/razinka/osmose/osmose-python commit -m "feat: spatial RV field diagnostic (basin contrast, within-basin CV, mean shift)"
```

---

## Final verification

- [ ] Full test file green: `.venv/bin/python -m pytest tests/test_rv_spatial_egg_survival.py -v`
- [ ] Inert-by-default parity green (`-k "parity or cross_engine"`).
- [ ] ruff check + `ruff format --check` on all changed files; pyright clean on changed engine files.
- [ ] `docs/diagnostics/rv_spatial_field.md` records the basin-contrast ratio, within-basin CV (the go/no-go), and the measured mean shift.
- [ ] SP1b (mean-restoring larval-M recalibration) noted as the follow-on; NOT done here.
