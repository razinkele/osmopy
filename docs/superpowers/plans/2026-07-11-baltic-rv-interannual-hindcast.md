# Baltic cod interannual reproductive-volume hindcast — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the stationary SP1 cod reproductive-volume egg-survival gate to a real 1993–2021 **interannual** field and run an honest A/B skill test (climatology vs interannual, shared `RV_ref`) of whether interannual RV forcing moves modeled cod toward observed ICES SSB.

**Architecture:** The engine already indexes an arbitrary-length RV field by `step % tlen` and the loader already validates `tlen % ndtperyear == 0` (comment: "24 climatology / 696 interannual"). So the work is: a new *concatenating* field builder, a build script + committed NetCDF, a loader fail-fast against silent temporal wrap, the observed-cod data, an offline correlation, and an A/B hindcast harness. No change to `natural.py` (the egg-survival term is field-agnostic) and no change to deployed config defaults (gate ships off).

**Tech Stack:** Python 3, numpy/xarray, `osmose.forcing.reproductive_volume`, `osmose.engine.config._load_rv_spatial`, `osmose.engine.PythonEngine`, `OsmoseResults`, pandas/scipy, pytest.

## Global Constraints

- **Deployed config behavior stays byte-identical, and `data/baltic/` is fully unchanged.** The gate ships **off** by default. The new interannual field is committed to a **separate** `data/baltic_rv/` dir — NOT under `data/baltic/` — so the demo generators (which `copytree` all of `data/baltic/` into every run) don't copy the 6 MB file into every baltic/baltic_a2 instantiation. The harness references it by absolute path. No config CSV edits, no preset, no default flip.
- **`build_rv_field` (climatology) stays byte-identical** — the A/B null arm and SP1's shipped behavior both depend on it. The new builder is a *separate* function.
- **Do NOT recalibrate larval-M** to fit the hindcast (A/B-skill-delta decision).
- **A/B shares one `RV_ref`** (the interannual field's), forced on both arms via `reproduction.rv.spatial.ref` — the fields do NOT auto-share it.
- **No CI gate on emergent hindcast outcomes** (biomass, skill delta) — non-reproducible across CI cores. Phases 0/3 are local documented runs.
- Species: sp0 = cod. Cod-only; percids are the accepted structural residual.
- Field grid: (696, 40, 50), north-first, land→NaN, `float32`. o₂ threshold 89.3 mmol/m³ (=2 mL/L), salinity 11 (SP1 defaults).
- Tests via `.venv/bin/python -m pytest`; lint `ruff check osmose/ tests/ scripts/` is not required for `scripts/` (out of CI scope) but `osmose/ tests/` must stay ruff-clean.

---

## File Structure

- **Modify** `osmose/forcing/reproductive_volume.py` — add `build_rv_field_interannual` (climatology `build_rv_field` untouched).
- **Modify** `scripts/build_baltic_rv_field.py` — add `--interannual` mode + tie-back assertion.
- **Create** `data/baltic_rv/baltic_rv_field_interannual.nc` — the 696-step field (committed, zlib-compressed ~2 MB; a NEW dir, not under `data/baltic/`, so no demo generator copies it).
- **Modify** `osmose/engine/config.py` — `_load_rv_spatial` fail-fast wrap guard.
- **Create** `docs/diagnostics/ices_cod_2732_observed.csv` — observed eastern-Baltic cod SSB+recruitment 1993–2021 (baked from the ICES 2023 assessment).
- **Create** `scripts/baltic_rv_cod_offline.py` — Phase 0 offline correlation → `docs/diagnostics/baltic_rv_cod_correlation.md`.
- **Create** `scripts/baltic_rv_hindcast.py` — Phase 3 A/B harness → `docs/diagnostics/baltic_rv_hindcast.md`.
- **Create** `tests/test_rv_interannual.py` — CI-safe unit tests (builder, guard, skill funcs).

---

## Task 1: `build_rv_field_interannual`

**Files:**
- Modify: `osmose/forcing/reproductive_volume.py`
- Test: `tests/test_rv_interannual.py`

**Interfaces:**
- Produces: `build_rv_field_interannual(phy_years, bgc_years, grid, *, sal_thresh=11.0, o2_thresh=89.3, ocean_mask, spawning_mask, start_year) -> xr.Dataset` with var `reproductive_volume` shape `(len(phy_years)*24, nlat, nlon)`, chronological, `RV_ref` + `start_year` + `units` attrs. Consumes the same helpers as `build_rv_field` (`_rv_year`, `regrid`, `resample_to_24`, `get_coords`, `target_coords`).

- [ ] **Step 1: Write the failing tests** (`tests/test_rv_interannual.py`)

```python
import numpy as np
import xarray as xr

from osmose.forcing.reproductive_volume import build_rv_field, build_rv_field_interannual
from osmose.maps.builder import GridSpec

# Real GridSpec (dx/dy are @property, computed from the corners — a plain stub lacks them and
# regrid()/target_coords() would AttributeError).
GRID = GridSpec(nlon=5, nlat=4, upleft_lat=65.5, upleft_lon=10.5, lowright_lat=54.5, lowright_lon=29.5)


def _fake_year(seed):
    # minimal (time=12, depth=3, lat=4, lon=5) so/o2 datasets on a source grid
    rng = np.random.default_rng(seed)
    depth = np.array([5.0, 20.0, 40.0])
    lat = np.linspace(54.5, 65.5, 4)
    lon = np.linspace(10.5, 29.5, 5)
    so = 6.0 + 8.0 * rng.random((12, 3, 4, 5))   # spans the 11-psu threshold
    o2 = 50.0 + 100.0 * rng.random((12, 3, 4, 5))  # spans the 89.3 threshold
    coords = {"time": np.arange(12), "depth": depth, "latitude": lat, "longitude": lon}
    phy = xr.Dataset({"so": (("time", "depth", "latitude", "longitude"), so)}, coords=coords)
    bgc = xr.Dataset({"o2": (("time", "depth", "latitude", "longitude"), o2)}, coords=coords)
    return phy, bgc


def _masks():
    ocean = np.ones((4, 5), dtype=bool)
    spawning = np.zeros((4, 5), dtype=bool)
    spawning[1:3, 1:4] = True
    return ocean, spawning


def test_interannual_shape_and_chronological():
    ph = [_fake_year(i) for i in range(3)]
    phy_years = [p for p, _ in ph]
    bgc_years = [b for _, b in ph]
    ocean, spawning = _masks()
    ds = build_rv_field_interannual(
        phy_years, bgc_years, GRID, ocean_mask=ocean, spawning_mask=spawning, start_year=1993
    )
    rv = ds["reproductive_volume"].values
    assert rv.shape == (3 * 24, 4, 5)              # concatenate, not average
    assert ds["reproductive_volume"].attrs["start_year"] == 1993
    # Year k's 24-step block equals that single year's standalone climatology-of-one build.
    for k in range(3):
        one = build_rv_field(
            [phy_years[k]], [bgc_years[k]], GRID, ocean_mask=ocean, spawning_mask=spawning
        )["reproductive_volume"].values
        block = rv[k * 24:(k + 1) * 24]
        np.testing.assert_allclose(np.nan_to_num(block), np.nan_to_num(one), rtol=1e-6)


def test_interannual_differs_from_climatology_mean():
    ph = [_fake_year(i) for i in range(3)]
    phy_years, bgc_years = [p for p, _ in ph], [b for _, b in ph]
    ocean, spawning = _masks()
    inter = build_rv_field_interannual(
        phy_years, bgc_years, GRID, ocean_mask=ocean, spawning_mask=spawning, start_year=1993
    )["reproductive_volume"].values
    clim = build_rv_field(
        phy_years, bgc_years, GRID, ocean_mask=ocean, spawning_mask=spawning
    )["reproductive_volume"].values
    # interannual is 72 steps; its per-year blocks are NOT all equal to the 24-step climatology
    assert inter.shape[0] == 72 and clim.shape[0] == 24
    assert not np.allclose(np.nan_to_num(inter[:24]), np.nan_to_num(inter[24:48]))


def test_climatology_builder_deterministic():
    # build_rv_field is not edited by this task; same inputs -> identical output (sanity).
    phy, bgc = _fake_year(7)
    ocean, spawning = _masks()
    a = build_rv_field([phy], [bgc], GRID, ocean_mask=ocean, spawning_mask=spawning)
    b = build_rv_field([phy], [bgc], GRID, ocean_mask=ocean, spawning_mask=spawning)
    np.testing.assert_array_equal(
        np.nan_to_num(a["reproductive_volume"].values),
        np.nan_to_num(b["reproductive_volume"].values),
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_interannual.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_rv_field_interannual'`.

- [ ] **Step 3: Implement `build_rv_field_interannual`** (append to `osmose/forcing/reproductive_volume.py`, after `build_rv_field`)

```python
def build_rv_field_interannual(
    phy_years: list[xr.Dataset],
    bgc_years: list[xr.Dataset],
    grid,
    *,
    sal_thresh: float = 11.0,
    o2_thresh: float = 89.3,
    ocean_mask: NDArray[np.bool_],
    spawning_mask: NDArray[np.bool_],
    start_year: int,
) -> xr.Dataset:
    """Build the CHRONOLOGICAL interannual RV field (per-year 24-step blocks concatenated).

    Same per-year viable-thickness metric and regridding as build_rv_field, but the years are
    stacked in order (year 0 = start_year, steps 0-23; year 1, steps 24-47; ...) instead of
    averaged. Returns var `reproductive_volume` (len(phy_years)*24, nlat, nlon), north-first,
    land -> NaN, with RV_ref (over RV>0 spawning cells across ALL steps) + start_year attrs.
    """
    per_year_24 = []
    for phy, bgc in zip(phy_years, bgc_years):
        rv_src = _rv_year(phy, bgc, sal_thresh, o2_thresh)
        src_lat, src_lon = get_coords(phy)
        rv_grid = regrid(rv_src, src_lat, src_lon, grid)
        per_year_24.append(resample_to_24(rv_grid))  # (24, nlat, nlon)
    rv = np.concatenate(per_year_24, axis=0).astype(np.float32)  # (nyear*24, nlat, nlon)
    rv[:, ~ocean_mask] = np.nan  # land -> NaN

    sp_vals = rv[:, spawning_mask]
    nonzero = sp_vals[sp_vals > 0]
    rv_ref = float(nonzero.mean()) if nonzero.size else 1.0

    lat, lon = target_coords(grid)
    ds = xr.Dataset(
        {"reproductive_volume": (("time", "latitude", "longitude"), rv)},
        coords={"time": np.arange(rv.shape[0]), "latitude": lat, "longitude": lon},
    )
    ds["reproductive_volume"].attrs["RV_ref"] = rv_ref
    ds["reproductive_volume"].attrs["start_year"] = int(start_year)
    ds["reproductive_volume"].attrs["units"] = "m"
    return ds
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_rv_interannual.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add osmose/forcing/reproductive_volume.py tests/test_rv_interannual.py
git commit -m "feat(forcing): build_rv_field_interannual (chronological per-year RV field)"
```

---

## Task 2: build script `--interannual` + generate & commit the field

**Files:**
- Modify: `scripts/build_baltic_rv_field.py`
- Create: `data/baltic_rv/baltic_rv_field_interannual.nc`

**Interfaces:**
- Consumes: Task 1's `build_rv_field_interannual`; the local 54 GB cache `data/cmems_cache/cmems_downloads/` (`*phy_monthly_reanalysis_so_*.nc`, `*bgc_monthly_reanalysis_o2_*.nc`); `docs/diagnostics/baltic_rv_fraction.csv` (offline series, for tie-back).
- Produces: the committed NetCDF.

> **Note:** not a TDD task — it runs the real builder over the local reanalysis cache and commits the artifact. Requires the 54 GB cache (not reproducible from a bare clone; documented).

- [ ] **Step 1: Add `--interannual` mode** to `scripts/build_baltic_rv_field.py`

Add near the top: `import argparse` and `from osmose.forcing.reproductive_volume import build_rv_field_interannual`. In `main()`, parse `--interannual`. When set: derive `start_year` from the first sorted phy filename (parse the 4-digit year), call `build_rv_field_interannual(..., start_year=start_year)`, write to `OUT_INTER = ROOT / "data/baltic_rv/baltic_rv_field_interannual.nc"` with `encoding={"reproductive_volume": {"dtype": "float32"}}`.

Tie-back assertion (before writing): compute the field's spawning-cell-mean per step, average to per-year (mean of each 24-block), load `docs/diagnostics/baltic_rv_fraction.csv` (`rv_fraction`), average to per-year, and assert `np.corrcoef(field_annual, offline_annual)[0,1] > 0.6` (raise with the value on failure). Print the correlation.

```python
# inside main(), when args.interannual:
years = sorted({int(p.name.split("_so_")[1][:4]) for p in phy})
start_year = years[0]
ds = build_rv_field_interannual(
    phy_years, bgc_years, grid, ocean_mask=ocean, spawning_mask=spawning, start_year=start_year
)
rv = ds["reproductive_volume"].values                      # (nyear*24, ny, nx)
n_year = rv.shape[0] // 24
field_annual = np.nan_to_num(rv[:, spawning]).reshape(n_year, 24, -1).mean(axis=(1, 2))
import pandas as pd
off = pd.read_csv(ROOT / "docs/diagnostics/baltic_rv_fraction.csv")
off["yr"] = pd.to_datetime(off["time"]).dt.year
off_annual = off.groupby("yr")["rv_fraction"].mean().reindex(range(start_year, start_year + n_year)).values
r = float(np.corrcoef(field_annual, off_annual)[0, 1])
# SOFT check only. The offline series is a bottom-slice areal FRACTION; the engine field is a
# full-column viable THICKNESS (different metrics — correlation is scale-invariant but not
# guaranteed high even for a correct field). The real correctness gate is Step 3 (shape +
# finite-fraction). A genuine orientation/np.flipud error shows up there as a degenerate/NaN-heavy
# field, NOT as a marginal correlation — do not "fix" orientation to force this number up.
if r < 0.3:
    print(f"WARNING: field vs offline annual corr={r:.3f} (<0.3) — inspect the field before use.")
else:
    print(f"tie-back OK-ish: field vs offline annual corr={r:.3f} (metrics differ; soft check).")
OUT_INTER = ROOT / "data" / "baltic_rv" / "baltic_rv_field_interannual.nc"
OUT_INTER.parent.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(
    OUT_INTER,
    encoding={"reproductive_volume": {"dtype": "float32", "zlib": True, "complevel": 4}},
)
print(f"wrote {OUT_INTER}: {rv.shape}, RV_ref={ds['reproductive_volume'].attrs['RV_ref']:.2f}")
return 0
```

- [ ] **Step 2: Generate the field**

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/build_baltic_rv_field.py --interannual`
Expected: prints the tie-back correlation (soft — the two metrics differ, so any positive corr is reassuring; a `<0.3` warning warrants inspection but does not block) and `wrote …baltic_rv_field_interannual.nc: (696, 40, 50), RV_ref=…`. The real correctness gate is **Step 3** (shape 696×40×50, positive RV_ref, high finite-fraction). If Step 3 shows a degenerate/NaN-heavy field, THEN suspect the `np.flipud` orientation gotcha.

- [ ] **Step 3: Sanity-check the file**

Run:
```bash
cd /home/razinka/osmopy && .venv/bin/python -c "
import xarray as xr, numpy as np
d = xr.open_dataset('data/baltic_rv/baltic_rv_field_interannual.nc')['reproductive_volume']
print('shape', d.shape, 'RV_ref', d.attrs['RV_ref'], 'start_year', d.attrs['start_year'])
print('finite frac', float(np.isfinite(d.values).mean()))
"
```
Expected: shape (696, 40, 50), positive RV_ref, start_year 1993.

- [ ] **Step 4: Commit the field + script**

```bash
git add scripts/build_baltic_rv_field.py data/baltic_rv/baltic_rv_field_interannual.nc
git commit -m "feat(baltic): build + commit interannual RV field (1993-2021, tie-back to offline series)"
```

---

## Task 3: loader fail-fast wrap guard

**Files:**
- Modify: `osmose/engine/config.py` (`_load_rv_spatial`)
- Test: `tests/test_rv_interannual.py`

**Interfaces:**
- Consumes: `_load_rv_spatial(cfg, n_species)` — already returns `(None, None)` when the gate is off.
- Produces: raises `ValueError` when the RV field is interannual (`tlen > ndtperyear`) and `nyear*ndtperyear > tlen` (a run that would silently wrap past the forcing period).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rv_interannual.py`)

```python
from pathlib import Path

import pytest


def _write_rv_nc(path, n_steps):
    lat = np.linspace(54.5, 65.5, 40)
    lon = np.linspace(10.5, 29.5, 50)
    rv = np.ones((n_steps, 40, 50), dtype=np.float32)
    ds = xr.Dataset(
        {"reproductive_volume": (("time", "latitude", "longitude"), rv)},
        coords={"time": np.arange(n_steps), "latitude": lat, "longitude": lon},
    )
    ds["reproductive_volume"].attrs["RV_ref"] = 10.0
    ds.to_netcdf(path)


def _rv_cfg(tmp_path, n_steps, nyear):
    from osmose.engine.config import _load_rv_spatial  # noqa: F401 (import path check)
    p = tmp_path / "rv.nc"
    _write_rv_nc(p, n_steps)
    return {
        "reproduction.rv.spatial.enabled": "true",
        "reproduction.rv.spatial.field.file": str(p),
        "reproduction.rv.spatial.species.enabled.sp0": "true",
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": str(nyear),
        "_osmose.config.dir": str(tmp_path),
    }


def test_wrap_guard_raises_when_run_exceeds_forcing(tmp_path, monkeypatch):
    monkeypatch.chdir("/home/razinka/osmopy")  # cod_spawning.csv fallback path
    from osmose.engine.config import _load_rv_spatial
    cfg = _rv_cfg(tmp_path, n_steps=696, nyear=30)  # 30*24=720 > 696 -> would wrap
    with pytest.raises(ValueError, match="wrap|exceed|forcing"):
        _load_rv_spatial(cfg, n_species=1)


def test_wrap_guard_ok_when_exact_and_climatology(tmp_path, monkeypatch):
    monkeypatch.chdir("/home/razinka/osmopy")
    from osmose.engine.config import _load_rv_spatial
    field, en = _load_rv_spatial(_rv_cfg(tmp_path, n_steps=696, nyear=29), 1)  # 696==696 ok
    assert field is not None and en[0]
    field2, _ = _load_rv_spatial(_rv_cfg(tmp_path, n_steps=24, nyear=50), 1)   # climatology cycles
    assert field2 is not None
```

(The tests rely on `_load_rv_spatial` reading `simulation.time.nyear`/`ndtperyear` from `cfg`. The `_osmose.config.dir` + a `maps/cod_spawning.csv` under it is not present, so the loader falls back to `data/baltic/maps/cod_spawning.csv` — hence `monkeypatch.chdir` to the repo root.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_interannual.py -k wrap_guard -v`
Expected: FAIL — `test_wrap_guard_raises...` does not raise (no guard yet).

- [ ] **Step 3: Add the guard** in `osmose/engine/config.py::_load_rv_spatial`, immediately after the existing `tlen % n_dt` validation block (after line ~1200, before the NaN-at-spawning check)

```python
    # Interannual field (tlen > one year): a run longer than the forcing period would silently
    # wrap (step % tlen repeats year 0). Fail fast rather than hindcast against wrapped years.
    if tlen > n_dt:
        nyear = int(float(cfg.get("simulation.time.nyear", "0") or 0))
        if nyear * n_dt > tlen:
            raise ValueError(
                f"RV field is interannual ({tlen} steps = {tlen // n_dt} yr) but the run is "
                f"{nyear} yr ({nyear * n_dt} steps) — it would wrap past the forcing period. "
                f"Set simulation.time.nyear <= {tlen // n_dt}."
            )
```

- [ ] **Step 4: Run to verify pass + gate-off parity**

Run: `.venv/bin/python -m pytest tests/test_rv_interannual.py -k wrap_guard -v`
Expected: PASS (2 passed).
Run: `.venv/bin/python -m pytest tests/ -k "rv_spatial or reproductive or larva_mortality" -q`
Expected: PASS — the existing RV parity/behavior tests unchanged (gate-off path untouched; the guard only fires for enabled interannual configs).

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/config.py tests/test_rv_interannual.py
git commit -m "feat(engine): fail-fast when an interannual RV run would wrap past the forcing period"
```

---

## Task 4: observed cod data (ICES 2023 assessment, baked)

**Files:**
- Create: `docs/diagnostics/ices_cod_2732_observed.csv`

**Interfaces:**
- Produces: `docs/diagnostics/ices_cod_2732_observed.csv` with columns `year,ssb_t,recruitment_thousands` for 1993–2021 (eastern-Baltic cod cod.27.24-32, ICES 2023 assessment; SSB in tonnes, recruitment in thousands).

> No MCP dependency at execution — the values are baked (fetched during planning via `mcp__ices__get_stock_assessment("cod.27.24-32", 2023)`).

- [ ] **Step 1: Write the CSV** (`docs/diagnostics/ices_cod_2732_observed.csv`)

```
# Eastern Baltic cod (cod.27.24-32), ICES 2023 assessment. ssb_t = tonnes, recruitment_thousands.
# Source: ICES stock assessment graphs (fetched via ICES MCP get_stock_assessment, 2023).
year,ssb_t,recruitment_thousands
1993,103145,2016780
1994,120533,1970220
1995,132252,1464130
1996,93870.9,2742310
1997,63302.6,2790870
1998,56050.2,2867140
1999,51971.4,2227150
2000,61608.1,2905690
2001,75403,1910600
2002,85029.2,2343260
2003,86703.8,4042790
2004,75587.4,3176490
2005,94282.5,3953130
2006,94985.6,4184580
2007,93791.2,3957610
2008,134284,4147550
2009,148363,3543400
2010,152917,3781720
2011,136020,5134910
2012,108942,5235180
2013,102055,3245730
2014,111502,2602870
2015,132347,1757580
2016,113740,2756890
2017,85336.7,2165040
2018,73681.7,1279890
2019,68094,2586840
2020,64835.4,2344850
2021,69026.4,1600960
```

- [ ] **Step 2: Verify it loads**

Run: `cd /home/razinka/osmopy && .venv/bin/python -c "import pandas as pd; d=pd.read_csv('docs/diagnostics/ices_cod_2732_observed.csv', comment='#'); print(len(d), d['year'].min(), d['year'].max(), d['ssb_t'].idxmax())"`
Expected: `29 1993 2021 17` (SSB peak at index 17 = 2010).

- [ ] **Step 3: Commit**

```bash
git add docs/diagnostics/ices_cod_2732_observed.csv
git commit -m "data(baltic): observed eastern-Baltic cod SSB+R 1993-2021 (ICES 2023 assessment)"
```

---

## Task 5: Phase 0 — offline RV↔cod correlation

**Files:**
- Create: `scripts/baltic_rv_cod_offline.py`
- Create: `docs/diagnostics/baltic_rv_cod_correlation.md`
- Test: `tests/test_rv_interannual.py`

**Interfaces:**
- Consumes: `docs/diagnostics/baltic_rv_fraction.csv` (offline RV series), `docs/diagnostics/ices_cod_2732_observed.csv` (Task 4).
- Produces: a `lagged_correlations(rv_annual, cod_series, max_lag)` helper (unit-tested) + a results doc.

- [ ] **Step 1: Write the failing test** (append to `tests/test_rv_interannual.py`)

```python
def test_lagged_correlations_recovers_known_lag():
    import sys
    sys.path.insert(0, "/home/razinka/osmopy/scripts")
    from baltic_rv_cod_offline import lagged_correlations
    rng = np.random.default_rng(0)
    rv = rng.random(29)
    cod = np.empty(29)
    cod[2:] = rv[:-2]          # cod lags rv by 2 yr
    cod[:2] = rng.random(2)
    lc = lagged_correlations(rv, cod, max_lag=4)   # dict lag->corr
    best = max(lc, key=lambda k: lc[k])
    assert best == 2 and lc[2] > 0.9
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_interannual.py -k lagged -v`
Expected: FAIL — `ModuleNotFoundError: baltic_rv_cod_offline`.

- [ ] **Step 3: Write `scripts/baltic_rv_cod_offline.py`**

```python
#!/usr/bin/env python
"""Phase 0: offline correlation of the real 1993-2021 reproductive-volume series vs observed
eastern-Baltic cod (recruitment/SSB). Informative (soft gate) — does NOT block the engine work."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def lagged_correlations(rv_annual, cod_series, max_lag: int) -> dict[int, float]:
    """corr(rv[year], cod[year+lag]) for lag in 0..max_lag (cod responds AFTER rv)."""
    rv = np.asarray(rv_annual, float)
    cod = np.asarray(cod_series, float)
    out = {}
    for lag in range(max_lag + 1):
        a, b = rv[: len(rv) - lag], cod[lag:]
        n = min(len(a), len(b))
        out[lag] = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 2 else float("nan")
    return out


def main() -> int:
    rv = pd.read_csv(ROOT / "docs/diagnostics/baltic_rv_fraction.csv")
    rv["yr"] = pd.to_datetime(rv["time"]).dt.year
    rv_annual = rv.groupby("yr")["rv_fraction"].mean()
    cod = pd.read_csv(ROOT / "docs/diagnostics/ices_cod_2732_observed.csv", comment="#").set_index("year")
    years = sorted(set(rv_annual.index) & set(cod.index))
    rv_a = rv_annual.reindex(years).values
    lines = ["# Phase 0 — offline reproductive-volume vs observed cod (cod.27.24-32)\n"]
    for col in ("recruitment_thousands", "ssb_t"):
        lc = lagged_correlations(rv_a, cod[col].reindex(years).values, max_lag=4)
        best = max((k for k in lc if np.isfinite(lc[k])), key=lambda k: lc[k], default=None)
        lines.append(f"\n## RV vs {col}\n")
        lines.append("| lag (yr) | corr |\n|---|---|\n")
        lines += [f"| {k} | {lc[k]:.3f} |\n" for k in sorted(lc)]
        lines.append(f"\n**Best lag = {best} (corr {lc[best]:.3f}).**\n")
    lines.append(
        "\n*Caveat:* eastern-Baltic cod was downgraded to data-limited (~2019); SSB/R post-2014 "
        "uncertain. Soft gate — the engine hindcast (Phases 1-3) is built regardless.\n"
    )
    out = ROOT / "docs/diagnostics/baltic_rv_cod_correlation.md"
    out.write_text("".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 4: Run test + the analysis**

Run: `.venv/bin/python -m pytest tests/test_rv_interannual.py -k lagged -v`
Expected: PASS.
Run: `cd /home/razinka/osmopy && .venv/bin/python scripts/baltic_rv_cod_offline.py && sed -n '1,40p' docs/diagnostics/baltic_rv_cod_correlation.md`
Expected: writes the doc; report the best lag/correlation for recruitment and SSB in the task report (whatever it is — a weak correlation is a legitimate, informative result).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_rv_cod_offline.py docs/diagnostics/baltic_rv_cod_correlation.md tests/test_rv_interannual.py
git commit -m "feat(baltic): Phase 0 offline RV-vs-observed-cod correlation"
```

---

## Task 6: Phase 3 harness code + skill functions

**Files:**
- Create: `scripts/baltic_rv_hindcast.py`
- Test: `tests/test_rv_interannual.py`

**Interfaces:**
- Consumes: `osmose.demo.osmose_demo("baltic")`, `OsmoseConfigReader`, `PythonEngine`, the two RV field files, `docs/diagnostics/ices_cod_2732_observed.csv`.
- Produces: `arm_overrides(mode, rv_ref, inter_path, clim_path) -> dict` and `skill_delta(model_a, model_b, observed) -> float` (both unit-tested); a `run_hindcast()` entry point (invoked in Task 7).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rv_interannual.py`)

```python
def test_arm_overrides_shared_ref_and_files():
    import sys
    sys.path.insert(0, "/home/razinka/osmopy/scripts")
    from baltic_rv_hindcast import arm_overrides
    off = arm_overrides("off", rv_ref=12.0, inter_path="i.nc", clim_path="c.nc")
    clim = arm_overrides("clim", rv_ref=12.0, inter_path="i.nc", clim_path="c.nc")
    inter = arm_overrides("inter", rv_ref=12.0, inter_path="i.nc", clim_path="c.nc")
    assert off.get("reproduction.rv.spatial.enabled", "false") == "false"
    # both enabled arms share the forced RV_ref, differ only in the field file
    assert clim["reproduction.rv.spatial.ref"] == "12.0"
    assert inter["reproduction.rv.spatial.ref"] == "12.0"
    assert clim["reproduction.rv.spatial.field.file"].endswith("c.nc")
    assert inter["reproduction.rv.spatial.field.file"].endswith("i.nc")
    assert clim["reproduction.rv.spatial.species.enabled.sp0"] == "true"
    # SSB enabled on ALL arms (incl. off) so .ssb() works uniformly
    assert off["output.ssb.enabled"] == "true" and inter["output.ssb.enabled"] == "true"


def test_skill_delta_positive_when_b_tracks_observed():
    import sys
    sys.path.insert(0, "/home/razinka/osmopy/scripts")
    from baltic_rv_hindcast import skill_delta
    obs = np.array([1.0, 2, 3, 2, 1, 2, 3], float)
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98], float)  # nonzero var, ~uncorrelated
    b = obs * 0.5 + 0.1                                            # tracks observed
    assert skill_delta(a, b, obs) > 0.5


def test_skill_delta_nan_safe_on_collapsed_arm():
    import sys
    sys.path.insert(0, "/home/razinka/osmopy/scripts")
    from baltic_rv_hindcast import skill_delta
    obs = np.array([1.0, 2, 3, 2, 1, 2, 3], float)
    flat = np.ones(7)  # a collapsed arm (zero variance) -> nan, NOT a crash / not 0
    assert np.isnan(skill_delta(flat, obs, obs))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_rv_interannual.py -k "arm_overrides or skill_delta" -v`
Expected: FAIL — `ModuleNotFoundError: baltic_rv_hindcast`.

- [ ] **Step 3: Write `scripts/baltic_rv_hindcast.py`**

```python
#!/usr/bin/env python
"""Phase 3: A/B reproductive-volume hindcast. Three arms over 1993-2021 (nyear=29):
off (no RV mechanism), clim (stationary climatology), inter (real interannual) — the two
enabled arms share one forced RV_ref so the A/B isolates temporal structure. Scores modeled
cod SSB vs observed ICES SSB (skill delta). NOT a CI gate (emergent)."""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTER = ROOT / "data/baltic_rv/baltic_rv_field_interannual.nc"
CLIM = ROOT / "data/baltic/forcing/baltic_rv_field.nc"
N_YEAR = 29           # 1993-2021
WINDOW = slice(6, 16)  # usable window sim-yr 6-15 (~1999-2008); intrinsic collapse dominates later


def arm_overrides(mode: str, rv_ref: float, inter_path: str, clim_path: str) -> dict:
    # output.ssb.enabled in the SHARED base so ALL arms (incl. "off") emit a real maturity-based
    # SSB — otherwise the "off" arm's in-memory results have no "SSB" entry and .ssb() raises.
    base = {"simulation.time.nyear": str(N_YEAR), "output.ssb.enabled": "true"}
    if mode == "off":
        return {**base, "reproduction.rv.spatial.enabled": "false"}
    path = inter_path if mode == "inter" else clim_path
    return {
        **base,
        "reproduction.rv.spatial.enabled": "true",
        "reproduction.rv.spatial.field.file": path,
        "reproduction.rv.spatial.field.varname": "reproductive_volume",
        "reproduction.rv.spatial.ref": str(rv_ref),   # SHARED across arms
        "reproduction.rv.spatial.species.enabled.sp0": "true",
    }


def skill_delta(model_a, model_b, observed) -> float:
    """corr(B, obs) - corr(A, obs) over the overlap (window applied by caller). A zero-variance
    (collapsed) arm has no correlation signal -> nan; callers aggregate with nanmean/nanstd."""
    o = np.asarray(observed, float)

    def c(m):
        m = np.asarray(m, float)
        n = min(len(m), len(o))
        if n <= 2 or np.std(m[:n]) == 0 or np.std(o[:n]) == 0:
            return float("nan")
        with np.errstate(invalid="ignore"):
            return float(np.corrcoef(m[:n], o[:n])[0, 1])

    return c(model_b) - c(model_a)


def _rv_ref_of(path: Path) -> float:
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return float(ds["reproductive_volume"].attrs["RV_ref"])


def _cod_ssb(raw: dict, seed: int) -> np.ndarray:
    # Real maturity-based spawning-stock biomass (length>=maturity_size AND age>=maturity_age),
    # matched to observed ICES ssb_t — enabled via output.ssb.enabled in arm_overrides' base.
    from osmose.engine import PythonEngine
    b = PythonEngine().run_in_memory(raw, seed=seed).ssb()
    return b["cod"].to_numpy(dtype=float)


def run_hindcast(seeds=(0, 1, 2, 3, 4)) -> dict:
    from osmose.config.reader import OsmoseConfigReader
    from osmose.demo import osmose_demo
    rv_ref = _rv_ref_of(INTER)
    tmp = Path(tempfile.mkdtemp())
    base = dict(OsmoseConfigReader().read(str(osmose_demo("baltic", tmp)["config_file"])))
    obs = pd.read_csv(ROOT / "docs/diagnostics/ices_cod_2732_observed.csv", comment="#")["ssb_t"].to_numpy()
    obs_win = obs[WINDOW]
    series = {m: [] for m in ("off", "clim", "inter")}
    for seed in seeds:
        for m in series:
            raw = {**base, **arm_overrides(m, rv_ref, str(INTER), str(CLIM))}
            series[m].append(_cod_ssb(raw, seed))
    means = {m: np.mean(np.stack(v), axis=0) for m, v in series.items()}
    deltas = [
        skill_delta(np.stack(series["clim"])[i][WINDOW], np.stack(series["inter"])[i][WINDOW], obs_win)
        for i in range(len(seeds))
    ]
    return {"means": means, "skill_delta_per_seed": deltas, "rv_ref": rv_ref, "obs": obs}


if __name__ == "__main__":
    sys.exit(0 if run_hindcast() else 0)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_rv_interannual.py -k "arm_overrides or skill_delta" -v`
Expected: PASS (3 passed — `test_arm_overrides_shared_ref_and_files`, `test_skill_delta_positive_when_b_tracks_observed`, `test_skill_delta_nan_safe_on_collapsed_arm`).

- [ ] **Step 5: Commit**

```bash
git add scripts/baltic_rv_hindcast.py tests/test_rv_interannual.py
git commit -m "feat(baltic): Phase 3 RV hindcast harness (3-arm shared-ref A/B + skill delta)"
```

---

## Task 7: Run the hindcast + results doc

**Files:**
- Create: `docs/diagnostics/baltic_rv_hindcast.md`
- (No CI test — documented emergent run.)

**Interfaces:** Consumes Tasks 2/3/4/6.

- [ ] **Step 1: Run the hindcast** via a small runner (`scripts/_run_rv_hindcast.py`)

```python
import sys
import numpy as np
sys.path.insert(0, "/home/razinka/osmopy/scripts")
from baltic_rv_hindcast import run_hindcast

r = run_hindcast()
d = np.array(r["skill_delta_per_seed"], float)
print("skill_delta per seed:", [round(float(x), 3) for x in d])
print("mean skill delta (clim->inter):", round(float(np.nanmean(d)), 3),
      "+/-", round(float(np.nanstd(d)), 3))
for m, s in r["means"].items():
    print(m, "cod proxy (sim yr6-15):", [round(float(x)) for x in s[6:16]])
```

Run: `cd /home/razinka/osmopy && PYTHONPATH=. .venv/bin/python scripts/_run_rv_hindcast.py`
Expected: completes (5 seeds × 3 arms × 29 yr — several minutes); prints per-seed skill deltas + per-arm cod trajectories over the window. **Report the actual numbers honestly** — a mean skill delta ≤ 0 (interannual adds no skill over climatology) is a legitimate result.

- [ ] **Step 2: Write `docs/diagnostics/baltic_rv_hindcast.md`**

Record: the 3-arm cod trajectories (sim-yr6–15 / 1999–2008), the per-seed and mean±sd skill delta (clim vs inter, via nanmean over seeds — report how many seeds were non-nan), the 2004-MBI feature check, and the honest verdict. State the framing:
- Intrinsic boom-bust is the null; the A/B isolates interannual structure via the shared `RV_ref` (with the Jensen caveat — the clip is nonlinear, so the arms don't have identical mean suppression).
- **Low correlation power:** over 1999–2008 the model's intrinsic cod is in post-peak *decline* while observed SSB *rises* (post-2003 recovery), so raw `corr` with observed is expected to be weak/negative for *both* arms — the **2004-MBI feature test is the primary signal**, not the absolute correlation. Report the skill *delta* (does interannual beat climatology) as the headline, not either arm's absolute skill.
- The 2016 MBI is beyond the usable window (cod collapsed). Model SSB vs observed `ssb_t` is like-for-like (the harness uses the engine's maturity-based SSB).
- Phase 0 (bottom-slice areal fraction) and Phase 3 (full-column thickness) are *related but distinct* RV metrics — note qualitative agreement, don't over-equate the numbers. Phase 0's cod SSB is data-limited post-~2014.

- [ ] **Step 3: Verify deployed config untouched + full CI-safe sweep**

Run: `git status --porcelain data/baltic/` (expect **empty** — the interannual field lives in `data/baltic_rv/`, so `data/baltic/` is entirely unchanged: no config CSV edits, no new file).
Run: `.venv/bin/python -m pytest tests/test_rv_interannual.py -v && .venv/bin/ruff check osmose/ tests/ && .venv/bin/ruff format --check osmose/ tests/`
Expected: all pass; ruff clean.

- [ ] **Step 4: Commit**

```bash
git add docs/diagnostics/baltic_rv_hindcast.md
git commit -m "docs(baltic): interannual RV cod hindcast results (A/B skill delta)"
```

---

## Self-Review (completed during authoring)

- **Spec coverage:** Phase 0 → T4+T5; Phase 1 → T1+T2; Phase 2 → T3; Phase 3 → T6+T7. Shared-`RV_ref` (T6 `arm_overrides`), window/2004-MBI honesty (T6 `WINDOW`, T7 doc), tie-back (T2), wrap-guard (T3), climatology byte-identical (T1 regression test), gate-off parity (T3), deployed-config-untouched (T7 Step 3).
- **Placeholder scan:** all code, the observed CSV, and commands are literal. (T7 Step 1's inline `-c` has a deliberate note to use a runner file — the implementer may write `scripts/_run_hindcast.py`; the harness API is fixed.)
- **Type consistency:** `build_rv_field_interannual`, `_load_rv_spatial` guard, `lagged_correlations`, `arm_overrides`, `skill_delta`, `run_hindcast` signatures are consistent across tasks and tests.
- **Resolved by the workflow review (2026-07-11):** the harness now scores the engine's **real maturity-based SSB** (`.ssb()["cod"]`, enabled via `output.ssb.enabled` in the shared arm base) against observed ICES `ssb_t` — like-for-like, and mechanistically matched (the RV gate acts on eggs → spawner reproduction). Also folded in: nan-safe `skill_delta` (collapsed arms → nan, aggregated with nanmean), real `GridSpec` in the T1 tests, soft tie-back (metrics differ), and the field committed to `data/baltic_rv/` (out of the demo copytree).

---

## Multi-agent review incorporation (2026-07-11)

A 4-lens adversarial workflow review (22 agents, **16 confirmed / 0 refuted**, all verified against the real code) surfaced 8 deduplicated issues, all folded in:

1. **[HIGH] `skill_delta` NaN.** `corrcoef` of a constant (collapsed) arm → NaN; the test asserted `NaN>0.5` (False) and a collapsed seed silently poisoned the ensemble mean. → nan-safe `c()` (zero-variance → nan, errstate-guarded) + nonzero-variance test array + a `test_skill_delta_nan_safe_on_collapsed_arm`; runner aggregates with `nanmean`/`nanstd` and reports non-nan seed count.
2. **[MED] Test `_Grid` stub errored** (missing `dx`/`dy` `@property`). → real `GridSpec` in the T1 tests.
3. **[MED, ×several] SSB vs total biomass.** The A/B scored *total* cod biomass against observed *SSB*; the engine has a real maturity-based `.ssb()` (gated on `output.ssb.enabled`, allowlisted). → `output.ssb.enabled=true` in the **shared** arm base (so the `off` arm also emits SSB, else `.ssb()` raises) + `_cod_ssb` reads `.ssb()["cod"]`. Mechanistically matched (RV acts on eggs → SSB) so zero-lag correlation is defensible.
4. **[MED] Tie-back hard-fail.** `corr>0.6` `SystemExit` between two *different* metrics (full-column thickness vs bottom-slice fraction) could false-fail the build; the plan also misattributed failures to `np.flipud`. → soft warn (`<0.3`), Step-3 finite/shape is the real gate, corrected the orientation misattribution.
5. **[LOW] Field bloat.** 6 MB field in `data/baltic/forcing/` was copytree'd into every baltic/baltic_a2 run. → committed to a separate `data/baltic_rv/` (out of the copytree) + zlib compression; `data/baltic/` now fully unchanged.
6. **[LOW] Jensen.** Spec's "identical mean suppression" was false (nonlinear clip). → reworded in the spec.
7/8. **[LOW] Phase-0 data-limited window / low correlation power** → Task-7 doc must report skill *delta* as headline, 2004-MBI feature test as primary signal, and note the metric distinction + post-2014 uncertainty.

Verified end-to-end: `.ssb()["cod"]` returns the same wide annual shape as `.biomass()["cod"]`; `output.ssb.enabled` is allowlisted (`config_validation.py:215`); the 29 phy `so` + 29 bgc `o2` cache files match the build globs.
