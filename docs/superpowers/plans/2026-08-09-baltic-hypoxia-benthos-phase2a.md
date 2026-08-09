# Baltic Hypoxia→Benthos Coupling (Phase 2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bottom dissolved oxygen scales benthos carrying capacity per cell — the first physics→biology coupling of spec Phase 2 (C2a), making hypoxic-area benthic food loss (the Baltic's defining pressure) explicit instead of implicit.

**Architecture:** A bottom-O₂ monthly climatology NetCDF (CMEMS `o2b`, 2024, regridded to the model grid, duplicated to 24 frames) is loaded through the existing `PhysicalData.from_netcdf` — decoupled from the bioen gate — and applied as a per-cell factor on benthos K inside `ResourceState.update`, using the existing `f_o2` Michaelis–Menten dose–response. Gate protocol identical to Phase 1: A/B off/on via the harness, identity-pinned certification decides adoption.

**Tech Stack:** Python 3.12, `.venv/bin/python`, xarray/netCDF4, copernicusmarine (creds in `.env`), pytest, ruff (line 100).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-baltic-improvement-avenues-design.md` §4 Phase 2, C2(a) only. C2(b) computed-RV, B1 interannual and F1 fishing are OUT of this plan.
- **Depletion stays OFF** — Phase 1 closed negative; this coupling modulates the non-depletable K reset. No `ltl.depletable.*` key is touched.
- O₂ response: `f_o2(o2, c1, c2)` from `osmose/engine/processes/oxygen_function.py` with defaults `c1 = 1.0`, `c2 = 60.0` (mmol m⁻³ half-saturation; benthic die-off literature centers ~2 ml/L ≈ 90 mmol m⁻³, giving f ≈ 0.6 there). Both are config keys so the A/B can vary them without code changes.
- New config keys (engine bucket, exact spellings):
  `ltl.oxygen.benthos.enabled` (default false), `ltl.oxygen.benthos.c1` (default 1.0),
  `ltl.oxygen.benthos.c2` (default 60.0), `ltl.oxygen.benthos.rsc` (default `Benthos` — resource
  NAME, not index, so species reindexing cannot silently retarget it).
- Oxygen forcing keys `oxygen.filename`, `oxygen.varname`, `oxygen.nsteps.year`, `oxygen.factor`, `oxygen.offset` MOVE from the Java-only allowlist bucket to the Python-honored bucket (`osmose/engine/config_validation.py` ~182–186), with the mirrored guard list in `tests/test_issue_123_known_but_unread_keys.py` updated — the spec §5 planned bucket move.
- **Frame alignment trap:** `PhysicalData.get_value` indexes `step % data.shape[0]` with the SIMULATION step (24/yr). A 12-frame monthly file misaligns from step 13 onward. The forcing writer MUST duplicate each month ×2 → 24 frames, and a unit test must pin frame count == `simulation.time.ndtperyear`.
- Certification gate: identical to Phase 1 — identity-pinned set {cod_west, cod_east, herring, sprat, flounder, perch, stickleback}, 50 yr × 5 seeds, via `scripts/baltic_depletable_ab.py --skip-default-arms --extra-arm`. GATE [off] PASS precondition, then GATE [o2on] decides adoption.
- Expected direction: flounder and cod lose benthic food in hypoxic cells; flounder sits at 40.5 kt vs a 20 kt floor (51% headroom) and cod_east at 83.0 kt vs a 60 kt floor (28% headroom) — record per-species deltas regardless of verdict.
- No concurrent engine jobs during A/B runs. Tests `.venv/bin/python -m pytest`; lint `.venv/bin/ruff check`.

---

### Task 1: Bottom-O₂ forcing file

**Files:**
- Create: `scripts/make_baltic_oxygen_forcing.py`
- Create: `data/baltic/baltic_oxygen_bottom.nc` (generated artifact, committed like `baltic_ltl_biomass.nc`)
- Test: `tests/test_baltic_oxygen_forcing.py`

**Interfaces:**
- Consumes: `copernicusmarine` (creds via `.env`, `load_dotenv` pattern from `mcp_servers/copernicus/server.py`), the Baltic BGC product carrying `o2b` (see the dataset catalog in that server, ~lines 100–132), `osmose.forcing` regrid helpers and the grid from `mcp_servers/copernicus/server.py:_baltic_grid`.
- Produces: `data/baltic/baltic_oxygen_bottom.nc` — variable `o2b`, dims `(time=24, y, x)` on the 50×40 model grid, units mmol m⁻³, 2024 monthly climatology with each month duplicated ×2 (frames 0–1 = Jan, …, 22–23 = Dec), land cells NaN or 0 consistently with `baltic_ltl_biomass.nc`'s convention (read that file and match it).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baltic_oxygen_forcing.py
"""Bottom-O2 forcing file: grid, frames, units, plausibility (spec Phase 2a)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

NC = Path(__file__).resolve().parents[1] / "data" / "baltic" / "baltic_oxygen_bottom.nc"

pytestmark = pytest.mark.skipif(not NC.exists(), reason="oxygen forcing not generated yet")


def test_dims_and_frames():
    ds = xr.open_dataset(NC)
    o2 = ds["o2b"]
    assert o2.dims[0] in ("time", "t")
    # 24 frames == simulation.time.ndtperyear — PhysicalData.get_value indexes step % nframes,
    # so 12 monthly frames would silently misalign from step 13 onward (plan Global Constraints).
    assert o2.shape[0] == 24
    assert o2.shape[1:] == (40, 50) or o2.shape[1:] == (50, 40)
    ds.close()


def test_month_duplication():
    ds = xr.open_dataset(NC)
    v = ds["o2b"].values
    for m in range(12):
        np.testing.assert_array_equal(v[2 * m], v[2 * m + 1])
    ds.close()


def test_values_plausible_mmol_m3():
    ds = xr.open_dataset(NC)
    v = ds["o2b"].values
    wet = v[~np.isnan(v)]
    wet = wet[wet != 0.0]
    # Baltic bottom O2 spans anoxic deeps (~0, occasionally negative-as-H2S-proxy in ERGOM,
    # clipped to >= 0 by the writer) to well-oxygenated coasts (~300-400 mmol/m3)
    assert wet.min() >= 0.0
    assert 150.0 <= wet.max() <= 600.0
    # hypoxia must actually exist in the domain or the coupling is vacuous
    assert (wet < 90.0).mean() > 0.02
    ds.close()
```

- [ ] **Step 2: Run it (skips — file absent), write the generator**

The generator script: `load_dotenv` exactly as `mcp_servers/copernicus/server.py` does; download the 2024 monthly `o2b` field from the Baltic BGC analysis product (reuse the dataset id from the server's catalog and its `download_field` logic — subset to 10–30°E, 54–66°N); regrid to the model grid the same way `generate_osmose_ltl`'s pipeline does (nearest-neighbour via `osmose.forcing` helpers); clip negatives to 0; duplicate months ×2 to 24 frames; write `data/baltic/baltic_oxygen_bottom.nc` with `o2b(time, y, x)`, units attribute `mmol m-3`, and a provenance `history` attribute (product id, year, generation date, script name).

- [ ] **Step 3: Generate the file and run the tests**

Run: `PYTHONPATH=. .venv/bin/python scripts/make_baltic_oxygen_forcing.py`
Then: `.venv/bin/python -m pytest tests/test_baltic_oxygen_forcing.py -v`
Expected: 3 PASS. If CMEMS credentials fail, STOP and report NEEDS_CONTEXT (do not fabricate data).

- [ ] **Step 4: Lint and commit**

```bash
.venv/bin/ruff check scripts/make_baltic_oxygen_forcing.py tests/test_baltic_oxygen_forcing.py
git add scripts/make_baltic_oxygen_forcing.py tests/test_baltic_oxygen_forcing.py data/baltic/baltic_oxygen_bottom.nc
git commit -m "feat(baltic): bottom-O2 forcing file (CMEMS o2b 2024 monthly, 24 frames, model grid)"
```

---

### Task 2: Engine coupling — O₂ scales benthos K

**Files:**
- Modify: `osmose/engine/resources.py` (accept oxygen data, apply factor to the named resource's K)
- Modify: `osmose/engine/simulate.py` (load oxygen NetCDF independent of bioen when the coupling or bioen needs it; pass into `ResourceState`)
- Modify: `osmose/engine/config_validation.py` (bucket move + new `ltl.oxygen.benthos.*` keys)
- Modify: `tests/test_issue_123_known_but_unread_keys.py` (mirror list)
- Test: `tests/test_engine_oxygen_benthos.py`

**Interfaces:**
- Consumes: `PhysicalData.from_netcdf(path, varname, nsteps_year, factor, offset)` and `.get_value(step, y, x)` / `._data` (osmose/engine/physical_data.py); `f_o2(o2, c1, c2)`; `ResourceState.update`'s `k_row` (built per resource just before the depletable/reset branch, resources.py ~250–270).
- Produces: `ResourceState(config, grid, oxygen: PhysicalData | None = None)` — new optional kwarg (default None keeps every existing caller/test working); when `ltl.oxygen.benthos.enabled=true` and oxygen is present, the resource whose `name` equals `ltl.oxygen.benthos.rsc` gets `k_row *= f_o2(o2_row, c1, c2)` each step, where `o2_row` is the oxygen field for that step flattened to grid order. Exposes `ResourceState.oxygen_factor_last` (the factor row from the most recent update) for tests/diagnostics.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_engine_oxygen_benthos.py
"""O2->benthos K coupling: factor math, gating, config validation (spec Phase 2a)."""

from __future__ import annotations

import numpy as np
from osmose.engine.grid import Grid
from osmose.engine.physical_data import PhysicalData
from osmose.engine.processes.oxygen_function import f_o2
from osmose.engine.resources import ResourceState


def _cfg(enabled="true"):
    return {
        "simulation.nresource": "1",
        "simulation.time.ndtperyear": "24",
        "ltl.name.rsc0": "Benthos",
        "ltl.size.min.rsc0": "0.5",
        "ltl.size.max.rsc0": "10.0",
        "ltl.tl.rsc0": "2.5",
        "ltl.accessibility2fish.rsc0": "0.8",
        "ltl.biomass.total.rsc0": "1000.0",
        "ltl.oxygen.benthos.enabled": enabled,
        "ltl.oxygen.benthos.c1": "1.0",
        "ltl.oxygen.benthos.c2": "60.0",
        "ltl.oxygen.benthos.rsc": "Benthos",
    }


def _oxygen(ny=2, nx=2, values=(0.0, 60.0, 90.0, 300.0)):
    data = np.array(values, dtype=np.float64).reshape(1, ny, nx)
    return PhysicalData(data=data, constant=None, nsteps_year=1)


def test_factor_applied_to_benthos_k():
    grid = Grid.from_dimensions(ny=2, nx=2)
    rs = ResourceState(config=_cfg(), grid=grid, oxygen=_oxygen())
    rs.update(step=0)
    base = 1000.0 / 4 * 0.8  # uniform per-cell K without oxygen
    expected = base * f_o2(np.array([0.0, 60.0, 90.0, 300.0]), 1.0, 60.0)
    np.testing.assert_allclose(rs.biomass[0], expected, rtol=1e-12)
    # anoxic cell -> zero benthos; well-oxygenated cell -> ~0.83 * base
    assert rs.biomass[0][0] == 0.0
    assert rs.biomass[0][3] > 0.8 * base


def test_disabled_or_no_oxygen_is_identity():
    grid = Grid.from_dimensions(ny=2, nx=2)
    base = 1000.0 / 4 * 0.8
    for rs in (
        ResourceState(config=_cfg(enabled="false"), grid=grid, oxygen=_oxygen()),
        ResourceState(config=_cfg(), grid=grid, oxygen=None),
    ):
        rs.update(step=0)
        np.testing.assert_allclose(rs.biomass[0], np.full(4, base), rtol=1e-12)


def test_named_resource_only():
    grid = Grid.from_dimensions(ny=2, nx=2)
    cfg = _cfg()
    cfg["ltl.oxygen.benthos.rsc"] = "SomethingElse"
    rs = ResourceState(config=cfg, grid=grid, oxygen=_oxygen())
    rs.update(step=0)
    np.testing.assert_allclose(rs.biomass[0], np.full(4, 1000.0 / 4 * 0.8), rtol=1e-12)


def test_new_keys_validate_clean():
    from osmose.engine.config_validation import validate

    issues = validate(_cfg())
    assert not [i for i in issues if "ltl.oxygen" in getattr(i, "key", "")]
```

- [ ] **Step 2: Run to verify failure** (`TypeError: unexpected keyword 'oxygen'`), **implement**

In `resources.py`: add the `oxygen` kwarg and the four `ltl.oxygen.benthos.*` config reads in `__init__`; in `update`, after `k_row` is assembled and before the depletable/reset branch, when enabled and `rsc.name == self._oxygen_rsc_name` and oxygen is not None, build `o2_row` for this step (use `PhysicalData._data` frame `step % nframes` flattened, or `get_value` per cell — prefer the vectorized frame path; regrid via `self._regrid_to_model` if the oxygen grid differs from the model grid) and apply `k_row = k_row * f_o2(o2_row, c1, c2)`; store `self.oxygen_factor_last = factor_row`.

In `simulate.py` (~1545–1552): hoist oxygen loading out of the bioen gate — load when `bioen_enabled` OR `ltl.oxygen.benthos.enabled` is true; support NetCDF mode: if `oxygen.filename` is set, `PhysicalData.from_netcdf(resolved_path, config.raw_config.get("oxygen.varname", "o2b"), int(nsteps), factor, offset)` (resolve the path with the same helper the resource forcing uses — `osmose/engine/path_resolution.py`); else fall back to `oxygen.value` constant. Pass `oxygen=o2_data` into the `ResourceState` construction (~line 1499).

In `config_validation.py`: move the five `oxygen.*` keys to the Python-honored bucket; add the four `ltl.oxygen.benthos.*` keys; update the mirror list in `tests/test_issue_123_known_but_unread_keys.py` (both directions: removed from Java-only, present in engine bucket).

- [ ] **Step 3: Run the tests + validation suite**

Run: `.venv/bin/python -m pytest tests/test_engine_oxygen_benthos.py tests/test_engine_resources.py tests/test_issue_123_known_but_unread_keys.py tests/test_engine_config_validation.py -v`
Expected: all PASS, validation warning-free.

- [ ] **Step 4: Lint and commit**

```bash
.venv/bin/ruff check osmose/engine/resources.py osmose/engine/simulate.py osmose/engine/config_validation.py tests/test_engine_oxygen_benthos.py
git add osmose/engine/resources.py osmose/engine/simulate.py osmose/engine/config_validation.py tests/test_issue_123_known_but_unread_keys.py tests/test_engine_oxygen_benthos.py
git commit -m "feat(engine): bottom-O2 scales benthos K (f_o2 dose-response, config-gated, bioen-independent oxygen loading)"
```

---

### Task 3: Gate run (decision)

**Files:**
- Create: `data/baltic/calibration_results/o2_benthos_arm.json` (arm params, committed with `git add -f` per precedent)
- Create: `docs/baltic_hypoxia_benthos_ab_2026-08-09.md` (adjust date)

**Interfaces:**
- Consumes: Tasks 1–2; `scripts/baltic_depletable_ab.py --skip-default-arms --extra-arm`.

- [ ] **Step 1: Write the arm file and run the gate**

`o2_benthos_arm.json`:

```json
{
  "ltl.oxygen.benthos.enabled": "true",
  "ltl.oxygen.benthos.c1": "1.0",
  "ltl.oxygen.benthos.c2": "60.0",
  "ltl.oxygen.benthos.rsc": "Benthos",
  "oxygen.filename": "baltic_oxygen_bottom.nc",
  "oxygen.varname": "o2b",
  "oxygen.nsteps.year": "24"
}
```

Run: `PYTHONPATH=. .venv/bin/python scripts/baltic_depletable_ab.py --skip-default-arms --extra-arm o2on data/baltic/calibration_results/o2_benthos_arm.json --out docs/baltic_hypoxia_benthos_ab_2026-08-09.md`
(~35 min. Note: the arm passes `oxygen.filename` as a config override — confirm the path resolves relative to the staged config dir; if the resolver needs an absolute path, use the absolute repo path in the JSON and note it in the report.)

- [ ] **Step 2: Two-key rule**

1. `GATE [off]: PASS` required (baseline sanity).
2. `GATE [o2on]: PASS` → Task 4 adoption. FAIL → STOP, commit the report as the negative
   result; parameter variations (c2 sweep) are a follow-up decision, not an automatic retry.
3. Either way, record flounder/cod deltas and the hypoxic-area benthos-K reduction
   (mean of `oxygen_factor_last` over wet cells) in the report commit message.

- [ ] **Step 3: Commit**

```bash
git add -f data/baltic/calibration_results/o2_benthos_arm.json
git add docs/baltic_hypoxia_benthos_ab_2026-08-09.md
git commit -m "docs(baltic): hypoxia->benthos gate verdict"
```

---

### Task 4: Adoption (only on GATE [o2on] PASS)

**Files:**
- Create: `data/baltic/baltic_param-oxygen.csv`
- Modify: `data/baltic/baltic_all-parameters.csv` (include line `osmose.configuration.oxygen;baltic_param-oxygen.csv`)
- Modify: `osmose/engine/config_validation.py` + `tests/test_issue_123_known_but_unread_keys.py` (allowlist the include key `osmose.configuration.oxygen` in BOTH copies — the a2 precedent; include keys are enumerated exactly)
- Test: `tests/test_baltic_oxygen_config.py`
- Create: `docs/baltic_hypoxia_certification_2026-08-09.md` + CLAUDE.md gotcha

Steps mirror the Phase 1 plan's Task 4/5 exactly (failing loading-assertion test first: raw keys match the arm JSON via the harness-style `_raw_pairs`, engine loads with `ResourceState.oxygen` active and factor applied; then overlay file with provenance comments citing the gate report; include line; allowlist both copies; full certification run `--params current`; CLAUDE.md gotcha noting the oxygen coupling, its keys, and that the Java engine does not implement it — `certify --java` needs `ltl.oxygen.benthos.enabled` added to `JAVA_INCOMPATIBLE_PINS` in `scripts/baltic_stability_certify.py` as part of this task, with a pinning test).

---

## Self-review notes

- Spec coverage: C2(a) fully; the §5 oxygen bucket move done in Task 2; Java pinning extension in Task 4; C2(b)/B1/F1 explicitly excluded.
- The frame-alignment trap (12 vs 24) is pinned by a dedicated test; the named-resource key avoids sp-index staleness (this repo's documented failure class).
- Every existing `ResourceState(config, grid)` call site keeps working (kwarg default None); the coupling is double-gated (enabled flag AND oxygen present).
- Falsifiability: `test_values_plausible_mmol_m3` requires >2% hypoxic wet cells — if the forcing has no hypoxia the coupling would be vacuously adopted; the test forbids that.
- Open science question for review: `c2=60` half-saturation vs a hard-threshold ramp at 90 mmol m⁻³ — flagged for the plan-review workflow's literature check.
