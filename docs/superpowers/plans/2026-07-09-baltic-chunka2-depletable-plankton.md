# Chunk A2 — depletable plankton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in per-resource logistic regrowth (depletable plankton) to `ResourceState.update()`, default off and byte-identical when off, then test whether the self-limiting feedback relaxes the Baltic overshoot and/or creates the cod↔sprat regime-shift bistability.

**Architecture:** The engine already grazes `resources.biomass` down within each timestep (Numba `mortality.py:1035` + Python fallbacks) and it persists. The only thing erasing it is `ResourceState.update()` overwriting biomass from forcing each step. Chunk A2 gates that overwrite on a `depletable` flag: off → verbatim reset (parity); on → treat the forced value as carrying capacity K and logistically regrow the carried-over biomass toward K, with a floor. All changes live in `osmose/engine/resources.py` + a config-validation allowlist; **no Numba/mortality/predation change.**

**Tech Stack:** Python 3.11+, numpy (already used). Tests via pytest reusing the `ResourceState` + `Grid.from_dimensions` pattern in `tests/test_engine_resources.py`. Ruff for lint/format.

## Global Constraints

- **No new dependencies; no change to `mortality.py` / `predation.py`.** `ruff check` + `ruff format --check` clean.
- **Parity is the critical guard:** `ltl.depletable.enabled` defaults **false** → `update()` is byte-identical to today, and the **entire existing engine test suite passes unchanged**.
- **Resources are sp8–13** (loaded via `_load_config_species_type` with `species.*.sp{i}` keys) in the Baltic config; the legacy `ltl.*.rsc{i}` path also exists and is used in unit tests.
- **Regrowth (verbatim):** per resource, `K` = the current reset value (`forcing × multiplier × accessibility`, per-cell, seasonal). depletable: `B = max(B_carried, floor·K)`; `B_new = min(K, B + r·B·(1 − B/K))`; where `K ≤ 0`, `B_new = 0`. not depletable: `B_new = K`.
- **New config keys:** `ltl.depletable.enabled` (bool, default `false`), `ltl.depletable.floor` (float, default `0.05`), per-resource `species.regrowth.rate.sp{i}` / `ltl.regrowth.rate.rsc{i}` (float) with global default `ltl.regrowth.rate.default` (default `1.0`). Only consulted when depletable.
- **CI discipline:** unit tests are CI-safe (no engine run); real Baltic runs (Task 3) are CLI-only, excluded from CI per `feedback-ci-fragile-emergent-tests`.
- **Test command:** `.venv/bin/python -m pytest tests/test_engine_resources.py -q` (+ the full engine suite for the parity gate in Task 2).

---

### Task 1: `logistic_regrow` helper + per-resource regrowth-rate plumbing

**Files:**
- Modify: `osmose/engine/resources.py` (add `logistic_regrow` at module scope; add `regrowth_rate` to `ResourceSpeciesInfo`; parse the depletable config in `ResourceState.__init__` and the per-resource rate in both loaders)
- Test: `tests/test_engine_resources.py` (append)

**Interfaces:**
- Produces:
  - `logistic_regrow(biomass: NDArray, k: NDArray, rate: float, floor: float) -> NDArray` — the regrowth equation above, vectorised over cells. Pure/deterministic.
  - `ResourceSpeciesInfo.regrowth_rate: float` (default `1.0`).
  - `ResourceState.depletable: bool`, `ResourceState.depletable_floor: float`, `ResourceState._regrowth_default: float` (parsed in `__init__`); each `ResourceSpeciesInfo.regrowth_rate` populated by the loaders.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_resources.py`:

```python
import numpy as np  # noqa: E402  (if not already imported at top)

from osmose.engine.resources import logistic_regrow  # noqa: E402


def test_logistic_regrow_at_capacity_is_stable():
    k = np.array([100.0, 100.0])
    # B already at K -> no growth (1 - B/K = 0)
    out = logistic_regrow(np.array([100.0, 100.0]), k, rate=0.5, floor=0.05)
    assert np.allclose(out, k)


def test_logistic_regrow_partial_leaves_below_k():
    k = np.array([100.0])
    out = logistic_regrow(np.array([50.0]), k, rate=0.5, floor=0.05)
    # 50 + 0.5*50*(1-0.5) = 62.5  -> depletion persists (< K)
    assert np.allclose(out, [62.5])
    assert out[0] < k[0]


def test_logistic_regrow_floor_recovers_from_zero():
    k = np.array([100.0])
    out = logistic_regrow(np.array([0.0]), k, rate=0.5, floor=0.05)
    # seeded to floor*K = 5 -> 5 + 0.5*5*(1-0.05) = 7.375
    assert out[0] >= 5.0


def test_logistic_regrow_caps_at_k_and_handles_zero_k():
    k = np.array([100.0, 0.0])
    out = logistic_regrow(np.array([90.0, 50.0]), k, rate=5.0, floor=0.05)
    assert out[0] == 100.0  # capped at K
    assert out[1] == 0.0  # K<=0 -> 0 (no NaN)
    assert not np.isnan(out).any()


def test_regrowth_rate_parsed_per_resource(_grid=None):
    from osmose.engine.grid import Grid
    from osmose.engine.resources import ResourceState

    grid = Grid.from_dimensions(ny=3, nx=3)
    config = {
        "simulation.nresource": "1",
        "ltl.name.rsc0": "Zoo",
        "ltl.size.min.rsc0": "0.01",
        "ltl.size.max.rsc0": "0.1",
        "ltl.tl.rsc0": "2.0",
        "ltl.accessibility2fish.rsc0": "0.5",
        "ltl.biomass.total.rsc0": "900.0",
        "ltl.depletable.enabled": "true",
        "ltl.depletable.floor": "0.05",
        "ltl.regrowth.rate.rsc0": "0.3",
        "ltl.regrowth.rate.default": "1.0",
    }
    rs = ResourceState(config=config, grid=grid)
    assert rs.depletable is True
    assert rs.depletable_floor == 0.05
    assert rs.species[0].regrowth_rate == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_resources.py -k "logistic_regrow or regrowth_rate_parsed" -v`
Expected: FAIL — `ImportError: cannot import name 'logistic_regrow'`

- [ ] **Step 3: Add `logistic_regrow` at module scope**

In `osmose/engine/resources.py`, after the imports (before `class ResourceSpeciesInfo`), add:

```python
def logistic_regrow(
    biomass: NDArray[np.float64],
    k: NDArray[np.float64],
    rate: float,
    floor: float,
) -> NDArray[np.float64]:
    """Per-cell logistic regrowth of a depletable resource toward carrying capacity K.

    B = max(B_carried, floor*K);  B_new = min(K, B + rate*B*(1 - B/K));  K<=0 -> 0.
    The floor seeds recovery so a fully-grazed cell (B_carried=0) is not a permanent dead
    zone; the min(.,K) caps at carrying capacity; K<=0 cells (land / off-season) stay empty.
    """
    k = np.asarray(k, dtype=np.float64)
    b = np.maximum(np.asarray(biomass, dtype=np.float64), floor * k)
    with np.errstate(divide="ignore", invalid="ignore"):
        grown = b + rate * b * (1.0 - b / k)
    grown = np.minimum(grown, k)
    return np.where(k > 0.0, grown, 0.0)
```

- [ ] **Step 4: Add `regrowth_rate` to `ResourceSpeciesInfo`**

In the `ResourceSpeciesInfo` dataclass, add the field after `accessibility_ts`:

```python
    regrowth_rate: float = 1.0  # per-step logistic regrowth rate (used only when depletable)
```

- [ ] **Step 5: Parse the depletable config**

In `ResourceState.__init__`, immediately after `self.n_resources = int(config.get("simulation.nresource", "0"))`, add:

```python
        self.depletable = str(config.get("ltl.depletable.enabled", "false")).lower() == "true"
        self.depletable_floor = float(config.get("ltl.depletable.floor", "0.05"))
        self._regrowth_default = float(config.get("ltl.regrowth.rate.default", "1.0"))
```

In `_load_config_ltl`, add `regrowth_rate` to the `ResourceSpeciesInfo(...)` construction:

```python
                    regrowth_rate=float(
                        cfg.get(f"ltl.regrowth.rate.rsc{i}", str(self._regrowth_default))
                    ),
```

In `_load_config_species_type`, add `regrowth_rate` to the `ResourceSpeciesInfo(...)` construction:

```python
                    regrowth_rate=float(
                        cfg.get(f"species.regrowth.rate.sp{fi}", str(self._regrowth_default))
                    ),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_resources.py -k "logistic_regrow or regrowth_rate_parsed" -q`
Expected: PASS (5 passed)

- [ ] **Step 7: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check osmose/engine/resources.py tests/test_engine_resources.py
.venv/bin/ruff format osmose/engine/resources.py tests/test_engine_resources.py
git add osmose/engine/resources.py tests/test_engine_resources.py
git commit -m "feat(engine): logistic_regrow helper + per-resource regrowth-rate config (inert)"
```

---

### Task 2: gate `ResourceState.update()` on `depletable` (+ parity, + key registration)

**Files:**
- Modify: `osmose/engine/resources.py` (`update()`), `osmose/engine/config_validation.py` (allowlist)
- Test: `tests/test_engine_resources.py` (append)

**Interfaces:**
- Consumes: `logistic_regrow`, `self.depletable`, `self.depletable_floor`, `rsc.regrowth_rate` (Task 1).
- Produces: `update(step)` regrows toward K when depletable, resets to K otherwise; the three new keys pass config validation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_engine_resources.py`:

```python
def _uniform_resource_config(depletable: bool, rate: str = "0.3") -> dict:
    cfg = {
        "simulation.nresource": "1",
        "ltl.name.rsc0": "Zoo",
        "ltl.size.min.rsc0": "0.01",
        "ltl.size.max.rsc0": "0.1",
        "ltl.tl.rsc0": "2.0",
        "ltl.accessibility2fish.rsc0": "0.5",
        "ltl.biomass.total.rsc0": "900.0",
    }
    if depletable:
        cfg.update(
            {
                "ltl.depletable.enabled": "true",
                "ltl.depletable.floor": "0.05",
                "ltl.regrowth.rate.rsc0": rate,
            }
        )
    return cfg


def test_update_off_is_full_reset_parity():
    from osmose.engine.grid import Grid
    from osmose.engine.resources import ResourceState

    grid = Grid.from_dimensions(ny=3, nx=3)
    rs = ResourceState(config=_uniform_resource_config(depletable=False), grid=grid)
    rs.update(step=0)
    k = rs.biomass.copy()
    # Deplete, then update again: NON-depletable resets fully to K
    rs.biomass[:] = 0.0
    rs.update(step=0)
    assert np.allclose(rs.biomass, k)


def test_update_on_regrows_instead_of_resetting():
    from osmose.engine.grid import Grid
    from osmose.engine.resources import ResourceState

    grid = Grid.from_dimensions(ny=3, nx=3)
    rs = ResourceState(config=_uniform_resource_config(depletable=True, rate="0.3"), grid=grid)
    rs.update(step=0)
    k = rs.biomass.copy()
    # Graze it down, then update: depletable REGROWS toward K (not a full reset), stays < K
    rs.biomass[:] = 0.5 * k
    rs.update(step=0)
    assert np.all(rs.biomass < k - 1e-9)  # not reset to K
    assert np.all(rs.biomass > 0.5 * k)  # but grew
    # fully grazed cell recovers above the floor, not stuck at 0
    rs.biomass[:] = 0.0
    rs.update(step=0)
    assert np.all(rs.biomass >= 0.05 * k - 1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_resources.py -k "update_off_is_full_reset or update_on_regrows" -v`
Expected: FAIL — `test_update_on_regrows_instead_of_resetting` fails (current `update` always resets to K).

- [ ] **Step 3: Gate `update()` on `depletable`**

In `ResourceState.update()`, replace the per-resource assignment so each case builds a full `k_row` (carrying capacity) and then either resets or regrows. Replace the body of the `for i in range(self.n_resources):` loop's assignment section (the three `self.biomass[i, ...] = ...` branches) with:

```python
            k_row = np.zeros(grid.ny * grid.nx, dtype=np.float64)
            if self._forcing_data is not None and rsc.name in self._forcing_data:
                step_in_year = step % n_dt_per_year
                forcing_idx = int(step_in_year * self._n_forcing_steps / n_dt_per_year)
                forcing_idx = min(forcing_idx, self._n_forcing_steps - 1)
                data = self._forcing_data[rsc.name].isel(time=forcing_idx).values
                biomass_2d = self._regrid_to_model(data)
                cell_biomass = biomass_2d.flatten() * rsc.multiplier * access
                k_row[: len(cell_biomass)] = cell_biomass[: grid.ny * grid.nx]
            elif self._uniform_biomass[i] > 0:
                k_row[:] = rsc.multiplier * (self._uniform_biomass[i] + rsc.offset) * access
            # else: k_row stays 0.0

            if self.depletable:
                self.biomass[i, :] = logistic_regrow(
                    self.biomass[i, :], k_row, rsc.regrowth_rate, self.depletable_floor
                )
            else:
                self.biomass[i, :] = k_row
```

(The `access` computation above this block is unchanged. This preserves the exact K math; the non-depletable branch writes the full `k_row`, which equals the prior partial assignment because the regrid always yields `ny*nx` cells — verified by the parity test and the full engine suite.)

- [ ] **Step 4: Register the new keys in config validation**

In `osmose/engine/config_validation.py`, add to the `_SUPPLEMENTARY_ALLOWLIST` frozenset:

```python
        "ltl.depletable.enabled",
        "ltl.depletable.floor",
        "ltl.regrowth.rate.default",
        "ltl.regrowth.rate.rsc{idx}",
        "species.regrowth.rate.sp{idx}",
```

- [ ] **Step 5: Run the new tests + the FULL engine suite (parity gate)**

Run: `.venv/bin/python -m pytest tests/test_engine_resources.py -q`
Expected: PASS (all resource tests).

Run (parity gate — the whole engine suite must be unchanged with depletable off):
`.venv/bin/python -m pytest tests/ -k "engine" -q`
Expected: PASS with the same pass/skip counts as before this task (no new failures). If anything fails, the parity of the non-depletable path is broken — fix `update()` before proceeding.

- [ ] **Step 6: Lint and commit**

```bash
cd /home/razinka/osmopy
.venv/bin/ruff check osmose/engine/resources.py osmose/engine/config_validation.py tests/test_engine_resources.py
.venv/bin/ruff format osmose/engine/resources.py osmose/engine/config_validation.py tests/test_engine_resources.py
git add osmose/engine/resources.py osmose/engine/config_validation.py tests/test_engine_resources.py
git commit -m "feat(engine): depletable-plankton regrowth in ResourceState.update (opt-in, parity-off)"
```

---

### Task 3: Real-engine — depletion sanity, ICES relaxation, bistability, write-up

**This task is real-engine, CLI-only, NOT CI.** Run after Tasks 1–2. Verification is that the sanity gate passes, the sweeps complete, and the results doc is written from the outputs. Regrowth-rate defaults for the run: phytoplankton (sp8,9) high (e.g. `5.0` ≈ fast), zooplankton (sp10–12) moderate (sweep target), benthos (sp13) low (e.g. `0.3`). Set via `species.regrowth.rate.sp{i}` overrides.

**Files:**
- Create: `docs/baltic_chunka2_results_2026-07-09.md`
- Produces: `docs/diagnostics/baltic_chunka2_*.json`

- [ ] **Step 1: Depletion sanity gate (STOP if pathological)**

Run one real Baltic run at a cod-established scale (larva ×0.3) with depletion on (zooplankton rate 0.3), and confirm resource biomass is drawn below K and recovers — not stuck at floor, not NaN/blow-up:

```bash
cd /home/razinka/osmopy
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "scripts")
import baltic_bistability_chunk0 as c0
from calibrate_baltic import run_simulation
base = c0.read_base_config()
rates = c0.read_base_larva_rates(base)
driver = c0.larva_scale_override(0.3, rates)
depl = {"ltl.depletable.enabled": "true", "ltl.depletable.floor": "0.05"}
for i in (8, 9): depl[f"species.regrowth.rate.sp{i}"] = "5.0"    # phyto fast
for i in (10, 11, 12): depl[f"species.regrowth.rate.sp{i}"] = "0.3"  # zoo depletable
depl["species.regrowth.rate.sp13"] = "0.3"                       # benthos slow
off = run_simulation(base, {**driver}, n_years=15, seed=0)
on = run_simulation(base, {**driver, **depl}, n_years=15, seed=0)
for sp in ("cod", "herring", "sprat", "perch"):
    print(f"{sp}: off={off.get(sp+'_mean'):.0f} on={on.get(sp+'_mean'):.0f}")
PY
```
Expected: the run completes (no NaN/1e22), and depletion measurably changes fish biomass (typically **lower** — less food) vs the control. If it crashes, NaNs, or every stock collapses to ~0, **STOP** and reassess the rates/floor before the sweeps (record the finding).

- [ ] **Step 2: ICES relaxation — zooplankton regrowth-rate sweep**

Sweep the zooplankton regrowth rate at the deployed larval mortality (×1.0) and at ×0.3; compare cod/herring/sprat to the ICES bands to find whether depletion pulls the community off the overshoot toward the bands. Use the same override dict as Step 1 with `species.regrowth.rate.sp{10,11,12}` ∈ {0.1, 0.3, 0.6}, `run_simulation(..., n_years=25)`, and print each stock vs its band (reuse the ICES-check pattern from `docs/superpowers/plans/2026-07-09-baltic-chunkc-clupeid-cod-egg-predation.md` Task 3 Step 4). Record the rate that best relaxes the overshoot.

- [ ] **Step 3: Bistability — warm-start regime-shift sweep with depletion on**

At the best-relaxing zooplankton rate from Step 2, run the warm-start regime-shift sweep with depletion on and write the result. Because depletion is config keys (not a harness flag), pass it by editing the base config in a short driver script that calls `c0.run_bistability_sweep(...)` with `base_config` merged with the depletion overrides, `warmstart=True, contrast="regime-shift", clupeid_targets=<herring+sprat>`, `n_years=25`, writing `docs/diagnostics/baltic_chunka2_regime-shift.json`. (Optionally layer Chunk C by also overriding `predation.accessibility.file` with a variant from `scripts/chunkc_accessibility.py`.) A determinate `regime-shift` outcome = bistability created.

- [ ] **Step 4: Write the results doc**

Create `docs/baltic_chunka2_results_2026-07-09.md` mirroring the prior results docs: the depletion sanity, the ICES relaxation table (per zooplankton rate), the regime-shift verdict (and Chunk C layered if run), and the honest interpretation — created bistability / relaxed the overshoot / negative (→ v2 total-pool depletion, or a combined lever).

- [ ] **Step 5: Commit results + diagnostics**

```bash
cd /home/razinka/osmopy
git add docs/baltic_chunka2_results_2026-07-09.md docs/diagnostics/baltic_chunka2_*.json
git commit -m "docs(baltic): Chunk A2 depletable-plankton results (2026-07-09)"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-07-09-baltic-chunka2-depletable-plankton-design.md`):
- per-resource logistic regrowth + floor → Task 1 `logistic_regrow` (exact equation) + Task 2 `update()` gate. ✓
- `regrowth_rate` on `ResourceSpeciesInfo`, parsed per resource with global default → Task 1 Steps 4–5. ✓
- config keys (`ltl.depletable.enabled`, `ltl.depletable.floor`, `species.regrowth.rate.sp{i}` / `ltl.regrowth.rate.rsc{i}`, `ltl.regrowth.rate.default`) → Task 1 Step 5 (parse) + Task 2 Step 4 (validation allowlist). ✓
- parity off = byte-identical → Task 2 Step 3 keeps the K math verbatim; Task 2 Step 5 runs the full engine suite as the gate; `test_update_off_is_full_reset_parity`. ✓
- no Numba/mortality/predation change → confirmed in the architecture; only `resources.py` + `config_validation.py` touched. ✓
- phyto-fast / zoo-slow / benthos-slow defaults respecting the 15-day timestep → Task 3 rate assignments. ✓
- depletion sanity STOP-gate, ICES relaxation, bistability test, Chunk C layering → Task 3 Steps 1–3. ✓
- outputs (JSONs + results doc) → Task 3 Steps 4–5. ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code; commands show expected output and STOP conditions.

**Type consistency:** `logistic_regrow(biomass, k, rate, floor)` signature matches its call in `update()`; `regrowth_rate` field name matches the loader assignments and the `rsc.regrowth_rate` read in `update()`; `depletable` / `depletable_floor` / `_regrowth_default` attribute names match between `__init__` and `update()`. Config-key spellings match between the parser (Task 1 Step 5) and the allowlist (Task 2 Step 4).
