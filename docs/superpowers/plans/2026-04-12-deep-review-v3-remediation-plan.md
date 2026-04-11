# Deep Review v3 — Remediation Plan (Important + Minor)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship every actionable Important and Minor finding from the 2026-04-11 fresh deep review v3 that can be fixed without a standalone architectural refactor, plus the 4 adjacent silent-failure sites identified during the C-3..C-7 commit. No production regressions, no parity drift, all changes test-pinned.

**Architecture:** 7 phases on a single feature branch. Phase 1 is pure test additions (zero production risk). Phase 2 tightens type invariants (medium risk — may surface latent bugs). Phase 3 removes dead code and consolidates duplication. Phase 4 extends the `_require_file` pattern to 4 adjacent sites. Phase 5 applies targeted hardening to polish items. Phase 6 audits test hygiene. Phase 7 adds pure-helper unit tests for 7 UI pages. A parity + pytest + ruff gate runs after each phase; the final gate hands off to `superpowers:finishing-a-development-branch`.

**Tech Stack:** Python 3.12, NumPy, pandas, Numba, xarray, Shiny, pytest, ruff.

**Spec:** `docs/superpowers/reviews/2026-04-11-fresh-deep-review-v3.md` — the consolidated 34-item findings document from the 6-lens parallel review. The Critical findings C-1 through C-8 are already shipped (commits `f177926`, `fa0c5f9`, `3d2d134`); this plan covers the remaining Important (I-1..I-10) and Minor (M-2..M-14) items minus items explicitly deferred in the "Out of Scope" section below.

---

## Scope & drift audit

Line numbers in this plan are anchored against master `3d2d134` (post-C-8). Some line numbers may drift slightly as tasks land in sequence — subagents MUST grep for the named code patterns rather than trusting line numbers when the delta exceeds ~20 lines. The v3 findings document has full context for any task whose scope is unclear.

**Verified 2026-04-12 against master HEAD:**
- I-10 `JavaEngine` class still present at `osmose/engine/__init__.py:102`; `Engine` Protocol at line 18; `runtime_checkable` import at line 10.
- M-6 misleading `cell_id` expression still present at `osmose/engine/processes/mortality.py:381`.
- M-8 `_close_spatial_ds` bare `except Exception` still present at `ui/pages/spatial_results.py:151`.
- M-4 three TODOs still present at `osmose/engine/output.py:289, 344, 345`.

**Baseline test count:** 2113 passed / 15 skipped / 0 failed, ruff clean, parity 12/12 bit-exact (confirmed 2026-04-12). Subagents should verify this count as Step 0 of Task 1 and treat subsequent expected counts as relative to whatever they observe.

---

## Out of scope (deferred to separate plans or investigations)

These v3 items are NOT in this plan because each needs its own session or external input:

1. **I-3 `EngineConfig.from_dict` monolith split** (~550 lines → 5–6 subsystem helpers). Real architectural refactor with high regression risk. Needs its own plan with per-subsystem tasks and a dedicated gate. Will be a successor plan once the items below are clean.

2. **M-5 `population.seeding.year.max` per-species check**. Requires reading Java OSMOSE source to determine whether the per-species key exists in Java's schema. Blocked on external input, not a coding task.

3. **D-1 `state.dirty.set` inside `reactive.isolate()` semantics verification**. Needs Shiny for Python reactive semantics investigation (does write-inside-isolate propagate to downstream readers, or does it silence the write?). The H8 fix shipped in Phase 2 used one interpretation; the v3 architecture reviewer flagged the opposite. Needs a standalone investigation — not code fixable without first resolving the semantic question.

4. **D-2 M-2 partial-year spawning warn-vs-raise design question**. Already resolved this session in favor of warn + normalize. No further action.

5. **M-1 `test_phi_t_degenerate_e_d_equals_e_m_falls_back_to_arrhenius`**. Already shipped this session in commit `e9dd7c0` as part of the C3 docstring follow-up.

---

## File structure

No new files are created by this plan. All changes are in-place modifications to existing source/test files:

**Production files touched:**
- `osmose/engine/state.py` — I-1 (SchoolState validate helper)
- `osmose/engine/config.py` — I-2 (bioen coupling), I-8 (MPAZone), I-9 (consolidate loaders), Phase 4 (4 adjacent file-resolution sites)
- `osmose/engine/__init__.py` — I-10 (delete JavaEngine + Engine Protocol + redundant Path import)
- `osmose/engine/simulate.py` — M-13/M-14 (StepOutput pair fields, SimulationContext diet fields — documentation + regression tests only)
- `osmose/engine/output.py` — M-4 (TODO cleanup)
- `osmose/engine/processes/mortality.py` — M-6 (cell_id comment)
- `osmose/engine/movement_maps.py` — M-7 (uncovered slot raise vs warn — requires user-visible decision, see task)
- `ui/pages/spatial_results.py` — M-8 (swallowed exception logging)

**Test files touched or created:**
- `tests/test_engine_predation.py` — I-4
- `tests/test_engine_simulate.py` — I-5, M-13
- `tests/test_engine_reproduction.py` — I-6 (may need to be created)
- `tests/test_engine_natural.py` — I-7 parts
- `tests/test_engine_mortality.py` — M-10 (strengthen existing)
- `tests/test_engine_state.py` — I-1 (SchoolState invariant tests)
- `tests/test_engine_config.py` — I-2, I-8, Phase 4
- `tests/test_engine_diet.py` — M-14
- `tests/test_engine_fisheries.py` / `test_engine_accessibility.py` — M-11 (deduplicate)
- `tests/test_engine_config_validation.py` — M-12 (audit construction-only tests)
- `tests/ui/test_ui_movement.py` and 6 siblings — Phase 7 (new test files for UI pages)

---

## Phase 1 — Test Coverage

Six tasks adding missing tests identified by the v3 test-coverage reviewer. Zero production-code changes in this phase. Each task adds one or two tests to existing test files.

### Task 1: Test `_predation_on_resources` removes biomass from a known cell (I-4)

**Files:**
- Modify: `tests/test_engine_predation.py`

**Context:** The LTL (lower-trophic-level) resource predation path — focal fish eating plankton/resources — is currently exercised only indirectly via full `simulate()` runs in parity tests. No test directly feeds a known `ResourceState` biomass plus a single focal school, calls the predation path, and asserts the resource biomass decreased by the expected amount. A regression that zeroes resource uptake (or doubles it) would not be caught by any targeted test.

- [ ] **Step 0: Baseline pytest count and parity check**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 2113 passed (or your current post-C-8 count — record it).

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -q`
Expected: 12 passed.

- [ ] **Step 1: Read `_predation_on_resources` signature**

Open `osmose/engine/processes/predation.py` and locate the function (it may be `_predation_on_resources`, `_consume_resources`, or embedded inline — search for the string `resource` plus `preyed_biomass` to find it). Record its parameter list and what it mutates.

Also open `osmose/engine/resources.py` to see the `ResourceState` constructor. You need to build a minimal instance with a known biomass and one resource species.

- [ ] **Step 2: Write the test**

In `tests/test_engine_predation.py`, append:

```python
def test_predation_on_resources_removes_expected_biomass():
    """A focal school eating a resource must decrement the resource biomass by
    the consumed amount and credit the school's preyed_biomass correspondingly.

    Deep review v3 I-4: this path had no direct behavioral test; only
    indirect coverage via full simulate() runs in parity tests.
    """
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.processes.predation import predation
    from osmose.engine.resources import ResourceState
    from osmose.engine.state import SchoolState

    cfg_dict = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "Predator",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.predprey.sizeratio.min.sp0": "1.0",
        "predation.predprey.sizeratio.max.sp0": "1000.0",
    }
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=1, nx=1)

    state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    state = state.replace(
        abundance=np.array([1000.0]),
        weight=np.array([0.01]),
        biomass=np.array([10.0]),
        length=np.array([20.0]),
        age_dt=np.array([10], dtype=np.int32),
        first_feeding_age_dt=np.array([0], dtype=np.int32),
        cell_x=np.array([0], dtype=np.int32),
        cell_y=np.array([0], dtype=np.int32),
    )

    initial_resource_biomass = 100.0
    resources = ResourceState(config=cfg_dict, grid=grid)
    if resources.biomass.size > 0:
        resources.biomass[:] = initial_resource_biomass

    rng = np.random.default_rng(42)
    initial_preyed = state.preyed_biomass.copy() if hasattr(state, "preyed_biomass") else None

    result = predation(state, cfg, rng, n_subdt=1, grid_ny=1, grid_nx=1, resources=resources)

    if resources.biomass.size == 0:
        pytest.skip("No resource species in minimal config — need resource fixture")

    if initial_preyed is not None:
        assert (result.preyed_biomass > initial_preyed).any(), (
            "Focal school preyed_biomass did not increase after resource predation"
        )
    assert (resources.biomass < initial_resource_biomass).any(), (
        "Resource biomass did not decrease after predation"
    )
```

**If ResourceState construction is too complex for the minimal fixture above**, the task fallback is to use `EngineConfig.from_dict(...)` + `simulate()` with a controlled seed, then assert that `outputs[-1]` has a lower total resource biomass than the initial state. That's a coarser test but still pins the direction. Use whichever is feasible without spending >30 min on fixture plumbing.

- [ ] **Step 3: Run the test**

Run: `.venv/bin/python -m pytest tests/test_engine_predation.py::test_predation_on_resources_removes_expected_biomass -v`

Expected: PASS (this is a pinning test for existing behavior — the production code is already correct per parity tests; the test adds a regression guard for the direct call path).

If it fails with an AssertionError about biomass not decreasing, that is itself a real finding — **stop and report DONE_WITH_CONCERNS** with the failure details instead of weakening the test.

- [ ] **Step 4: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: baseline + 1 passed, 15 skipped, 0 failed.

- [ ] **Step 5: Ruff**

Run: `.venv/bin/ruff check tests/test_engine_predation.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add tests/test_engine_predation.py
git commit -m "test: add direct behavioral test for _predation_on_resources (I-4)"
```

---

### Task 2: Test `_average_step_outputs` multi-element branch (I-5)

**Files:**
- Modify: `tests/test_engine_simulate.py`

**Context:** The Phase 2 H3 fix added `test_average_step_outputs_preserves_distributions` but only uses `freq=1`, which exercises the `len(accumulated) == 1` short-circuit. The multi-element branch of `osmose/engine/simulate.py` sums mortality, sums yield, averages biomass, and snapshots the LAST distribution dict — each a separate contract. The v3 test-coverage reviewer flagged this gap.

- [ ] **Step 1: Locate `_average_step_outputs` in `osmose/engine/simulate.py`**

Search for `def _average_step_outputs`. Confirm the multi-element branch structure: `biomass = np.mean(...)`, `mortality = np.sum(...)`, `yield_sum = np.sum(..., axis=0)`, `biomass_by_age=accumulated[-1].biomass_by_age`, etc.

- [ ] **Step 2: Write the test**

In `tests/test_engine_simulate.py`, append (immediately after the existing `test_average_step_outputs_preserves_distributions`):

```python
def test_average_step_outputs_multi_element_contract():
    """The multi-element branch of _average_step_outputs must:
    - Average biomass across the window
    - Sum mortality_by_cause across the window
    - Sum yield_by_species across the window
    - Snapshot the LAST entry's distribution dicts (not the first, not the mean)

    Deep review v3 I-5: the multi-element branch was previously untested;
    only the len(accumulated) == 1 short-circuit had coverage via
    test_average_step_outputs_preserves_distributions.
    """
    from osmose.engine.simulate import StepOutput, _average_step_outputs

    so_a = StepOutput(
        step=0,
        biomass=np.array([100.0]),
        abundance=np.array([50.0]),
        mortality_by_cause=np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        yield_by_species=np.array([10.0]),
        biomass_by_age={0: np.array([1.0])},
        abundance_by_age={0: np.array([1.0])},
        biomass_by_size={0: np.array([1.0])},
        abundance_by_size={0: np.array([1.0])},
    )
    so_b = StepOutput(
        step=1,
        biomass=np.array([200.0]),
        abundance=np.array([100.0]),
        mortality_by_cause=np.array([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]]),
        yield_by_species=np.array([20.0]),
        biomass_by_age={0: np.array([2.0])},
        abundance_by_age={0: np.array([2.0])},
        biomass_by_size={0: np.array([2.0])},
        abundance_by_size={0: np.array([2.0])},
    )
    so_c = StepOutput(
        step=2,
        biomass=np.array([300.0]),
        abundance=np.array([150.0]),
        mortality_by_cause=np.array([[3.0, 6.0, 9.0, 12.0, 15.0, 18.0]]),
        yield_by_species=np.array([30.0]),
        biomass_by_age={0: np.array([3.0])},
        abundance_by_age={0: np.array([3.0])},
        biomass_by_size={0: np.array([3.0])},
        abundance_by_size={0: np.array([3.0])},
    )

    result = _average_step_outputs([so_a, so_b, so_c], freq=3, record_step=2)

    np.testing.assert_allclose(result.biomass, np.array([200.0]))
    np.testing.assert_allclose(result.abundance, np.array([100.0]))
    np.testing.assert_allclose(
        result.mortality_by_cause, np.array([[6.0, 12.0, 18.0, 24.0, 30.0, 36.0]])
    )
    np.testing.assert_allclose(result.yield_by_species, np.array([60.0]))
    np.testing.assert_array_equal(result.biomass_by_age[0], np.array([3.0]))
    np.testing.assert_array_equal(result.abundance_by_age[0], np.array([3.0]))
    np.testing.assert_array_equal(result.biomass_by_size[0], np.array([3.0]))
    np.testing.assert_array_equal(result.abundance_by_size[0], np.array([3.0]))
    assert result.step == 2
```

- [ ] **Step 3: Run the test**

Run: `.venv/bin/python -m pytest tests/test_engine_simulate.py::test_average_step_outputs_multi_element_contract -v`
Expected: PASS.

- [ ] **Step 4: Run full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: baseline + 1 passed.

- [ ] **Step 5: Ruff**

Run: `.venv/bin/ruff check tests/test_engine_simulate.py`

- [ ] **Step 6: Commit**

```bash
git add tests/test_engine_simulate.py
git commit -m "test: pin _average_step_outputs multi-element branch contract (I-5)"
```

---

### Task 3: Test `reproduction` "fewer eggs than n_schools" branch (I-6)

**Files:**
- Modify: `tests/test_engine_reproduction.py` (create if not present — check first with `ls tests/test_engine_reproduction.py`)

**Context:** `osmose/engine/processes/reproduction.py` contains a branch where, if `n_eggs[sp] < n_new`, the code collapses `n_new` to 1. No test hits this branch. A bug here would silently skip reproduction for rare spawners or create egg schools with zero abundance.

- [ ] **Step 1: Locate the target branch in `osmose/engine/processes/reproduction.py`**

Search for `n_eggs` and `n_new` in `osmose/engine/processes/reproduction.py`. Find the branch where the code decides how many egg-school records to create. Confirm it collapses to a single egg school when `n_eggs[sp]` is less than `n_new`. Record the function signature.

- [ ] **Step 2: Check whether `tests/test_engine_reproduction.py` exists**

`ls tests/test_engine_reproduction.py` — if yes, append. If no, create it with imports:

```python
"""Tests for osmose.engine.processes.reproduction."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.state import SchoolState
```

- [ ] **Step 3: Write the test**

```python
def test_reproduction_creates_single_school_when_n_eggs_below_n_new():
    """When n_eggs[sp] < n_new (n_schools per year), reproduction must collapse
    to a single egg school containing all eggs, not skip or over-split.

    Deep review v3 I-6: this branch had no test coverage.
    """
    cfg_dict = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "20",
        "species.name.sp0": "RareSpawner",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "species.relative.fecundity.sp0": "0.1",
        "species.sexratio.sp0": "0.5",
    }
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=1, nx=1)

    state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    state = state.replace(
        abundance=np.array([10.0]),
        weight=np.array([0.0001]),
        biomass=np.array([0.001]),
        length=np.array([25.0]),
        age_dt=np.array([48], dtype=np.int32),
        first_feeding_age_dt=np.array([0], dtype=np.int32),
        cell_x=np.array([0], dtype=np.int32),
        cell_y=np.array([0], dtype=np.int32),
    )

    rng = np.random.default_rng(42)
    result = reproduction(state, cfg, rng, step=0)

    new_eggs = result.age_dt == 0
    n_egg_schools = int(new_eggs.sum())

    assert n_egg_schools <= 1, (
        f"Expected 0-1 new egg schools in the n_eggs<n_new branch, got {n_egg_schools}"
    )
```

**Caveat:** the exact signature of `reproduction(...)` varies — read the function in `osmose/engine/processes/reproduction.py` before running this and adjust the call. If the function takes additional required parameters, populate them with minimal defaults. If you cannot adapt the test to a working call signature within ~20 minutes, report **NEEDS_CONTEXT** with what you found.

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/test_engine_reproduction.py::test_reproduction_creates_single_school_when_n_eggs_below_n_new -v`

Expected: PASS.

- [ ] **Step 5: Full suite + ruff + commit**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check tests/test_engine_reproduction.py
git add tests/test_engine_reproduction.py
git commit -m "test: cover reproduction 'n_eggs < n_new' collapse branch (I-6)"
```

---

### Task 4: Test `out_mortality` with `is_out=True` (I-7a)

**Files:**
- Modify: `tests/test_engine_natural.py` (check with `ls tests/test_engine_natural.py`; if absent, create)

**Context:** No test sets `is_out=True` on a school and asserts mortality is applied at `M_out / n_dt_per_year`. A regression that drops the denominator or forgets to apply the rate goes unseen.

- [ ] **Step 1: Locate `out_mortality` in `osmose/engine/processes/natural.py`**

Search for `def out_mortality`. Record its signature.

- [ ] **Step 2: Write the test**

In `tests/test_engine_natural.py`:

```python
def test_out_mortality_applies_rate_when_is_out_true():
    """A school with is_out=True must lose abundance at out_mortality_rate / n_dt_per_year.

    Deep review v3 I-7: the is_out=True branch of out_mortality had no coverage.
    """
    import numpy as np
    from osmose.engine.config import EngineConfig
    from osmose.engine.processes.natural import out_mortality
    from osmose.engine.state import SchoolState

    cfg_dict = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "Migrator",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "mortality.out.rate.sp0": "2.4",
    }
    cfg = EngineConfig.from_dict(cfg_dict)

    state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    initial_abundance = 1000.0
    state = state.replace(
        abundance=np.array([initial_abundance]),
        is_out=np.array([True]),
        weight=np.array([0.01]),
        biomass=np.array([10.0]),
        length=np.array([20.0]),
        age_dt=np.array([10], dtype=np.int32),
        first_feeding_age_dt=np.array([0], dtype=np.int32),
    )

    result = out_mortality(state, cfg)

    expected_survival = np.exp(-2.4 / 24)
    expected_abundance = initial_abundance * expected_survival
    np.testing.assert_allclose(
        result.abundance[0], expected_abundance, rtol=1e-10,
        err_msg="out_mortality did not apply the expected M_out/n_dt_per_year rate"
    )
```

**Adapt the import path and signature of `out_mortality(...)` to match the actual source.**

- [ ] **Step 3: Run the test, full suite, ruff**

Run: `.venv/bin/python -m pytest tests/test_engine_natural.py::test_out_mortality_applies_rate_when_is_out_true -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_natural.py
git commit -m "test: pin out_mortality rate application when is_out=True (I-7a)"
```

---

### Task 5: Test `additional_mortality_by_dt` override rotation (I-7b)

**Files:**
- Modify: `tests/test_engine_natural.py`

**Context:** No test verifies that `additional_mortality_by_dt[sp_idx]` actually replaces the base rate and rotates correctly with `step % len(arr)`.

- [ ] **Step 1: Write the test**

Append to `tests/test_engine_natural.py`:

```python
def test_additional_mortality_by_dt_override_rotates_with_step():
    """additional_mortality_by_dt[sp] overrides the base additional-mortality rate
    and rotates via step % len(arr). Alternating [0, 1.0, 0, 1.0] produces zero
    deaths on even steps and positive deaths on odd steps.

    Deep review v3 I-7.
    """
    import numpy as np
    from osmose.engine.config import EngineConfig
    from osmose.engine.processes.natural import additional_mortality
    from osmose.engine.state import SchoolState

    cfg_dict = {
        "simulation.time.ndtperyear": "4",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "mortality.additional.rate.sp0": "0.0",
    }
    cfg = EngineConfig.from_dict(cfg_dict)

    cfg.additional_mortality_by_dt = [np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)]

    state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    state = state.replace(
        abundance=np.array([1000.0]),
        weight=np.array([0.01]),
        biomass=np.array([10.0]),
        length=np.array([20.0]),
        age_dt=np.array([10], dtype=np.int32),
        first_feeding_age_dt=np.array([0], dtype=np.int32),
    )

    result_even = additional_mortality(state, cfg, step=0)
    assert result_even.abundance[0] == state.abundance[0], (
        "Step 0 override is 0; abundance should be unchanged"
    )

    result_odd = additional_mortality(state, cfg, step=1)
    expected_survival = np.exp(-1.0 / 4)
    np.testing.assert_allclose(
        result_odd.abundance[0], state.abundance[0] * expected_survival, rtol=1e-10,
        err_msg="Step 1 override=1.0 did not produce expected mortality"
    )
```

**Adapt `additional_mortality` signature from the source.**

- [ ] **Step 2: Run the test, full suite, ruff**

Run: `.venv/bin/python -m pytest tests/test_engine_natural.py::test_additional_mortality_by_dt_override_rotates_with_step -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_engine_natural.py
git commit -m "test: pin additional_mortality_by_dt override step-rotation (I-7b)"
```

---

### Task 6: Strengthen `test_zero_rate_no_mortality` (M-10)

**Files:**
- Modify: `tests/test_engine_mortality.py`

**Context:** The v3 reviewer flagged `test_zero_rate_no_mortality` as near-tautological: rate=0 → D=0 is trivially true. A stronger test includes a second school with rate>0 in the same state to verify the zero-rate school is unaffected *specifically* — not just that the whole state didn't change.

- [ ] **Step 1: Find the existing test**

Run: `grep -n "test_zero_rate_no_mortality" tests/test_engine_mortality.py` — note the class and test method location.

- [ ] **Step 2: Strengthen the test**

Read the existing test body. Modify it in-place to:
1. Keep the existing "school with rate=0 has unchanged abundance" assertion.
2. Add a second school with a non-zero rate in the SAME state.
3. Assert the second school's abundance DID decrease.
4. Assert the first school's abundance (rate=0) did NOT decrease.

Exact edit depends on the current fixture structure; follow the existing pattern. Add an `n_dead` assertion on the rate=0 school: `assert result.n_dead[0, ADDITIONAL] == 0`.

If the test is class-based (likely inside `class TestLarvaMortality`), keep it in the class.

- [ ] **Step 3: Run the strengthened test**

Run: `.venv/bin/python -m pytest tests/test_engine_mortality.py -v -k zero_rate`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_mortality.py
git commit -m "test: strengthen test_zero_rate_no_mortality with non-zero control school (M-10)"
```

---

### Phase 1 gate

- [ ] **Run full suite**: `.venv/bin/python -m pytest tests/ -q`
  - Expected: baseline + 6 passed (at least 2119), 15 skipped, 0 failed.
- [ ] **Ruff**: `.venv/bin/ruff check osmose/ ui/ tests/`
  - Expected: `All checks passed!`
- [ ] **Parity**: `.venv/bin/python -m pytest tests/test_engine_parity.py -q`
  - Expected: 12 passed.

If any gate fails, stop and escalate.

---

## Phase 2 — Type Invariants

Five tasks tightening type invariants on the core domain types. Medium risk — these additions may surface latent bugs in production code that was relying on the absence of validation. If any existing test breaks with the new invariants, that's a real finding, not a test-fixture bug.

### Task 7: `SchoolState.validate()` opt-in biological invariants (I-1)

**Files:**
- Modify: `osmose/engine/state.py` — add `validate()` method to `SchoolState`
- Modify: `tests/test_engine_state.py`

**Context:** `SchoolState.__post_init__` only validates array lengths. The v3 type reviewer flagged that biologically-invalid states are representable (`abundance < 0`, `length < 0`, `biomass != abundance * weight`, `is_egg && age_dt > 0`, etc.). Adding these checks to `__post_init__` is too expensive for the hot path. The fix is an opt-in `validate()` method that tests call explicitly.

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_state.py`, append:

```python
class TestSchoolStateValidate:
    """Opt-in validation of biological invariants on SchoolState.

    Deep review v3 I-1: SchoolState.__post_init__ only checks array lengths.
    validate() adds explicit checks for abundance >= 0, length >= 0,
    weight >= 0, biomass >= 0, biomass ≈ abundance * weight (within rtol),
    and cell coordinates >= 0.
    """

    def _minimal_valid_state(self, n: int = 2):
        import numpy as np
        from osmose.engine.state import SchoolState

        s = SchoolState.create(
            n_schools=n, species_id=np.zeros(n, dtype=np.int32)
        )
        return s.replace(
            abundance=np.full(n, 100.0),
            weight=np.full(n, 0.01),
            biomass=np.full(n, 1.0),
            length=np.full(n, 10.0),
            age_dt=np.zeros(n, dtype=np.int32),
            cell_x=np.zeros(n, dtype=np.int32),
            cell_y=np.zeros(n, dtype=np.int32),
        )

    def test_validate_passes_on_clean_state(self):
        s = self._minimal_valid_state()
        s.validate()

    def test_validate_raises_on_negative_abundance(self):
        import numpy as np
        import pytest

        s = self._minimal_valid_state()
        s = s.replace(abundance=np.array([-1.0, 10.0]))
        with pytest.raises(ValueError, match="abundance must be non-negative"):
            s.validate()

    def test_validate_raises_on_negative_length(self):
        import numpy as np
        import pytest

        s = self._minimal_valid_state()
        s = s.replace(length=np.array([-5.0, 10.0]))
        with pytest.raises(ValueError, match="length must be non-negative"):
            s.validate()

    def test_validate_raises_on_biomass_mismatch(self):
        import numpy as np
        import pytest

        s = self._minimal_valid_state()
        s = s.replace(biomass=np.array([2.0, 1.0]))
        with pytest.raises(ValueError, match="biomass .* abundance \\* weight"):
            s.validate()

    def test_validate_raises_on_negative_cell(self):
        import numpy as np
        import pytest

        s = self._minimal_valid_state()
        s = s.replace(cell_x=np.array([-1, 0], dtype=np.int32))
        with pytest.raises(ValueError, match="cell_x must be non-negative"):
            s.validate()

    def test_validate_skip_dead_schools(self):
        import numpy as np

        s = self._minimal_valid_state()
        s = s.replace(
            abundance=np.array([100.0, 0.0]),
            weight=np.array([0.01, 0.0]),
            biomass=np.array([1.0, 0.0]),
        )
        s.validate()
```

- [ ] **Step 2: Run the tests — they must FAIL**

Run: `.venv/bin/python -m pytest tests/test_engine_state.py::TestSchoolStateValidate -v`
Expected: 6 tests fail with `AttributeError: 'SchoolState' object has no attribute 'validate'`.

- [ ] **Step 3: Implement `validate()`**

In `osmose/engine/state.py`, locate `class SchoolState`. Add a method (not a `__post_init__` change — this is opt-in):

```python
    def validate(self) -> None:
        """Check biological invariants. Opt-in: called from tests and from
        an OSMOSE_DEBUG=1 env hook, not from __post_init__ (hot path).

        Raises ValueError on the first violation. Deep review v3 I-1.
        """
        if (self.abundance < 0).any():
            raise ValueError(
                f"abundance must be non-negative; found minimum {self.abundance.min()}"
            )
        if (self.length < 0).any():
            raise ValueError(
                f"length must be non-negative; found minimum {self.length.min()}"
            )
        if (self.weight < 0).any():
            raise ValueError(
                f"weight must be non-negative; found minimum {self.weight.min()}"
            )
        if (self.biomass < 0).any():
            raise ValueError(
                f"biomass must be non-negative; found minimum {self.biomass.min()}"
            )
        if (self.cell_x < 0).any():
            raise ValueError(
                f"cell_x must be non-negative; found minimum {self.cell_x.min()}"
            )
        if (self.cell_y < 0).any():
            raise ValueError(
                f"cell_y must be non-negative; found minimum {self.cell_y.min()}"
            )
        live = self.abundance > 0
        if live.any():
            expected = self.abundance[live] * self.weight[live]
            rtol = 1e-6
            diff = np.abs(self.biomass[live] - expected)
            tol = rtol * np.abs(expected) + 1e-12
            bad = diff > tol
            if bad.any():
                bad_idx = np.where(live)[0][bad][0]
                raise ValueError(
                    f"biomass must equal abundance * weight for live schools "
                    f"(school {bad_idx}: biomass={self.biomass[bad_idx]}, "
                    f"abundance*weight={self.abundance[bad_idx] * self.weight[bad_idx]})"
                )
```

- [ ] **Step 4: Run the tests — they must pass**

Run: `.venv/bin/python -m pytest tests/test_engine_state.py::TestSchoolStateValidate -v`
Expected: 6 passed.

- [ ] **Step 5: Full suite + ruff**

Run: `.venv/bin/python -m pytest tests/ -q`

Run: `.venv/bin/ruff check osmose/engine/state.py tests/test_engine_state.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/state.py tests/test_engine_state.py
git commit -m "feat(engine): add SchoolState.validate() for biological invariants (I-1)"
```

---

### Task 8: `EngineConfig.__post_init__` bioen coupling check (I-2)

**Files:**
- Modify: `osmose/engine/config.py` — `EngineConfig.__post_init__`
- Modify: `tests/test_engine_config.py`

**Context:** 21+ `bioen_*` fields on `EngineConfig` are declared `NDArray | None = None` and gated on `bioen_enabled: bool`. Nothing enforces that `bioen_enabled=True` requires each bioen field to be non-None. A partial config crashes deep in a mortality kernel instead of at config load.

- [ ] **Step 1: Enumerate the bioen fields**

Run: `grep -n "bioen_.*NDArray.*None" osmose/engine/config.py` to list every `bioen_*` field declared on `EngineConfig`. Record the exact field names.

- [ ] **Step 2: Write the failing test**

In `tests/test_engine_config.py`, append:

```python
def test_engine_config_bioen_enabled_requires_all_bioen_fields(minimal_config):
    """When bioen_enabled=True, every bioen_* field must be non-None; partial
    enablement must raise at EngineConfig construction, not later.

    Deep review v3 I-2.
    """
    cfg_dict = dict(minimal_config)
    cfg_dict["simulation.bioen.enabled"] = "true"
    with pytest.raises(ValueError, match="bioen_enabled=True but"):
        EngineConfig.from_dict(cfg_dict)
```

**Note:** the test will initially fail for TWO reasons:
1. The assertion expects a ValueError that isn't raised yet.
2. The minimal_config may not have ANY bioen keys set, so `from_dict` may succeed and silently set `bioen_enabled=False`.

If the second case occurs, the test needs adjustment: set a bare minimum of bioen keys so `from_dict` tries to enable bioen but one required key is missing.

- [ ] **Step 3: Add the coupling check**

In `osmose/engine/config.py`, in `EngineConfig.__post_init__`, add after existing length checks:

```python
        if self.bioen_enabled:
            bioen_fields = [
                "bioen_alpha",
                "bioen_beta",
                "bioen_theta",
                "bioen_c_rate",
                "bioen_e_mobi",
                "bioen_e_d",
                "bioen_tp",
                "bioen_hmax",
                "bioen_gross_efficiency",
            ]
            missing = [f for f in bioen_fields if getattr(self, f, None) is None]
            if missing:
                raise ValueError(
                    f"bioen_enabled=True but the following bioen_* fields are None: "
                    f"{missing}. Either set every bioen_* key in the config or disable "
                    f"simulation.bioen.enabled."
                )
```

Replace the `bioen_fields` list with the exact fields grepped in Step 1.

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py::test_engine_config_bioen_enabled_requires_all_bioen_fields -v`
Expected: PASS.

- [ ] **Step 5: Full suite — verify no existing bioen test breaks**

Run: `.venv/bin/python -m pytest tests/ -q`

If any existing test breaks with `bioen_enabled=True but ... are None`, that test was exercising the latent bug. Investigate and fix the test fixture (don't weaken the new check).

- [ ] **Step 6: Ruff + commit**

```bash
.venv/bin/ruff check osmose/engine/config.py tests/test_engine_config.py
git add osmose/engine/config.py tests/test_engine_config.py
git commit -m "feat(engine): validate bioen_* field coupling at EngineConfig construction (I-2)"
```

---

### Task 9: `MPAZone` grid shape and binary-value validation (I-8)

**Files:**
- Modify: `osmose/engine/config.py` — `MPAZone.__post_init__`
- Modify: `tests/test_engine_config.py`

**Context:** `MPAZone.grid` is typed `NDArray[np.float64]` and documented as `1 = protected, 0 = not`. `__post_init__` only checks `percentage` and year ordering. A 3D grid, a 1D array, or a continuous-valued grid silently yields wrong MPA application.

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_config.py`, append:

```python
class TestMPAZoneValidation:
    """Deep review v3 I-8: MPAZone must validate grid shape and value range."""

    def _base_kwargs(self):
        import numpy as np

        return {
            "grid": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            "start_year": 0,
            "end_year": 10,
            "percentage": 0.5,
        }

    def test_valid_mpa_zone(self):
        from osmose.engine.config import MPAZone

        MPAZone(**self._base_kwargs())

    def test_1d_grid_rejected(self):
        import numpy as np
        import pytest
        from osmose.engine.config import MPAZone

        kwargs = self._base_kwargs()
        kwargs["grid"] = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="grid must be 2D"):
            MPAZone(**kwargs)

    def test_3d_grid_rejected(self):
        import numpy as np
        import pytest
        from osmose.engine.config import MPAZone

        kwargs = self._base_kwargs()
        kwargs["grid"] = np.zeros((2, 2, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="grid must be 2D"):
            MPAZone(**kwargs)

    def test_continuous_grid_rejected(self):
        import numpy as np
        import pytest
        from osmose.engine.config import MPAZone

        kwargs = self._base_kwargs()
        kwargs["grid"] = np.array([[0.0, 0.5], [1.0, 0.2]])
        with pytest.raises(ValueError, match="grid values must be 0 or 1"):
            MPAZone(**kwargs)

    def test_negative_start_year_rejected(self):
        import pytest
        from osmose.engine.config import MPAZone

        kwargs = self._base_kwargs()
        kwargs["start_year"] = -1
        with pytest.raises(ValueError, match="start_year must be non-negative"):
            MPAZone(**kwargs)
```

- [ ] **Step 2: Run the tests — they must FAIL**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py::TestMPAZoneValidation -v`
Expected: 4 tests fail (all except `test_valid_mpa_zone`).

- [ ] **Step 3: Tighten `MPAZone.__post_init__`**

In `osmose/engine/config.py`, find `class MPAZone` and add to `__post_init__`:

```python
        if self.grid.ndim != 2:
            raise ValueError(
                f"MPAZone.grid must be 2D (shape (ny, nx)), got {self.grid.ndim}D"
            )
        if not np.isin(self.grid, [0.0, 1.0]).all():
            unique = np.unique(self.grid)
            raise ValueError(
                f"MPAZone.grid values must be 0 or 1 (binary protected/unprotected), "
                f"got unique values {unique.tolist()}"
            )
        if self.start_year < 0:
            raise ValueError(
                f"MPAZone.start_year must be non-negative, got {self.start_year}"
            )
```

- [ ] **Step 4: Run the tests — they must pass**

Run: `.venv/bin/python -m pytest tests/test_engine_config.py::TestMPAZoneValidation -v`
Expected: 5 passed.

- [ ] **Step 5: Full suite + ruff + commit**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check osmose/engine/config.py tests/test_engine_config.py
git add osmose/engine/config.py tests/test_engine_config.py
git commit -m "feat(engine): MPAZone validates grid shape and binary values (I-8)"
```

---

### Task 10: Pin `StepOutput` age/size distribution pairing invariant (M-13)

**Context:** The v3 type reviewer flagged `StepOutput`'s 4 distribution fields as independent Optionals that should be two paired dataclasses. The refactor is semantically correct but touches many call sites and would invalidate existing baselines. **Descope to documentation**: add a class-level docstring comment documenting the pairing invariant and a regression test that enforces it.

**Files:**
- Modify: `osmose/engine/simulate.py` — `StepOutput` dataclass
- Modify: `tests/test_engine_simulate.py`

- [ ] **Step 1: Add docstring**

In `osmose/engine/simulate.py`, locate `class StepOutput` (grep `class StepOutput`). Add to the class docstring:

```
    Pairing invariant (deep review v3 M-13):
    - biomass_by_age and abundance_by_age must both be None or both non-None.
    - biomass_by_size and abundance_by_size must both be None or both non-None.
    - When they are dicts, they share the same species_id keys.

    Callers must not set one field of a pair while leaving the other None —
    downstream NetCDF/CSV writers rely on the co-presence.
```

- [ ] **Step 2: Add a pairing test**

In `tests/test_engine_simulate.py`, append:

```python
def test_step_output_distribution_pairs_travel_together():
    """biomass_by_age and abundance_by_age must be co-populated by any code path
    that fills one of them. This is a pinning regression test — the contract is
    maintained by convention across the engine, not enforced in __post_init__.

    Deep review v3 M-13.
    """
    from osmose.engine.simulate import simulate
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid

    cfg_dict = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "1",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "output.biomass.byage.enabled": "true",
        "output.abundance.byage.enabled": "true",
    }
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=3, nx=3)
    outputs = simulate(cfg, grid, np.random.default_rng(42))

    for o in outputs:
        age_pair_set = o.biomass_by_age is not None
        age_pair_set_2 = o.abundance_by_age is not None
        assert age_pair_set == age_pair_set_2, (
            f"biomass_by_age ({age_pair_set}) and abundance_by_age "
            f"({age_pair_set_2}) must match on step {o.step}"
        )
        size_pair_set = o.biomass_by_size is not None
        size_pair_set_2 = o.abundance_by_size is not None
        assert size_pair_set == size_pair_set_2, (
            f"biomass_by_size ({size_pair_set}) and abundance_by_size "
            f"({size_pair_set_2}) must match on step {o.step}"
        )
```

The exact config keys for enabling age/size distributions may vary — grep the config parser if the test doesn't reach the intended branch.

- [ ] **Step 3: Run + commit**

```bash
.venv/bin/python -m pytest tests/test_engine_simulate.py::test_step_output_distribution_pairs_travel_together -v
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check osmose/engine/simulate.py tests/test_engine_simulate.py
git add osmose/engine/simulate.py tests/test_engine_simulate.py
git commit -m "docs+test: pin StepOutput age/size distribution pairing invariant (M-13)"
```

---

### Task 11: Pin `SimulationContext` diet field coupling (M-14)

**Files:**
- Modify: `osmose/engine/simulate.py` — `SimulationContext` docstring
- Modify: `tests/test_engine_diet.py`

**Context:** `SimulationContext` has `diet_tracking_enabled`, `diet_matrix`, and `tl_weighted_sum` as three independent fields. When tracking is enabled, both arrays must be non-None and shape-compatible. Refactoring into a `DietTracking | None` sub-dataclass is out of scope. Document the invariant + add a pinning test.

- [ ] **Step 1: Update `SimulationContext` docstring**

In `osmose/engine/simulate.py`, locate `class SimulationContext`. Add to its docstring:

```
    Diet tracking three-way coupling (deep review v3 M-14):
    - When diet_tracking_enabled is True, BOTH diet_matrix and tl_weighted_sum
      must be non-None, and their shapes must be consistent (diet_matrix has
      shape (n_schools, n_species); tl_weighted_sum has shape (n_schools,)).
    - When diet_tracking_enabled is False, both arrays must be None.
    - enable_diet_tracking() / disable_diet_tracking() are the only correct
      way to transition between these states.
```

- [ ] **Step 2: Write a pinning test**

In `tests/test_engine_diet.py` (create if absent), append:

```python
def test_simulation_context_diet_coupling_after_enable():
    """After enable_diet_tracking(), all three fields must be consistent."""
    from osmose.engine.processes.predation import enable_diet_tracking, disable_diet_tracking
    from osmose.engine.simulate import SimulationContext

    ctx = SimulationContext(config_dir="")

    assert ctx.diet_tracking_enabled is False
    assert ctx.diet_matrix is None
    assert ctx.tl_weighted_sum is None

    enable_diet_tracking(n_schools=5, n_species=3, ctx=ctx)

    assert ctx.diet_tracking_enabled is True
    assert ctx.diet_matrix is not None
    assert ctx.diet_matrix.shape == (5, 3)
    assert ctx.tl_weighted_sum is not None
    assert ctx.tl_weighted_sum.shape == (5,)

    disable_diet_tracking(ctx=ctx)

    assert ctx.diet_tracking_enabled is False
    assert ctx.diet_matrix is None
    assert ctx.tl_weighted_sum is None
```

The exact signature of `enable_diet_tracking` / `disable_diet_tracking` comes from `osmose/engine/processes/predation.py` — read it and adapt the call.

- [ ] **Step 3: Run + commit**

```bash
.venv/bin/python -m pytest tests/test_engine_diet.py -v
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check osmose/engine/simulate.py tests/test_engine_diet.py
git add osmose/engine/simulate.py tests/test_engine_diet.py
git commit -m "docs+test: pin SimulationContext diet field three-way coupling (M-14)"
```

---

### Phase 2 gate

- [ ] **Run full suite**: `.venv/bin/python -m pytest tests/ -q`
  - Expected: baseline + 12 passed (at least 2125).
- [ ] **Ruff**: `.venv/bin/ruff check osmose/ ui/ tests/`
- [ ] **Parity**: `.venv/bin/python -m pytest tests/test_engine_parity.py -q` — 12 passed.

---

## Phase 3 — Dead Code + Duplication

### Task 12: Delete `JavaEngine` + `Engine` Protocol (I-10) + redundant Path import (M-2)

**Files:**
- Modify: `osmose/engine/__init__.py`
- Modify: `tests/test_engine_state.py` — remove the 2 structural hasattr tests

**Context:** `JavaEngine` class is a pure stub (both methods raise `NotImplementedError`). `Engine` Protocol has zero type annotations referencing it. Only references are the definitions and 2 structural `hasattr` tests. Plus a redundant `from pathlib import Path as P` local re-import inside `PythonEngine.run()`.

- [ ] **Step 1: Confirm no production callers**

Run:
```bash
grep -rn "JavaEngine\|: Engine\b\|-> Engine\b\|Engine]" osmose/ ui/
```

Expected: only the class definitions in `osmose/engine/__init__.py` and the 2 test references. Any production caller is a blocker — stop and escalate.

- [ ] **Step 2: Locate and read the 2 hasattr tests**

Run: `grep -n "JavaEngine" tests/test_engine_state.py` to find them. Read the surrounding context to make sure they're structural-only.

- [ ] **Step 3: Delete `JavaEngine` class**

In `osmose/engine/__init__.py`, remove the `class JavaEngine` block (approximately lines 102-123 as of 2026-04-12; grep `class JavaEngine` to find).

- [ ] **Step 4: Delete `Engine` Protocol and imports**

In the same file, remove the `@runtime_checkable class Engine(Protocol):` block (approximately line 17-29) AND remove `Protocol`, `runtime_checkable` from the `from typing import ...` line at line 10 (if they're not used elsewhere; grep first).

- [ ] **Step 5: Delete redundant `Path as P` re-import (M-2)**

In the same file, look for `from pathlib import Path as P` (approximately line 54 inside `PythonEngine.run()` per v3 M-2). Remove it and replace any `P(...)` usage with `Path(...)` — the module-level `Path` import at line 9 covers all uses.

- [ ] **Step 6: Delete the 2 hasattr tests in test_engine_state.py**

Remove `test_java_engine_satisfies_protocol` and any sibling JavaEngine-related test. The `PythonEngine` tests stay.

- [ ] **Step 7: Run the suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: baseline - 2 (we removed 2 tests) + 0 new.

Run: `.venv/bin/ruff check osmose/engine/__init__.py tests/test_engine_state.py`

- [ ] **Step 8: Commit**

```bash
git add osmose/engine/__init__.py tests/test_engine_state.py
git commit -m "chore: delete unused JavaEngine stub + Engine Protocol + redundant Path import (I-10, M-2)"
```

---

### Task 13: Consolidate `_load_fishing_rate_by_year` and `_load_additional_mortality_by_dt` (I-9)

**Files:**
- Modify: `osmose/engine/config.py`

**Context:** Two ~11-line functions with identical control flow, differing only in the config key pattern. Consolidate into a single `_load_per_species_timeseries` helper; the wrappers become 1-2 line passthroughs.

- [ ] **Step 1: Read both functions**

Confirm the near-duplicate shape. Current state should be: for each species, check `cfg.get(f"...file.sp{i}", "")`, if non-empty call `_require_file`, `np.loadtxt`, flatten, store in list.

- [ ] **Step 2: Add the helper**

In `osmose/engine/config.py`, near `_load_additional_mortality_by_dt` (before or after it), add:

```python
def _load_per_species_timeseries(
    cfg: dict[str, str], n_species: int, key_pattern: str, context_prefix: str
) -> list[NDArray[np.float64] | None] | None:
    """Load a per-species time-varying CSV into a list of flattened arrays.

    Shared implementation for _load_fishing_rate_by_year and
    _load_additional_mortality_by_dt. key_pattern should contain ``{i}`` which
    is formatted per species. context_prefix is prepended to _require_file's
    error message for debuggability.

    Returns None if no species has the key set; otherwise a list of length
    n_species with None for unconfigured species and an ndarray for configured ones.
    """
    result: list[NDArray[np.float64] | None] = [None] * n_species
    found_any = False
    for i in range(n_species):
        file_key = cfg.get(key_pattern.format(i=i), "")
        if not file_key:
            continue
        path = _require_file(file_key, _cfg_dir(cfg), key_pattern.format(i=i))
        values = np.loadtxt(path, dtype=np.float64)
        result[i] = values.flatten()
        found_any = True
    return result if found_any else None
```

- [ ] **Step 3: Replace the two wrappers**

Change `_load_fishing_rate_by_year` to:

```python
def _load_fishing_rate_by_year(
    cfg: dict[str, str], n_species: int
) -> list[NDArray[np.float64] | None] | None:
    """Load time-varying annual fishing rate CSV for each species."""
    return _load_per_species_timeseries(
        cfg, n_species, "mortality.fishing.rate.byyear.file.sp{i}", "fishing_rate_by_year"
    )
```

And `_load_additional_mortality_by_dt` to:

```python
def _load_additional_mortality_by_dt(
    cfg: dict[str, str], n_species: int
) -> list[NDArray[np.float64] | None] | None:
    """Load time-varying additional mortality CSV (BY_DT scenario)."""
    return _load_per_species_timeseries(
        cfg, n_species, "mortality.additional.rate.bytdt.file.sp{i}", "additional_mortality_by_dt"
    )
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: unchanged count (consolidation doesn't add or remove tests).

- [ ] **Step 5: Ruff + commit**

```bash
.venv/bin/ruff check osmose/engine/config.py
git add osmose/engine/config.py
git commit -m "refactor(engine): consolidate per-species timeseries CSV loaders into one helper (I-9)"
```

---

### Task 14: Consolidate `_load_accessibility` / `_load_stage_accessibility` call path (M-3)

**Files:**
- Modify: `osmose/engine/config.py`

**Context:** Both functions look up `predation.accessibility.file`. The caller always invokes both on the same cfg. Only one can be non-None in practice, so the second `_resolve_file` is wasted work.

- [ ] **Step 1: Read both functions and the caller**

Run: `grep -n "_load_accessibility\|_load_stage_accessibility" osmose/engine/config.py`

Locate the definitions AND the call site in `from_dict`. Confirm both functions grep `cfg.get("predation.accessibility.file", "")` and the caller passes the same `cfg` to both.

- [ ] **Step 2: Choose the safer consolidation approach**

The accessibility parsing logic is involved. Rather than rewriting it in a new helper, use a **cache approach** or **scope guardrail**:

**Option A (safe):** Add a module-level `@functools.lru_cache`-free cache pattern — extract the file-existence check into a tiny helper that both loaders call:

```python
def _accessibility_path_or_none(cfg: dict[str, str]) -> Path | None:
    """Resolve predation.accessibility.file once. Returns None if not set or
    raises FileNotFoundError via _require_file if set but missing."""
    file_key = cfg.get("predation.accessibility.file", "")
    if not file_key:
        return None
    return _require_file(file_key, _cfg_dir(cfg), "predation.accessibility.file")
```

Then both `_load_accessibility` and `_load_stage_accessibility` call this helper at the top. The cost is still "resolve twice" but we've factored out the duplication.

**Option B (scope guardrail):** If consolidation is too tangled, mark the task as DONE_WITH_CONCERNS and add a `# TODO(M-3): consolidate accessibility loaders` comment.

Pick Option A first; fall back to Option B if any test regresses.

- [ ] **Step 3: Apply Option A**

Replace the `file_key = cfg.get(...); path = _resolve_file(...)` prefix in both `_load_accessibility` and `_load_stage_accessibility` with:

```python
    path = _accessibility_path_or_none(cfg)
    if path is None:
        return None
```

Keep the rest of each function untouched.

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: unchanged count. Any test failure means the consolidation broke behavior — revert and use Option B.

- [ ] **Step 5: Ruff + commit**

```bash
.venv/bin/ruff check osmose/engine/config.py
git add osmose/engine/config.py
git commit -m "refactor(engine): extract _accessibility_path_or_none helper (M-3)"
```

If you took Option B, commit the TODO comment instead:

```bash
git add osmose/engine/config.py
git commit -m "docs(engine): note M-3 accessibility loader duplication as future cleanup"
```

---

### Task 15: Deduplicate `test_parse_label` (M-11)

**Files:**
- Modify: `tests/test_engine_accessibility.py` OR `tests/test_engine_fisheries.py` (delete from one)

**Context:** Duplicate `test_parse_label` tests exist in both files. Delete from whichever file is less semantically appropriate.

- [ ] **Step 1: Find both tests and determine which parses what**

Run:
```bash
grep -n "def test_parse_label" tests/test_engine_accessibility.py tests/test_engine_fisheries.py
```

Read both. They probably test the same `parse_label` function. Find the original definition — whichever test file is closer to the module containing `parse_label` is the canonical home.

- [ ] **Step 2: Delete the duplicate**

Remove `test_parse_label` from the file that is NOT the parser's primary test module. If both are equally appropriate, delete from `tests/test_engine_fisheries.py` (arbitrary tiebreaker).

- [ ] **Step 3: Run suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: baseline - 1 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_engine_accessibility.py tests/test_engine_fisheries.py
git commit -m "test: deduplicate test_parse_label (M-11)"
```

---

### Phase 3 gate

- [ ] **Run full suite, ruff, parity.** Note that the expected test count has now decreased by ~3 (Task 12 removed 2, Task 15 removed 1) while growing by 12 from Phase 1/2 additions. Adjust expectations accordingly.

---

## Phase 4 — Adjacent Silent-Failure Fixes

### Task 16: Apply `_require_file` to the 4 adjacent sites

**Files:**
- Modify: `osmose/engine/config.py`
- Modify: `tests/test_engine_config.py`

**Context:** During C-3..C-7 implementation 3 more sites in `config.py` with the same silent-fallback anti-pattern were found that weren't listed in the v3 findings document: `fisheries.discards.file`, `mpa.file.mpa{i}`, `fisheries.movement.file.map0`. Plus `_load_accessibility` passes `file_key` to `_resolve_file` without the empty-check guard.

- [ ] **Step 1: Locate each site**

Run:
```bash
grep -n "fisheries.discards.file\|mpa.file.mpa\|fisheries.movement.file.map0\|predation.accessibility.file" osmose/engine/config.py
```

Confirm all 4 sites still use `_resolve_file` and still have the silent-fallback shape.

- [ ] **Step 2: Convert `_load_discard_rates`**

Replace:
```python
    file_key = cfg.get("fisheries.discards.file", "")
    path = _resolve_file(file_key, _cfg_dir(cfg))
    if path is None:
        return None
```

with:
```python
    file_key = cfg.get("fisheries.discards.file", "")
    if not file_key:
        return None
    path = _require_file(file_key, _cfg_dir(cfg), "fisheries.discards.file")
```

- [ ] **Step 3: Convert `_parse_mpa_zones`**

Replace:
```python
        file_key = cfg.get(f"mpa.file.mpa{i}", "")
        if not file_key:
            break
        path = _resolve_file(file_key, _cfg_dir(cfg))
        if path is None:
            i += 1
            continue
```

with:
```python
        file_key = cfg.get(f"mpa.file.mpa{i}", "")
        if not file_key:
            break
        path = _require_file(file_key, _cfg_dir(cfg), f"mpa.file.mpa{i}")
```

Drop the `i += 1; continue` path — if path resolution fails it now raises.

- [ ] **Step 4: Convert shared fishing map**

Find the block (approximately line 841):

```python
        shared_fishing_map_file = cfg.get("fisheries.movement.file.map0", "")
        shared_fishing_map: np.ndarray | None = None
        if shared_fishing_map_file:
            shared_path = _resolve_file(shared_fishing_map_file, _cfg_dir(cfg))
            if shared_path is not None:
                shared_fishing_map = _load_spatial_csv(shared_path)
```

Replace with:

```python
        shared_fishing_map_file = cfg.get("fisheries.movement.file.map0", "")
        shared_fishing_map: np.ndarray | None = None
        if shared_fishing_map_file:
            shared_path = _require_file(
                shared_fishing_map_file, _cfg_dir(cfg), "fisheries.movement.file.map0"
            )
            shared_fishing_map = _load_spatial_csv(shared_path)
```

- [ ] **Step 5: Convert `_load_accessibility`**

Replace:
```python
    file_key = cfg.get("predation.accessibility.file", "")
    path = _resolve_file(file_key, _cfg_dir(cfg))
    if path is not None:
        df = pd.read_csv(path, sep=";", index_col=0)
        return df.values.astype(np.float64)
    return None
```

with:
```python
    file_key = cfg.get("predation.accessibility.file", "")
    if not file_key:
        return None
    path = _require_file(file_key, _cfg_dir(cfg), "predation.accessibility.file")
    df = pd.read_csv(path, sep=";", index_col=0)
    return df.values.astype(np.float64)
```

**Note:** this task may conflict with Task 14 (M-3 consolidation). If Task 14 already rewrote `_load_accessibility`, skip this step for that function and apply only the `discards`, `mpa`, and `shared_fishing_map` changes. Check git log before editing.

- [ ] **Step 6: Add 4 regression tests**

In `tests/test_engine_config.py`, in `TestRequireFileRaisesOnMissing`, append:

```python
    def test_fisheries_discards_file_missing_raises(self, tmp_path):
        from osmose.engine.config import _load_discard_rates

        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "fisheries.discards.file": "typo.csv",
        }
        with pytest.raises(FileNotFoundError, match="typo.csv"):
            _load_discard_rates(cfg, ["Anchovy"], n_species=1)

    def test_mpa_file_missing_raises(self, tmp_path):
        from osmose.engine.config import _parse_mpa_zones

        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "mpa.file.mpa0": "typo.csv",
        }
        with pytest.raises(FileNotFoundError, match="typo.csv"):
            _parse_mpa_zones(cfg)

    def test_shared_fishing_map_missing_raises(self, tmp_path):
        """fisheries.movement.file.map0 non-empty key with missing file raises."""
        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "simulation.time.ndtperyear": "24",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "5",
            "species.name.sp0": "Anchovy",
            "species.linf.sp0": "20.0",
            "species.k.sp0": "0.3",
            "species.t0.sp0": "-0.1",
            "species.egg.size.sp0": "0.1",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.lifespan.sp0": "3",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.efficiency.critical.sp0": "0.57",
            "fisheries.movement.file.map0": "ghost_shared_map.csv",
        }
        with pytest.raises(FileNotFoundError, match="ghost_shared_map.csv"):
            EngineConfig.from_dict(cfg)

    def test_predation_accessibility_file_missing_raises(self, tmp_path):
        from osmose.engine.config import _load_accessibility

        cfg = {
            "_osmose.config.dir": str(tmp_path),
            "predation.accessibility.file": "typo_access.csv",
        }
        with pytest.raises(FileNotFoundError, match="typo_access.csv"):
            _load_accessibility(cfg, n_species=2)
```

- [ ] **Step 7: Run suite + ruff + commit**

```bash
.venv/bin/python -m pytest tests/test_engine_config.py::TestRequireFileRaisesOnMissing -v
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check osmose/engine/config.py tests/test_engine_config.py
git add osmose/engine/config.py tests/test_engine_config.py
git commit -m "fix(engine): apply _require_file to 4 adjacent silent-failure sites"
```

---

### Phase 4 gate

- [ ] Same as previous gates: pytest + ruff + parity.

---

## Phase 5 — Hardening Polish

### Task 17: Clean up stale TODOs in `output.py` (M-4)

**Files:**
- Modify: `osmose/engine/output.py`

**Context:** 3 TODOs at lines 289, 344, 345 describe deferred spatial/bioen output variants. They should either link to a tracking doc or be deleted.

- [ ] **Step 1: Read all 3 TODOs for context**

Run: `grep -n "TODO\|FIXME" osmose/engine/output.py`

Read the surrounding code to understand which output types are missing.

- [ ] **Step 2: Check whether `docs/parity-roadmap.md` covers these**

Run: `grep -i "spatial.*bioen\|size.*age.*distribution\|spatial.*biomass" docs/parity-roadmap.md 2>/dev/null`

If the roadmap has matching sections, update each TODO to reference them:

```python
# Deferred: see docs/parity-roadmap.md §5.3 (spatial bioen) — not blocking v0.6.0.
```

If the roadmap does NOT cover them, keep the TODOs but make them actionable:

```python
# TODO(v0.7): spatial bioen output variants — not implemented because
# SpatialEnetOutput requires per-cell StepOutput (blocked on M-13 type refactor).
```

- [ ] **Step 3: Ruff + commit**

```bash
.venv/bin/ruff check osmose/engine/output.py
git add osmose/engine/output.py
git commit -m "docs: make output.py TODOs actionable / link to roadmap (M-4)"
```

---

### Task 18: Document `cell_id` expression in mortality.py (M-6)

**Files:**
- Modify: `osmose/engine/processes/mortality.py`

**Context:** `cell_id = cell_y * (resources.grid.nx if resources else 0) + cell_x` at line 381 produces a garbage value when `resources is None`, but the value is never read in that branch. The expression is misleading to readers.

- [ ] **Step 1: Read the surrounding context**

Run: `sed -n '370,420p' osmose/engine/processes/mortality.py` to see how `cell_id` is used downstream.

Confirm that `cell_id` is only read inside an `if resources is not None:` guard. If it IS read in the None branch, this isn't a minor documentation issue — it's a real bug. Escalate.

- [ ] **Step 2: Restructure the expression**

Replace:
```python
        cell_id = cell_y * (resources.grid.nx if resources else 0) + cell_x
```

with:
```python
        # cell_id is only meaningful when resources are present (used inside the
        # `if resources is not None:` branch below). When resources is None we
        # assign a sentinel to satisfy the variable-always-defined rule, but the
        # value is never read. Deep review v3 M-6.
        if resources is not None:
            cell_id = cell_y * resources.grid.nx + cell_x
        else:
            cell_id = -1
```

- [ ] **Step 3: Run suite**

Run: `.venv/bin/python -m pytest tests/test_engine_mortality.py tests/test_engine_predation.py -q`
Expected: unchanged count.

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "docs(engine): clarify cell_id expression when resources is None (M-6)"
```

---

### Task 19: Opt-in strict mode for movement map coverage (M-7)

**Files:**
- Modify: `osmose/engine/movement_maps.py`
- Modify: `osmose/engine/config.py` (to read the new config key)
- Modify: `tests/test_engine_map_movement.py`

**Context:** The H5 fix made the warning aggregated but didn't escalate to an error. The v3 reviewer argued that silent fallback to random walk for uncovered slots is a correctness problem. Escalate from warning to ValueError when strict-mode is on, but keep default as warn to avoid breaking existing configs.

- [ ] **Step 1: Write the failing test**

In `tests/test_engine_map_movement.py`, append:

```python
def test_movement_map_uncovered_raises_in_strict_mode(tmp_path):
    """With movement.strict_coverage=true, any uncovered (age_dt, step) slot
    must raise ValueError instead of warning-and-fallback.

    Deep review v3 M-7.
    """
    import numpy as np
    import pytest

    # Build a MovementMapSet with strict_coverage=True and at least one uncovered slot.
    # Read osmose/engine/movement_maps.py for the actual constructor signature and
    # adapt this stub to build the minimal fixture that triggers the uncovered-slot path.
    from osmose.engine.movement_maps import MovementMapSet

    # Example: create a MovementMapSet with lifespan_dt=4 and n_total_steps=4 (16 slots),
    # but only provide a map for (0, 0) so 15 slots are uncovered.
    # (Exact fixture depends on MovementMapSet's __init__ — adapt accordingly.)
    with pytest.raises(ValueError, match="have no movement map assigned"):
        # Constructor call with strict_coverage=True and an uncovered configuration
        pass  # replace with actual constructor call
```

**This task has a real design question.** Before implementing, flag whether strict mode should be:
- (a) opt-in via a new config key `movement.strict_coverage` (default: False, current behavior)
- (b) unconditional (every uncovered slot raises)

Option (a) is safer. Implement (a).

- [ ] **Step 2: Implement the strict check**

In `osmose/engine/movement_maps.py`, around the aggregated warning:

```python
        # --- Validate: warn about uncovered (age, step) slots ---
        uncovered = int((self.index_maps == -1).sum())
        if uncovered > 0:
            total_slots = lifespan_dt * n_total_steps
            message = (
                f"Species {species_name!r}: {uncovered} of {total_slots} "
                f"(age_dt, step) slots have no movement map assigned"
            )
            if strict_coverage:
                raise ValueError(message)
            logger.warning(message)
```

Add a `strict_coverage: bool = False` parameter to the relevant method signature and plumb it from `EngineConfig.from_dict` which reads:

```python
strict_movement_coverage = cfg.get("movement.strict_coverage", "false").lower() == "true"
```

- [ ] **Step 3: Test + commit**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py -q`

```bash
git add osmose/engine/movement_maps.py tests/test_engine_map_movement.py osmose/engine/config.py
git commit -m "feat(engine): movement.strict_coverage config raises on uncovered slots (M-7)"
```

---

### Task 20: Log exc_info in `_close_spatial_ds` (M-8)

**Files:**
- Modify: `ui/pages/spatial_results.py:145-152`

**Context:** `except Exception: pass` swallows any close-time exception. Change to `logger.warning(..., exc_info=True)`.

- [ ] **Step 1: Locate the function**

Run: `grep -n "_close_spatial_ds\|except Exception" ui/pages/spatial_results.py` — confirm the bare `except: pass` is still present.

- [ ] **Step 2: Replace the pass**

Change:
```python
            try:
                old_ds.close()
            except Exception:
                pass
```

to:
```python
            try:
                old_ds.close()
            except Exception:
                _log.warning(
                    "Failed to close previous spatial dataset during swap",
                    exc_info=True,
                )
```

Ensure `_log` is available at module scope — if not, add:
```python
from osmose.logging import setup_logging
_log = setup_logging("osmose.spatial_results_ui")
```

- [ ] **Step 3: Run suite + ruff + commit**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check ui/pages/spatial_results.py
git add ui/pages/spatial_results.py
git commit -m "fix(ui): log exc_info in _close_spatial_ds instead of bare pass (M-8)"
```

---

### Phase 5 gate

- [ ] pytest + ruff + parity as before.

---

## Phase 6 — Test Hygiene

### Task 21: Audit `test_engine_config_validation.py` for construction-only tests (M-12)

**Files:**
- Modify: `tests/test_engine_config_validation.py`

**Context:** The v3 reviewer flagged several tests in this file as "construction-only assertions" — they build an EngineConfig with placeholder arrays and implicitly test that `__init__` accepts them, rather than exercising an actual validation branch.

- [ ] **Step 1: Read the file**

Run: `wc -l tests/test_engine_config_validation.py; grep -n "^def test_\|^    def test_" tests/test_engine_config_validation.py`

Scan each test. For each, ask: does this test assert on behavior (a property of the returned object, an error raised, a mutation), or does it just create the object and move on?

- [ ] **Step 2: Annotate weak tests**

For each test that's construction-only, either:
- Add a meaningful assertion (ideal) — e.g., if the test builds an EngineConfig with zero fishing, assert `ec.fishing_rate.sum() == 0` or similar.
- Add a `# NOTE(M-12): construction-only — consider strengthening to a behavior assertion` comment so the next reader knows.
- Delete the test if strengthening requires significant fixture work.

Apply comments/strengthening to 2-4 of the weakest tests. Don't rewrite all of them — time-box this task.

- [ ] **Step 3: Commit**

```bash
.venv/bin/python -m pytest tests/test_engine_config_validation.py -v
.venv/bin/ruff check tests/test_engine_config_validation.py
git add tests/test_engine_config_validation.py
git commit -m "test: strengthen construction-only assertions in config validation tests (M-12)"
```

---

### Phase 6 gate: same as previous.

---

## Phase 7 — UI Test Coverage

Seven tasks, one per untested UI page. The goal is NOT to cover the reactive handler (which needs a Shiny test harness) but to **extract and test pure helpers** — functions that take data in and return data out without touching reactive state.

### Task 22–28: Per-page UI helper tests

For each of these pages, do the following workflow:

| # | Page |
|---|---|
| 22 | `ui/pages/movement.py` |
| 23 | `ui/pages/fishing.py` |
| 24 | `ui/pages/forcing.py` |
| 25 | `ui/pages/economic.py` |
| 26 | `ui/pages/spatial_results.py` |
| 27 | `ui/pages/diagnostics.py` |
| 28 | `ui/pages/map_viewer.py` |

For each page:

- [ ] **Step A: Read the page file and identify one pure helper function**

A pure helper is any function that:
- Does NOT take `input`, `output`, `session` as parameters
- Does NOT call `reactive.*` or `ui.*` to build UI
- DOES take data (dict, DataFrame, array) and return data

Examples from already-tested pages: `derive_map_label`, `parse_movement_steps`, `list_movement_species` in `ui/pages/grid_helpers.py`. Look for similar standalone functions.

If NO pure helper exists in the page, the task becomes "extract one" — identify a small (~10 line) chunk of data-transformation logic inside a reactive handler, extract it into a module-level function, then test it. If that's too large for one task, mark the page as DONE_WITH_CONCERNS and move to the next.

- [ ] **Step B: Create the test file**

Create `tests/ui/test_ui_<pagename>.py` with imports and 2-3 tests for the identified helper. Follow the existing `tests/test_ui_*.py` style.

Example structure:

```python
"""Tests for pure helpers in ui/pages/<pagename>.py."""

import pytest
from ui.pages.<pagename> import <helper_function>


def test_<helper>_happy_path():
    result = <helper_function>(<valid_input>)
    assert result == <expected>


def test_<helper>_empty_input():
    result = <helper_function>(<empty_input>)
    assert result == <expected_for_empty>


def test_<helper>_edge_case():
    # Exercise one boundary or error path
    ...
```

- [ ] **Step C: Run the tests**

Run: `.venv/bin/python -m pytest tests/ui/test_ui_<pagename>.py -v`
Expected: all pass.

- [ ] **Step D: Commit**

```bash
git add tests/ui/test_ui_<pagename>.py
git commit -m "test(ui): add pure-helper unit tests for ui/pages/<pagename>.py (M-9)"
```

**Per-task time budget: 20 minutes.** If you can't find a pure helper in 20 minutes, report DONE_WITH_CONCERNS with "no pure helper found — reactive handler is tightly coupled, recommend extracting one in a dedicated refactor." Do NOT try to test reactive handlers directly — that's out of scope.

**Expected total for Phase 7: 7 × 20 min = 140 minutes.** May be shorter if some pages already have pure helpers to grab.

---

### Phase 7 gate: pytest + ruff + parity. Expect ~14-21 new tests depending on how many helpers each page yielded.

---

## Final gate

- [ ] **Run full suite**: `.venv/bin/python -m pytest tests/ -q`
  - Expected: ≥2130 passed (numbers depend on Phase 7 helper discovery; record actual).
- [ ] **Ruff**: `.venv/bin/ruff check osmose/ ui/ tests/`
- [ ] **Parity**: `.venv/bin/python -m pytest tests/test_engine_parity.py -q` — 12 passed.
- [ ] **Commit count**: `git log --oneline master..HEAD` should show 22–28 commits (one per task, plus any follow-ups for Important review findings).
- [ ] **Invoke `superpowers:finishing-a-development-branch`** to merge/PR/cleanup.

---

## Out of scope (explicit non-goals — repeated here for clarity)

- **I-3** `EngineConfig.from_dict` 550-line monolith split — needs its own plan with per-subsystem tasks.
- **M-5** `population.seeding.year.max` per-species — blocked on Java source verification.
- **D-1** `state.dirty.set` inside `reactive.isolate` semantics — needs Shiny reactive-model investigation.
- **D-2** Already resolved this session (M-2 partial-year warn vs raise).
- **M-1** Already shipped this session.

These items remain documented in `docs/superpowers/reviews/2026-04-11-fresh-deep-review-v3.md` and can be picked up individually when the blockers clear.
