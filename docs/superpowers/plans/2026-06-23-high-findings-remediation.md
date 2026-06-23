# Deep-Review High-Findings Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three remaining deep-review high findings — egg-retention no-op (production Numba path), fleet-effort dropped on the Python fishing fallback, and `_worker_eval` swallowing programming errors — each test-first, with a Java/EEC/BoB parity gate on the egg fix.

**Architecture:** Targeted fixes in `osmose/engine/processes/mortality.py` (egg + fleet) and `osmose/calibration/problem.py` (worker_eval). The egg fix threads `egg_retained` through the production Numba predation kernel chain AND the pure-Python fallback so both paths gate egg predation on the released fraction. Fleet-effort is extracted into a shared helper used by both rate paths. The critical (bioen double-starvation) is already fixed (`457ac55` + `c273c8f`).

**Tech Stack:** Python 3.12+, NumPy, Numba (`@njit`), pytest, ruff, pyright; Java engine via `osmose.runner.OsmoseRunner` (bundled 4.3.3 JAR) for the cross-check.

**Spec:** `docs/superpowers/specs/2026-06-23-high-findings-remediation-design.md`

## Global Constraints

- Use `.venv/bin/python` (NOT `python`) for all pytest/python invocations.
- Ruff line length 100; lint = BOTH `.venv/bin/ruff check` AND `.venv/bin/ruff format --check` on changed files; `.venv/bin/pyright` clean on changed files.
- `_HAS_NUMBA=True` is the production/CI path. The pure-Python `_mortality_in_cell`/`_get_mortality_causes`/`_apply_*_for_school` path runs ONLY when `_HAS_NUMBA=False` — force it in tests with `mock.patch("osmose.engine.processes.mortality._HAS_NUMBA", False)`.
- Egg-retention model: eatable egg prey = `max(inst_abd[q] - egg_retained[q], 0)`; `egg_retained` is 0 for non-eggs, so non-egg/resource prey are unchanged.
- **Parity gate (Fix 1 only):** Java cross-check is the source of truth; regenerate EEC/BoB `.npz` baselines ONLY after the Java check confirms the shift is toward Java. Never loosen tolerance or re-bless a baseline to make a run pass — stop and report deltas.
- Commit after each task.

---

### Task 1: Fix 3 — `_worker_eval` stops swallowing programming errors

**Files:**
- Modify: `osmose/calibration/problem.py:93-99` (the `_worker_eval` function)
- Test: `tests/test_calibration_worker_eval.py` (new)

**Interfaces:**
- Consumes: `osmose.calibration.problem._worker_eval`, `_WORKER_PROBLEM` (module global), `OsmoseCalibrationProblem._evaluate_candidate`.
- Produces: nothing for later tasks.

**Background:** `_evaluate_candidate` (problem.py:245) already catches `_python_engine_errors` internally and returns `[inf]*n_obj` for expected model failures, so those never reach `_worker_eval`. `_worker_eval`'s `except Exception` therefore only ever catches *unexpected* programming errors (TypeError/AttributeError), silently turning a real bug into `inf`. Remove the swallow.

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration_worker_eval.py`:

```python
"""_worker_eval must let unexpected (programming) errors propagate, not swallow
them to inf — expected model errors are already handled in _evaluate_candidate.
Deep review 2026-06-22 (HIGH-3, narrowed)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import osmose.calibration.problem as problem


def test_worker_eval_propagates_unexpected_error(monkeypatch):
    stub = MagicMock(n_obj=1)
    stub._evaluate_candidate.side_effect = TypeError("objective bug")
    monkeypatch.setattr(problem, "_WORKER_PROBLEM", stub)

    with pytest.raises(TypeError):
        problem._worker_eval(0, np.zeros(2))


def test_worker_eval_returns_evaluate_candidate_result(monkeypatch):
    # Expected-error handling lives in _evaluate_candidate; _worker_eval just
    # returns whatever it produces (here, a normal objective vector).
    stub = MagicMock(n_obj=1)
    stub._evaluate_candidate.return_value = [1.5]
    monkeypatch.setattr(problem, "_WORKER_PROBLEM", stub)

    assert problem._worker_eval(0, np.zeros(2)) == [1.5]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_worker_eval.py -q`
Expected: `test_worker_eval_propagates_unexpected_error` FAILS — currently `_worker_eval` catches the `TypeError` and returns `[inf]`, so `pytest.raises(TypeError)` does not see the exception. (The second test passes already.)

- [ ] **Step 3: Remove the swallow**

In `osmose/calibration/problem.py`, replace the `_worker_eval` body (lines 93-99):

```python
def _worker_eval(run_id: int, params: np.ndarray) -> list[float]:
    """Evaluate one candidate in the worker.

    Expected model failures are caught inside `_evaluate_candidate` and returned
    as `[inf]*n_obj`. UNEXPECTED errors (e.g. a TypeError/AttributeError bug in an
    objective) propagate so the pool future re-raises and the bug surfaces, rather
    than being silently turned into `inf` (which would poison the Pareto front).
    """
    assert _WORKER_PROBLEM is not None
    return _WORKER_PROBLEM._evaluate_candidate(run_id, params)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_worker_eval.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Regression + lint**

Run: `.venv/bin/python -m pytest tests/test_calibration_problem.py -q`
Expected: PASS.
Run: `.venv/bin/ruff check osmose/calibration/problem.py tests/test_calibration_worker_eval.py && .venv/bin/ruff format --check osmose/calibration/problem.py tests/test_calibration_worker_eval.py && .venv/bin/pyright osmose/calibration/problem.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add osmose/calibration/problem.py tests/test_calibration_worker_eval.py
git commit -m "fix(calibration): let _worker_eval propagate unexpected errors (no inf-swallow)"
```

---

### Task 2: Fix 2 — fleet-effort on the Python fishing fallback

**Files:**
- Modify: `osmose/engine/processes/mortality.py` (add `_fleet_effort_factor`; use it in `_precompute_effective_rates:786-802` and `_apply_fishing_for_school:180`; pass `ctx.fleet_state` at the Python fishing call site `:1745`)
- Test: `tests/test_engine_fishing_fleet_python_path.py` (new)

**Interfaces:**
- Consumes: `work_state.cell_y/cell_x`, `fleet_state.fleets[].target_species`, `fleet_state.effort_map` (shape `(n_fleets, ny, nx)`).
- Produces: `_fleet_effort_factor(sp_id, cell_y, cell_x, fleet_state) -> float`.

**Background:** the fleet-effort scaling exists only in `_precompute_effective_rates` (Numba path). `_apply_fishing_for_school` (Python fallback, `_HAS_NUMBA=False` only) lacks it. Extract a shared helper.

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_fishing_fleet_python_path.py`:

```python
"""Fleet-effort scaling must apply on the pure-Python fishing fallback too, not
only the Numba path. Deep review 2026-06-22 (HIGH-2, _HAS_NUMBA=False-scoped)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from osmose.engine.processes.mortality import _fleet_effort_factor


def _fleet_state(target_species, effort):
    # effort_map shape (n_fleets, ny, nx); one fleet, 2x2 grid.
    emap = np.zeros((1, 2, 2), dtype=np.float64)
    emap[0] = effort
    return SimpleNamespace(
        fleets=[SimpleNamespace(target_species=set(target_species))],
        effort_map=emap,
    )


def test_factor_none_when_no_fleet_state():
    assert _fleet_effort_factor(0, 0, 0, None) == 1.0


def test_factor_one_for_non_targeted_species():
    fs = _fleet_state(target_species=[1], effort=np.full((2, 2), 3.0))
    assert _fleet_effort_factor(0, 0, 0, fs) == 1.0  # sp 0 not targeted


def test_factor_sums_effort_for_targeted_species_in_cell():
    fs = _fleet_state(target_species=[0], effort=np.array([[2.0, 0.0], [0.0, 5.0]]))
    assert _fleet_effort_factor(0, 0, 0, fs) == 2.0
    assert _fleet_effort_factor(0, 1, 1, fs) == 5.0


def test_factor_zero_when_targeted_cell_out_of_bounds():
    fs = _fleet_state(target_species=[0], effort=np.full((2, 2), 3.0))
    assert _fleet_effort_factor(0, 9, 9, fs) == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_fleet_python_path.py -q`
Expected: FAIL — `ImportError: cannot import name '_fleet_effort_factor'`.

- [ ] **Step 3: Add the helper**

In `osmose/engine/processes/mortality.py`, add above `_precompute_effective_rates` (before line 661):

```python
def _fleet_effort_factor(sp_id, cell_y, cell_x, fleet_state) -> float:
    """Multiplicative fishing-effort factor for a school. Returns 1.0 when
    fleet_state is None OR sp_id is not targeted by any fleet (base F unchanged).
    For a TARGETED species: the sum of effort_map across fleets at (cell_y,
    cell_x), or 0.0 if that cell is out of bounds. Mirrors mortality.py:786-802."""
    if fleet_state is None:
        return 1.0
    targeted: set[int] = set()
    for f in fleet_state.fleets:
        targeted.update(f.target_species)
    if int(sp_id) not in targeted:
        return 1.0
    ny, nx = fleet_state.effort_map.shape[1], fleet_state.effort_map.shape[2]
    if 0 <= cell_y < ny and 0 <= cell_x < nx:
        return float(fleet_state.effort_map[:, cell_y, cell_x].sum())
    return 0.0
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_fleet_python_path.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Refactor `_precompute_effective_rates` to use the helper (behavior-preserving)**

In `_precompute_effective_rates`, replace the fleet block at lines **786-802** (the `if fleet_state is not None:` two-loop block ending at `eff_fishing[i] *= effort_factor[i]`) with:

```python
        # Scale fishing by fleet effort when economic module is active
        if fleet_state is not None:
            for i in range(n):
                eff_fishing[i] *= _fleet_effort_factor(
                    work_state.species_id[i],
                    work_state.cell_y[i],
                    work_state.cell_x[i],
                    fleet_state,
                )
```

Note: for a non-targeted species the helper returns 1.0 (no-op), and for a targeted out-of-bounds cell it returns 0.0 — identical to the old block (old block only multiplied targeted species; non-targeted were left untouched == ×1.0).

- [ ] **Step 6: Apply the factor in `_apply_fishing_for_school`**

In `_apply_fishing_for_school`, the function currently takes `(idx, state, config, n_subdt, inst_abd, step=0)`. Add a `fleet_state=None` parameter:

```python
def _apply_fishing_for_school(
    idx: int,
    state: SchoolState,
    config: EngineConfig,
    n_subdt: int,
    inst_abd: NDArray[np.float64],
    step: int = 0,
    fleet_state=None,
) -> None:
```

Then, immediately AFTER the MPA-reduction block and BEFORE the seasonality/`F = ...` block (i.e. right after the `for mpa in config.mpa_zones:` loop, ~line 252), insert:

```python
    # Fleet-effort (DSVM economics) — parity with the Numba path's
    # _precompute_effective_rates. 1.0 when no fleet_state / species not targeted.
    f_rate = f_rate * _fleet_effort_factor(
        sp, int(state.cell_y[idx]), int(state.cell_x[idx]), fleet_state
    )
```

- [ ] **Step 7: Pass `ctx.fleet_state` at the Python fishing call site**

At `mortality.py:1745` (inside `_mortality_in_cell`'s pure-Python loop, the `elif cause == _FISHING:` branch starts ~`:1740`), change the call:

```python
                _apply_fishing_for_school(
                    f_idx, state, config, n_subdt, inst_abd, step=step,
                    fleet_state=(ctx.fleet_state if ctx is not None else None),
                )
```

- [ ] **Step 8: Write the fallback-path integration assertion**

Append to `tests/test_engine_fishing_fleet_python_path.py` a test that exercises `_apply_fishing_for_school` under forced `_HAS_NUMBA=False`, confirming effort doubles the fishing deaths. Use the existing single-school config/state builders from `tests/test_engine_mortality_loop.py` as the model (`_make_config()` returns a config dict; `SchoolState.create(...)`). Concretely:

```python
def test_apply_fishing_for_school_scales_by_fleet_effort(monkeypatch):
    import osmose.engine.processes.mortality as m
    from osmose.engine.config import EngineConfig
    from osmose.engine.state import MortalityCause, SchoolState
    from tests.test_engine_mortality_loop import _make_config

    monkeypatch.setattr(m, "_HAS_NUMBA", False)
    cfg_dict = _make_config()
    cfg_dict["mortality.fishing.rate.sp0"] = "0.5"
    cfg = EngineConfig.from_dict(cfg_dict)

    def _run(fleet_state):
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state.abundance[0] = 1000.0
        state.age_dt[0] = 50
        state.length[0] = 30.0
        inst = state.abundance.copy()
        m._apply_fishing_for_school(0, state, cfg, n_subdt=1, inst_abd=inst, step=0,
                                    fleet_state=fleet_state)
        return state.n_dead[0, int(MortalityCause.FISHING)] + state.n_dead[0, int(MortalityCause.DISCARDS)]

    base = _run(None)
    scaled = _run(_fleet_state(target_species=[0], effort=np.full((2, 2), 2.0)))
    assert base > 0
    assert scaled > base  # ~2x effort -> more fishing deaths
```

If `_make_config`'s species/grid defaults don't match this setup (e.g. selectivity returns early, or the cell is out of the 2×2 effort map), adjust the state fields (`cell_y/cell_x=0`, `feeding_stage`, selectivity params) so a base fishing death occurs — the assertion is `scaled > base > 0`, not an exact value.

- [ ] **Step 9: Run tests + lint + pyright**

Run: `.venv/bin/python -m pytest tests/test_engine_fishing_fleet_python_path.py tests/test_vectorized_rates.py tests/test_economics_choice.py -q`
Expected: PASS (the vectorized-rates and economics tests guard the `_precompute_effective_rates` refactor).
Run: `.venv/bin/ruff check osmose/engine/processes/mortality.py tests/test_engine_fishing_fleet_python_path.py && .venv/bin/ruff format --check osmose/engine/processes/mortality.py tests/test_engine_fishing_fleet_python_path.py && .venv/bin/pyright osmose/engine/processes/mortality.py`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add osmose/engine/processes/mortality.py tests/test_engine_fishing_fleet_python_path.py
git commit -m "fix(engine): apply fleet-effort on the pure-Python fishing fallback path"
```

---

### Task 3: Fix 1 — egg-retention gates predation on the released fraction

**Files:**
- Modify: `osmose/engine/processes/mortality.py` — `_apply_predation_numba` (def `:810`, prey-avail `:894`) + its three callers (`_mortality_in_cell_numba` call `:1139`, `_mortality_all_cells_numba` call `:1306`, `_mortality_all_cells_parallel` call `:1485`) + the driver signatures (`:1077`/`:1219`/`:1386`) + the `mortality()` dispatch call (`:~1939`) + the `_mortality_in_cell` dispatch; and `_apply_predation_for_school` (prey-avail `:397`); plus `tests/test_engine_functional_response.py:780` (existing direct kernel call gets the new arg).
- Test: `tests/test_engine_egg_retention.py` (new)

**Interfaces:**
- Consumes: `work_state.egg_retained` (float64 array, current per sub-dt), `state.egg_retained` (Python path).
- Produces: nothing for later tasks (Task 4 gates this).

**Background:** `egg_retained` is computed and decremented per sub-dt but never read by predation. Thread it into the predation reads so eatable egg prey = `max(inst_abd[q] - egg_retained[q], 0)`.

- [ ] **Step 1: Write the failing test**

The full-`mortality()` path is awkward to make falsifiable (config size-ratio parsing). Instead drive `_apply_predation_for_school` directly — the pure-Python predation path that reads `state.egg_retained` after the fix — using the **verified working harness** at `tests/test_engine_functional_response.py:551-668` (its config dict, weights, and the exact `_apply_predation_for_school(...)` positional call). The Numba production kernel gets the same one-line clamp and is validated end-to-end by Task 4's Java cross-check + parity baselines. Create `tests/test_engine_egg_retention.py`:

```python
"""Egg-retention: predation must only see the RELEASED egg fraction, not the full
egg cohort. Drives _apply_predation_for_school directly with the verified harness
from tests/test_engine_functional_response.py:551. Deep review 2026-06-22 (HIGH-1)."""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.mortality import _apply_predation_for_school
from osmose.engine.resources import ResourceState
from osmose.engine.state import MortalityCause, SchoolState

_CFG = {
    "simulation.time.ndtperyear": "24", "simulation.time.nyear": "1",
    "simulation.nspecies": "2", "simulation.nschool.sp0": "1", "simulation.nschool.sp1": "1",
    "species.name.sp0": "Egg", "species.name.sp1": "Predator",
    "species.linf.sp0": "15.0", "species.linf.sp1": "50.0",
    "species.k.sp0": "0.5", "species.k.sp1": "0.2",
    "species.t0.sp0": "-0.1", "species.t0.sp1": "-0.1",
    "species.egg.size.sp0": "0.1", "species.egg.size.sp1": "0.1",
    "species.length2weight.condition.factor.sp0": "0.006",
    "species.length2weight.condition.factor.sp1": "0.006",
    "species.length2weight.allometric.power.sp0": "3.0",
    "species.length2weight.allometric.power.sp1": "3.0",
    "species.lifespan.sp0": "5", "species.lifespan.sp1": "10",
    "species.vonbertalanffy.threshold.age.sp0": "1.0",
    "species.vonbertalanffy.threshold.age.sp1": "1.0",
    "mortality.subdt": "10",
    "predation.ingestion.rate.max.sp0": "3.5", "predation.ingestion.rate.max.sp1": "3.5",
    "predation.efficiency.critical.sp0": "0.57", "predation.efficiency.critical.sp1": "0.57",
    # NOTE: the parser keys are ALL-LOWERCASE (config.py:646-647); camelCase
    # sizeRatio keys are silently ignored -> defaults. Use lowercase + the real
    # operating window so the test exercises the guard, not a default fallback.
    "predation.predprey.sizeratio.min.sp0": "1.0", "predation.predprey.sizeratio.min.sp1": "1.0",
    "predation.predprey.sizeratio.max.sp0": "3.5", "predation.predprey.sizeratio.max.sp1": "3.5",
    "mortality.additional.rate.sp0": "0.0", "mortality.additional.rate.sp1": "0.0",
    "mortality.starvation.rate.max.sp0": "0.0", "mortality.starvation.rate.max.sp1": "0.0",
    "simulation.fishing.mortality.enabled": "false",
}


def _eaten_eggs(egg_retained_frac: float) -> float:
    n_subdt = 10
    cfg = EngineConfig.from_dict(dict(_CFG))
    grid = Grid.from_dimensions(ny=1, nx=1)
    rs = ResourceState(config=cfg.raw_config, grid=grid)
    state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
    pred_w = 0.006 * 30**3 * 1e-6
    prey_w = 0.006 * 10**3 * 1e-6
    pred_abundance = 100.0
    pred_biomass = pred_abundance * pred_w
    max_eatable = pred_biomass * 3.5 / (24 * n_subdt)
    prey_abundance = (2.0 * max_eatable) / prey_w  # r=2: prey plentiful, predator appetite-bound
    state = state.replace(
        abundance=np.array([pred_abundance, prey_abundance]),
        length=np.array([30.0, 10.0]),  # predator/prey length ratio 3.0, within [1.0, 3.5)
        weight=np.array([pred_w, prey_w]),
        biomass=np.array([pred_biomass, prey_abundance * prey_w]),
        age_dt=np.array([48, 24], dtype=np.int32),
        cell_x=np.array([0, 0], dtype=np.int32),
        cell_y=np.array([0, 0], dtype=np.int32),
        feeding_stage=np.array([0, 0], dtype=np.int32),
        is_egg=np.array([False, True]),
        egg_retained=np.array([0.0, egg_retained_frac * prey_abundance]),
    )
    rng = np.random.default_rng(42)
    cell_indices = np.array([0, 1], dtype=np.int32)
    _apply_predation_for_school(
        0, cell_indices, state, cfg, rs, 0, 0, rng, n_subdt,
        None, False, False, None, None, inst_abd=state.abundance.copy(),
    )
    return float(state.preyed_biomass[0])  # eaten biomass by the predator


def test_fully_retained_eggs_are_not_eaten():
    # egg_retained == full abundance -> eatable (inst_abd - egg_retained) == 0.
    assert _eaten_eggs(egg_retained_frac=1.0) == 0.0


def test_released_eggs_are_eaten():
    # Baseline: with nothing retained the predator eats eggs (proves the harness bites).
    assert _eaten_eggs(egg_retained_frac=0.0) > 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_egg_retention.py -q`
Expected: `test_fully_retained_eggs_are_not_eaten` FAILS — current code reads `inst_abd[q_idx]` (ignores `egg_retained`), so the predator eats the plentiful prey and `preyed_biomass[0] > 0`. `test_released_eggs_are_eaten` PASSES already (confirms the harness produces predation). If `test_released_eggs_are_eaten` is 0 (no predation at all), the size-ratio/age setup is wrong — re-check against the cited template before proceeding.

- [ ] **Step 3: Thread `egg_retained` into the Numba kernel (prey-avail clamp)**

In `_apply_predation_numba` (def at `:810`), add `egg_retained` as the FINAL positional parameter. At the prey-availability read inside the prey scan (the `abd_q = inst_abd[q_idx]` line, **`:894`**), change to:

```python
            abd_q = inst_abd[q_idx] - egg_retained[q_idx]
            if abd_q < 0.0:
                abd_q = 0.0
```

**Also update the existing direct kernel-call test** `tests/test_engine_functional_response.py:780` (`_mort._apply_predation_numba(...)`) — it passes the kernel's full positional arg list and will fail to JIT-compile with the new param. Append a final `egg_retained` argument there: `np.zeros(n_schools, dtype=np.float64)`. (This is the only direct `_apply_predation_numba` call in `tests/`.)

- [ ] **Step 4: Thread `egg_retained` through the three driver kernels**

Add `egg_retained` as the FINAL positional parameter to the signatures of `_mortality_in_cell_numba` (`:1077`), `_mortality_all_cells_numba` (`:1219`), and `_mortality_all_cells_parallel` (`:1386`); and pass `egg_retained` as the final arg at each `_apply_predation_numba(...)` call site (`:1139`, `:1306`, `:1485`).

- [ ] **Step 5: Pass `work_state.egg_retained` at the dispatch sites**

In `mortality()`, at the `_batch_fn(...)` call (`:~1939`), add `work_state.egg_retained` as the final argument — this is the **production hot path**. For the `else:` per-cell fallback (only reached when `_HAS_NUMBA=False`), add `egg_retained` as a final param to `_mortality_in_cell` and pass `work_state.egg_retained`; thread it to the internal `_mortality_in_cell_numba` call for signature consistency. (Note: that internal Numba branch is dead under `_HAS_NUMBA=False` — `use_full_numba` is always False there, `:1623` — so the *active* fallback fix is Step 6's pure-Python `_apply_predation_for_school` change. Still thread the param to keep arg counts consistent.)

- [ ] **Step 6: Thread into the pure-Python predation fallback**

In `_apply_predation_for_school`, at the school-prey availability read (`inst_abd_q = inst_abd[q_idx]`, `:397`) change to:

```python
        inst_abd_q = inst_abd[q_idx] - state.egg_retained[q_idx]
        if inst_abd_q < 0.0:
            inst_abd_q = 0.0
```

(`state` is already a parameter; resource prey are read elsewhere and are never eggs.)

- [ ] **Step 7: Run the egg test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_engine_egg_retention.py -q`
Expected: PASS — survivors now exceed `0.1 * egg_abundance`.

- [ ] **Step 8: Engine regression (NOT the parity baselines yet)**

Run: `.venv/bin/python -m pytest tests/test_engine_mortality_loop.py tests/test_engine_predation.py tests/test_engine_mortality.py tests/test_engine_bioen_integration.py tests/test_engine_functional_response.py -q`
Expected: PASS — `test_engine_functional_response.py` must be green (proves the `:780` kernel-call update + the new param compile). Predation outputs change for egg prey, but these tests assert invariants/directions, not stored baselines. If any asserts an exact pre-fix egg-predation number, that's an expected shift — report it; do not blindly edit the assertion.

**Baseline-staleness note (for the reviewer):** this commit shifts BoB/EEC biomass, so the bit-exact `TestBaselineParity` cases in `tests/test_engine_parity.py` (local-only) WILL fail until Task 4 regenerates the baselines. Do NOT run `tests/test_engine_parity.py` in this task and do NOT treat its failure as a Task-3 regression — Task 4 restores it after the Java cross-check confirms the shift is Java-correct.
Run: `.venv/bin/ruff check osmose/engine/processes/mortality.py tests/test_engine_egg_retention.py && .venv/bin/ruff format --check osmose/engine/processes/mortality.py tests/test_engine_egg_retention.py && .venv/bin/pyright osmose/engine/processes/mortality.py`
Expected: clean.

- [ ] **Step 9: Commit (the code fix; parity gate is Task 4)**

```bash
git add osmose/engine/processes/mortality.py tests/test_engine_egg_retention.py tests/test_engine_functional_response.py
git commit -m "fix(engine): gate egg predation on the released fraction (Numba + Python paths)"
```

---

### Task 4: Fix 1 parity gate — Java cross-check + EEC/BoB baselines

**Files:**
- Create: `tests/test_egg_retention_java_parity.py` (Python-vs-Java cross-check)
- Modify: `scripts/save_parity_baseline.py` (emit an EEC baseline), `tests/test_engine_parity.py` (add the EEC case), `pyproject.toml` (register the `slow` marker)
- New baselines (committed only after the Java check passes): `tests/baselines/parity_baseline_eec_1yr_seed42.npz`, regenerated `parity_baseline_bob_1yr_seed42.npz`

**Interfaces:**
- Consumes: `osmose.runner.OsmoseRunner(jar_path).run(...)` (async), `osmose.results.OsmoseResults`, the Python `simulate(...)` path (Task 3).

**This task is a verification gate, not a behavior change. Its DELIVERABLE is evidence that Fix 1 moves Python toward Java, plus regenerated baselines.**

- [ ] **Step 0: Register the `slow` marker**

In `pyproject.toml` add to the `[tool.pytest.ini_options] markers` list (currently only `e2e`, `visual`, around `:95-98`):

```toml
    "slow: long-running tests incl. the Java cross-check (opt-in; run with -m slow)",
```

This silences the unknown-marker warning. Note `addopts` (`-m 'not e2e and not visual'`) does NOT exclude `slow`, so isolation relies on the `OSMOSE_JAR` skip guard above — do not rely on the marker alone.

- [ ] **Step 1: Write the Java cross-check test**

Create `tests/test_egg_retention_java_parity.py`. **Gate it on the `OSMOSE_JAR` env var** (the repo's opt-in pattern at `tests/test_calibration_problem_python_engine.py:113-116`) — NOT on a JAR-glob `skipif`, because the JAR is present in-tree so a glob guard never skips and the multi-minute Java run would execute on every default `pytest`. Also register the `slow` marker (Step 0 below). Use the **cross-engine ratio band** `0.1 <= py/java <= 10.0` (the "within 1 OoM" check the repo uses at `tests/test_calibration_problem_python_engine.py:135`), NOT `test_engine_parity.py`'s 5% Python-vs-Python tolerance.

```python
"""Python-vs-Java cross-check for the egg-retention fix: the FIXED Python engine
must agree with Java within ~1 order of magnitude per species (Java implements
graduated egg release). If Python DIVERGES from Java, the fix's direction is
wrong — STOP. Opt-in via OSMOSE_JAR; excluded from the default suite."""

import os
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_JAR = os.environ.get("OSMOSE_JAR")


@pytest.mark.skipif(not _JAR, reason="set OSMOSE_JAR to the 4.3.3 jar to run")
def test_python_matches_java_eec_biomass():
    # 1. Java: OsmoseRunner(Path(_JAR)).run(config_path, output_dir) is async ->
    #    asyncio.run(...); read with OsmoseResults(output_dir, prefix="eec")
    #    (EEC sets output.file.prefix=eec; the default "osm" prefix finds no files).
    # 2. Python: run the SAME EEC config via simulate(...) (model the load +
    #    final-year mean biomass extraction on tests/test_engine_parity.py::_run_engine,
    #    but pointed at data/eec_full/).
    # 3. For each species assert 0.1 <= py_biomass/java_biomass <= 10.0; on failure
    #    print the per-species ratio so a divergence is diagnosable.
    ...
```

**NOTE to implementer:** `OsmoseRunner.run` (`osmose/runner.py:131`) is async — wrap in `asyncio.run`. Use a short horizon (1-2 years). The 4.3.3 jar has historically struggled to load `eec_full` — if the Java run errors on config load (not a biomass divergence), that is a JAR/config-compat blocker, not a fix problem: report it and treat Task 4 as blocked pending a runnable Java EEC config (the fix's unit test in Task 3 still stands).

- [ ] **Step 2: Run the Java cross-check (with Fix 1 applied)**

Run: `OSMOSE_JAR=osmose-java/<the-4.3.3-jar>.jar .venv/bin/python -m pytest tests/test_egg_retention_java_parity.py -q -m slow`
Expected: PASS — fixed Python agrees with Java within the 0.1–10× band. (Without `OSMOSE_JAR` set, it SKIPS — confirm it's skipped, not silently passing, in the default suite.)
**STOP CONDITION:** if it FAILS with Python *further* from Java than before Fix 1, do not proceed. Report the per-species ratios — this falsifies the assumption that Java gradually releases eggs, and the fix direction needs human review before any baseline is touched.

- [ ] **Step 3: Add the EEC baseline tooling**

In `scripts/save_parity_baseline.py`, add an EEC variant: a `--config {bob,eec}` flag (default `bob`) selecting the EEC config under `data/eec_full/` and writing `tests/baselines/parity_baseline_eec_<years>yr_seed<seed>.npz`. **Parameterize the base directory, not just the CSV path:** the existing `save_baseline` hardcodes `data/examples/` as the base for grid/NetCDF resolution (`EXAMPLES_CONFIG` + the `grid.netcdf.file` lookup); for EEC the base must be `data/eec_full/` (its grid `eec_grid-mask.nc` and background-species NetCDFs live there). Mirror the rest of the body (read config, set nyear, build grid via `Grid.from_netcdf(base / grid_file, ...)`, `simulate`, save biomass/abundance/mortality/species_names).

In `tests/test_engine_parity.py`, add an EEC case mirroring the BoB fixture: an `_eec_baseline_path()`, an `_run_engine_eec()` (or parametrize `_run_engine` on config), and a parity test that loads `parity_baseline_eec_*.npz` and compares (same tolerance structure as the BoB case; skip if the baseline is absent).

- [ ] **Step 4: Capture pre-fix deltas, then regenerate baselines (ONLY after Step 2 is green)**

Generate the EEC and BoB baselines on the FIXED engine:

```bash
.venv/bin/python scripts/save_parity_baseline.py --config eec --years 1 --seed 42
.venv/bin/python scripts/save_parity_baseline.py --config bob --years 1 --seed 42
```

Record the biomass deltas vs the prior BoB baseline (and the EEC pre/post if a pre-fix EEC baseline was captured) in the commit message — the egg fix is *expected* to shift these; the Java cross-check (Step 2) is what justifies the regeneration.

- [ ] **Step 5: Run the parity suite against the regenerated baselines**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -q`
Expected: PASS (now matches the regenerated EEC + BoB baselines).
Run: `.venv/bin/ruff check scripts/save_parity_baseline.py tests/test_engine_parity.py tests/test_egg_retention_java_parity.py && .venv/bin/ruff format --check scripts/save_parity_baseline.py tests/test_engine_parity.py tests/test_egg_retention_java_parity.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/test_egg_retention_java_parity.py scripts/save_parity_baseline.py tests/test_engine_parity.py pyproject.toml tests/baselines/parity_baseline_eec_1yr_seed42.npz tests/baselines/parity_baseline_bob_1yr_seed42.npz
git commit -m "test(engine): Java cross-check + EEC/BoB parity baselines for egg-retention fix

Java cross-check confirms the fixed Python engine agrees with Java within
the parity band; EEC + BoB baselines regenerated to the post-fix outputs.
Biomass deltas vs prior baseline: <fill in from Step 4>."
```

---

### Final verification (before finishing the branch)

- [ ] Full suite (default): `.venv/bin/python -m pytest -n auto -q -m "not e2e and not visual and not slow"` → green. (Plain `-m "not slow"` would OVERRIDE `addopts` and re-admit e2e/visual — keep all three exclusions.)
- [ ] `.venv/bin/ruff check osmose/ ui/ tests/ scripts/` and `.venv/bin/ruff format --check osmose/ ui/ tests/ scripts/` → clean.
- [ ] `.venv/bin/pyright osmose/engine/processes/mortality.py osmose/calibration/problem.py` → no new errors.
- [ ] Java cross-check ran green (Task 4 Step 2) — the egg fix is Java-confirmed.
- [ ] Critical-fix regression intact: `.venv/bin/python -m pytest tests/test_engine_mortality_causes.py tests/test_engine_bioen_starvation_rate_suppressed.py -q` → pass.

## Spec coverage map

- Fix 3 (`_worker_eval` remove swallow + `_WORKER_PROBLEM`-patched test) → Task 1.
- Fix 2 (`_fleet_effort_factor` helper, `_precompute_effective_rates` refactor, `_apply_fishing_for_school` + call site, `_HAS_NUMBA`-patched test) → Task 2.
- Fix 1 egg-retention (Numba kernel + 3 drivers + dispatch + Python fallback, falsifiable Numba-path test) → Task 3.
- Fix 1 parity gate (Java cross-check first, then EEC + BoB baseline regen) → Task 4.
- Global: `_HAS_NUMBA` patching, eatable model, parity stop-and-report, lint/pyright → per-task steps + Final verification.
