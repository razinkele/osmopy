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
- Modify: `osmose/engine/processes/mortality.py` (add `_fleet_effort_factor`; use it in `_precompute_effective_rates:778-794` and `_apply_fishing_for_school:180`; pass `ctx.fleet_state` at the Python fishing call site `:1737`)
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
    cell_x), or 0.0 if that cell is out of bounds. Mirrors mortality.py:778-794."""
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

In `_precompute_effective_rates`, replace the fleet block at lines 778-794 (the `if fleet_state is not None:` two-loop block ending at `eff_fishing[i] *= effort_factor[i]`) with:

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

At `mortality.py:1737` (inside `_mortality_in_cell`'s pure-Python loop, the `elif cause == _FISHING:` branch), change the call:

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
- Modify: `osmose/engine/processes/mortality.py` — `_apply_predation_numba` (def `:810`, prey-avail `:886`) + its three callers (`_mortality_in_cell_numba` call `:1139`, `_mortality_all_cells_numba` call `:1306`, `_mortality_all_cells_parallel` call `:1485`) + the driver signatures + the `mortality()` dispatch call (`:~1939`) + the `_mortality_in_cell` dispatch; and `_apply_predation_for_school` (prey-avail `:397`).
- Test: `tests/test_engine_egg_retention.py` (new)

**Interfaces:**
- Consumes: `work_state.egg_retained` (float64 array, current per sub-dt), `state.egg_retained` (Python path).
- Produces: nothing for later tasks (Task 4 gates this).

**Background:** `egg_retained` is computed and decremented per sub-dt but never read by predation. Thread it into the predation reads so eatable egg prey = `max(inst_abd[q] - egg_retained[q], 0)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_egg_retention.py`. Model the config/state on `tests/test_engine_mortality_loop.py::TestPredation` (it builds a 2-species predator/prey cell). Set the prey school as an egg cohort and give the predator an appetite strictly between `egg_abundance/n_subdt` and `egg_abundance`:

```python
"""Egg-retention: a graduated egg cohort must NOT be fully eaten in sub-dt 1.
egg_retained is released abundance/n_subdt per sub-dt; predation must only see the
released fraction. Deep review 2026-06-22 (HIGH-1). Runs on the default (Numba) path.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.mortality import mortality
from osmose.engine.resources import ResourceState
from osmose.engine.state import SchoolState
from tests.test_engine_mortality_loop import TestPredation  # reuse its _make_2sp_config


def test_egg_cohort_not_fully_consumed_in_first_subdt():
    cfg = EngineConfig.from_dict(TestPredation()._make_2sp_config())
    grid = Grid.from_dimensions(ny=1, nx=1)
    # School 0 = predator (sp1), school 1 = egg-cohort prey (sp0).
    state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
    # ... configure: prey is_egg=True with a known abundance; predator large+hungry;
    #     same cell; n_subdt >= 4 so the released slice << full cohort.
    # (Use the field setup from TestPredation.test_total_eaten_never_exceeds_max_eatable
    #  as the template for sizes/weights/feeding_stage/accessibility.)
    egg_abundance = 1e6
    # ... assign state.is_egg[1]=True, state.abundance[1]=egg_abundance, etc.
    resources = ResourceState.empty(grid, cfg)
    rng = np.random.default_rng(0)
    out = mortality(state, cfg, grid, resources, n_subdt=4, step=0, rng=rng)
    survivors = out.abundance[1]
    # Buggy code eats the whole cohort in sub-dt 1 -> ~0 survivors.
    # Fixed code can eat at most the released slices -> a clear positive remainder.
    assert survivors > 0.1 * egg_abundance
```

**NOTE to implementer:** fill the `...` setup by copying the concrete field assignments from `TestPredation.test_total_eaten_never_exceeds_max_eatable` (`tests/test_engine_mortality_loop.py:180`), then set `state.is_egg[1] = True` and a predator appetite large enough to clear the whole cohort in one sub-dt. The assertion threshold (`> 0.1 * egg_abundance`) must FAIL on current code (cohort wiped) — verify that in Step 2; if current code leaves survivors (appetite too small), increase predator size/ingestion until the buggy run wipes the cohort, so the test is falsifiable.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_egg_retention.py -q`
Expected: FAIL — `survivors` is ~0 (egg cohort fully consumed in sub-dt 1). If it does not fail, the predator appetite is too small; increase it until the buggy path wipes the cohort.

- [ ] **Step 3: Thread `egg_retained` into the Numba kernel (prey-avail clamp)**

In `_apply_predation_numba` (def at `:810`), add `egg_retained` as the FINAL positional parameter. At the prey-availability read inside the prey scan (the `abd_q = inst_abd[q_idx]` line, `:886`), change to:

```python
            abd_q = inst_abd[q_idx] - egg_retained[q_idx]
            if abd_q < 0.0:
                abd_q = 0.0
```

- [ ] **Step 4: Thread `egg_retained` through the three driver kernels**

Add `egg_retained` as the FINAL positional parameter to the signatures of `_mortality_in_cell_numba` (`:1069`), `_mortality_all_cells_numba` (`:1211`), and `_mortality_all_cells_parallel` (`:1377`); and pass `egg_retained` as the final arg at each `_apply_predation_numba(...)` call site (`:1139`, `:1306`, `:1485`).

- [ ] **Step 5: Pass `work_state.egg_retained` at the dispatch sites**

In `mortality()`, at the `_batch_fn(...)` call (`:~1939`), add `work_state.egg_retained` as the final argument. In the `_mortality_in_cell` fallback dispatch (the `else:` per-cell loop calling `_mortality_in_cell`), pass `work_state.egg_retained` through to `_mortality_in_cell` (add the param there too, final position, and forward to its internal `_mortality_in_cell_numba` call).

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

Run: `.venv/bin/python -m pytest tests/test_engine_mortality_loop.py tests/test_engine_predation.py tests/test_engine_mortality.py tests/test_engine_bioen_integration.py -q`
Expected: PASS. (Predation outputs change for egg prey, but these tests assert invariants/directions, not stored baselines. If any asserts an exact pre-fix egg-predation number, that's an expected shift — report it; do not blindly edit the assertion.)
Run: `.venv/bin/ruff check osmose/engine/processes/mortality.py tests/test_engine_egg_retention.py && .venv/bin/ruff format --check osmose/engine/processes/mortality.py tests/test_engine_egg_retention.py && .venv/bin/pyright osmose/engine/processes/mortality.py`
Expected: clean.

- [ ] **Step 9: Commit (the code fix; parity gate is Task 4)**

```bash
git add osmose/engine/processes/mortality.py tests/test_engine_egg_retention.py
git commit -m "fix(engine): gate egg predation on the released fraction (Numba + Python paths)"
```

---

### Task 4: Fix 1 parity gate — Java cross-check + EEC/BoB baselines

**Files:**
- Create: `tests/test_egg_retention_java_parity.py` (Python-vs-Java cross-check)
- Modify: `scripts/save_parity_baseline.py` (emit an EEC baseline), `tests/test_engine_parity.py` (add the EEC case)
- New baselines (committed only after the Java check passes): `tests/baselines/parity_baseline_eec_1yr_seed42.npz`, regenerated `parity_baseline_bob_1yr_seed42.npz`

**Interfaces:**
- Consumes: `osmose.runner.OsmoseRunner(jar_path).run(...)` (async), `osmose.results.OsmoseResults`, the Python `simulate(...)` path (Task 3).

**This task is a verification gate, not a behavior change. Its DELIVERABLE is evidence that Fix 1 moves Python toward Java, plus regenerated baselines.**

- [ ] **Step 1: Write the Java cross-check test**

Create `tests/test_egg_retention_java_parity.py`. Mark it `@pytest.mark.slow` and skip when the JAR is absent. Run one short EEC config (`data/eec_full/`) on BOTH engines and compare per-species final biomass within the established parity band (the repo's parity tolerance — "within 1 OoM"; reuse the tolerance constant/asserts from `tests/test_engine_parity.py`'s statistical/portable section).

```python
"""Python-vs-Java cross-check for the egg-retention fix: the FIXED Python engine
must agree with Java within the parity band (Java implements graduated egg
release). If Python diverges from Java, the fix's direction is wrong — STOP.
Slow/JAR-gated; not in the default suite."""

import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_REPO = pathlib.Path(__file__).resolve().parent.parent
_JAR = next((_REPO / "osmose-java").glob("*jar-with-dependencies*.jar"), None)


@pytest.mark.skipif(_JAR is None, reason="OSMOSE Java JAR not present")
def test_python_matches_java_eec_biomass():
    # 1. Run EEC on Java via OsmoseRunner(_JAR).run(config_path, output_dir) (async ->
    #    asyncio.run); read with OsmoseResults(output_dir).
    # 2. Run the same EEC config on the Python engine (simulate(...)) with matching
    #    nyear/seed.
    # 3. Compare per-species final-year mean biomass within the parity tolerance.
    # Assert each species agrees within the band; on failure, print the per-species
    # ratio so a divergence is diagnosable.
    ...
```

**NOTE to implementer:** model the Java run on the existing runner usage (`osmose/runner.py:OsmoseRunner.run` is async — wrap in `asyncio.run`) and the output read on `osmose.results.OsmoseResults`. Model the Python run + biomass extraction on `tests/test_engine_parity.py::_run_engine`. Use a short horizon (1-2 years) to keep it tractable. If no existing Python-vs-Java comparison helper exists, this test defines the first one — keep the tolerance identical to the repo's stated parity band; do NOT invent a looser one.

- [ ] **Step 2: Run the Java cross-check (with Fix 1 applied)**

Run: `.venv/bin/python -m pytest tests/test_egg_retention_java_parity.py -q -m slow`
Expected: PASS — fixed Python agrees with Java within tolerance.
**STOP CONDITION:** if it FAILS with Python *further* from Java than before Fix 1, do not proceed. Report the per-species ratios — this falsifies the assumption that Java gradually releases eggs, and the fix direction needs human review before any baseline is touched.

- [ ] **Step 3: Add the EEC baseline tooling**

In `scripts/save_parity_baseline.py`, add an EEC variant: a `--config {bob,eec}` flag (default `bob`) selecting `data/eec_full/` (find its top-level `*all-parameters*.csv`) and writing `tests/baselines/parity_baseline_eec_<years>yr_seed<seed>.npz`. Mirror the existing `save_baseline` body (read config, set nyear, build grid, `simulate`, save biomass/abundance/mortality/species_names).

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
git add tests/test_egg_retention_java_parity.py scripts/save_parity_baseline.py tests/test_engine_parity.py tests/baselines/parity_baseline_eec_1yr_seed42.npz tests/baselines/parity_baseline_bob_1yr_seed42.npz
git commit -m "test(engine): Java cross-check + EEC/BoB parity baselines for egg-retention fix

Java cross-check confirms the fixed Python engine agrees with Java within
the parity band; EEC + BoB baselines regenerated to the post-fix outputs.
Biomass deltas vs prior baseline: <fill in from Step 4>."
```

---

### Final verification (before finishing the branch)

- [ ] Full suite (default, no slow/e2e): `.venv/bin/python -m pytest -n auto -q -m "not slow"` → green.
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
