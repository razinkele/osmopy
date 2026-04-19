# Phase 7.1 — Reconcile dual predation paths: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the standalone `osmose/engine/processes/predation.py::predation()` batch orchestrator by migrating its ~30 test call sites to a new public `predation_for_cell()` API, without any behavior change to production.

**Architecture:** Add `predation_for_cell` as the public per-cell predation entry point. Refactor the existing batch `predation()` to delegate to it in a cell loop (commit 1 — behavior-preserving). Migrate tests file-by-file to call `predation_for_cell` directly with explicit `cell_indices` (commit 2). Delete `predation()` (commit 3). Update parity roadmap (commit 4).

**Tech Stack:** Python 3.12, NumPy, Numba JIT (optional), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-19-phase71-predation-reconciliation-design.md` (HEAD `4fec5c8`, 4 review rounds converged).

**Baseline:** master `4fec5c8`, **2510 passed / 15 skipped / 41 deselected**, ruff clean. This is a refactor — target is unchanged counts (no new tests, no regressions).

**Ship target:** v0.9.3 (or bundled with any other post-v0.9.2 cleanup).

**Known spec decisions carried into this plan:**
- Default `use_numba=_HAS_NUMBA`. Silent fallback when True + Numba absent.
- In-place mutation of four `state` fields (`abundance`, `pred_success_rate`, `preyed_biomass`, `feeding_stage`). Callers with multi-call sequences must manage `+=` accumulation on `pred_success_rate` / `preyed_biomass`.
- `feeding_stage` populated via `state.feeding_stage[:] = compute_feeding_stages(state, config)` — in-place numpy write (frozen dataclass allows mutable-value mutation).
- Caller owns `cell_indices` correctness; no internal validation.
- `predation_for_cell` returns `None`; all modifications land on the passed `state` reference.

---

## Pre-flight

- [ ] Confirm baseline. Run `.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3`. Expected: `2510 passed, 15 skipped, 41 deselected, 62 warnings`. If different, record the number and use `baseline+0` framing in all task-success checks (this is a refactor — expect no delta).
- [ ] Confirm ruff baseline. Run `.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2`. Expected: `All checks passed!`.
- [ ] Verify line anchors (grep rather than trust):
  - `osmose/engine/processes/predation.py` — `def predation(` at `:521`; `_predation_in_cell_python` at `:255`; `_predation_in_cell_numba` at `:130`; `_predation_on_resources` at `:371`; `_DUMMY_ACCESS = np.zeros` at `:518`; `_DUMMY_DIET` defined at `:125`, used inside the Numba dispatch branch around `:610-612`.
  - `osmose/engine/state.py` — `@dataclass(frozen=True)` at `:30`; `class SchoolState` at `:31`; `feeding_stage: NDArray[np.int32]` at `:59`.
  - `osmose/engine/processes/feeding_stage.py` — `compute_feeding_stages` at `:20`, returns `np.zeros(n, dtype=np.int32)` for empty states at `:40-43`.
- [ ] Verify production path is untouched. Grep `osmose/engine/simulate.py` for `from osmose.engine.processes.predation import predation` — should find NO such import (only `enable_diet_tracking` and `disable_diet_tracking` imports at `:1199` and `:1216`).

---

## Task 1: Add `predation_for_cell` + route `predation()` through it (behavior-preserving)

**Goal:** Add the new public function; refactor the existing `predation()` batch orchestrator to call it in its cell loop. After this commit both functions exist and both produce identical output, proven by the full unchanged test suite still passing. No test changes in this commit.

**Files:**
- Modify: `osmose/engine/processes/predation.py` (add `predation_for_cell`, refactor `predation()` body).

---

- [ ] **Step 1: Read the current `predation()` body end-to-end**

Open `osmose/engine/processes/predation.py` and re-read lines 521-683 (the full current `predation()` implementation). The new function will factor out everything the current code does per-cell; the refactored `predation()` will keep only the cross-cell batching (cell-id argsort, searchsorted boundaries, accumulating the new biomass at the end).

Also read `_predation_in_cell_python` at `:255-363` to understand what fields it mutates (`abundance`, `pred_success_rate`, `preyed_biomass` via `+=` accumulation) and `_predation_on_resources` at `:371` to understand its signature.

---

- [ ] **Step 2: Insert `predation_for_cell` ABOVE the existing `predation()` function**

Add this new function at `osmose/engine/processes/predation.py` just above the `def predation(` line (around `:521`). Paste verbatim, don't abbreviate:

```python
def predation_for_cell(
    cell_indices: NDArray[np.int32],
    state: SchoolState,
    config: EngineConfig,
    rng: np.random.Generator,
    n_subdt: int,
    *,
    use_numba: bool = _HAS_NUMBA,
    ctx: SimulationContext | None = None,
    species_rngs: list[np.random.Generator] | None = None,
    resources: ResourceState | None = None,
    cell_y: int = 0,
    cell_x: int = 0,
) -> None:
    """Apply predation (+ optional resource predation) within a single cell.

    In-place modification of state arrays. Public contract for test harnesses
    and any caller that needs predation isolation without running the full
    mortality pipeline. Production code uses mortality.mortality() instead.

    Fields mutated in place on ``state``:
      - abundance       (absolute assignment)
      - pred_success_rate  (accumulating +=)
      - preyed_biomass     (accumulating +=)
      - feeding_stage   (fresh overwrite via [:])

    Caller owns cell_indices correctness (unique, in-range, all in cell
    (cell_y, cell_x)). No internal validation.
    """
    if len(cell_indices) < 2:
        return

    # Ensure feeding_stage is current. compute_feeding_stages returns a fresh
    # np.int32 array of length n_schools; frozen dataclass permits in-place
    # buffer mutation even though attribute reassignment is blocked.
    state.feeding_stage[:] = compute_feeding_stages(state, config)

    # Precompute accessibility info for this call. Matches the precomputation
    # block in the current batch predation() at lines 564-590, but scoped to
    # this single call. Per-call recomputation sidesteps the in-loop rebinding
    # quirk on _DUMMY_ACCESS that the batch function has as a local-variable
    # side effect.
    if config.stage_accessibility is not None:
        sa = config.stage_accessibility
        prey_access_idx = sa.compute_school_indices(
            state.species_id,
            state.age_dt,
            config.n_dt_per_year,
            config.all_species_names,
            role="prey",
        )
        pred_access_idx = sa.compute_school_indices(
            state.species_id,
            state.age_dt,
            config.n_dt_per_year,
            config.all_species_names,
            role="pred",
        )
        access_matrix = sa.raw_matrix
        has_access = True
        use_stage_access = True
    else:
        prey_access_idx = np.zeros(len(state), dtype=np.int32)
        pred_access_idx = np.zeros(len(state), dtype=np.int32)
        has_access = config.accessibility_matrix is not None
        access_matrix = config.accessibility_matrix if has_access else _DUMMY_ACCESS
        use_stage_access = False

    cell_indices_i32 = cell_indices.astype(np.int32, copy=False)

    # Dispatch to Numba or Python backend. Silent fallback if Numba requested
    # but unavailable — matches current predation() behavior.
    if use_numba and _HAS_NUMBA:
        if species_rngs is not None and len(cell_indices_i32) > 0:
            first_pred_sp = int(state.species_id[cell_indices_i32[0]])
            _cell_rng = (
                species_rngs[first_pred_sp]
                if first_pred_sp < len(species_rngs)
                else rng
            )
        else:
            _cell_rng = rng
        pred_order = _cell_rng.permutation(len(cell_indices_i32)).astype(np.int32)
        _diet_en = ctx.diet_tracking_enabled if ctx else False
        _diet_mat = ctx.diet_matrix if ctx else None
        diet_mat = _diet_mat if _diet_en and _diet_mat is not None else _DUMMY_DIET
        if access_matrix is None:
            access_matrix = _DUMMY_ACCESS
        _predation_in_cell_numba(
            cell_indices_i32,
            pred_order,
            state.abundance,
            state.length,
            state.weight,
            state.age_dt,
            state.first_feeding_age_dt,
            state.species_id,
            state.pred_success_rate,
            state.preyed_biomass,
            config.size_ratio_min,
            config.size_ratio_max,
            config.ingestion_rate,
            access_matrix,
            has_access,
            n_subdt,
            config.n_dt_per_year,
            state.feeding_stage,
            prey_access_idx,
            pred_access_idx,
            use_stage_access,
            diet_mat,
            _diet_en,
        )
    else:
        _predation_in_cell_python(
            cell_indices_i32,
            state,
            config,
            rng,
            n_subdt,
            prey_access_idx=prey_access_idx if use_stage_access else None,
            pred_access_idx=pred_access_idx if use_stage_access else None,
            stage_access_matrix=access_matrix if use_stage_access else None,
            ctx=ctx,
        )

    # Resource predation: focal schools eat LTL plankton/detritus.
    if resources is not None and resources.n_resources > 0:
        _predation_on_resources(
            cell_indices_i32,
            state,
            config,
            resources,
            cell_y,
            cell_x,
            rng,
            n_subdt,
            pred_access_idx=pred_access_idx if use_stage_access else None,
            stage_access_matrix=access_matrix if use_stage_access else None,
            ctx=ctx,
        )
```

---

- [ ] **Step 3: Refactor `predation()` body to delegate to `predation_for_cell`**

Replace the existing `predation()` body (from `predation.py:521` through the end of the function, around `:683`) with this delegating implementation. The function keeps the cross-cell batching and the final `state.replace(...)` return so its external contract is unchanged:

```python
def predation(
    state: SchoolState,
    config: EngineConfig,
    rng: np.random.Generator,
    n_subdt: int,
    grid_ny: int,
    grid_nx: int,
    resources: ResourceState | None = None,
    species_rngs: list[np.random.Generator] | None = None,
    ctx: SimulationContext | None = None,
) -> SchoolState:
    """Apply predation across all grid cells.

    Batch orchestrator retained as a thin wrapper around predation_for_cell
    during the Phase 7.1 migration. Delete in commit 3 after all tests
    migrate to predation_for_cell directly.
    """
    if len(state) == 0:
        return state

    # Make working copies so the caller's state is not mutated.
    abundance = state.abundance.copy()
    pred_success_rate = state.pred_success_rate.copy()
    preyed_biomass = state.preyed_biomass.copy()
    feeding_stage = state.feeding_stage.copy()

    work_state = state.replace(
        abundance=abundance,
        pred_success_rate=pred_success_rate,
        preyed_biomass=preyed_biomass,
        feeding_stage=feeding_stage,
    )

    # Group schools by cell using searchsorted (fast boundary detection).
    cell_ids = work_state.cell_y * grid_nx + work_state.cell_x
    order = np.argsort(cell_ids, kind="mergesort")
    sorted_cells = cell_ids[order]

    n_cells = grid_ny * grid_nx
    boundaries = np.searchsorted(sorted_cells, np.arange(n_cells + 1))

    for cell in range(n_cells):
        start = boundaries[cell]
        end = boundaries[cell + 1]
        if end - start < 2:
            continue

        cell_indices = order[start:end].astype(np.int32)
        predation_for_cell(
            cell_indices,
            work_state,
            config,
            rng,
            n_subdt,
            use_numba=_HAS_NUMBA,
            ctx=ctx,
            species_rngs=species_rngs,
            resources=resources,
            cell_y=cell // grid_nx,
            cell_x=cell % grid_nx,
        )

    new_biomass = work_state.abundance * work_state.weight

    return state.replace(
        abundance=work_state.abundance,
        biomass=new_biomass,
        pred_success_rate=work_state.pred_success_rate,
        preyed_biomass=work_state.preyed_biomass,
    )
```

The old body's inline `feeding_stage = compute_feeding_stages(work_state, config)` and `work_state = work_state.replace(feeding_stage=feeding_stage)` lines are deleted — `predation_for_cell` handles this per-call via the in-place `[:]` write.

**Behavioral note:** the current batch `predation()` computes `feeding_stage` ONCE before the cell loop; the refactored version has `predation_for_cell` recompute it at every call. The values are identical because `compute_feeding_stages` reads only `state.species_id` / `age_dt` / `length` / `weight` / `trophic_level`, none of which are modified during predation. So the recomputation is redundant within a batch call but numerically faithful. In the terminal state (after commit 3 when batch `predation()` is deleted), tests each call `predation_for_cell` once per test invocation, so the redundancy vanishes.

**Edge case for batch caller:** if a `predation()` caller has a state where EVERY cell has `< 2` schools (so `predation_for_cell` early-exits for every cell), `feeding_stage` is never recomputed by the cell loop. The caller sees `feeding_stage` = the `.copy()` we made before the loop, i.e., whatever the caller passed in. This matches current behavior at the `state.replace(...)` return: the returned state also does not include `feeding_stage`, so the caller's original array is what they see. No regression.

---

- [ ] **Step 4: Run the full test suite — expect unchanged pass count**

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
```

Expected: `2510 passed, 15 skipped, 41 deselected`. If any test regresses, commit 1 is broken — investigate before proceeding. Most likely failure modes:

- `test_engine_predation.py::test_same_cell_predation_via_top_level` fails: the delegation chain has a bug. Check the `use_numba=_HAS_NUMBA` dispatch branch.
- `test_engine_predation_helpers.py::test_diet_tracking_*` fails: `ctx` not forwarded correctly. Grep for `ctx=ctx` in the new `predation()` body.
- `test_engine_rng_consumers.py::test_predation_accepts_species_rngs` fails: signature regression. Keep `species_rngs` in `predation()`'s public signature.
- Numba parity tests fail with a value drift: the per-call access-index recomputation may have a subtle difference from batch precomputation. Add a print-based diagnostic inside the new `predation_for_cell` to compare indices with the batch version.

---

- [ ] **Step 5: Run ruff — expect clean**

```bash
.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2
```

Expected: `All checks passed!`.

If ruff flags the new function (likely complaints: unused imports, line-length, type-annotation order), fix in place without changing behavior.

---

- [ ] **Step 6: Commit**

Write the commit message to `/tmp/task1.msg` using the Write tool (repo CLAUDE.md forbids heredocs with # lines inline):

```
feat(predation): public predation_for_cell API (Phase 7.1 commit 1)

Add a new public function osmose.engine.processes.predation.
predation_for_cell(cell_indices, state, config, rng, n_subdt, ...) that
applies predation (and optional resource predation) within a single cell.
In-place modification of state arrays. Public contract for test harnesses
and any caller that needs predation isolation without running the full
mortality pipeline.

Refactor the existing batch predation() function to delegate to
predation_for_cell in its cell loop. Both functions coexist during the
migration. This commit is behavior-preserving: the batch function keeps
its external contract (fresh working copies, returned state.replace),
and predation_for_cell is the SAME code path the batch runs today, just
extracted. Any behavior drift after this commit is a bug here, caught
before any test migrates.

Fields mutated in place on state:
  - abundance        (absolute assignment)
  - pred_success_rate (accumulating +=)
  - preyed_biomass    (accumulating +=)
  - feeding_stage    (fresh overwrite via state.feeding_stage[:] =
                      compute_feeding_stages(...), which works under
                      @dataclass(frozen=True) because numpy arrays
                      are mutable attribute values)

Spec: docs/superpowers/specs/2026-04-19-phase71-predation-
reconciliation-design.md. Full suite still 2510 passed / 15 skipped /
41 deselected. Ruff clean.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Stage and commit:

```bash
git add osmose/engine/processes/predation.py
git commit -F /tmp/task1.msg
```

---

## Task 2: Migrate all test call sites to `predation_for_cell`

**Goal:** Update every test that calls the top-level `predation()` to call `predation_for_cell` directly instead. Test-file-by-test-file, each file a sub-commit. Full suite must pass after each sub-commit. Uses the audit table produced in step 1.

**Files (all modifications; no creations):**
- `tests/test_engine_predation_helpers.py`
- `tests/test_engine_diet.py`
- `tests/test_engine_predation.py`
- `tests/test_engine_background.py`
- `tests/test_engine_feeding_stages.py`
- `tests/test_engine_rng_consumers.py`

---

- [ ] **Step 1: Produce the audit table**

Enumerate every `predation()` call site with its class tag (A-G per the spec). Write the audit script to a file (the repo's shell rules block heredocs with `#` lines inline), then execute it.

First, write the script using the Write tool to `/tmp/phase71_audit.py`:

```python
import re
import sys
from pathlib import Path

TEST_FILES = [
    "tests/test_engine_predation_helpers.py",
    "tests/test_engine_diet.py",
    "tests/test_engine_predation.py",
    "tests/test_engine_background.py",
    "tests/test_engine_feeding_stages.py",
    "tests/test_engine_rng_consumers.py",
]

CALL_RE = re.compile(r"(?<![a-zA-Z_])predation\(")

lines = ["| File | Line | Snippet | Class | Notes |",
         "|---|---:|---|---|---|"]

project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

for f in TEST_FILES:
    text = (project_root / f).read_text().splitlines()
    for i, line in enumerate(text, start=1):
        if not CALL_RE.search(line):
            continue
        stripped = line.strip()
        if stripped.startswith('"') or stripped.startswith("'"):
            continue
        if stripped.startswith("#"):
            continue
        if "assert" in stripped and "predation()" in stripped:
            continue
        snippet = stripped[:60].replace("|", "\\|")
        surrounding_before = "\n".join(text[max(0, i - 10):i])
        surrounding_after = "\n".join(text[i:i + 5])
        class_tag = "A"
        if "cell_x=np.array([0, 5]" in surrounding_before or "different cell" in surrounding_before:
            class_tag = "B"
        if "use_numba" in surrounding_before + surrounding_after:
            class_tag = "C"
        if "resources=" in line or "ResourceState" in surrounding_before:
            class_tag = "D"
        if "inspect.signature" in line:
            class_tag = "F"
        if "new_state =" in line:
            class_tag = f"{class_tag} (return captured — check G)"
        lines.append(f"| {f} | {i} | `{snippet}` | {class_tag} |  |")

Path("/tmp/phase71-audit.md").write_text("\n".join(lines))
print(f"Wrote {len(lines) - 2} call sites to /tmp/phase71-audit.md")
```

Run it:

```bash
.venv/bin/python /tmp/phase71_audit.py /home/razinka/osmose/osmose-python
```

Expected: `Wrote <N> call sites to /tmp/phase71-audit.md` where N is in the 20-30 range. Review `/tmp/phase71-audit.md` manually.

The heuristic class assignments are starting points — refine by hand:

- Any row where the snippet captures the return value AND the test later uses `new_state is state`, `new_state is not state`, or chains `new_state` into another call → **class G**.
- Any row in `test_engine_rng_consumers.py` that wraps `inspect.signature(predation)` → **class F**.
- Any row where the 10 lines above the call construct `cell_x=np.array([..., ...])` with two distinct values → **class B**.
- Any row with `ctx=ctx` in the args → add a note "**ctx** — preserve in migrated call".
- Any row where the SAME test function calls `predation()` more than once (look for pattern `_run_predation` or pairs of `predation(...)` calls in the same `def test_*`) → add a note "**multi-call** — check if fresh-copy semantics was load-bearing, zero `pred_success_rate`/`preyed_biomass` between migrated calls if so".

Expected rough counts at HEAD `4fec5c8`:
- Total call sites across 6 files: ~22-26 (varies by how strictly you filter — the target counts in commit 2's verification use actual grep).
- Class A: majority (simple single-cell).
- Class B: ~2 sites (`test_engine_predation.py:181` and any similar in `test_engine_background.py`).
- Class C: the `_run_predation` helper in `test_engine_predation_helpers.py` (2 or 3 calls using Numba mocking).
- Class D: 2 sites (`test_engine_diet.py` and `test_engine_predation.py` for resource predation).
- Class F: 1 site (`test_engine_rng_consumers.py::test_predation_accepts_species_rngs`).
- Class G: spot-check; likely 0-3 sites.

---

- [ ] **Step 2: Migrate `tests/test_engine_predation_helpers.py`**

This file has the largest number of call sites and the Numba/Python parity helper (`_run_predation`). It's the trickiest migration.

**Import update.** Change the top of the file:

```python
# Before
from osmose.engine.processes.predation import (
    compute_appetite,
    compute_size_overlap,
    disable_diet_tracking,
    enable_diet_tracking,
    get_diet_matrix,
    predation,
)
```

```python
# After
from osmose.engine.processes.predation import (
    compute_appetite,
    compute_size_overlap,
    disable_diet_tracking,
    enable_diet_tracking,
    get_diet_matrix,
    predation_for_cell,
)
```

**Call-site migration (representative samples).** For each call matching `predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10, ctx=ctx)`:

```python
# Before
predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10, ctx=ctx)

# After — tests in this file place both schools in cell (0, 0); so cell_indices = [0, 1]
cell_indices = np.array([0, 1], dtype=np.int32)
predation_for_cell(cell_indices, state, cfg, rng, n_subdt=10, ctx=ctx)
```

**Numba/Python parity helper (`_run_predation`).** The current helper uses `mock.patch.object(predation_module, "_HAS_NUMBA", use_numba)` to force dispatch. With `use_numba` now an explicit kwarg on `predation_for_cell`, the mock is unnecessary — the new API honors the flag directly.

**Important: Numba availability guard.** The guard belongs on the **parity-comparison test** (the test that runs both arms and compares), NOT inside `_run_predation` itself. If `_run_predation(use_numba=True)` did `pytest.importorskip("numba")` internally, calling the helper with `use_numba=False` would also run the import check and still work — but a parity test that calls both arms and compares outputs must skip entirely on Numba-less systems (you can't compare to a non-existent Numba result).

Pattern:

```python
# Before (simplified)
def _run_predation(self, use_numba: bool, seed: int = 42) -> tuple:
    state, cfg = _make_pred_prey_state()
    rng = np.random.default_rng(seed)
    ctx = SimulationContext(...)
    enable_diet_tracking(ctx, ...)
    with mock.patch.object(predation_module, "_HAS_NUMBA", use_numba):
        new_state = predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10, ctx=ctx)
    diet = get_diet_matrix(ctx)
    return new_state, diet

# In parity-comparison test (simplified)
def test_numba_python_parity(self):
    state_numba, diet_numba = self._run_predation(use_numba=True, seed=42)
    state_python, diet_python = self._run_predation(use_numba=False, seed=42)
    np.testing.assert_allclose(state_numba.abundance, state_python.abundance)
```

```python
# After — helper is now simple, guard lives on the parity test
def _run_predation(self, use_numba: bool, seed: int = 42) -> tuple:
    state, cfg = _make_pred_prey_state()
    rng = np.random.default_rng(seed)
    ctx = SimulationContext(...)
    enable_diet_tracking(ctx, ...)
    cell_indices = np.array([0, 1], dtype=np.int32)
    predation_for_cell(
        cell_indices, state, cfg, rng, n_subdt=10, use_numba=use_numba, ctx=ctx
    )
    diet = get_diet_matrix(ctx)
    return state, diet  # in-place; return `state` not `new_state`

def test_numba_python_parity(self):
    pytest.importorskip("numba")  # skip whole comparison if Numba absent
    state_numba, diet_numba = self._run_predation(use_numba=True, seed=42)
    state_python, diet_python = self._run_predation(use_numba=False, seed=42)
    np.testing.assert_allclose(state_numba.abundance, state_python.abundance)
```

Callers of `_run_predation` assert on `state_numba == state_python` — the return-value name change doesn't affect them since they only use positional unpacking. Drop `mock` and `predation_module` imports from the file's top if no other test uses them.

**Per-call migration pattern.** Each `predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10, ctx=ctx)` in this file follows the same shape because all tests use two schools in cell (0, 0). Use `cell_indices = np.array([0, 1], dtype=np.int32)` for two-school cases; adjust indices list for tests that create more schools.

**Run the affected tests after each 4-5 line migration block** — don't migrate every site then run once at the end.

```bash
.venv/bin/python -m pytest tests/test_engine_predation_helpers.py -v 2>&1 | tail -20
```

Expected: all pre-existing test names pass. No changes to test count.

**Commit sub-step** (after all helpers tests pass):

```bash
git add tests/test_engine_predation_helpers.py
git commit -m "refactor(tests): migrate test_engine_predation_helpers to predation_for_cell (Phase 7.1 commit 2a)

Update imports (predation -> predation_for_cell). Rewrite every
predation() call site in this file to use cell_indices + in-place
state mutation. The Numba/Python parity helper _run_predation now
passes use_numba explicitly; the parity-comparison test itself
adds pytest.importorskip('numba') so the whole comparison skips
cleanly when Numba is absent.

No behavior change. Suite still 2510 passed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 3: Migrate `tests/test_engine_diet.py`**

Same import swap + call-site migration. This file has 5 call sites, mostly class A with `ctx=ctx` and a couple of class D (with `resources=`).

**Import update** — as above, swap `predation` for `predation_for_cell` in the `from osmose.engine.processes.predation import (...)` block.

**Class A migration (4 sites):** same as helpers file:

```python
# Before
predation(state, cfg, rng, n_subdt=10, grid_ny=1, grid_nx=1, ctx=ctx)
# After
predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10, ctx=ctx)
```

Note `grid_ny=1, grid_nx=1` is gone — the per-cell API doesn't need grid dimensions.

**Class D migration (1 site):** the test around `test_engine_diet.py:261` uses `resources=`:

```python
# Before
result = predation(
    state, cfg, rng, n_subdt=10, grid_ny=1, grid_nx=1, resources=resources, ctx=ctx
)
# After
cell_indices = np.array([0, 1], dtype=np.int32)
predation_for_cell(
    cell_indices, state, cfg, rng, n_subdt=10, resources=resources, ctx=ctx
)
result = state  # in-place; no return value to capture
```

Adjust assertions on `result` to read from `state` directly — if the previous assertion was `assert result.abundance[0] < 50.0`, the new form is `assert state.abundance[0] < 50.0`.

**Run:**
```bash
.venv/bin/python -m pytest tests/test_engine_diet.py -v 2>&1 | tail -15
```

**Commit:**
```bash
git add tests/test_engine_diet.py
git commit -m "refactor(tests): migrate test_engine_diet to predation_for_cell (Phase 7.1 commit 2b)

Five call sites migrated (4 class-A single-cell, 1 class-D resource).
ctx= kwarg preserved at every site for diet-tracking continuity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 4: Migrate `tests/test_engine_predation.py`**

This file has 4 call sites: 2 class A (same-cell), 1 class B (different-cell isolation), 1 already using `predation_in_cell` (aliased private — just needs the import rename).

**Import update.** Change:

```python
# Before
from osmose.engine.processes.predation import (
    _predation_on_resources,
    predation,
    _predation_in_cell_python as predation_in_cell,
)
```

```python
# After
from osmose.engine.processes.predation import (
    _predation_on_resources,
    predation_for_cell,
)
# Locally alias for back-compat with the existing `predation_in_cell(...)`
# call at line 159.
predation_in_cell = predation_for_cell
```

Actually, simpler — rewrite the call at line 159 to use `predation_for_cell` directly, then drop the alias:

```python
# At line 159 (approximately)
# Before
predation_in_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
# After (identical, just the new name)
predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
```

Then the import block becomes:

```python
from osmose.engine.processes.predation import (
    _predation_on_resources,
    predation_for_cell,
)
```

**Class B migration (`test_schools_in_different_cells_dont_interact`) — rewrite to test the per-cell `cell_indices` boundary.** The original test verified the BATCH orchestrator's cell-grouping: "when `predation()` loops over cells, a school in cell (0,0) does not eat a school in cell (5,5)." After Phase 7.1 that batch orchestrator is gone, but there's a corresponding useful invariant at the per-cell API: **`predation_for_cell` must only mutate schools whose indices appear in `cell_indices`**. Rewrite the test to verify this, using a three-school state where one school is a bystander outside `cell_indices`:

Bad alternatives (do NOT use either):
- `predation_for_cell(np.array([0], dtype=np.int32), ...)` — the function early-exits when `len(cell_indices) < 2` so nothing runs; the assertion is trivially true.
- Two sequential calls with one school each — same early-exit on both calls.

Good alternative:

```python
# Before (lines ~167-183)
def test_schools_in_different_cells_dont_interact(self):
    cfg = EngineConfig.from_dict(_make_predation_config())
    state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
    state = state.replace(
        abundance=np.array([50.0, 500.0]),
        length=np.array([25.0, 10.0]),
        weight=np.array([78.125, 6.0]),
        biomass=np.array([3906.25, 3000.0]),
        age_dt=np.array([24, 24], dtype=np.int32),
        cell_x=np.array([0, 5], dtype=np.int32),
        cell_y=np.array([0, 5], dtype=np.int32),
    )
    rng = np.random.default_rng(42)
    new_state = predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10)
    np.testing.assert_allclose(new_state.abundance[1], 500.0)
```

```python
# After — rename to reflect the new invariant. Three schools: a
# predator + prey in the cell being tested, plus a bystander not in
# cell_indices. The per-cell API must not mutate the bystander.
def test_school_outside_cell_indices_is_untouched(self):
    cfg = EngineConfig.from_dict(_make_predation_config())
    state = SchoolState.create(
        n_schools=3, species_id=np.array([1, 0, 0], dtype=np.int32)
    )
    state = state.replace(
        abundance=np.array([50.0, 500.0, 123.0]),
        length=np.array([25.0, 10.0, 10.0]),
        weight=np.array([78.125, 6.0, 6.0]),
        biomass=np.array([3906.25, 3000.0, 738.0]),
        age_dt=np.array([24, 24, 24], dtype=np.int32),
    )
    rng = np.random.default_rng(42)
    # Only schools 0 and 1 are in the cell we're exercising;
    # school 2 is the bystander outside cell_indices.
    predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
    # Predation ran: prey was eaten.
    assert state.abundance[1] < 500.0
    # Bystander was NOT in cell_indices — abundance unchanged.
    np.testing.assert_allclose(state.abundance[2], 123.0)
```

Pass count is preserved (one `def test_*` in, one `def test_*` out). The test name changes (`test_schools_in_different_cells_dont_interact` → `test_school_outside_cell_indices_is_untouched`); mention the rename in the commit message so anyone searching git log finds the connection.

**Class A migration (same-cell tests, `test_empty_state` and `test_same_cell_predation_via_top_level`):**

```python
# test_empty_state: migrate by calling predation_for_cell on an empty index set
def test_empty_state(self):
    cfg = EngineConfig.from_dict(_make_predation_config())
    state = SchoolState.create(n_schools=0)
    rng = np.random.default_rng(42)
    predation_for_cell(np.array([], dtype=np.int32), state, cfg, rng, n_subdt=10)
    assert len(state) == 0

# test_same_cell_predation_via_top_level: rename to reflect new API
def test_same_cell_predation(self):
    cfg = EngineConfig.from_dict(_make_predation_config())
    state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
    state = state.replace(
        abundance=np.array([50.0, 500.0]),
        length=np.array([25.0, 10.0]),
        weight=np.array([78.125, 6.0]),
        biomass=np.array([3906.25, 3000.0]),
        age_dt=np.array([24, 24], dtype=np.int32),
        cell_x=np.array([3, 3], dtype=np.int32),
        cell_y=np.array([2, 2], dtype=np.int32),
    )
    rng = np.random.default_rng(42)
    predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
    assert state.abundance[1] < 500.0
```

The rename `test_same_cell_predation_via_top_level` → `test_same_cell_predation` is not required but clearer after migration (nothing is "top-level" any more).

**Run:**
```bash
.venv/bin/python -m pytest tests/test_engine_predation.py -v 2>&1 | tail -20
```

**Commit:**
```bash
git add tests/test_engine_predation.py
git commit -m "refactor(tests): migrate test_engine_predation to predation_for_cell (Phase 7.1 commit 2c)

Four call sites migrated (3 class-A, 1 class-B). Drop the
_predation_in_cell_python-as-predation_in_cell alias — replaced by
direct predation_for_cell call. Rename
test_same_cell_predation_via_top_level (no longer top-level).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 5: Migrate `tests/test_engine_background.py`**

Four call sites, all class E (background-species edge cases). Pattern is identical to class A.

**Import update:**

```python
# Before (line 554)
from osmose.engine.processes.predation import predation  # noqa: E402
# After
from osmose.engine.processes.predation import predation_for_cell  # noqa: E402
```

**Call migration (4 sites, lines ~608, ~616, ~645, ~803):**

```python
# Before
result = predation(state, config, rng, n_subdt=10, grid_ny=3, grid_nx=3)
# After — most tests here have 2-3 schools in a single cell
cell_indices = np.array([0, 1, ...], dtype=np.int32)  # adjust based on n_schools
predation_for_cell(cell_indices, state, config, rng, n_subdt=10)
result = state  # in-place
```

Check the test body for how many schools exist — adjust `cell_indices` accordingly. For `n_schools=2`, use `np.array([0, 1], dtype=np.int32)`.

If any of these 4 sites is passing `config=ec` (some tests use `ec` as the variable name), the migration is the same — variable name preserved.

**Run:**
```bash
.venv/bin/python -m pytest tests/test_engine_background.py -v 2>&1 | tail -15
```

**Commit:**
```bash
git add tests/test_engine_background.py
git commit -m "refactor(tests): migrate test_engine_background to predation_for_cell (Phase 7.1 commit 2d)

Four class-E (background-species) call sites migrated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 6: Migrate `tests/test_engine_feeding_stages.py`**

Four class A call sites, all in cells (0, 0) with `grid_ny=1, grid_nx=1` (minimal grid).

**Import update:**

```python
# Before (line 8)
from osmose.engine.processes.predation import predation
# After
from osmose.engine.processes.predation import predation_for_cell
```

**Call migration:** identical pattern to diet file. Check how many schools each test creates — tests that use 2 schools get `cell_indices = np.array([0, 1], dtype=np.int32)`; tests with more schools expand the indices list.

```python
# Before
new_state = predation(state, ec, rng, n_subdt=10, grid_ny=1, grid_nx=1)
# After
predation_for_cell(np.array([0, 1], dtype=np.int32), state, ec, rng, n_subdt=10)
new_state = state  # in-place
```

Note: because this file exercises `feeding_stage` semantics, verify post-migration that `state.feeding_stage` has been populated correctly after the call (it should — `predation_for_cell` does `state.feeding_stage[:] = compute_feeding_stages(...)` in step 3 of its body).

**Run:**
```bash
.venv/bin/python -m pytest tests/test_engine_feeding_stages.py -v 2>&1 | tail -15
```

**Commit:**
```bash
git add tests/test_engine_feeding_stages.py
git commit -m "refactor(tests): migrate test_engine_feeding_stages to predation_for_cell (Phase 7.1 commit 2e)

Four class-A call sites migrated. feeding_stage is now populated
by predation_for_cell itself via in-place numpy write
(state.feeding_stage[:] = compute_feeding_stages(state, config))
matching the prior batch predation() semantics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 7: Migrate `tests/test_engine_rng_consumers.py`**

Three signature-introspection checks (class F) + one actual call (class A/D).

**Import update:**

```python
# Before (line 11)
from osmose.engine.processes.predation import predation
# After
from osmose.engine.processes.predation import predation_for_cell
```

**Class F migration (line 34-39):**

```python
# Before
def test_predation_accepts_species_rngs(self):
    """predation() should accept species_rngs as optional keyword argument."""
    sig = inspect.signature(predation)
    assert "species_rngs" in sig.parameters, "predation() must accept species_rngs parameter"
    param = sig.parameters["species_rngs"]
    assert param.default is None, "species_rngs default should be None"
```

```python
# After
def test_predation_for_cell_accepts_species_rngs(self):
    """predation_for_cell() should accept species_rngs as optional keyword argument."""
    sig = inspect.signature(predation_for_cell)
    assert "species_rngs" in sig.parameters, "predation_for_cell() must accept species_rngs parameter"
    param = sig.parameters["species_rngs"]
    assert param.default is None, "species_rngs default should be None"
```

**Class A/D migration (line 162 — call site):**

```python
# Before
result = predation(
    state, cfg, rng, n_subdt=10, grid_ny=1, grid_nx=1, species_rngs=None
)
# After
cell_indices = np.array([0, 1], dtype=np.int32)  # two-school fixture
predation_for_cell(
    cell_indices, state, cfg, rng, n_subdt=10, species_rngs=None
)
result = state
```

**Run:**
```bash
.venv/bin/python -m pytest tests/test_engine_rng_consumers.py -v 2>&1 | tail -15
```

**Commit:**
```bash
git add tests/test_engine_rng_consumers.py
git commit -m "refactor(tests): migrate test_engine_rng_consumers to predation_for_cell (Phase 7.1 commit 2f)

Signature-introspection test now inspects predation_for_cell.
species_rngs kwarg and default-None remain wired through the new API.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

- [ ] **Step 8: Full-suite verification + RNG-drift canary**

After all 6 files migrated:

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
```

Expected: `2510 passed, 15 skipped, 41 deselected`. Exact match.

Then the RNG-drift canary — run the affected test files three times with the same seed order and confirm identical output:

Run the pytest suite three times (each as its own Bash call; shell rules forbid `>>` redirection). Record the final summary line from each:

```bash
.venv/bin/python -m pytest tests/test_engine_predation.py tests/test_engine_predation_helpers.py tests/test_engine_feeding_stages.py tests/test_engine_diet.py tests/test_engine_background.py tests/test_engine_rng_consumers.py -q --no-header 2>&1 | tail -3
```

Run this command three times. Compare the final `N passed, M skipped` summary lines by eye (they should be identical). If you want a persistent log, write the expected summary to a file via the Write tool after the first run, then diff each subsequent run against it mentally — but there's no substitute for visually confirming "same N passed across all three runs" since RNG-drift test failures would manifest as different failure counts, which are immediately visible in the terminal.

**Grep verification:**

```bash
grep -rn "from osmose.engine.processes.predation import predation\b" tests/ osmose/
```

Expected: zero hits (the `\b` word-boundary excludes `predation_for_cell` which SHOULD still import legitimately).

```bash
grep -c "predation_for_cell(" tests/test_engine_*.py
```

Expected: non-zero per file, summing to ≥ 20 (one per migrated call site).

---

- [ ] **Step 9: Run ruff**

```bash
.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2
```

Expected: `All checks passed!`. Fix any lint issues in place.

If ruff flags unused imports (e.g., `Grid` or `numpy` imports that were only used via `predation(..., grid_ny=..., grid_nx=...)` and are no longer referenced), remove them.

---

## Task 3: Delete the batch `predation()` function

**Goal:** Remove `predation()` from `osmose/engine/processes/predation.py`. Small commit.

**Files:**
- Modify: `osmose/engine/processes/predation.py`.

---

- [ ] **Step 1: Remove the `predation()` function body**

Open `osmose/engine/processes/predation.py` and delete the full `def predation(...)` block (including its docstring and body) at the end of the file — the function that now just delegates to `predation_for_cell`.

---

- [ ] **Step 2: Audit module-level state for orphans**

Run:

```bash
grep -c "_DUMMY_ACCESS" osmose/engine/processes/predation.py
grep -c "_DUMMY_DIET" osmose/engine/processes/predation.py
```

Expected: both should still have ≥ 2 hits each — they're used by the Numba path inside `predation_for_cell`. If either drops to 1 (just the definition), investigate whether it can be removed; otherwise leave.

Module-level constants that should stay:
- `_DUMMY_ACCESS = np.zeros((1, 1), dtype=np.float64)` — used in Numba dispatch branch for `access_matrix is None` fallback.
- `_DUMMY_DIET = ...` — used for diet-matrix fallback in Numba dispatch.
- `_HAS_NUMBA` — used by the default value of `use_numba` in `predation_for_cell`.

---

- [ ] **Step 3: Update the module docstring**

At the top of `osmose/engine/processes/predation.py`, update any docstring or comment that references the batch `predation()` function. If the module has a top-level docstring mentioning "two entry points" or "predation() orchestrator", rewrite it to reflect the new single-entry API:

```python
"""Predation process.

Public API:
    predation_for_cell(cell_indices, state, config, rng, n_subdt, ...)
        Apply predation within a single cell. In-place on state.
    enable_diet_tracking / disable_diet_tracking / get_diet_matrix
        Diet-tracking helpers.
    compute_size_overlap, compute_appetite, compute_feeding_stages
        Utility predicates.

Test-exposed private helpers (leading underscore — not stable API):
    _predation_in_cell_python, _predation_in_cell_numba
    _predation_on_resources
        Used by targeted tests that need to exercise a specific backend
        or the resource-predation path in isolation. Tests that import
        these take on the maintenance burden if signatures change.

Production code uses mortality.mortality() rather than this module
directly; predation_for_cell is exposed for predation-isolated
testing.
"""
```

---

- [ ] **Step 4: Run the full suite**

```bash
.venv/bin/python -m pytest -q --no-header 2>&1 | tail -3
```

Expected: `2510 passed, 15 skipped, 41 deselected`. Must be unchanged.

If ANY test fails, the migration in Task 2 missed a site. Look at the failure:
- `ImportError: cannot import name 'predation'` → some test still imports the deleted name. Re-run the grep from Task 2 Step 8 to find it.
- `NameError: name 'predation' is not defined` → a call site wasn't migrated in commit 2. Fix it and bundle the fix into this commit (or revert this commit and extend Task 2).

---

- [ ] **Step 5: Run grep invariants**

```bash
grep -n "^def predation\b" osmose/engine/processes/predation.py
```

Expected: empty (no `def predation(` at top level).

```bash
grep -rn "from osmose.engine.processes.predation import.*\bpredation\b" osmose/ tests/
```

Expected: zero hits for the bare `predation` name (distinct from `predation_for_cell`).

```bash
grep -rn "\.predation(" osmose/ tests/
```

Expected: zero hits. (Catches any attribute-style access like `predation_module.predation(...)`.)

---

- [ ] **Step 6: Run ruff**

```bash
.venv/bin/ruff check osmose/ tests/ 2>&1 | tail -2
```

Expected: `All checks passed!`.

---

- [ ] **Step 7: Commit**

Write to `/tmp/task3.msg`:

```
refactor(predation): delete batch predation() orchestrator (Phase 7.1 commit 3)

Remove the standalone predation() batch function from
osmose/engine/processes/predation.py. All tests migrated to
predation_for_cell in commit 2; production already uses
mortality.mortality() as of Phase 5.

Module-level constants _DUMMY_ACCESS, _DUMMY_DIET, _HAS_NUMBA stay —
used by predation_for_cell's Numba dispatch branch.

Module docstring updated to reflect the new single-entry public API.

Spec: docs/superpowers/specs/2026-04-19-phase71-predation-
reconciliation-design.md. Suite still 2510 passed / 15 skipped /
41 deselected. Ruff clean.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

```bash
git add osmose/engine/processes/predation.py
git commit -F /tmp/task3.msg
```

---

## Task 4: Mark Phase 7.1 STATUS-COMPLETE on the parity roadmap

**Goal:** One-line update to `docs/parity-roadmap.md` §7.1 with the SHIPPED banner and commit anchor.

**Files:**
- Modify: `docs/parity-roadmap.md`.

---

- [ ] **Step 1: Open the parity roadmap section**

Read `docs/parity-roadmap.md` around line 316 (`## Phase 7: Code Quality & Architecture (ongoing)` and subsection `### 7.1 Reconcile dual predation paths`).

---

- [ ] **Step 2: Add the SHIPPED banner**

Find the current `### 7.1 Reconcile dual predation paths` heading and prepend SHIPPED status with a pointer to the commit range:

```markdown
### 7.1 Reconcile dual predation paths — SHIPPED (2026-04-19)
**STATUS:** SHIPPED — migrated 6 test files to `predation_for_cell` and deleted the standalone `predation()` batch orchestrator. Commits `<sha1>..<sha3>` (where sha1 = task 1 commit, sha3 = task 3 commit). Spec at `docs/superpowers/specs/2026-04-19-phase71-predation-reconciliation-design.md`.
```

Replace `<sha1>..<sha3>` with the actual short SHAs from `git log --oneline -6` after Task 3 commits.

---

- [ ] **Step 3: Commit**

```bash
git add docs/parity-roadmap.md
git commit -m "docs: parity-roadmap §7.1 STATUS-COMPLETE (Phase 7.1)

Mark Phase 7.1 (reconcile dual predation paths) as SHIPPED with
commit anchor. Predation now has one public per-cell entry and
one production orchestrator (mortality.mortality()).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review checklist

Before executing, the implementer verifies:

- [ ] Every spec requirement maps to a task — `predation_for_cell` public API (Task 1), test migration (Task 2), delete old orchestrator (Task 3), roadmap update (Task 4). ✓
- [ ] No placeholder text: no TBD, TODO, "fill in", or "similar to task N". Every code block is pasteable. ✓
- [ ] Type / symbol consistency: every mention of `predation_for_cell` uses the same signature defined in Task 1 Step 2. Every mention of `compute_feeding_stages` uses the import path `osmose.engine.processes.feeding_stage`. `_HAS_NUMBA` is referenced, not `HAS_NUMBA`. ✓
- [ ] Test runner: `.venv/bin/python -m pytest` throughout. No bare `python`. ✓
- [ ] Commit granularity: 1 (API) + 6 (per-file migrations) + 1 (delete) + 1 (docs) = **9 commits total**. Each reviewable in isolation (modulo the commit 2-3 rollback coupling noted in the spec). ✓
- [ ] Baseline check: pre-flight asserts 2510 passed / 15 skipped / 41 deselected. Each task's verification pins the same number. ✓
- [ ] Line anchors verified at plan writing time against HEAD `4fec5c8`. ✓

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-phase71-predation-reconciliation-plan.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent to execute Task 1 (the API + delegation), review, then dispatch per-file subagents for Task 2 sub-steps, then Task 3, then Task 4. Each with its own spec-compliance + code-quality review.

**2. Inline Execution** — I execute all tasks in this session using `superpowers:executing-plans` with checkpoints after Task 1 and Task 3.

Which approach?
