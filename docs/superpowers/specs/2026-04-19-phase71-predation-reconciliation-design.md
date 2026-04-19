# Phase 7.1 — Reconcile dual predation paths (design)

**Status:** design / ready for implementation plan
**Target:** `osmose/engine/processes/predation.py` + six test files
**Ship target:** v0.9.3 (or bundled with any other post-v0.9.2 cleanup)
**Baseline:** master `0e59258` (v0.9.2), 2510 tests passing
**Parity-roadmap anchor:** `docs/parity-roadmap.md` §7.1

## Problem statement

Two parallel implementations apply predation to schools:

1. **`predation.predation()`** at `osmose/engine/processes/predation.py:521` — a batch entry point that loops over all grid cells and applies predation only (no fishing/starvation/additional mortality). Kept alive by 34 call sites in six test files. **NOT called by production.**
2. **`mortality.mortality()`** at `osmose/engine/processes/mortality.py:1665` — the production path. Per-cell, per-school, with predation / starvation / fishing / additional mortality interleaved.

Both paths share the actual predation math via `_predation_in_cell_python` / `_predation_in_cell_numba` / `_predation_on_resources`. The duplication is only at the orchestration layer — and the standalone orchestrator exists solely because tests need predation isolation.

Keeping `predation()` around is low-cost but it has two ongoing costs:

- **Drift risk.** When someone modifies predation semantics (e.g., accessibility matrix handling, diet tracking wiring), they must remember to update both orchestrators. A past regression audit surfaced at least one case where `predation()` lagged behind production.
- **Dead-code signal.** New contributors reading `predation.py` see two functions named `predation` and `_predation_in_cell_python` with overlapping purpose and spend time figuring out which is canonical. The test-driven orchestrator is an archaeological artefact of an earlier refactor (pre-Phase-5).

The parity roadmap §7.1 (`docs/parity-roadmap.md:316-321`) flags this: "Not suitable for autonomous execution — needs test-parity audit first." This design is that audit.

## Goals

- Delete the standalone `predation()` function.
- Give tests a clean, honest public surface for predation-isolated testing.
- Preserve every existing numerical invariant the 34 call sites verify today.
- Zero production-path changes (no touching `simulate.py` or `mortality.py`).

## Non-goals

- Refactoring `mortality.py`'s per-cell machinery.
- Aligning Java's predation orchestration with Python's — Java has no predation-isolation surface and the parity roadmap tracks behavior, not API shape.
- Consolidating `_predation_in_cell_python` and `_predation_in_cell_numba` into one dispatcher — they have genuinely different signatures because Numba requires unpacked arrays.
- Promoting `_predation_on_resources` to public. It stays private, invoked via `predation_for_cell(resources=...)`.

## Architecture

### Public API

One new public function in `osmose/engine/processes/predation.py`:

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
    """
```

Design decisions behind this signature:

- **`cell_indices` accepted, not computed internally.** The caller knows which schools occupy the cell. Tests that fabricate a two-school scenario in cell (0,0) pass `np.array([0, 1], dtype=np.int32)` directly. No redundant `argsort`/`searchsorted` per call.
- **`use_numba` keyword-only, default `_HAS_NUMBA`.** Lets the Numba-vs-Python parity tests explicitly pick a backend. Honors the same `_HAS_NUMBA` fallback as the batch `predation()` — requesting `use_numba=True` when Numba is unavailable silently falls back to Python, matching current behavior.
- **`ctx`, `species_rngs`, `resources`, `cell_y`, `cell_x` all optional.** Minimum call is `predation_for_cell(indices, state, cfg, rng, n_subdt=10)`. Diet tracking, resource predation, and species-specific RNGs opt in via kwargs.
- **In-place on `state`.** Matches `_predation_in_cell_python`'s existing contract. No `.replace(...)` reassignment at the callsite. This is a deliberate divergence from batch `predation()`, which returned a new state — tests that currently capture the return value will be rewritten to read `state` after the call.

Internally, `predation_for_cell`:

1. Handles the early-exit case `len(cell_indices) < 2` (no predation possible).
2. Precomputes `prey_access_idx` / `pred_access_idx` / `access_matrix` for the call (redundant if a test calls it in a loop over cells, but test performance isn't a concern).
3. Ensures `feeding_stage` is populated on `state`. Population check is `state.feeding_stage is None or state.feeding_stage.size == 0`; if either, call `compute_feeding_stages(state, config)` and reassign via `state.replace(feeding_stage=...)`. Tests that call `predation_for_cell` twice on overlapping `cell_indices` get the same cached `feeding_stage` for both calls — overwrite is avoided after the first call. This is a deliberate departure from batch `predation()`, which always recomputed.
4. Dispatches to `_predation_in_cell_numba` or `_predation_in_cell_python` based on `use_numba` + `_HAS_NUMBA`. **Silent fallback when `use_numba=True` and Numba is unavailable** — matches the current batch `predation()` behavior. Numba-parity tests that need to assert "this run actually exercised Numba" must independently guard with `pytest.importorskip("numba")` or an explicit `if not _HAS_NUMBA: pytest.skip(...)` at the top of the test body. The audit (class C) flags every parity site so this guard is consistently added during migration.
5. If `resources is not None and resources.n_resources > 0`, invokes `_predation_on_resources` for the same cell coordinates.

**Caller contract for `cell_indices`:** the caller owns correctness. No internal validation. Indices must be unique, in-range (`0 <= idx < state.n_schools`), and represent schools that all genuinely occupy cell `(cell_y, cell_x)`. Passing duplicates or out-of-range indices is undefined behavior — expect silent wrong math, not a raised error. This is the same contract `_predation_in_cell_python` has today; the new public function doesn't tighten it.

**Parameter naming:** the existing private `_predation_in_cell_python` takes `indices` as its first parameter. The new public API uses `cell_indices` — a deliberate rename for public clarity. Internal call sites adapt accordingly.

Helpers that remain public and untouched:

- `enable_diet_tracking(ctx)`, `disable_diet_tracking(ctx)`, `get_diet_matrix(ctx)`
- `compute_size_overlap(pred, prey, sizeratio_min, sizeratio_max)`
- `compute_appetite(...)`
- `compute_feeding_stages(state, config)`

Helpers that remain private:

- `_predation_in_cell_python`, `_predation_in_cell_numba`
- `_predation_on_resources`
- `_DUMMY_ACCESS`, `_DUMMY_DIET`

### Test-migration classification

Every call site in the six affected files falls into one of seven classes (A-G). The implementation plan's first task is an audit pass that produces a `file:line → class → replacement snippet` table for all 34 sites, plus a separate audit column flagging `ctx=` preservation (orthogonal to A-G, captured in its own checklist). Classes:

| Class | Meaning | Replacement shape |
|---|---|---|
| A | Single-cell, single `predation()` call | `predation_for_cell(np.array([...], dtype=np.int32), state, cfg, rng, n_subdt=10)` |
| B | Multi-cell (verifying isolation) | One `predation_for_cell` call per occupied cell, OR a single call asserting schools outside `cell_indices` are untouched |
| C | Numba/Python parity | Pass `use_numba=True`/`False` explicitly. Add `pytest.importorskip("numba")` or `_HAS_NUMBA` guard if the test asserts Numba was actually exercised — silent fallback would otherwise give a false pass. |
| D | Resource predation | Pass `resources=...` |
| E | Background-species edge cases | Same as A; state just happens to include background species |
| F | Signature introspection (`test_engine_rng_consumers.py`) | Introspect `predation_for_cell` instead of `predation` |
| G | Return-value semantics | Any test that does `new_state = predation(...)` and then either (a) asserts `new_state is not state` (immutability check) or (b) passes `new_state` into a subsequent call. Must be rewritten because `predation_for_cell` modifies in place. Class-G sites get flagged explicitly during the audit — migration rewrites to read `state` directly and, if the immutability assertion was load-bearing, deletes it (the new API's in-place contract makes it inapplicable). |

### Call-site distribution (measured at baseline)

| File | Calls |
|---|---:|
| `tests/test_engine_predation_helpers.py` | 12 |
| `tests/test_engine_diet.py` | 5 |
| `tests/test_engine_predation.py` | 5 |
| `tests/test_engine_background.py` | 4 |
| `tests/test_engine_feeding_stages.py` | 4 |
| `tests/test_engine_rng_consumers.py` | 4 |
| **Total** | **34** |

The parity roadmap said "20+" — the actual count is 34. Not a concern, just a scale update.

## Implementation sequence

Four commits, each reviewable in isolation:

### Commit 1 — `feat(predation): public predation_for_cell API`

Adds `predation_for_cell` alongside the existing `predation()`. **Refactors `predation()` to call `predation_for_cell` in its cell loop**, so from this commit onward the two paths are provably the same code — any migration-induced drift is caught here, before any test changes. No test changes in this commit.

New file size: `predation.py` grows by ~50 lines (the new public function) but the `predation()` body shrinks by ~30 (delegating to the new function), net ~+20 lines.

Verification: full suite passes unchanged (2510 passed, 15 skipped, 41 deselected at baseline). The predation-isolation tests still pass because `predation()` still exists and still works — it just internally delegates.

### Commit 2 — `refactor(tests): migrate predation-isolation tests to predation_for_cell`

All 34 call sites migrated. Grouped by file (so the diff is reviewable one module at a time). Each test file's diff should be ≤ ~200 lines; if a single file exceeds that, split it into sub-commits per logical test group.

After migration, run the RNG-drift canary: three consecutive runs of the six affected test files with the same seed, asserting identical outputs. If `pytest-repeat` is installed, use `pytest --count=3`; otherwise a three-iteration bash loop.

Verification: `grep -c "predation_for_cell(" tests/test_engine_*.py` ≥ 34. Full suite still passes. No test file imports the bare `predation` name any more — `grep -rn "from osmose.engine.processes.predation import predation\b" tests/` returns zero hits (the `\b` word-boundary excludes `predation_for_cell`, which still imports legitimately).

### Commit 3 — `refactor(predation): delete batch predation() orchestrator`

Removes the `predation()` function. Audits module-level state — `_DUMMY_ACCESS` is used by the Numba path for the `access_matrix is None` fallback, so it stays. `_DUMMY_DIET` likewise. Nothing else to remove. Updates the module docstring to reflect the new public surface.

Small diff: most of the work landed in commit 1 already.

Verification: `grep -n "^def predation\b" osmose/engine/processes/predation.py` empty. Full suite passes. `ruff check` clean.

### Commit 4 — `docs: parity-roadmap Phase 7.1 STATUS-COMPLETE`

Updates `docs/parity-roadmap.md` §7.1 with a SHIPPED banner and the commit anchor (hash of commit 3). One-line diff.

## Risks and mitigations

| Risk | Manifestation | Mitigation |
|---|---|---|
| Numerical drift between old and new paths | Tests pinning exact abundance values diverge because per-cell precomputation order changes | Commit 1 routes `predation()` through `predation_for_cell` — they're the same code path. Any drift is a bug in commit 1, caught before any migration begins. |
| RNG consumption order change | Tests draw RNG between migrated calls in different order than before | Test migrations don't insert RNG draws between calls. `species_rngs` handling preserved (first-predator-species RNG for cell shuffle). |
| Feeding-stage recomputation per cell | Per-call recomputation vs batch precomputation | Both paths compute the same values deterministically from state. Commit 1's behavior-preserving property covers this. |
| `_predation_on_resources` coupling | Resource predation currently invoked inside batch `predation()` per-cell | `predation_for_cell(resources=...)` forwards `cell_y`, `cell_x` kwargs; resource step runs after predation step for the same cell. |
| Signature-introspection tests | `inspect.signature(predation)` asserts `"species_rngs" in sig.parameters` | Adapt to introspect `predation_for_cell`. `species_rngs` is in the new signature. |
| Diet matrix state leak | Migration drops `ctx=` kwarg; diet assertions silently fail | Audit pass explicitly flags every site that passes `ctx=` in the current code — orthogonal to the A-G class taxonomy and tracked as a separate audit column. Migration preserves `ctx=` when present; the plan writer adds a checklist item "every ctx= site in the before-code has ctx= in the after-code". |
| Numba unavailability | Parity tests require both backends | `_HAS_NUMBA` already gates. `predation_for_cell(use_numba=True)` with Numba absent silently falls back to Python (matching current `predation()` behavior). Class-C parity tests therefore must not rely on silent fallback — the audit adds `pytest.importorskip("numba")` or an explicit `_HAS_NUMBA` guard to every class-C migration so the test skips rather than silently passing under the wrong backend. |

## Success criteria

Ship criteria (all must hold):

1. `grep -n "^def predation\b" osmose/engine/processes/predation.py` → empty.
2. `grep -rn "from osmose.engine.processes.predation import.*\bpredation\b" osmose/ tests/` → zero hits of the bare `predation` name (distinct from `predation_for_cell`).
3. `grep -c "predation_for_cell(" tests/test_engine_*.py` summed ≥ 34.
4. Full suite: `pytest -q` reports **≥ baseline passed, ≤ baseline skipped, zero new failures, zero new skips introduced by this branch**. The baseline at spec time is 2510 passed / 15 skipped / 41 deselected; if unrelated commits land on master first, the plan's implementer updates the target to the pre-branch baseline they observe and keeps the "no regression" framing.
5. `ruff check osmose/ tests/` → clean.
6. RNG-drift canary: three consecutive runs of the six affected test files, same seed, identical outputs.
7. `docs/parity-roadmap.md` §7.1 carries a SHIPPED banner.
8. `simulate.py` untouched; EEC 14/14 and Bay of Biscay 8/8 parity tests still pass with unchanged biomass values.

## Rollback

Each commit is individually revertible, but the revertibility window is **directional and tied to commit ordering**:

- **Commit 4 (docs):** trivially revertible at any point.
- **Commit 3 (delete `predation()`):** revert at any point → `predation()` comes back. Tests still pass because they've been migrated to `predation_for_cell` and both functions would then exist.
- **Commit 2 (test migration):** per-file partial revert is safe **only before commit 3 lands**. After commit 3, a single-file revert of commit 2 would put that file back on `predation()` — a function that no longer exists, breaking test collection. Post-commit-3, commit 2 and commit 3 must be reverted together (or not at all). The plan should treat commits 2+3 as a paired unit for any rollback beyond docs.
- **Commit 1 (add API):** behavior-preserving by construction. If this ever needs to revert, the whole branch goes.

**Practical implication:** the safest rollback posture is to review commits 1-3 together before any of them push to a shared branch. Once they're all on master, individual-file regressions surface as test failures on the per-file scope they originated from — revert the whole trio, not a subset.

## Out-of-scope follow-ups

- Phase 7.1 does not try to collapse the `_predation_in_cell_python` / `_predation_in_cell_numba` duplication. Numba's array-unpacking requirement makes a single dispatcher impractical without significant rework. A future phase could revisit.
- If `_predation_on_resources` ever grows test coverage, it may deserve promotion. Not now.
- A similar audit could be run for `mortality.py` internals (the per-cell helpers `_apply_predation_for_school`, `_apply_starvation_for_school`, etc. have tests that peek at them), but that's a separate effort.
