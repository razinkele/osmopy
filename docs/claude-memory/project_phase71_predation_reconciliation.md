---
name: Phase 7.1 predation-reconciliation implementation (SHIPPED)
description: How predation testing moved from the batch predation() orchestrator to the per-cell predation_for_cell() API — the mutation contract, the in-place caveats, and what Task 3's deletion left behind.
type: project
originSessionId: 62ae1657-034b-4171-9e07-85306c7671a8
---
**Shipped:** 2026-04-19 (master `72f80a3..0b82e68`), staged for v0.9.3 under `[Unreleased]`.

**What changed.** `osmose.engine.processes.predation` now exposes ONE public entry point for predation: `predation_for_cell(cell_indices, state, config, rng, n_subdt, *, use_numba, ctx, species_rngs, resources, cell_y, cell_x) -> None`. The old batch `predation()` orchestrator was deleted. All 6 test files previously using `predation()` migrated to the per-cell API. Production (`mortality.mortality()`) is untouched.

**The mutation contract (the gotcha).** `predation_for_cell` operates in-place on four fields of `state`:
- `abundance` — absolute assignment (set by predation loop)
- `pred_success_rate` — **accumulating** via `+=`
- `preyed_biomass` — **accumulating** via `+=`
- `feeding_stage` — fresh overwrite via `state.feeding_stage[:] = compute_feeding_stages(state, config)`

The batch `predation()` used to copy these fields before delegating so every call returned a fresh-copy-isolated state. The new per-cell API does not — sequential calls on a shared state **accumulate** `pred_success_rate` and `preyed_biomass`. Tests that called `predation()` twice in a row expecting isolation were identified in Task 2's audit and fixed (either zero fields between calls, or construct a fresh state per call).

**Caller owns `cell_indices`.** No internal validation. Unique, in-range, all-in-same-cell — caller's responsibility. `cell_x`/`cell_y` kwargs are only used by `_predation_on_resources` for grid lookups; `predation_for_cell`'s own dispatch is governed by `cell_indices`, not by `state.cell_x/y`.

**Frozen-dataclass trick.** `SchoolState` is `@dataclass(frozen=True)` but numpy array values ARE mutable. `state.feeding_stage[:] = new_array` works (buffer mutation); `state.feeding_stage = new_array` does not (attribute assignment). All Phase 7.1 in-place writes exploit this.

**Class taxonomy used in the audit (see plan):**
- A — single-cell, single call (majority)
- B — rewritten from "schools in different cells" (batch orchestrator invariant) to "schools outside cell_indices" (per-cell API invariant); see `test_school_outside_cell_indices_is_untouched`
- C — Numba/Python parity tests; `pytest.importorskip("numba")` on the parity-comparison test, `use_numba=True/False` kwarg on `_run_predation`
- D — resource predation (`resources=` kwarg)
- E — background-species edge cases (same as A structurally)
- F — signature-introspection tests (renamed to inspect `predation_for_cell`)
- G — return-value semantics (old `new_state = predation(...)` → in-place `state`)

**Key files:**
- `osmose/engine/processes/predation.py` — `predation_for_cell` at ~line 530; batch `predation()` deleted
- `tests/test_engine_predation_helpers.py` — `_run_predation(use_numba)` helper + parity tests with `importorskip`
- `tests/test_engine_predation.py` — `test_school_outside_cell_indices_is_untouched` is the post-migration version of the old spatial-isolation test
- `osmose/engine/state.py:30` — `@dataclass(frozen=True)` SchoolState (context for the in-place trick)

**Parity roadmap §7.1:** SHIPPED 2026-04-19, commits `72f80a3..009c781`. Phase 7 is now fully closed (7.1, 7.2 pre-v0.9.0, 7.3 in v0.9.2).
