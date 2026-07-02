---
name: Python-side perf plan outcome
description: 2026-05-08 — A1 + A2 shipped, A3 dropped; cumulative 37.8 % wall-time reduction on eec_full 5-yr from vectorising the non-JIT'd hot paths
type: project
originSessionId: d2b7f4a5-d107-4042-a473-f491e81f4df1
---
The Python-side perf plan (PR #34) targeted three Python-side hot paths the K4 profile gate identified as larger wins than the kernel-surgery K-list. Outcome:

| Item | Surface | Plan estimate | Measured | PR | Status |
|---|---|---:|---:|---:|---|
| A1 | `AccessibilityMatrix.compute_school_indices` (vectorise via per-species `np.searchsorted`) | 2-7 % | **17.7 %** | #35 | shipped |
| A2 | `_precompute_map_indices` (vectorise per-species 2D index) | 2-12 % | **24.4 %** | #36 | shipped |
| A3 | fused `compute_school_indices_both` for prey+pred | 1-4 % | 0.7 % | #37 | **dropped** (post-mortem only) |

**Cumulative master → post-A2:** 4.872 s → 3.030 s = **37.8 % wall-time reduction** on eec_full 5-yr, exact bit-for-bit parity on all 14 species, 12/12 parity tests still bit-exact.

Plan-target comparison: floor target was 5 %, ceiling was 23 %. Realised 37.8 % = **1.6× the ceiling**.

## Why A3 dropped

A3's saving estimate (1-4 %) assumed the per-call cost was dominated by per-species mask + age-conversion work. After A1 collapsed `compute_school_indices` from ~1460 µs/call to ~290 µs/call, halving the already-small remaining overhead produced sub-noise-floor results (20 ms over 3.030 s = 0.7 %). The fused method itself is functionally correct (12/12 parity tests passed during the speculative implementation, all 6 new tests passed) — A3 simply has no measurable surface left to optimise after A1.

## Patterns that worked

1. **K-style profile gate first.** K4 (PR #33) ran cProfile on the warm Numba cache and identified that the largest wins were *outside* the JIT'd kernel, in pure Python orchestration. Without K4 the perf plan would still have targeted the K1/K2/K3 kernel-surgery items (combined ceiling: 12 % on optimistic estimates).
2. **Per-species mask + vectorised inner ops.** Both A1 and A2 share the same template: `for sp_idx, ms in lookup.items(): mask = species_id == sp_idx; if not mask.any(): continue; <vectorised work on subset>`. This converts an n_schools-Python-loop into an n_species-Python-loop with vectorised inner work — same pattern as DSVM accumulator (H7) and biomass_by_cell (vectorised in PR #29).
3. **Keep the loop implementation as `_*_loop` for cross-check tests.** Critical for randomised parity tests against the vectorised path. Both A1 and A2 retain `_compute_school_indices_loop` / `_precompute_map_indices_loop` as static cross-check references.
4. **Pre-mask BEFORE indexing on negative indices.** A2's r2 review caught that NumPy's negative-index wrap-around (e.g. `arr[-1, step]` returns last row) makes `prev_age = age - 1` for `age == 0` schools silently produce a wrong result. Always explicit-mask before indexing on candidate-negative indices.

## Followups (deferred)

- **A4 candidate:** `compute_feeding_stages` per-cell call inside `_predation_in_cell_python`/`_predation_in_cell_numba` dispatch (predation.py:565). cProfile showed feeding-stage work below the 2 % gate on eec_full pre-A1/A2; needs a fresh profile post-A2 to re-prioritise.
- **K1 (kernel surgery):** still in the kernel-surgery plan as conditional-pursue (3-12 % straddles the gate). Whether to attempt it depends on whether the post-A2 hot-path moves it above or below the gate. Re-profile before deciding.
- **Post-A2 hot-path:** the original cProfile top-3 (`compute_school_indices`, `_precompute_map_indices`, `mortality()` Python wrapper) all collapsed. Future perf work needs a fresh profile against the post-A2 master to identify the new top-3 — don't extrapolate from the K4 baseline.

## Reference

- Plan: `docs/plans/2026-05-08-python-side-perf-plan.md`
- Profile predecessor: `docs/perf/2026-05-08-eec_full-5yr-profile.md`
- A3 post-mortem: `docs/perf/2026-05-08-A3-not-shipping.md`
- Worktree benchmark gotcha: `feedback_pythonpath_worktree_benchmark.md`
