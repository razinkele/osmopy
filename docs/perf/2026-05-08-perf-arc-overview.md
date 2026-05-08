# Engine perf arc — 2026-05-08 session

> Overview of the perf work completed across the 2026-05-08 session, written as an artefact-of-record for future readers who want to know *what was tried*, *what shipped*, *what dropped*, and *where the perf surface stands now* after the dust settled.

## Headline

Wall-time on `eec_full` 5-yr 7-repeat median:

```
Pre-A1 baseline:  4.872 s
Post-K1 + P2:     2.881 s   ← current master
Cumulative:       −40.9 % wall-time reduction
```

12/12 Java parity tests bit-exact across every shipped change. B-H opt-in default unchanged. No public API regressions.

## Plans (in order)

1. **Kernel-surgery plan** (`docs/plans/2026-05-08-kernel-surgery-plan.md`) — three K-items (K1/K2/K3) targeting JIT-internal allocations in the predation kernel.
2. **K4 profile gate** (`docs/perf/2026-05-08-eec_full-5yr-profile.md`) — read-only diagnostic that gated K1/K2/K3 against measured kernel-internal cost. **K3 dropped** at 0.41 % (post-mortem `K3-not-shipping.md`); **K2 dropped** at ~1 % + RNG-stream cost (post-mortem `K2-not-shipping.md`); **K1 conditional** at 3-12 % straddling the 2 % gate.
3. **Python-side perf plan** (`docs/plans/2026-05-08-python-side-perf-plan.md`) — pivot to the Python orchestration layer above the JIT'd kernel, which K4 surfaced as the larger surface (compute_school_indices at 17.3 %, _precompute_map_indices at 16.7 %).
4. **Post-v0.12.0 perf plan** (`docs/plans/2026-05-08-post-v012-perf-plan.md`) — picks up K1 (re-evaluate after Python-side wins shifted the ratio), `np.add.at` replacement, and `state.replace` machinery.
5. **P4b plan** (`docs/plans/2026-05-08-p4b-post-init-bypass-plan.md`) — `__post_init__` bypass via `object.__setattr__` on `cls.__new__(cls)`. Implementation correct, measurement below gate.

## Items

| Item | Surface | Plan estimate | Measured (eec_full) | Status |
|---|---|---:|---:|---|
| **A1** | Vectorise `AccessibilityMatrix.compute_school_indices` via per-species `np.searchsorted` over precomputed `_stages_by_role` arrays | 2-7 % | **17.7 %** | shipped (PR #35) |
| **A2** | Vectorise `_precompute_map_indices` per-species with explicit pre-mask before 2D indexing | 2-12 % | **24.4 %** | shipped (PR #36) |
| **A3** | Fused `compute_school_indices_both` for the prey + pred double-call | 1-4 % | 0.7 % | dropped (PR #37) |
| **K1** | Hoist `_apply_predation_numba` scratch buffers (3 arrays at lines 846-848) per-cell | 3-12 % | **6.4 %** | shipped (PR #40) |
| **K2** | Pool `cause_orders` + 4-element Fisher-Yates | ~5 % | est ~1 % | dropped at K4 |
| **K3** | Pool `larva_deaths` / `inst_abd` scratch via `SimulationContext` | 2 % | est 0.41 % | dropped at K4 |
| **P2** | Replace per-species `np.add.at` in `_collect_spatial_outputs` with `np.bincount` on flat `(y * nx + x)` index | 3-6 % | 1.8-3.1 % (likely noise — function is gated off) | shipped (PR #41) |
| **P3 / A4** | Vectorise `compute_feeding_stages` | 0-1 % | 0.14 % | dropped (PR #42) — already vectorised |
| **P4** | `state.replace` field-name caching | 0-2 % | 0.17 % | dropped (PR #43) |
| **P4b** | `__post_init__` bypass via `object.__setattr__` | 2-4 % | 1.8 % eec / 0.9 % baltic | dropped (PR #44) |

**Net shipped:** A1 + A2 + K1 + P2. **Net dropped:** A3 + K2 + K3 + P3 + P4 + P4b.

## What we learned

### 1. The K4 profile gate is the most important pattern in this arc

Without K4 — the read-only diagnostic that ran cProfile + alloc micro-bench against the warm Numba cache before any kernel change — we would have shipped K3 (0.41 % saving for a parity-test scope), pushed harder on K2 (~1 % for a calibration-cache invalidation cost), and missed the *much larger* Python-side wins (A1 + A2 = 42 % combined) entirely.

**Pattern:** before a perf-plan that changes Numba kernels, run cProfile against a warm JIT cache. Numba kernels are mostly opaque to cProfile, but the profile shows the Python-orchestration layer above the kernel — and that layer is where the big wins usually hide once parity-bound work is done.

### 2. The vectorisation pattern that shipped twice

Both A1 (accessibility) and A2 (movement maps) used the same template:

```python
# Before: per-school Python loop with dict lookup
for i in range(n_schools):
    sp = species_id[i]
    ...inner work...

# After: per-species mask + vectorised inner ops
for sp_idx, stages in stages_by_sp.items():
    mask = species_id == sp_idx
    if not mask.any():
        continue
    bin_idx = np.searchsorted(stages.thresholds, age_years[mask], side="right")
    ...vectorised work on subset...
```

This converts an `O(n_schools)` Python interpreter loop into an `O(n_species)` Python loop with vectorised inner work — same algorithm, ~10× speedup in practice on EEC's 14 focal species + 3 000 schools.

The key constraint: the per-species precomputed cache (`_stages_by_role` for accessibility, the `map_sets` dict for movement) must be built once at config-load time, not per-call. Both used the same dataclass-cache-after-construction pattern.

### 3. NumPy negative-index wrap-around is a silent correctness trap

A2's r2 review caught it: in `_precompute_map_indices`, schools with `age_dt == 0` produce `prev_age = -1`. NumPy treats negative indices as wrap-around, so `ms.index_maps[-1, prev_step]` silently returns the last row's value instead of the `-1` sentinel the legacy code emitted. The vectorised path needs an explicit pre-mask + scatter:

```python
prev_in_bounds = (prev_ages >= 0) & (prev_ages < n_ages) & (0 <= prev_step < n_steps)
prev_local = np.full(ages.shape, -1, dtype=np.int32)
if prev_in_bounds.any():
    prev_local[prev_in_bounds] = ms.index_maps[prev_ages[prev_in_bounds], prev_step]
```

The reviewer caught this before code was written. Test `test_age_zero_no_wrap_to_last_row` exercises a map whose last-row values are non-(-1), so the wrap would produce a detectable wrong answer.

### 4. The in-loop reviewer pattern catches real bugs

Six plans in this arc went through 2-3 reviewer iterations each. Real bugs caught (each would have been a parity regression or wasted PR if shipped):

- **A3 union-of-roles fix.** Iterating only the prey species would have left predator-only species (seals/cormorants on Baltic) with `pred_idx = -1` — a silent parity break.
- **A2 negative-index wrap.** As above.
- **A3 second consumer at predation.py:573.** The plan's scope said "fused method only used by mortality" — wrong; predation_for_cell has the same double-call.
- **P2 attribution.** The plan said "H7 in biomass_by_cell" caused the 0.242 s of `np.ufunc.at`; actual hot site was `_collect_spatial_outputs` per-species loop (and even that was gated off on the benchmark fixtures).
- **P4 ceiling overstatement.** Field-name caching alone can't recover `__post_init__` validation cost.
- **P4b scope gap.** Two more internal hot-path callers (reproduction.py:191, simulate.py:622) had the same construction pattern as the targeted hot paths.

The "two clean reviewer rounds" rule + rotating reviewer types (`feature-dev:code-reviewer` + `general-purpose` + `superpowers:code-reviewer`) caught everything before code was written.

### 5. The benchmark-measurement gotchas

Two real measurement bugs hit during the session, both saved as feedback memory:

- **PYTHONPATH for worktree benchmarks.** Running `scripts/benchmark_engine.py` against a worktree without `PYTHONPATH=<worktree>` silently imports master's code and produces a misleading measurement. A1's apparent 88 ms regression turned out to be PYTHONPATH-induced; correcting the import path showed a 17.7 % win. (`feedback_pythonpath_worktree_benchmark.md`)
- **Check call-path before perf gate.** P2's 3.1 % "improvement" was likely machine-state drift, not the change — `_collect_spatial_outputs` is gated on `output.spatial.enabled`, which is `false` on both eec_full and baltic. Always grep for config-flag gating before trusting a perf gate measurement. (`feedback_check_call_path_before_perf_gate.md`)

### 6. Where the perf surface ends

After the post-v0.12.0 plan + P4b dropped at the noise floor, the eec_full / baltic perf surface is genuinely closed. Remaining hot spots are:

| Surface | Status | Why no more wins |
|---|---|---|
| `mortality()` Python wrapper (1.0-1.2 s tottime) | Mostly JIT-call orchestration — the Numba kernel is the wall-clock cost | Numba kernel isn't visible to optimisation without rewriting in C/Rust |
| `np.add.at` calls (other 17 sites) | Below 2 % each | Each individual site is sub-2 % on the benchmark |
| `state.replace` machinery | Below 2 % even with `__post_init__` bypass | P4b confirmed |
| `compute_feeding_stages` | 0.14 % on eec_full | Already vectorised |

Future structural levers (none pursued in this session):

1. **`__slots__` on `SchoolState`** — eliminates per-field `__dict__` write, possibly 1-3 %. Needs invariant-coverage refactor.
2. **Mutable `SchoolState`** — remove the per-construction validation idiom entirely; switch to explicit invariant-check call sites. Largest design change, biggest potential gain.
3. **C/Rust extension for the predation kernel** — replace `_apply_predation_numba` with a hand-written native extension. Depends on whether further wall-time reduction matters more than the Numba parity-with-Java contract.

None of these are 2 %-via-single-edit wins. They need their own design work.

## Files of record

Plans (5):
- `docs/plans/2026-05-08-kernel-surgery-plan.md`
- `docs/plans/2026-05-08-python-side-perf-plan.md`
- `docs/plans/2026-05-08-post-v012-perf-plan.md`
- `docs/plans/2026-05-08-p4b-post-init-bypass-plan.md`

Profile + decision artefacts (5):
- `docs/perf/2026-05-08-eec_full-5yr-profile.md` (K4 gate)
- `docs/perf/2026-05-08-K2-not-shipping.md`
- `docs/perf/2026-05-08-K3-not-shipping.md`
- `docs/perf/2026-05-08-A3-not-shipping.md`
- `docs/perf/2026-05-08-A4-not-shipping.md`
- `docs/perf/2026-05-08-P4-not-shipping.md`
- `docs/perf/2026-05-08-P4b-not-shipping.md`

Shipped PRs (perf code):
- #29 — H7 vectorise biomass_by_cell DSVM accumulator
- #30 — perf bench infra (`--config` flag, fixture grid resolution)
- #31 — H6 partial drop redundant n_dead zeros allocation
- #35 — A1 vectorise compute_school_indices
- #36 — A2 vectorise _precompute_map_indices
- #40 — K1/P1 hoist predation scratch buffers
- #41 — P2 replace np.add.at with np.bincount in _collect_spatial_outputs

Post-mortems shipped (no code change):
- #37 — A3
- #42 — P3
- #43 — P4
- #44 — P4b

Release: `v0.12.0` tagged at master `31a2386`.
