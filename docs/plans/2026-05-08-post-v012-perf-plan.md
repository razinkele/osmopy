# Post-v0.12.0 Perf Plan — what's left after the 37.8 % cumulative win

> Created: 2026-05-08
> Branch: `docs/post-v012-perf-plan`
> Status: r3 (stale-references swept)
>
> **Revision log:**
> - 2026-05-08 r1 — initial draft.
> - 2026-05-08 r2 — applied review findings:
>   - **P2 attribution corrected.** The 0.242 s `numpy.ufunc.at` profiler row covers ~17 distinct `np.add.at` call sites across `simulate.py` and `output.py` — not just H7's DSVM call at `simulate.py:1236`. The DSVM site is gated on `ctx.fleet_state is not None`, which is **inactive on eec_full** (no economics module wired) — its contribution is zero on the benchmark. The dominant hot site is `simulate.py:871-873` (`_collect_spatial_outputs`, 3 calls per species per step = ~5 040 of the 6 453 hits on a 14-species 120-step run). P2's target moved to that site; np.bincount needs explicit flat-index linearisation for the 3D `(sp, ys, xs)` index.
>   - **P3 framing corrected.** `compute_feeding_stages` already uses `np.searchsorted` per species (`feeding_stage.py:88`); it is not a per-school Python loop analogous to pre-A1 `compute_school_indices`. Remaining overhead is the per-species mask + the pre-mask sort lookup. The "≥ 2 % profile-confirm" gate is preserved — likely drops without further work.
>   - **P4 ceiling corrected.** Field-name caching eliminates `fields()` overhead in the dict-building genexpr but **does not** eliminate `state.__post_init__`'s own `for f in fields(self)` validation loop (separate 0.046 s profiler entry). Recoverable ceiling drops from "2-3 %" to **~1.3-1.7 %** with caching alone. Full recovery (additional ~1.5 %) needs a `__post_init__` skip flag or `object.__setattr__` bypass on internal-construction paths — out of scope for P4 (tracked as P4b in the deferred section).
> - 2026-05-08 r3 — swept stale references the r2 patch missed: profile-table comment on `np.ufunc.at` (was still attributing to "H7's `np.add.at` in biomass_by_cell"), Sequencing-table P2 saving estimate (was still "6-7 %" instead of the conservative "3-6 %"), and dimension-count nits in the revision log + deferred-section label.
> Predecessors:
> - [`docs/perf/2026-05-08-eec_full-5yr-profile.md`](../perf/2026-05-08-eec_full-5yr-profile.md) — K4 profile gate (pre-A1/A2 baseline)
> - [`docs/plans/2026-05-08-kernel-surgery-plan.md`](2026-05-08-kernel-surgery-plan.md) — K1 conditional, K2/K3 dropped
> - [`docs/plans/2026-05-08-python-side-perf-plan.md`](2026-05-08-python-side-perf-plan.md) — A1+A2 shipped, A3 dropped
> - v0.12.0 release tag (commit `31a2386`)

## Where we are

After v0.12.0:
- **eec_full 5-yr median: 3.030 s** (was 4.872 s pre-A1/A2 = 37.8 % wall-time reduction)
- A1 and A2 collapsed the original cProfile top-3:
  - `compute_school_indices`: 1.269 s → **0.046 s** (96 % reduction)
  - `_precompute_map_indices`: 1.230 s → **0.068 s** (95 % reduction)
- 12/12 Java parity tests bit-exact; 2740 tests passing.

Two perf items from prior plans are still outstanding:
- **K1** (kernel-surgery): predation scratch buffer hoisting — conditional pursue per K4 gate, 3-12 % straddled the 2 % gate.
- **A4** (python-side perf): `compute_feeding_stages` per-cell call inside `_predation_in_cell_*` — flagged as a deferred candidate; no measured baseline.

This plan picks them both up against the **post-v0.12.0 baseline** (3.030 s), plus introduces one new item the fresh profile surfaced.

## Fresh profile (post-v0.12.0 master, `31a2386`)

cProfile against eec_full 5-yr warm cache. Total profiled run: 7.15 s (with cProfile overhead; uninstrumented runtime ~3.03 s). Top Python-side spots by tottime:

| tottime | calls | function | share of profiled run |
|--------:|------:|---|-----:|
| 1.199 s | 120 | `mortality.py:1665(mortality)` Python wrapper (most of cumtime is JIT-internal) | 16.8 % |
| 0.323 s | 120 | `movement.py:196(movement)` Python wrapper | 4.5 % |
| 0.242 s | 6 453 | `numpy.ufunc.at` (dominated by `_collect_spatial_outputs` per-species loop at `simulate.py:867-874`; see P2) | 3.4 % |
| 0.079 s | 14 | `movement_maps.py:93(__init__)` (one-shot at engine init) | 1.1 % |
| 0.078 s | 1 324 | `numpy.array` constructors | 1.1 % |
| 0.068 s | 119 | `_precompute_map_indices` (post-A2; was 1.230 s pre-A2) | 0.95 % |
| 0.060 s | 120 | `reproduction.py:62(reproduction)` | 0.84 % |
| 0.060 s | 120 | `simulate.py:703(_collect_distributions)` | 0.84 % |
| 0.046 s | 240 | `compute_school_indices` (post-A1; was 1.269 s pre-A1) | 0.64 % |
| 0.046 s | 5 122 | `state.py:83(__post_init__)` (frozen dataclass init) | 0.64 % |
| 0.034 s | 3 966 | `state.py:141(replace)` + cumulative dataclasses.fields machinery (~0.14 s with `getattr`/genexpr) | ~2 % |

Three new perf surfaces emerge:
1. **`mortality()` Python wrapper.** 1.199 s tottime is largely JIT-call-orchestration overhead (`larva_mortality`, 3× `state.replace`, `compute_feeding_stages`, output collection) — much of this is unavoidable, but specific Python-side hotspots inside it may be amenable to optimisation.
2. **`np.add.at`.** Known-slow NumPy primitive. H7 (PR #29) used it for the DSVM accumulator. Replacing with `np.bincount` / sort-then-reduce can be 5-10× faster.
3. **`state.replace` machinery.** ~140 ms cumulatively across 3 966 calls. Caching the field-name list or converting to `__slots__` could halve this.

## Items in scope

### P1 — K1 kernel-surgery re-evaluation (conditional pursue from K4)

**File.** `osmose/engine/processes/mortality.py:786+` (`_apply_predation_numba`), allocations at lines 846-848.

**Re-derive the gate.** K4's K1 estimate was 3-12 % of a 7.34 s baseline = 0.22-0.88 s. Post-v0.12.0 the baseline is 3.03 s and the **JIT-internal predation work is proportionally larger** (a higher fraction of total runtime). The K4 alloc-cost micro-bench gave 0.85 µs per Python-side alloc-triple; in-JIT it's likely 5-20× faster, so 50-200 ms over the run. Against the new 3.03 s baseline that's **1.6-6.6 %** — straddles the 2 % gate similarly to before but now closer to the upper end.

**Proceed criterion.** Re-run the K4 profile pattern (Python-side alloc micro-bench + cProfile against current master). If the post-A2 profile still shows kernel-internal allocation cost as ≥ 2 % of total, implement K1 with the K4 plan's design (hoist scratch buffers in `_apply_predation_numba`, pool via `SimulationContext`).

**Acceptance.**
1. eec_full 5-yr median improves by **≥ 2 %** vs post-v0.12.0 baseline.
2. 12/12 parity tests bit-exact.
3. New regression test pinning the pooled-scratch behaviour.
4. If measured < 2 %, drop with a `docs/perf/2026-05-08-K1-not-shipping.md`.

**Risk.** Medium. Numba kernel modification carries float-reduction reordering risk; the H9 cross-thread test has demonstrated the engine is currently deterministic across thread counts and K1 must preserve that.

### P2 — `np.add.at` replacement in `_collect_spatial_outputs`

**Files.** Primary: `osmose/engine/simulate.py:867-874` (`_collect_spatial_outputs`, 3 `np.add.at` calls inside a per-species loop). Secondary candidates by call frequency: `simulate.py:660-664` (per-species biomass aggregation in `_collect_outputs`), `simulate.py:606-607` (background biomass), `simulate.py:639-640` (per-species sums for averaging).

**Symptom.** cProfile shows `numpy.ufunc.at` at 0.242 s tottime / 6 453 calls. There are ~17 distinct `np.add.at` call sites across `simulate.py` and `output.py` (greppable list); on eec_full the dominant contributor is the per-species spatial-output loop at `simulate.py:867-874` — invoked once per timestep per species (120 × 14 = 1 680 species-step iterations × 3 metrics = ~5 040 calls). The H7 DSVM call at `simulate.py:1236` is gated on `ctx.fleet_state is not None` and is **inactive on eec_full** — it contributes zero to this row.

**Approach.** Replace each per-species 2D `np.add.at(out_3d[sp], (ys[m], xs[m]), values[m])` with a single vectorised `np.bincount` on a flat (sp, y, x) index:

```python
n_sp, ny, nx = sb.shape
flat_idx = sp_ids[focal] * (ny * nx) + ys * nx + xs
sb_flat = np.bincount(flat_idx, weights=biomass, minlength=n_sp * ny * nx)
sb = sb_flat.reshape(n_sp, ny, nx)
# repeat for sa, sy with the same flat_idx
```

This collapses the per-species Python loop into a single bincount per metric (3 calls instead of 3 × n_species = 42). Bit-exact for non-overlapping integer indices.

Alternative if the bincount allocation footprint is too large: per-species `np.bincount(ys * nx + xs, weights=values, minlength=ny*nx)` — keeps the per-species loop but eliminates the slow `np.add.at` per call. Roughly equivalent perf on eec_full (small grids), but worth trying both at impl time.

**Acceptance.**
1. eec_full 5-yr median improves by **≥ 2 %** vs post-v0.12.0 baseline.
2. 12/12 parity tests bit-exact (np.bincount is exact-equivalent for non-overlapping integer indices on float64 weights).
3. Existing spatial-output regression tests continue to pass.

**Estimated saving.** Hard to size from the profile row alone (the 6 453 calls are spread across 17 sites). The `simulate.py:867-874` site contributes ~5 040 / 6 453 = 78 % of `np.add.at` calls; if that fraction maps proportionally to time, fixing it saves ~0.19 s. Against 3.03 s baseline = **~6 %**. **Conservatism: estimate 3-6 % until measured**, gate at 2 %.

**Risk.** Low. `np.bincount` is bit-exact for integer indices and float64 weights. The flat-index linearisation is the only subtle bit — verify int dtype throughout to avoid silent overflow on a hypothetical 10 K × 10 K grid (eec_full is 22×22, no risk).

### P3 — A4: revisit `compute_feeding_stages` (likely drop after profile)

**File.** `osmose/engine/processes/feeding_stage.py:88` — call site at `mortality.py:1723` and `predation.py:565` (the latter is a `state.feeding_stage[:]` mutation path, not a fresh allocation).

**Reality check.** Contrary to the deferred-A4 framing in the python-side perf plan, `compute_feeding_stages` is **already vectorised** with `np.searchsorted` per species (similar template to post-A1 `compute_school_indices`). The function iterates ~15 species in Python with vectorised inner work. cProfile does not call it out by name in the post-v0.12.0 top-30, suggesting it's already below the 2 % gate.

**Proceed criterion.** Run `line_profiler` on `compute_feeding_stages` against eec_full 5-yr to attribute time. If self-time is ≥ 2 % of total run, identify which line dominates (probably the per-species mask `species_id == sp_idx` if anything) and address. If self-time is < 2 %, **drop with a `docs/perf/2026-05-08-A4-not-shipping.md`** — the function has already been vectorised in a prior shipped change and the remaining cost is below the gate.

**Acceptance** (only if proceed-criterion is met).
1. eec_full 5-yr median improves by ≥ 2 % once optimised.
2. 12/12 parity tests bit-exact.

**Estimated saving.** Likely 0-1 % (already vectorised). High likelihood of drop.

**Risk.** Low.

### P4 — `state.replace` field-list caching (borderline)

**File.** `osmose/engine/state.py:141`.

**Symptom.** `state.replace` calls `fields(self)` (dataclasses-machinery, 9 448 calls cumulative ~0.07 s) and then builds a dict via genexpr with `getattr` per field per call. Plus a separate **0.046 s** in `state.__post_init__` (line 83) — `__post_init__` runs its own `for f in fields(self)` validation loop on every constructed `SchoolState`.

**Approach (caching only).** Cache the field-name list at the SchoolState class level and use direct attribute access:
```python
_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(SchoolState))

def replace(self, **kwargs):
    values = {name: getattr(self, name) for name in self._FIELD_NAMES}
    values.update(kwargs)
    return SchoolState(**values)
```

This eliminates the `fields()` call + the genexpr's overhead but **still pays the `__post_init__` cost** because `SchoolState(**values)` triggers `__post_init__` on every replace.

**Recoverable ceiling.** `replace`-machinery contribution alone: ~40-60 ms = **1.3-2 %**. **Below the gate's measurement floor.** Full recovery (additional ~50 ms = 1.5 %) requires bypassing `__post_init__` via `object.__setattr__` on a pre-validated internal-construction path or a `__post_init__` skip-flag — explicitly out of scope for P4 (separate refactor with its own design + parity considerations).

**Acceptance** (only if pursued).
1. eec_full 5-yr median improves by ≥ 2 %.
2. 12/12 parity tests bit-exact.

**Recommendation.** **Skip in this plan.** P4's recoverable ceiling is at the gate's edge. Pursue only after P2 ships and the new profile shows the surface is still relevant. If pursued, a follow-on item to bypass `__post_init__` (call it P4b) can recover the remaining ~1.5 %.

**Risk.** Very low for the caching alone; medium for the `__post_init__` bypass (would need the validate-only-in-tests pattern that already exists for `validate()`).

## Sequencing

| Order | Item | Reasoning |
|---|---|---|
| 1 | **P2** (np.bincount) | Highest estimated saving (3-6 %), lowest risk, tightest scope. Ship first. |
| 2 | **P3** (A4 feeding_stage profile + vectorise if hot) | Independent of P2. Profile first, then decide. |
| 3 | **P1** (K1 re-evaluate + maybe ship) | Profile against post-P2/P3 baseline. Numba kernel touched — biggest review surface, ship last. |
| 4 | **P4** (state.replace cache) | Optional. Pursue only if cumulative budget hasn't hit a stop point. |

## Cumulative target

| Item | Floor | Ceiling |
|---|---|---|
| P1 (K1) | 0 % (drop) | 6 % |
| P2 (np.bincount) | 2 % | 6 % |
| P3 (A4) | 0 % (likely drop) | 1 % |
| P4 (state.replace) | 0 % (likely drop) | 2 % |
| **Total** | **2 %** | **15 %** |

Realistic mid-target: **3-8 % wall-time on top of v0.12.0** — bringing eec_full 5-yr to ~2.8-2.9 s. Most of the head-room is in P2; P3 will likely drop after the line_profiler check; P4 is at the gate's edge.

## Pre-requisites

### Baseline
```
.venv/bin/python scripts/benchmark_engine.py --config eec_full --years 5 --repeats 7
```
Post-v0.12.0 master median: **3.030 s** (606 ms/yr). 2 % noise floor: 60 ms.

### Measurement protocol
Same as A1/A2: 7-repeat median, discard run 1, side-by-side compare via `--compare baseline.json current.json`. **PYTHONPATH must point at the worktree** (per `feedback_pythonpath_worktree_benchmark.md`) — running from outside imports master's code instead and silently invalidates the measurement.

### Parity gate
```
.venv/bin/python -m pytest tests/test_engine_parity.py -q
```
12/12 must remain bit-exact. None of P1-P4 are pure RNG-stream changes (P1 may reorder reductions in JIT but should not change output via the H9 contract); fail the parity gate → debug or drop.

## What NOT to do

- **Do not Numba-JIT P3 or P4.** They're Python-orchestration overhead, not numerical inner-loops. Vectorisation collapses the interpreter cost without paying the JIT-cache invalidation tax.
- **Do not bundle K1 with P2/P3/P4 in one PR.** K1 carries a Numba parity contract; if it ships and a regression slips through, isolating the cause across multiple bundled items is harder than necessary.
- **Do not pre-commit to all four items.** Each has its own gate. If P2 ships at 6 % and P3 measures at 0.5 %, drop P3 without further work — chasing diminishing returns burns review cycles for no measurable gain.
- **Do not extend scope to non-engine paths.** UI / calibration / config-reader optimisations are out of scope — this plan is focused on the engine simulation hot path against eec_full 5-yr.

## Resolved decisions (formerly r1 open questions)

1. **P2 target site is `simulate.py:867-874` (`_collect_spatial_outputs`), not H7's DSVM call.** The DSVM call is gated on `ctx.fleet_state is not None` (inactive on eec_full); the spatial-output loop fires per species per step and dominates the `np.add.at` profiler row.
2. **`compute_feeding_stages` already uses np.searchsorted** (`feeding_stage.py:88`). P3's "vectorise" framing is wrong; the item is now a profile-then-likely-drop investigation, not a pre-committed perf change.
3. **Re-profile after each item's PR** before committing to the next — the post-v0.12.0 profile becomes stale once P2 collapses the spatial-output cost, so P3 / P4 decisions must use a fresh profile, not this plan's table.

## P4b — `__post_init__` bypass (deferred — not in this plan)

For reference if a P4b plan is later opened:

```python
class SchoolState:
    @classmethod
    def _internal(cls, **fields_dict):
        # Internal construction path that skips __post_init__ validation.
        # Use only when caller has already verified field shapes match.
        instance = cls.__new__(cls)
        for name, value in fields_dict.items():
            object.__setattr__(instance, name, value)
        return instance
```

Worth a separate plan with its own parity-test sweep — `__post_init__` enforces array-shape invariants that several test sites rely on, and a skip path needs to demonstrate every replace caller has pre-validated shapes.
