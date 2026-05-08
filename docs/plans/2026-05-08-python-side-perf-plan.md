# Python-side Perf Plan — vectorise the non-JIT'd hot paths

> Created: 2026-05-08
> Branch: `docs/python-side-perf-plan`
> Status: r3 (post-second-review)
> Predecessor: [`docs/perf/2026-05-08-eec_full-5yr-profile.md`](../perf/2026-05-08-eec_full-5yr-profile.md) — K4 profile gate
> Sibling: [`docs/plans/2026-05-08-kernel-surgery-plan.md`](2026-05-08-kernel-surgery-plan.md) — kernel-surgery plan (K1 still in-flight, K2/K3 dropped)
>
> **Revision log:**
> - 2026-05-08 r1 — initial draft.
> - 2026-05-08 r2 — applied review findings:
>   - C1: A3 fused method must iterate the **union** of prey + pred species (not only prey keys) so background-predator-only species (seals, cormorants on Baltic) still get a `pred_idx`.
>   - C2: A2 `prev_idx` block expanded — silent negative-index trap when `age_dt == 0` (prev_age = -1 would wrap-index `ms.index_maps`); explicit bounds-check required before indexing.
>   - I1: clarified A1 searchsorted-with-clamp correctness — clamp handles finite-last-threshold cases too; not conditional on `+inf` sentinel.
>   - I2: A3 saving estimate now states the dependency on prey/pred role overlap (50 % halving only when roles are fully symmetric).
>   - Q7: converted all three "open questions" to resolved decisions.
> - 2026-05-08 r3 — applied second-pass review findings:
>   - **R1:** A3 must also patch `predation.py:573-586` — same prey + pred double-call pattern. `predation_for_cell` is exposed for predation-isolated tests/benchmarks (not a hot production path), but A3 keeps the two call sites in sync to prevent drift.
>   - **R2:** Made the int-keyed `_stages_by_role` rebuild pseudocode explicit in A1: `prey_lookup` / `pred_lookup` are keyed by **CSV-label string**, the proposed `_stages_by_role[role][sp_idx]` is keyed by **int sp_idx**, so `from_csv` must call `resolve_name(species_names[sp_idx])` and only insert sp_idx entries whose name resolves (preserving the current `resolved.get(sp) is None: continue` skip).
>   - **R3:** Added construction-time assert that every species's `stages` list is non-empty (avoids the latent `matrix_indices[-1]` wrap-around if `len(thresholds) == 0` survives a malformed CSV).
>   - Added all-finite-thresholds species coverage to A1 acceptance test (a CSV without an open-ended adult label still returns `stages[-1]` correctly).

## What this plan covers

The K4 profile (warm-cache eec_full 5-year, 7.34 s) showed the largest wins on the engine hot path are *outside* the JIT'd Numba kernels — they're in Python orchestration around the kernels. Three items, all in the engine simulation loop:

- **A1 (accessibility)** — vectorise `StageAccessibility.compute_school_indices` and the underlying `get_index` lookup. **Profile share: 17.3 % (`compute_school_indices`) of which 6.7 % is inside `get_index`.**
- **A2 (movement)** — vectorise `_precompute_map_indices` per-school Python loop. **Profile share: 16.7 %.**
- **A3 (mortality wrapper)** — eliminate the redundant double-call to `compute_school_indices` in `mortality()` by computing prey + pred indices in a single pass. **Profile share: 17.6 % wrapper time, of which ~6 % is the duplicated index work.**

Each item is a pure functional refactor: outputs must be **bit-exact** versus the current implementation. None ship without (a) a measured ≥ 2 % wall-time improvement on the established eec_full 5-year benchmark, (b) bit-exact parity with the existing 12/12 Java parity tests, and (c) a regression test that pins the new behaviour.

## Why this plan exists separately from the K-plan

The kernel-surgery plan assumed the largest perf surface was inside the JIT'd kernels. K4 falsified that assumption — the Python orchestration above the kernels is 3-5× larger than any individual K-item and carries no Numba parity contract. Keeping the items in a separate plan:
- Lets K1 proceed independently (or be dropped) without entanglement.
- Avoids polluting the kernel-surgery plan's narrow scope.
- Gives this plan its own profile gate (already done — K4) rather than re-deriving one.

## Pre-requisites

### Baseline (already established by predecessor plan PR #30)

```
.venv/bin/python scripts/benchmark_engine.py --config eec_full --years 5 --repeats 7
```

Current master (post-#33) median: **5.025 s** (1.005 s/yr). 2 % noise floor: 100 ms.

### Measurement protocol (same for every A-item)

1. **Before the change, on the working branch:** capture a 7-repeat median (`--output baseline.json`).
2. **After the change:** repeat with `--output current.json`.
3. **Discard runs 1 of each set** to amortise Numba JIT compilation; the median over the remaining 6 is the comparison.
4. Use `scripts/benchmark_engine.py --compare baseline.json current.json` for a side-by-side report.
5. **Ship gate:** post-change median must be ≤ 0.98 × pre-change median (≥ 2 % faster) AND species-final-biomass exact parity (every species ratio == 1.0 in the compare output).

### Parity gate (same for every A-item)

```
.venv/bin/python -m pytest tests/test_engine_parity.py -q
```

12/12 must remain bit-exact. These items don't touch the predation/mortality JIT'd kernel directly, but they DO change the inputs the kernel sees (e.g., a different `prey_access_idx` ordering would feed different access values into the predation matrix lookup). Verify before assuming any of these items is "safe by inspection."

### Per-A parity contracts

| Item | Surface | Output type | Parity contract |
|---|---|---|---|
| A1 | `StageAccessibility.compute_school_indices` | `np.int32[n_schools]` | **Element-wise equality** with current implementation for the same inputs. The output is consumed as integer indices into a deterministic matrix — no float-reduction reordering risk. |
| A2 | `_precompute_map_indices` | `(np.int32[n], np.bool_[n])` tuple | **Element-wise equality** for both arrays. The current implementation is bounded-deterministic; vectorising must preserve the same out-of-bounds → -1 sentinel behaviour. |
| A3 | `mortality()` Python wrapper | (no new output — refactor of existing call site) | **No observable change**: same prey/pred index arrays consumed by the kernel, no RNG draws affected, no allocation order change visible to the kernel. |

## A1 — vectorise accessibility.compute_school_indices

**File.** `osmose/engine/accessibility.py:115-194`.

**Symptom.** Per-school Python loop over n_schools (~3 000 on eec_full):

```python
def compute_school_indices(self, species_id, age_dt, n_dt_per_year, all_species_names, role):
    n = len(species_id)
    indices = np.full(n, -1, dtype=np.int32)
    resolved: dict[int, str | None] = {}
    for sp_idx in range(len(all_species_names)):
        ...
    for i in range(n):              # ← 3 000 iterations × 240 calls/run × 5 yr
        sp = species_id[i]
        csv_name = resolved.get(sp)
        if csv_name is None:
            continue
        age_years = float(age_dt[i]) / n_dt_per_year
        indices[i] = self.get_index(csv_name, age_years, role)  # ← inner loop too
```

`get_index` itself does a Python loop over per-species "stages" (typically 2-4 stages: juvenile + sub-adult + adult thresholds), so the effective cost is **n_schools × avg_stages_per_species** Python ops per call. With 240 calls/run × 5 yr the aggregate is ~1.9 M `get_index` invocations and ~3.8 M `dict.get` calls (cProfile-confirmed).

**Approach.** Precompute per-species threshold + index arrays once at `StageAccessibility.from_*` construction time:

```python
@dataclass
class _PerSpeciesStages:
    thresholds: np.ndarray   # float64, sorted ascending. Last element = +inf if open-ended.
    matrix_indices: np.ndarray  # int32, same length

# Built at from_csv time. Keyed by INT species index, NOT csv-label string.
# self.prey_lookup / self.pred_lookup (existing) are string-keyed; the
# rebuild below translates each focal+background sp_idx to its csv name
# via self.resolve_name and inserts a stages entry only when the name
# resolves — preserving the current `resolved.get(sp) is None: continue`
# skip semantics.
self._stages_by_role: dict[str, dict[int, _PerSpeciesStages]] = {
    "prey": {}, "pred": {},
}
for sp_idx, name in enumerate(species_names):
    csv_name = self.resolve_name(name)
    if csv_name is None:
        continue  # sp_idx absent from both _stages_by_role[role] dicts → keeps -1
    for role, lookup in (("prey", self.prey_lookup), ("pred", self.pred_lookup)):
        stages = lookup.get(csv_name)
        if stages is None or len(stages) == 0:
            continue  # role-asymmetric species — handled by per-role .get(sp_idx)
        thresholds = np.array([s.threshold for s in stages], dtype=np.float64)
        matrix_indices = np.array([s.matrix_index for s in stages], dtype=np.int32)
        self._stages_by_role[role][sp_idx] = _PerSpeciesStages(
            thresholds=thresholds, matrix_indices=matrix_indices,
        )
        # Construction-time guard: assert non-empty (already checked above
        # via len(stages) == 0; this keeps the invariant explicit for the
        # later searchsorted+clamp at compute_school_indices time).
        assert len(thresholds) >= 1, f"{role} stages for {name!r} must be non-empty"
```

Then `compute_school_indices` becomes:

```python
def compute_school_indices(self, species_id, age_dt, n_dt_per_year, ...):
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    indices = np.full(species_id.shape, -1, dtype=np.int32)
    for sp_idx, stages in self._stages_by_role[role].items():
        mask = species_id == sp_idx
        if not mask.any():
            continue
        # Stage = first index where age < threshold; falls back to last stage.
        bin_idx = np.searchsorted(stages.thresholds, age_years[mask], side="right")
        # Clamp to last index (the open-ended "adult" stage).
        bin_idx = np.minimum(bin_idx, len(stages.thresholds) - 1)
        indices[mask] = stages.matrix_indices[bin_idx]
    return indices
```

**Subtleties.**
- Current `get_index` returns `-1` for unknown species names. The vectorised path inherits this — species without an entry in `_stages_by_role[role]` keep their `-1` initialisation.
- `n_dt_per_year` was an int divisor; ensure age conversion produces the same float values (NumPy's `int32 / int` → float64 should match `float(int) / int`).
- The threshold semantics in current code: `if age_years < stage.threshold: return stage.matrix_index`, with iteration over sorted-ascending stages, falling back to the last stage. `np.searchsorted(thresholds, age, side="right")` gives the index of the first stage where `age < threshold`; the subsequent `np.minimum(bin_idx, len(thresholds) - 1)` clamp reproduces the current loop's "fallback to last stage" behaviour. **The clamp is sufficient even if the last threshold is finite** — `searchsorted` returns `len(thresholds)` only when `age >= all_thresholds`, and the clamp drops it back to `len - 1`, matching the current loop's `return stages[-1]`. The `+inf` sentinel from `_parse_label` (for un-thresholded "adult" labels) just guarantees `searchsorted` never returns `len(thresholds)` in practice; correctness does not depend on it. Add a boundary-test sweep at `age == threshold ± 1e-12` (current code: `age_years < stage.threshold` is False at equality → falls through to next stage; `searchsorted(..., age, side="right")` at `age == threshold` returns the index past the equal element — same answer).
- Per-role precomputation: `_stages_by_role: dict[str, dict[int, _PerSpeciesStages]]` keyed by role first to avoid per-call branching. Build at `from_*` time.

**Acceptance.**
1. `tests/test_engine_accessibility.py::test_compute_school_indices_vectorised_matches_loop` — randomised inputs (multiple species, multiple ages including boundary == threshold), assert element-wise equality with the current loop implementation. **Do not delete the loop implementation** — keep as `_compute_school_indices_loop` with a `@staticmethod` decorator, used only by the test for cross-check.
2. eec_full 5-year benchmark median improves by ≥ 2 % vs post-#33 master.
3. `tests/test_engine_parity.py` 12/12 remain bit-exact.
4. Sweep test: every species + every age stage at `age == threshold ± 1e-12` returns the same matrix index as the loop implementation.
5. **Edge-case test: all-finite-thresholds species** — a CSV containing only finite-threshold labels for a species (no open-ended `+inf` "adult" label, e.g. `cod < 0.4` and `cod < 1.2` only). Verify the searchsorted + clamp returns `stages[-1].matrix_index` for ages `>= 1.2`, matching the current loop's `return stages[-1].matrix_index` fallback.
6. **Edge-case test: species with no `resolve_name` mapping** — a species in `all_species_names` whose name does not appear in the accessibility CSV. Schools of that species must keep the `-1` sentinel (matching the current `resolved.get(sp) is None: continue` skip).

**Estimated saving.** 4-7 % of total run (full `get_index` at 6.7 % + portion of the wrapper-loop overhead). Conservatively ≥ 2 %.

**Risk.** Low. The output is element-wise equality, not a float reduction.

## A2 — vectorise _precompute_map_indices

**File.** `osmose/engine/processes/movement.py:492-524`.

**Symptom.** Per-school Python loop over n_map_schools (~3 000 on eec_full) with a dict lookup + 2D array index:

```python
for k, i in enumerate(map_school_mask):
    sp = int(species_id[i])
    age = int(age_dt[i])
    if sp not in map_sets:
        continue
    ms = map_sets[sp]
    if 0 <= age < ms.index_maps.shape[0] and 0 <= step < ms.index_maps.shape[1]:
        current_idx[k] = ms.index_maps[age, step]
    ...
```

Aggregate workload: ~3 000 × 119 calls/run × 5 yr ≈ 1.8 M iterations. cProfile shows this is the single largest Python-side hot spot (16.7 % of total).

**Approach.** Group by species (similar to A1):

```python
def _precompute_map_indices(species_id, age_dt, uses_maps, map_sets, step):
    map_school_mask = np.where(uses_maps)[0]
    n = len(map_school_mask)
    current_idx = np.full(n, -1, dtype=np.int32)
    prev_idx = np.full(n, -1, dtype=np.int32)
    sp_at_mask = species_id[map_school_mask]  # int32[n]
    age_at_mask = age_dt[map_school_mask]      # int32[n]

    step_in_range = 0 <= step < ...  # bound per-species
    prev_step = step - 1

    for sp, ms in map_sets.items():
        sel = (sp_at_mask == sp)
        if not sel.any():
            continue
        ages = age_at_mask[sel]
        prev_ages = ages - 1  # int32; may be -1 for age==0
        n_ages, n_steps = ms.index_maps.shape
        sel_idx = np.where(sel)[0]

        # current_idx: bound (age, step) and scatter -1 for out-of-bounds.
        # CRITICAL: pre-mask BEFORE indexing ms.index_maps. NumPy treats negative
        # indices as wrap-around (age = -1 picks the last row), which would
        # silently produce a wrong lookup rather than the -1 sentinel.
        cur_in_bounds = (ages >= 0) & (ages < n_ages) & (0 <= step) & (step < n_steps)
        if cur_in_bounds.any():
            valid_ages = ages[cur_in_bounds]
            current_local = np.full(ages.shape, -1, dtype=np.int32)
            current_local[cur_in_bounds] = ms.index_maps[valid_ages, step]
            current_idx[sel_idx] = current_local

        # prev_idx: SAME pre-mask logic, applied to prev_ages and prev_step.
        # Equivalent to the current loop's `0 <= prev_age < ms.index_maps.shape[0]`
        # check before `ms.index_maps[prev_age, prev_step]`.
        prev_in_bounds = (
            (prev_ages >= 0) & (prev_ages < n_ages)
            & (0 <= prev_step) & (prev_step < n_steps)
        )
        if prev_in_bounds.any():
            valid_prev_ages = prev_ages[prev_in_bounds]
            prev_local = np.full(ages.shape, -1, dtype=np.int32)
            prev_local[prev_in_bounds] = ms.index_maps[valid_prev_ages, prev_step]
            prev_idx[sel_idx] = prev_local

    same_map = (current_idx == prev_idx) & (age_dt[map_school_mask] > 0) & (step > 0)
    return current_idx, same_map
```

**Subtleties.**
- The current code's `0 <= step < ms.index_maps.shape[1]` check varies per species (different `ms.index_maps` shapes). The vectorised path checks per species in the per-species loop — same semantics.
- `0 <= age < ms.index_maps.shape[0]` and the same for `prev_age = age - 1` must preserve `-1` sentinel for out-of-bounds, NOT a clamped value. **Critical correctness trap:** NumPy treats negative integer indices as wrap-around (`ms.index_maps[-1, step]` returns the last row's value at `step`, not an error). For schools with `age_dt == 0`, `prev_age = -1`, so a naive `ms.index_maps[prev_ages, prev_step]` without pre-masking would silently produce a wrong lookup. The pseudocode above pre-masks via `prev_in_bounds = (prev_ages >= 0) & ...` and indexes only the in-bounds subset (`ms.index_maps[valid_prev_ages, prev_step]`). Match the current loop's defensive `0 <= prev_age` check exactly.
- `species_id` and `age_dt` are int32 NumPy arrays already; no per-iter `int(...)` cast needed.
- Schools whose `sp` is not in `map_sets` keep the `-1` sentinel (initialised that way).

**Acceptance.**
1. `tests/test_engine_movement.py::test_precompute_map_indices_vectorised_matches_loop` — randomised inputs across multiple species + age ranges + steps including out-of-bounds (age < 0, step ≥ shape[1], unknown species), assert element-wise equality with the current loop. **Must include `age_dt == 0` schools** to exercise the prev_age = -1 path that the wrap-index trap would silently corrupt.
2. eec_full 5-year benchmark median improves by ≥ 2 % vs post-A1 master (or post-#33 if A2 ships first).
3. `tests/test_engine_parity.py` 12/12 remain bit-exact.

**Estimated saving.** 8-12 % of total run. The current 16.7 % includes the per-iter `int()` casts, the `if sp not in map_sets` Python check, and the dict lookup; vectorising should reclaim most of it.

**Risk.** Low-medium. The per-species 2D indexing requires careful handling of out-of-bounds — keep the loop implementation around for the parity test, as in A1.

## A3 — single-pass prey + pred index computation

**Files.** Two call sites both make the same prey + pred double-call:
- `osmose/engine/processes/mortality.py:1726-1742` — **production hot path** (called once per timestep, ~120 calls per 5-year run on eec_full; 17.6 % wrapper time per the K4 profile).
- `osmose/engine/processes/predation.py:573-586` — **test/benchmark entry point** (`predation_for_cell` is exposed for predation-isolated tests/benchmarks per the module docstring at predation.py:4-19; not in the production simulation path, but kept in sync to prevent drift between the two patterns).

**Symptom.** Both call sites call `compute_school_indices` twice in succession — once for prey, once for pred. Each call iterates over all schools. The per-school work for prey vs pred differs only in which lookup table (`prey_lookup` vs `pred_lookup`) is consulted.

```python
prey_access_idx = sa.compute_school_indices(..., role="prey")
pred_access_idx = sa.compute_school_indices(..., role="pred")
```

After A1 lands, this is two vectorised passes. Eliminating the second pass means walking schools once and doing both lookups inline.

**Approach.** Add a fused method:

```python
def compute_school_indices_both(
    self, species_id, age_dt, n_dt_per_year, all_species_names
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Return (prey_indices, pred_indices) in a single sweep over schools."""
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    prey_idx = np.full(species_id.shape, -1, dtype=np.int32)
    pred_idx = np.full(species_id.shape, -1, dtype=np.int32)

    # Iterate the UNION of prey + pred species. Background-predator-only
    # species (e.g., seals/cormorants on Baltic) appear ONLY in pred_lookup;
    # iterating prey keys alone would leave their pred_idx at -1 — a silent
    # parity bug versus the two-call implementation.
    all_sp = set(self._stages_by_role["prey"]) | set(self._stages_by_role["pred"])
    for sp_idx in all_sp:
        mask = species_id == sp_idx
        if not mask.any():
            continue
        ages_sub = age_years[mask]
        prey_stages = self._stages_by_role["prey"].get(sp_idx)
        pred_stages = self._stages_by_role["pred"].get(sp_idx)
        if prey_stages is not None:
            bin_idx = np.searchsorted(prey_stages.thresholds, ages_sub, side="right")
            bin_idx = np.minimum(bin_idx, len(prey_stages.thresholds) - 1)
            prey_idx[mask] = prey_stages.matrix_indices[bin_idx]
        if pred_stages is not None:
            bin_idx = np.searchsorted(pred_stages.thresholds, ages_sub, side="right")
            bin_idx = np.minimum(bin_idx, len(pred_stages.thresholds) - 1)
            pred_idx[mask] = pred_stages.matrix_indices[bin_idx]
    return prey_idx, pred_idx
```

The saving comes from sharing the per-species mask + age-conversion work between roles. The fraction of work saved is proportional to **prey/pred role overlap**: on EEC (where every focal species appears in both roles) the savings approach 50 % of the mask + searchsorted cost; on Baltic with predator-only seals + cormorants the overlap is partial and savings are smaller. The gate measurement on eec_full will expose the realised number — see I2 in the revision log.

**Subtleties.**
- A species may appear in `prey_lookup` but not `pred_lookup` (or vice versa). The union-of-roles iteration above covers both cases. Per-role `is not None` checks handle the asymmetry. Schools whose species lacks a role-specific entry retain the `-1` sentinel.
- The fused method must be added to the same class as A1 (StageAccessibility), and **both** `mortality()` and `predation_for_cell` updated to call it. Keep the single-role `compute_school_indices` method as the public single-role API — other call sites may use it, and the parity test can cross-check.
- The `predation_for_cell` patch is purely scope-keeping (call site is not in the production hot path). Production saving comes entirely from the `mortality()` patch. The gate measurement on eec_full reflects the production saving only.

**Acceptance.**
1. `tests/test_engine_accessibility.py::test_compute_school_indices_both_matches_separate` — assert that for the same inputs `(prey, pred) = compute_school_indices_both(...)` equals `(compute_school_indices(role="prey"), compute_school_indices(role="pred"))` element-wise. **Must include a species in `pred_lookup` but not `prey_lookup`** (a background-predator) to exercise the union-of-roles path that a prey-only iteration would silently break.
2. eec_full 5-year benchmark median improves by ≥ 2 % vs post-A2 master.
3. `tests/test_engine_parity.py` 12/12 remain bit-exact.
4. **Baltic parity check.** Run `.venv/bin/python -m pytest tests/test_engine_baltic.py -q` (if such a test exists) or a one-shot baltic 1-year smoke test, and verify the predation outputs for sp14/sp15 (seal, cormorant) are unchanged. The Baltic scenario is the only configuration in CI that exercises predator-only species and would catch a regression in the union-of-roles fix.

**Estimated saving.** 1-4 % of total run. Realised saving is proportional to prey/pred role overlap: on EEC (full overlap) the upper end is plausible; on scenarios with significant role asymmetry (Baltic with predator-only species) the lower end. **Gate is ≥ 2 % on eec_full**, not Baltic, so saving asymmetry is not a ship blocker.

**Risk.** Low. Equivalent computation, just refactored. The C1 union-of-roles fix is the single subtle correctness point and is now explicit in the pseudocode, the subtleties section, and the acceptance test list.

## Sequencing

| Order | Item | Reasoning |
|---|---|---|
| 1 | A1 | Foundation: A2 and A3 are independent of A1, but A3 directly extends A1's vectorisation. A1 first reduces dependency cycles. |
| 2 | A2 | Independent of A1. Can run in parallel with A1 if a different worktree wants to. |
| 3 | A3 | Depends on A1's vectorisation pattern. Must come after A1 lands. |

If A1 misses its gate, A3 is still meaningful (single-pass is faster even on the loop implementation), but the estimated saving drops to ~1-2 %.

## Cumulative target

| Item | Floor | Ceiling |
|---|---|---|
| A1 | 2 % | 7 % |
| A2 | 2 % | 12 % |
| A3 | 1 % | 4 % |
| **Total** | **5 %** | **23 %** |

A3's floor drops to 1 % after the union-of-roles correctness fix — the saving is now bounded by role-overlap (full overlap on EEC, partial on Baltic). If A3 measures < 2 % on eec_full it ships only as a documentation/clarity improvement (no regression).

The realistic combined target is **8-12 % wall-time improvement on eec_full 5-year**. This is materially larger than the K-plan's combined ceiling (K1 alone at 3-12 %, K2/K3 dropped).

## Risks (cross-cutting)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Vectorised path produces different float intermediate (e.g., `int32 / int` vs `float(int) / int`) → different `searchsorted` bucket | Low | Add boundary-test coverage in A1 acceptance #1; use `np.float64` divisor consistently. |
| Numba JIT downstream of the Python-side work depends on input array dtype/contiguity | Low | All current consumers expect `np.int32` C-contiguous arrays; vectorised path produces those by construction (NumPy default). |
| Hidden side effect in current loop (e.g., a per-iter mutation of `resolved` dict) | Very low | Code review of A1 / A2 confirms no side effects; loop implementation only writes to the result array. |
| Per-species mask cost dominates with very few species per call | Medium | On eec_full there are ~16 focal + ~14 background species; the per-species loop is small. On scenarios with 100+ species the linear-in-species cost could swamp savings — monitor with a baltic benchmark too. |

## What NOT to do in this plan

- Do not Numba-JIT these functions. cProfile shows they're large in the *Python interpretation* layer, not in numerical work; vectorising in NumPy already collapses the interpreter overhead. JIT'ing them would re-introduce the kernel-parity contract that K2/K3 carried.
- Do not pool or cache outputs across calls. The arrays change every step (school positions, ages, species composition all shift). Per-call reconstruction is correct.
- Do not extend scope to other Python-side hot spots (`reproduction.py`, `simulate.py:_collect_distributions`) — they're each under 1 % of run time and below the gate.

## Resolved decisions (formerly r1 open questions)

1. **Keep the single-role `compute_school_indices` method.** It remains the public single-role API; the fused method is additive (used by `mortality()` in the production hot path and by `predation_for_cell` to keep the test/benchmark entry point in sync). This avoids tying single-role consumers to the fused implementation's role-union iteration cost when they only need one role.
2. **Cache the per-species precomputed stages on `StageAccessibility` itself, at `from_*` construction time.** The lookup tables are immutable for the lifetime of an `EngineConfig`, so per-call rebuild would be wasted work.
3. **Do not add an `unique(species_id)` pre-filter to A2.** The `mask.any()` check is O(n) and runs once per species; `unique` itself is O(n log n). On scenarios with many no-op species per call the constant-factor wins are negligible compared to A2's main vectorisation gain.

## Deferred (post-A3)

- **A4 candidate:** `compute_feeding_stages` per-cell call inside `_predation_in_cell_python`/`_predation_in_cell_numba` dispatch context (predation.py:565). cProfile shows feeding-stage work below the 2 % gate on eec_full, so deferred until profiling on a larger fixture or a sensitivity scenario re-prioritises it.
