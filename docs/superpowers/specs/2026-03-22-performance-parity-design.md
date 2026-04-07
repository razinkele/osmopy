# Performance Parity: Batch Cell Loop + Parallel Mortality

**Date:** 2026-03-22
**Status:** Draft
**Goal:** Close the 5x performance gap between Python and Java engines on Bay of Biscay 5yr (11.5s → ~2s, Java reference 2.3s)

## Context

After Tier 1-3 optimizations (cached inst_abd, Numba predation, full Numba cell loop), the Python engine achieves Java parity on BoB 1yr (0.79s vs 0.80s) but remains 5x slower on 5yr (11.5s vs 2.3s).

The scaling difference (Python 14.5x from 1yr→5yr vs Java 2.9x) is caused by per-cell Python overhead that scales linearly with school count growth:
- 120k cell visits per 5yr (250 cells × 120 steps × 4 sub-timesteps)
- Each cell visit: 4× `rng.permutation()`, n_local× `rng.shuffle()`, array allocation, Numba dispatch
- The Numba-compiled inner loop is fast; the Python orchestration around it dominates

## Approach

Two phases, each independently shippable:

- **Phase A (Batch Cell Loop):** Move the `for cell in range(n_cells)` loop into a single Numba function. Pre-generate all RNG data for all cells in one Python pass, then make one Numba call per sub-timestep instead of 250.
- **Phase B (Parallel Cells):** Replace `range(n_cells)` with `prange(n_cells)` for multi-core execution. Safe because all school-level mutations are cell-local.

Statistical equivalence is accepted (same distribution of outcomes, not bit-identical to current code).

## Phase A: Batch Cell Loop

### RNG Pre-generation

Currently, each cell visit generates random data inline:
```python
seq_pred = rng.permutation(n_local)   # 4 calls per cell
...
for i in range(n_local):
    rng.shuffle(causes)               # n_local calls per cell
```

New approach: pre-generate ALL random data for all cells in one Python pass before the Numba call.

**Storage format — two flat buffers indexed by cell boundaries:**

```python
# seq_bufs[4][total_valid_schools] — the 4 shuffled sequences
#   seq_bufs[k][start:end] contains rng.permutation(n_local) for cell k
#   Values are in [0, n_local) — local indices into cell_indices
# cause_orders_buf[total_valid_schools, 4] — cause dispatch order per school slot
#   cause_orders_buf[start+i] contains a shuffled [0,1,2,3] for slot i of that cell
```

The existing `boundaries` array (from `searchsorted`) maps cell index → school range. Each cell's random data is at `buf[boundaries[cell]:boundaries[cell+1]]`.

**Pre-generation function:**
```python
def _pre_generate_cell_rng(rng, boundaries, n_cells):
    total = boundaries[n_cells]  # total valid schools
    seq_bufs = [np.empty(total, dtype=np.int32) for _ in range(4)]
    cause_orders_buf = np.empty((total, 4), dtype=np.int32)
    causes = [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING]

    for cell in range(n_cells):
        start = boundaries[cell]
        end = boundaries[cell + 1]
        n_local = end - start
        if n_local == 0:
            continue
        for k in range(4):
            seq_bufs[k][start:end] = rng.permutation(n_local).astype(np.int32)
        for i in range(n_local):
            rng.shuffle(causes)
            cause_orders_buf[start + i, 0] = causes[0]
            cause_orders_buf[start + i, 1] = causes[1]
            cause_orders_buf[start + i, 2] = causes[2]
            cause_orders_buf[start + i, 3] = causes[3]

    return seq_bufs, cause_orders_buf
```

Note: `causes` is a Python list reused across all cells. `rng.shuffle(causes)` shuffles in-place, so each iteration starts from the previous shuffle's result — this matches the current per-cell code's behavior.

Cost: <2ms per sub-timestep for 250 cells. Negligible.

### Numba Batch Function

Replace the per-cell Python dispatch loop with a single Numba function:

```python
@njit(cache=True)
def _mortality_all_cells_numba(
    # Cell structure (cell_id = cell index, row-major flat)
    sorted_indices, boundaries, n_cells,
    # Pre-generated RNG buffers
    seq_pred_buf, seq_starv_buf, seq_fish_buf, seq_nat_buf,
    cause_orders_buf,
    # School state arrays (unchanged from Tier 3)
    inst_abd, n_dead, species_id, length, weight, age_dt,
    first_feeding_age_dt, feeding_stage, pred_success_rate,
    preyed_biomass, trophic_level,
    # Pre-computed rates (unchanged from Tier 3)
    eff_starv, eff_additional, eff_fishing, fishing_discard,
    # Predation config (unchanged)
    size_ratio_min, size_ratio_max, ingestion_rate,
    n_dt_per_year, n_subdt,
    access_matrix, has_access, use_stage_access,
    prey_access_idx, pred_access_idx,
    # Resources (unchanged)
    rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl,
    rsc_access_rows, n_resources, n_species,
    # Tracking (passed as explicit arrays, NOT module globals)
    tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled,
):
    for cell in range(n_cells):
        start = boundaries[cell]
        end = boundaries[cell + 1]
        if end <= start:
            continue

        cell_indices = sorted_indices[start:end]
        n_local = end - start
        cell_id = cell  # flat row-major index, same as rsc_biomass column indexing

        # Index into pre-generated RNG buffers
        seq_pred = seq_pred_buf[start:end]
        seq_starv = seq_starv_buf[start:end]
        seq_fish = seq_fish_buf[start:end]
        seq_nat = seq_nat_buf[start:end]
        cause_orders = cause_orders_buf[start:end]

        # Inline interleaved mortality (same logic as _mortality_in_cell_numba)
        for i in range(n_local):
            for c in range(4):
                cause = cause_orders[i, c]
                if cause == 0:  # PREDATION
                    p_idx = cell_indices[seq_pred[i]]
                    _apply_predation_numba(
                        p_idx, cell_indices, inst_abd, n_dead,
                        ... # same params as current
                    )
                elif cause == 1:  # STARVATION
                    idx = cell_indices[seq_starv[i]]
                    D = eff_starv[idx]
                    if D > 0:
                        abd = inst_abd[idx]
                        if abd > 0:
                            dead = abd * (1.0 - np.exp(-D))
                            n_dead[idx, 1] += dead
                            inst_abd[idx] -= dead
                elif cause == 2:  # ADDITIONAL
                    idx = cell_indices[seq_nat[i]]
                    # ... same pattern
                elif cause == 3:  # FISHING
                    idx = cell_indices[seq_fish[i]]
                    # ... same pattern with discard split
```

The inner logic is identical to the existing `_mortality_in_cell_numba`. The only change is that the cell loop is now inside Numba, and RNG data comes from pre-generated buffers instead of inline generation.

### Important: inst_abd Lifecycle

`inst_abd` is initialized **once** from `work_state.abundance.copy()` before the sub-timestep loop and is **never reset** between sub-timesteps. It serves as the running availability tracker: each mortality event decrements it in-place, and these decrements accumulate across all sub-timesteps and all cells. The batch Numba function receives `inst_abd` by reference and modifies it in-place — the same array is passed to every sub-timestep iteration.

### Important: Grid Assumption

`cell_id = cell` assumes that `boundaries` uses the same row-major grid numbering as `rsc_biomass[r, cell_id]`. This holds because both derive from `cell_y * grid.nx + cell_x` using the same focal grid, and the OSMOSE model requires resources to share the focal species grid.

### Integration into mortality()

```python
def mortality(state, resources, config, rng, grid, step=0, ...):
    # ... existing pre-computation (Tier 1-3, unchanged) ...

    # inst_abd initialized ONCE before the loop, persists across sub-timesteps
    inst_abd = work_state.abundance.copy()

    for _sub in range(n_subdt):
        # Egg release (unchanged)
        ...

        if _HAS_NUMBA and len(valid_indices) > 0:
            # Pre-generate RNG for all cells (Python)
            seq_bufs, cause_orders_buf = _pre_generate_cell_rng(
                rng, boundaries, n_cells
            )

            # Extract tracking arrays from module globals BEFORE Numba call
            tl_ws = _tl_weighted_sum if _tl_weighted_sum is not None else _DUMMY_RSC_1D
            tl_track = _tl_weighted_sum is not None
            d_mat = _diet_matrix if _diet_tracking_enabled and _diet_matrix is not None else _DUMMY_DIET
            d_en = _diet_tracking_enabled and _diet_matrix is not None

            # Single Numba call for all cells
            _mortality_all_cells_numba(
                sorted_indices, boundaries, n_cells,
                seq_bufs[0], seq_bufs[1], seq_bufs[2], seq_bufs[3],
                cause_orders_buf,
                inst_abd, work_state.n_dead, ...
                tl_ws, tl_track, d_mat, d_en,
            )
        else:
            # Existing per-cell Python fallback
            for cell in range(n_cells):
                ...
```

### Important: Module Globals Must Be Extracted in Python

`_tl_weighted_sum` and `_diet_matrix` are module-level globals in `mortality.py` and `predation.py` respectively. They must be read and resolved to concrete array references in the Python orchestration layer (before the Numba call), then passed as explicit parameters. The Numba function must **never** reference these globals directly — only through its parameter list. This is already the pattern used by `_mortality_in_cell_numba` in Tier 3.

### Expected Performance

- Eliminates 250 Python→Numba dispatches per sub-timestep → 1 dispatch
- Eliminates per-cell array allocation overhead
- RNG pre-generation adds ~2ms per sub-timestep (negligible)
- **BoB 5yr: 11.5s → ~3-4s** (estimated 3-4x speedup)
- BoB 1yr improvement is uncertain — 1yr already has fewer schools per cell, so the dispatch overhead is a smaller fraction of total time. Expect modest improvement (~0.5-0.7s).

## Phase B: Parallel Cells

### Safety Analysis

All school-level mutations during the interleaved mortality loop are cell-local. The key invariant: **schools belong to exactly one cell** (via `sorted_indices[start:end]`). Schools don't move during mortality. Each cell's index range `[start, end)` is disjoint from every other cell's range.

| Array | Mutated by | Cross-cell? | prange-safe? |
|-------|-----------|-------------|-------------|
| `inst_abd[idx]` | All causes | No — idx from cell_indices, disjoint per cell | Yes |
| `n_dead[idx, cause]` | All causes | No — same indexing | Yes |
| `pred_success_rate[idx]` | Predation | No | Yes |
| `preyed_biomass[idx]` | Predation | No | Yes |
| `tl_weighted_sum[p_idx]` | Predation | No — p_idx is a school index, schools are cell-local | Yes |
| `diet_matrix[p_idx, sp]` | Predation | No — p_idx is a school index, rows are cell-local | Yes |
| `rsc_biomass[r, cell_id]` | Predation | No — cell_id is unique per loop iteration | Yes |

### Implementation

Requires adding `prange` to the Numba import:

```python
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
```

The parallel function:

```python
@njit(cache=True, parallel=True)
def _mortality_all_cells_parallel(
    ... # same signature as _mortality_all_cells_numba
):
    for cell in prange(n_cells):   # only change: range → prange
        ... # identical inner logic
```

### Parallel Flag

The parallel/sequential choice is controlled by a `parallel` argument on `mortality()`, defaulting to `True` when Numba is available. This avoids modifying `EngineConfig` for an implementation detail.

```python
def mortality(state, resources, config, rng, grid, step=0,
              species_rngs=None, parallel=True):
    ...
    use_parallel = _HAS_NUMBA and parallel
    if use_parallel:
        _mortality_all_cells_parallel(...)
    else:
        _mortality_all_cells_numba(...)  # sequential Numba (Phase A)
```

### Expected Performance

- 2-3x on top of Phase A on a 4-core machine
- **BoB 5yr: ~3-4s → ~1.5-2s** (Java parity at 2.3s)
- Diminishing returns beyond 4-8 cores (cell count limits parallelism)

## Testing & Validation

### Determinism (Exact Match)

Within a code version, same seed must produce bit-identical output:
- Phase A: deterministic because RNG is pre-generated sequentially, Numba loop is deterministic
- Phase B: deterministic because each cell writes to disjoint array slices — `prange` iteration order doesn't affect results (no cross-cell accumulation into shared locations)

Existing `TestDeterminism::test_same_seed_same_output` validates this. **After Phase A lands, the exact-match parity baseline (`parity_baseline_bob_1yr_seed42.npz`) must be regenerated** via `scripts/save_parity_baseline.py`, since Phase A changes the RNG consumption order. Phase B does not change results relative to Phase A, so no further regeneration is needed.

### Cross-Version Parity (Statistical)

Phase A changes RNG consumption order (all permutations generated before any cell processes). Results differ from Tier 3 at the FP level but are statistically equivalent.

New test: `TestStatisticalParity`
```python
def test_multi_seed_biomass_within_tolerance(self):
    """10 seeds, mean final biomass per species within 5% of baseline."""
    seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
    baseline_means = _load_multi_seed_baseline()
    current_means = np.mean([_run_engine(1, s)[0][-1] for s in seeds], axis=0)
    np.testing.assert_allclose(current_means, baseline_means, rtol=0.05)
```

### Baseline Management

Statistical parity baselines are stored alongside existing baselines in `tests/baselines/`:

- **File:** `statistical_baseline_bob_1yr_10seeds.npz`
- **Contents:** `mean_biomass[n_species]`, `std_biomass[n_species]`, `seeds[10]`
- **Generated by:** `scripts/save_parity_baseline.py --statistical --seeds 10`
- **When to regenerate:** After Phase A is complete (before Phase B), since Phase A changes the RNG consumption order. Phase B should not change results.
- **Committed to repo:** Yes, alongside existing `parity_baseline_bob_1yr_seed42.npz`

### Sanity Checks

Existing tests (non-negative biomass, species survival, mortality non-negative) apply unchanged.

### Benchmark Targets

| Config | Current | Phase A | Phase A+B | Java |
|--------|---------|---------|-----------|------|
| BoB 1yr | 0.79s | ~0.5-0.7s | ~0.3-0.5s | 0.80s |
| BoB 5yr | 11.5s | ~3-4s | ~1.5-2s | 2.3s |

Note: 1yr estimates are extrapolated. The dispatch overhead is a smaller fraction of 1yr total time, so the speedup may be less pronounced than for 5yr.

## What's NOT In Scope

- Optimizing non-mortality processes (movement, growth, reproduction)
- Custom Numba RNG implementation
- Changing the mortality algorithm (same interleaved cause dispatch)
- GPU acceleration
- Numba AOT compilation

## Files Modified

- `osmose/engine/processes/mortality.py` — new batch functions, updated `mortality()` orchestration, add `prange` import
- `tests/test_engine_parity.py` — add `TestStatisticalParity`, update baseline workflow
- `scripts/save_parity_baseline.py` — add `--statistical` mode
- `scripts/benchmark_engine.py` — 5yr benchmark support
- `tests/baselines/` — new statistical baseline files
