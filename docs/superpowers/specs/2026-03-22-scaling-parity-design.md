# Scaling Parity: Movement Numba + Vectorized Rates

**Date:** 2026-03-22
**Status:** Draft
**Goal:** Close the 2.2x performance gap on EEC 5yr (17.5s → ~6-8s, Java reference 8.1s) by Numba-compiling the movement loop and vectorizing `_precompute_effective_rates`.

## Context

After Phase A/B mortality optimizations, the Python engine achieves Java parity on Bay of Biscay (2.45s vs 2.3s) but remains 2.2x slower on the larger English Channel (EEC) config (17.5s vs 8.1s for 5yr).

Profiling reveals that mortality is no longer the dominant bottleneck on EEC:

| Process | Time (s) | % | Root Cause |
|---------|----------|---|------------|
| **movement** | 11.58 | 65.7% | Per-school Python scalar loop |
| mortality (numba core) | 2.53 | 14.3% | Already optimized |
| **_precompute_effective_rates** | 1.99 | 11.3% | Per-school Python loop |
| resource_update | 0.55 | 3.1% | — |
| everything else | 0.88 | 5.0% | — |

EEC characteristics: 14 species, 22×45 grid (990 cells), 24 dt/yr, 10 mortality sub-timesteps. School count grows from 0 → 13,744 over 5 years. All Python loop overhead scales linearly with school count.

## Part 1: Movement Numba Acceleration

### Current Implementation

`movement()` in `osmose/engine/processes/movement.py` has two paths:

1. **Random walk** (vectorized, fast) — batch NumPy displacement for "random" species
2. **Map-based** (scalar Python loop, slow) — `for i in np.where(uses_maps)[0]` calls `_map_move_school()` per school

`_map_move_school()` (lines 19-84) does:
- **Map lookup:** `map_set.get_index(age_dt, step)` — index into `MovementMapSet.index_maps[age_dt, step]`, a 2D int32 array
- **Same-map detection:** Compare current vs previous map index
- **New placement (rejection sampling):** Pick random flat cell index, check `current_map[j, i] > 0`, accept with probability `proba / max_p`. Up to 10,000 attempts.
- **Random walk (same map):** Nested loop over walk_range neighborhood, collect accessible cells into a Python list, pick one randomly.

### MovementMapSet Data Structure

```python
class MovementMapSet:
    index_maps: NDArray[np.int32]         # shape (lifespan_dt, n_total_steps)
    maps: list[NDArray[np.float64] | None]  # each (ny, nx) or None
    max_proba: NDArray[np.float64]         # shape (n_canonical_maps,)
    n_maps: int
```

The `maps` list contains `None` entries for "out-of-domain" maps (school goes out of grid). Non-None maps are float64 grids. The `index_maps` 2D array maps `(age_dt, step)` → map index (or -1).

### Approach: Pre-extract + Numba Batch

**Step 1: Pre-extract map data into Numba-compatible arrays** (once per species, at simulation start):

```python
def _flatten_map_set(map_set: MovementMapSet, ny: int, nx: int) -> tuple:
    """Convert MovementMapSet into flat arrays for Numba."""
    n_canonical = len(map_set.maps)
    # Stack all maps into a 3D array; None maps become all-zeros
    maps_3d = np.zeros((n_canonical, ny, nx), dtype=np.float64)
    is_null = np.zeros(n_canonical, dtype=np.bool_)
    for k, m in enumerate(map_set.maps):
        if m is not None:
            maps_3d[k] = m
        else:
            is_null[k] = True
    return maps_3d, is_null, map_set.max_proba, map_set.index_maps
```

This is called once per species during simulation setup (not per step). Cost: negligible.

**Step 2: Pre-compute per-school map indices** (per step, vectorized in Python):

```python
# For all map-using schools:
age_dts = state.age_dt[map_mask]       # int32 array
sp_ids = state.species_id[map_mask]    # int32 array

# Look up current and previous map indices
# Must replicate get_index() bounds-checking: return -1 if out of range
def _get_idx(index_maps, age, s):
    if age < 0 or age >= index_maps.shape[0]:
        return -1
    if s < 0 or s >= index_maps.shape[1]:
        return -1
    return int(index_maps[age, s])

current_idx = np.array([
    _get_idx(flat_data[sp].index_maps, a, step)
    for sp, a in zip(sp_ids, age_dts)
], dtype=np.int32)
prev_idx = np.array([
    _get_idx(flat_data[sp].index_maps, a - 1, step - 1)
    for sp, a in zip(sp_ids, age_dts)
], dtype=np.int32)
same_map = (current_idx == prev_idx) & (age_dts > 0) & (step > 0)
```

This is still a Python loop but only computes two integers per school — much cheaper than the full movement logic. An alternative is to batch all species' `index_maps` into a single lookup table, but species have different lifespan_dt dimensions, making this complex. The per-school index lookup is <1ms even for 14k schools — negligible vs. the 11.6s movement cost.

**Step 3: Numba batch function** (per step):

```python
@njit(cache=True)
def _map_move_batch_numba(
    rng_seed,
    # Per-school inputs
    school_indices,     # int32[n] — indices into state arrays
    current_map_idx,    # int32[n] — which map to use
    same_map,           # bool[n] — True if same map as last step
    cell_x, cell_y,     # int32[n] — current positions (may be -1)
    sp_ids,             # int32[n] — species ID per school
    # Per-species map data (ragged → use offsets)
    all_maps,           # float64[total_maps, ny, nx] — stacked maps for all species
    all_max_proba,      # float64[total_maps]
    all_is_null,        # bool[total_maps]
    sp_map_offset,      # int32[n_species] — offset into all_maps for each species
    # Grid
    ocean_mask,         # bool[ny, nx]
    walk_range,         # int32[n_species]
    ny, nx,
    # Output
    out_cx, out_cy, out_is_out,  # int32[n], int32[n], bool[n]
):
    np.random.seed(rng_seed)
    n_cells = ny * nx
    for k in range(len(school_indices)):
        idx = school_indices[k]
        sp = sp_ids[idx]
        map_idx = current_map_idx[k]

        # Resolve to global map array index
        global_map_idx = sp_map_offset[sp] + map_idx

        # Out-of-domain check
        if map_idx < 0 or all_is_null[global_map_idx]:
            out_cx[idx] = -1
            out_cy[idx] = -1
            out_is_out[idx] = True
            continue

        current_map = all_maps[global_map_idx]
        max_p = all_max_proba[global_map_idx]

        # New placement (rejection sampling)
        if not same_map[k] or cell_x[idx] < 0:
            placed = False
            for _ in range(10_000):
                flat_idx = int(round((n_cells - 1) * np.random.random()))
                j = flat_idx // nx
                i = flat_idx % nx
                proba = current_map[j, i]
                if proba > 0 and not np.isnan(proba):
                    if max_p == 0.0 or proba >= np.random.random() * max_p:
                        out_cx[idx] = i
                        out_cy[idx] = j
                        out_is_out[idx] = False
                        placed = True
                        break
            if not placed:
                out_cx[idx] = -1
                out_cy[idx] = -1
                out_is_out[idx] = True
            continue

        # Random walk (same map, school is located)
        cx_k = cell_x[idx]
        cy_k = cell_y[idx]
        wr = walk_range[sp]
        n_accessible = 0
        # First pass: count accessible cells
        y_lo = max(0, cy_k - wr)
        y_hi = min(ny, cy_k + wr + 1)
        x_lo = max(0, cx_k - wr)
        x_hi = min(nx, cx_k + wr + 1)
        for yi in range(y_lo, y_hi):
            for xi in range(x_lo, x_hi):
                if ocean_mask[yi, xi] and current_map[yi, xi] > 0:
                    if not np.isnan(current_map[yi, xi]):
                        n_accessible += 1

        if n_accessible == 0:
            out_cx[idx] = cx_k
            out_cy[idx] = cy_k
            out_is_out[idx] = False
            continue

        # Second pass: pick random accessible cell
        target = int(round((n_accessible - 1) * np.random.random()))
        count = 0
        for yi in range(y_lo, y_hi):
            for xi in range(x_lo, x_hi):
                if ocean_mask[yi, xi] and current_map[yi, xi] > 0:
                    if not np.isnan(current_map[yi, xi]):
                        if count == target:
                            out_cx[idx] = xi
                            out_cy[idx] = yi
                            out_is_out[idx] = False
                        count += 1
                        if count > target:
                            break
            if count > target:
                break
```

The double-pass approach (count then pick) avoids the Python `list.append()` accumulation pattern which isn't available in Numba. The walk_range is typically small (1-3 cells), so the neighborhood scan is at most ~49 cells — trivial cost per school when compiled.

### Integration into movement()

```python
def movement(state, grid, config, step, rng, map_sets=None, ...):
    # ... random walk (unchanged) ...

    # Map-based movement
    if uses_maps.any() and map_sets is not None:
        if _HAS_NUMBA:
            # Pre-compute per-school map indices (vectorized where possible)
            map_indices, prev_indices, same_map_flags = _precompute_map_indices(
                state, config, map_sets, step, uses_maps
            )
            rng_seed = int(rng.integers(0, 2**63))
            out_cx, out_cy, out_is_out = state.cell_x.copy(), state.cell_y.copy(), state.is_out.copy()
            _map_move_batch_numba(
                rng_seed,
                np.where(uses_maps)[0].astype(np.int32),
                map_indices, same_map_flags,
                out_cx, out_cy, state.species_id,
                flat_maps, flat_max_proba, flat_is_null, sp_offsets,
                grid.ocean_mask, config.random_walk_range.astype(np.int32),
                grid.ny, grid.nx,
                out_cx, out_cy, out_is_out,
            )
            state = state.replace(cell_x=out_cx, cell_y=out_cy, is_out=out_is_out)
        else:
            # Existing Python fallback (unchanged)
            ...
```

### Flat Map Data Preparation

The `all_maps` 3D array stacks all species' maps contiguously. `sp_map_offset[sp]` gives the starting index for species `sp`. This is computed once at simulation start:

```python
def _flatten_all_map_sets(map_sets, n_species, ny, nx):
    """Stack all species' maps into contiguous arrays for Numba."""
    total_maps = sum(len(ms.maps) for ms in map_sets.values())
    all_maps = np.zeros((total_maps, ny, nx), dtype=np.float64)
    all_max_proba = np.zeros(total_maps, dtype=np.float64)
    all_is_null = np.zeros(total_maps, dtype=np.bool_)
    sp_map_offset = np.full(n_species, -1, dtype=np.int32)

    pos = 0
    for sp, ms in map_sets.items():
        sp_map_offset[sp] = pos
        for k, m in enumerate(ms.maps):
            if m is not None:
                all_maps[pos + k] = m
            else:
                all_is_null[pos + k] = True
            all_max_proba[pos + k] = ms.max_proba[k]
        pos += len(ms.maps)

    return all_maps, all_max_proba, all_is_null, sp_map_offset
```

### Per-Species RNG

The current code uses per-species RNG when `config.movement_seed_fixed` is True. The Numba batch function uses a single seed. To maintain per-species independence, we can derive per-school seeds: `rng_seed + sp_id * 104729`. This changes the random stream but maintains statistical equivalence (accepted per the existing spec).

When `movement_seed_fixed` is False (the common case), a single seed is fine.

### Expected Performance

- Eliminates ~14,000 Python function calls per step (one per map-using school)
- Rejection sampling and walk logic compiled to machine code
- `_precompute_map_indices` is a lightweight Python loop (~0.5ms for 14k schools)
- **EEC 5yr: 11.58s → ~0.5-1.0s for movement**

## Part 2: Vectorize _precompute_effective_rates

### Current Implementation

`_precompute_effective_rates()` in `mortality.py` (lines 484-614) contains two Python `for i in range(n)` loops:

1. **Additional mortality loop** (lines 503-531): For each school, looks up `config.additional_mortality_rate[sp]`, optionally overrides with time-varying rate and spatial factor. EEC uses both spatial maps and time-varying rates.

2. **Fishing loop** (lines 537-610): For each school, applies selectivity (age or length-based), spatial maps, MPA zones, and seasonality. ~70 lines of branching logic per school.

With 13,744 schools at year 5, these loops cost 2.0s (11.3% of total).

### Approach: Vectorized NumPy

Replace scalar loops with species-indexed array operations. The key insight: most per-school config lookups are `config.some_array[species_id[i]]`, which vectorizes to `config.some_array[species_id]`.

**Additional mortality (vectorized):**

```python
# Base rates for all schools
sp = work_state.species_id
rates = config.additional_mortality_rate[sp].copy()

# Time-varying override (per-species, same value for all schools of same species)
if config.additional_mortality_by_dt is not None:
    for sp_id in range(config.n_species):
        arr = config.additional_mortality_by_dt[sp_id]
        if arr is not None:
            mask = sp == sp_id
            rates[mask] = arr[step % len(arr)]

# Spatial factor
if config.additional_mortality_spatial is not None:
    for sp_id in range(config.n_species):
        sp_map = config.additional_mortality_spatial[sp_id]
        if sp_map is not None:
            mask = sp == sp_id
            cy = work_state.cell_y[mask]
            cx = work_state.cell_x[mask]
            valid = (cy >= 0) & (cy < sp_map.shape[0]) & (cx >= 0) & (cx < sp_map.shape[1])
            # Index only valid elements to avoid negative-index wrapping
            factors = np.zeros(mask.sum(), dtype=np.float64)
            if valid.any():
                f_vals = sp_map[cy[valid], cx[valid]]
                f_vals = np.where(np.isnan(f_vals) | (f_vals <= 0), 0.0, f_vals)
                factors[valid] = f_vals
            rates[mask] *= factors

# Apply masks
rates[work_state.is_background] = 0.0
rates[work_state.age_dt == 0] = 0.0
rates[rates < 0] = 0.0
eff_additional = rates / denom
```

The outer loop is over `n_species` (14 for EEC), not `n_schools` (13,744). Each iteration does vectorized NumPy operations on the species mask. Cost: ~0.1ms vs the current 1.0ms.

**Fishing (vectorized):**

```python
if not config.fishing_enabled:
    eff_fishing = np.zeros(n, dtype=np.float64)
    fishing_discard = np.zeros(n, dtype=np.float64)
else:

# Base rates
f_rates = config.fishing_rate[sp].copy()

# Year-varying override
if config.fishing_rate_by_year is not None:
    year = step // config.n_dt_per_year
    for sp_id in range(config.n_species):
        arr = config.fishing_rate_by_year[sp_id] if sp_id < len(config.fishing_rate_by_year) else None
        if arr is not None and year < len(arr):
            f_rates[sp == sp_id] = arr[year]

# Selectivity
selectivity = np.ones(n, dtype=np.float64)
for sp_id in range(config.n_species):
    mask = sp == sp_id
    if not mask.any():
        continue
    sel_type = config.fishing_selectivity_type[sp_id]
    if sel_type == 0:  # age-based
        age_years = work_state.age_dt[mask] / config.n_dt_per_year
        a50 = config.fishing_selectivity_a50[sp_id]
        selectivity[mask] = np.where(age_years < a50, 0.0, 1.0)
    elif sel_type == 1:  # logistic
        l50 = config.fishing_selectivity_l50[sp_id]
        slope = config.fishing_selectivity_slope[sp_id]
        selectivity[mask] = 1.0 / (1.0 + np.exp(-slope * (work_state.length[mask] - l50)))
    else:  # length cutoff
        l50 = config.fishing_selectivity_l50[sp_id]
        selectivity[mask] = np.where((l50 > 0) & (work_state.length[mask] < l50), 0.0, 1.0)

# Spatial maps
spatial_factor = np.ones(n, dtype=np.float64)
for sp_id in range(config.n_species):
    sp_map = config.fishing_spatial_maps[sp_id] if sp_id < len(config.fishing_spatial_maps) else None
    if sp_map is None:
        continue
    mask = sp == sp_id
    cy, cx = work_state.cell_y[mask], work_state.cell_x[mask]
    valid = (cy >= 0) & (cy < sp_map.shape[0]) & (cx >= 0) & (cx < sp_map.shape[1])
    # Index only valid elements to avoid negative-index wrapping
    factors = np.zeros(mask.sum(), dtype=np.float64)
    if valid.any():
        f_vals = sp_map[cy[valid], cx[valid]]
        f_vals = np.where(np.isnan(f_vals) | (f_vals <= 0), 0.0, f_vals)
        factors[valid] = f_vals
    spatial_factor[mask] = factors

# MPA zones (apply to all species — Java MPA affects all focal species)
mpa_factor = np.ones(n, dtype=np.float64)
if config.mpa_zones is not None:
    year = step // config.n_dt_per_year
    for mpa in config.mpa_zones:
        if not (mpa.start_year <= year < mpa.end_year):
            continue
        cy = work_state.cell_y
        cx = work_state.cell_x
        valid = (cy >= 0) & (cy < mpa.grid.shape[0]) & (cx >= 0) & (cx < mpa.grid.shape[1])
        in_mpa = np.zeros(n, dtype=np.bool_)
        in_mpa[valid] = mpa.grid[cy[valid], cx[valid]] > 0
        mpa_factor *= np.where(in_mpa, 1.0 - mpa.percentage, 1.0)

# Seasonality
season = np.ones(n, dtype=np.float64)
if config.fishing_seasonality is not None:
    step_in_year = step % config.n_dt_per_year
    season = config.fishing_seasonality[sp, step_in_year]

# Combine — denominator differs based on seasonality
if config.fishing_seasonality is not None:
    eff_fishing = f_rates * selectivity * spatial_factor * mpa_factor * season / n_subdt
else:
    eff_fishing = f_rates * selectivity * spatial_factor * mpa_factor / denom

# Zero out background/eggs
eff_fishing[work_state.is_background] = 0.0
eff_fishing[work_state.age_dt == 0] = 0.0
eff_fishing[eff_fishing < 0] = 0.0

# Fishing discard rates — only set where fishing is active
fishing_discard = np.zeros(n, dtype=np.float64)
if config.fishing_discard_rate is not None:
    fishing_discard = np.where(eff_fishing > 0, config.fishing_discard_rate[sp], 0.0)
```

### Expected Performance

- Loops over `n_species` (14) instead of `n_schools` (13,744) — ~1000x reduction
- Each iteration is vectorized NumPy on species mask
- **EEC 5yr: 2.0s → ~0.05s**

## Testing & Validation

### Movement Tests

Existing movement tests in `tests/test_engine_movement.py` validate map-based movement behavior. After the Numba optimization:

1. **Determinism:** Same seed → same output (already tested by `TestDeterminism`)
2. **Statistical parity:** Multi-seed test validates mean biomass within 5% (already exists)
3. **Sanity checks:** Non-negative biomass, species survival (already exist)

New unit test: `test_map_move_batch_numba` — verify that the Numba batch function produces valid cell coordinates (within grid bounds, on ocean cells, matching map probabilities).

### Rates Tests

Add a targeted test: compute rates with the vectorized function and the original scalar function, assert exact match. This ensures the vectorization doesn't change any logic.

### Baseline Management

Both changes alter the RNG consumption order (movement uses different seeding, rates computation order changes). Regenerate exact-match and statistical baselines after implementation. Statistical parity (5% tolerance) should hold.

### Benchmark Targets

| Config | Current | After | Java |
|--------|---------|-------|------|
| EEC 1yr | 1.07s | ~0.8s | 2.71s |
| EEC 5yr | 17.5s | ~5-7s | 8.1s |
| BoB 1yr | 0.26s | ~0.25s | 0.80s |
| BoB 5yr | 2.45s | ~2.0s | 2.3s |

## What's NOT In Scope

- Optimizing resource_update (3.1%, I/O-bound NetCDF reads)
- Optimizing growth/reproduction (<3% combined)
- Algorithmic changes to predation (O(n²) is inherent)
- GPU acceleration
- Optimizing collect_outputs (1.2%)

## Files Modified

- `osmose/engine/processes/movement.py` — add Numba batch function, update `movement()` dispatch
- `osmose/engine/processes/mortality.py` — vectorize `_precompute_effective_rates()`
- `osmose/engine/simulate.py` — pre-flatten map data at simulation start
- `tests/test_engine_movement.py` — add Numba movement test
- `tests/test_engine_parity.py` — regenerate baselines
- `tests/baselines/` — updated baseline files
