# B1: Map-Based Movement — Design Spec

> **Date:** 2026-03-19
> **Feature:** Spatial movement using age/season-specific probability maps
> **Approach:** MovementMapSet loads CSV maps into indexMaps lookup table; per-school movement via rejection sampling + random walk

## Overview

Map-based movement replaces the simple random walk with spatially explicit probability maps that vary by species, age, and season. Each timestep, a school's age and the simulation step determine which map applies. If no map is defined (file="null"), the school is marked out-of-domain. If the map changed from the previous timestep, the school is re-placed via rejection sampling. If the map is the same, the school does a random walk within map-positive cells.

This coexists with the existing random walk method — species with `movement.distribution.method = "random"` use the existing code unchanged.

## Config Keys

### Method selection (per species)
```
movement.distribution.method.sp{N} = maps      # or "random"
```

### Map definitions (per map entry)
```
movement.species.map{N} = lesserSpottedDogfish  # species name (matches species.name.sp{X})
movement.initialAge.map{N} = 0                  # start age in years (inclusive)
movement.lastAge.map{N} = 2                     # end age in years (exclusive after conversion)
movement.file.map{N} = maps/1Roussette_01.csv   # CSV file path, or "null" for out-of-domain
movement.steps.map{N} = 0;1;2;...;23            # step indices within year (default: all)
```

Optional year range:
```
movement.years.map{N} = 0;1;2                   # explicit year list
# OR:
movement.initialYear.map{N} = 0                 # start year (default: 0)
movement.lastYear.map{N} = 4                    # end year inclusive (default: nyear-1)
```

### Other movement config
```
movement.randomwalk.range.sp{N} = 1             # cells (default: 1, used by both random and maps)
movement.randomseed.fixed = true                # deterministic seeds (default: false)
movement.netcdf.enabled = false                 # NetCDF map loading (default: false)
movement.checks.enabled = false                 # output map validation file (default: false)
```

### NetCDF map loading (when `movement.netcdf.enabled = true`)
```
movement.file.map{N} = path/to/file.nc
movement.variable.map{N} = varname
movement.nsteps.year.map{N} = 12
```

## Data Model: `MovementMapSet`

Core data structure per species, matching Java's `MapSet`:

```python
class MovementMapSet:
    index_maps: NDArray[np.int32]       # (lifespan_dt, n_total_steps) → map_index, -1 = no map
    maps: list[NDArray[np.float64] | None]  # 2D grids (ny, nx), None = out-of-domain
    max_proba: NDArray[np.float64]      # (n_maps,) max cell value per map
    n_maps: int
```

**`n_total_steps`** = `max(n_years * n_dt_per_year, n_dt_per_year)` — the second dimension covers the full simulation duration. This matches Java's `Math.max(getNStep(), getNStepYear())`. The `step` parameter in `get_map(age_dt, step)` is the full simulation step index (not within-year), ranging `[0, n_total_steps)`.


### Construction algorithm

1. Scan `movement.species.map{N}` keys matching this species' name
2. For each matched map:
   - Age range: `round(initialAge * n_dt_per_year)` to `min(round(lastAge * n_dt_per_year), lifespan_dt - 1)`
   - Step range: parse `movement.steps.map{N}` (semicolon-separated, filter values >= n_dt_per_year)
   - Year range: from `movement.years.map{N}` or `initialYear`/`lastYear` (default: all years, lastYear inclusive)
   - File: CSV path or `"null"` → None
3. Fill `index_maps[age_dt][year * n_dt_per_year + season] = map_idx` for all age/year/season combinations
4. Load CSV files into 2D float arrays
5. Compute `max_proba[i]` per map:
   - If max cell value >= 1.0 (presence/absence): set `max_proba = 0.0` (uniform acceptance trick)
   - Otherwise: store actual max
6. Deduplicate: maps referencing the same CSV file share one loaded array AND `index_maps` entries are remapped to point to the canonical map index. This ensures `same_map` detection works correctly across map-entry boundaries that share the same CSV file.
7. Bounds check: silently skip `index_maps[age_dt][step]` assignments where `step >= n_total_steps` (matching Java's `iStep < indexMaps[iAge].length` guard). This handles configs with year ranges beyond the simulation duration.

### Lookup

```python
def get_map(self, age_dt: int, step: int) -> NDArray[np.float64] | None:
    idx = self.index_maps[age_dt, step]
    if idx < 0:
        return None
    return self.maps[idx]
```

### CSV Grid Format

- Semicolon-separated 2D grid, no header
- Row 0 in CSV = grid row `ny - 1` (North at top, grid y=0 is South — rows flipped on load)
- Values: -99 = land/absent (rejected by `proba > 0`), positive = presence/probability
- `"na"`/`"nan"` strings → `NaN` (rejected by `not isnan(proba)`)

### Validation

After building `index_maps`, check that every `[age_dt][step]` for step in `[0, n_total_steps)` has a valid map index (not -1). Log warnings for missing entries. Java calls `error()` (fatal) on missing entries; Python logs warnings and continues for robustness. Missing entries produce `get_map()` returning None → school marked out-of-domain.

**File:** `osmose/engine/movement_maps.py`

## Movement Algorithm

### Per-school movement (species with `method="maps"`)

**Step 1 — Out-of-domain check:**
```
map = map_set.get_map(school.age_dt, step)
if map is None:
    school.is_out = True
    school.cell_x = school.cell_y = -1
    return
```

**Step 2 — Same-map detection:**
```
index_map = index_maps[age_dt, step]
same_map = False
if age_dt > 0 and step > 0:
    prev_index = index_maps[age_dt - 1, step - 1]
    same_map = (index_map == prev_index)
```

**Step 3a — New placement** (if `not same_map` OR `cell_x < 0` i.e. unlocated):

Rejection sampling over full grid (nx * ny cells, including land):
```
for attempt in range(10_000):
    flat_idx = round((nx * ny - 1) * rng.random())   # Math.round bias
    j = flat_idx // nx
    i = flat_idx % nx
    proba = map[j, i]
    if proba > 0 and not isnan(proba):
        if max_proba == 0 or proba >= rng.random() * max_proba:
            school.cell_x = i
            school.cell_y = j
            return
raise RuntimeError("placement failed after 10000 attempts")
```

**Step 3b — Random walk** (if `same_map` AND school is located):
```
neighbors = all cells in [x-range, x+range] × [y-range, y+range], clipped to grid bounds
accessible = [c for c in neighbors
              if ocean_mask[c.y, c.x] and map[c.y, c.x] > 0 and not isnan(map[c.y, c.x])]
if len(accessible) == 0:
    # Edge case: school stranded — no valid neighbors. Stay in place.
    # Java would crash with IndexOutOfBoundsException here.
    return
idx = round((len(accessible) - 1) * rng.random())   # Math.round bias
school.cell_x, school.cell_y = accessible[idx]
```

### RNG usage

Java uses three separate `Random` instances per species (rd1 for cell sampling, rd2 for probability threshold, rd3 for random walk), each with distinct seeds when `movement.randomseed.fixed=true`. Python uses a single `np.random.Generator` passed to the movement function. Since we don't target bit-for-bit RNG reproduction (only statistical behavior), a single RNG is acceptable. The call order within each school's movement step matches Java's logical sequence.

### `Math.round` bias replication

Java's `(int) Math.round((N-1) * rnd)` gives edge indices half the probability of middle indices. Python equivalent: `int(round((N-1) * rng.random()))`. This must be replicated for statistical parity, though bit-for-bit RNG reproduction is not required.

### `max_proba = 0` trick for presence/absence maps

When all cell values are 0 or 1 and `max >= 1.0`, `max_proba` is set to 0.0. The acceptance condition `proba >= rng.random() * 0.0` becomes `proba >= 0`, which is always true when `proba > 0`. Result: uniform random placement among all occupied cells (no probability weighting).

### Out-of-domain lifecycle

1. Timestep T: map lookup returns None → `is_out = True`, `cell_x = cell_y = -1`
2. `out_mortality` applies mortality to out-of-domain schools (existing code)
3. Timestep T+1: `_reset_step_variables()` resets `is_out = False`
4. Movement re-evaluates: if new map exists, school is unlocated (`cell_x < 0`) → rejection sampling re-places it
5. If still no map → `is_out = True` again

## Integration

### Movement orchestrator (`processes/movement.py`)

Extend the existing `movement()` function:

```python
def movement(state, grid, config, step, rng, map_sets=None):
    # Split by method
    for each school:
        if config.movement_method[sp] == "random":
            apply random_walk (existing batch code)
        elif config.movement_method[sp] == "maps":
            apply map_distribution (per-school, using map_sets[sp])
    return updated state
```

The random walk path stays vectorized (batch NumPy). The maps path is per-school (sequential) since each school has a unique age_dt and position.

### `map_sets` initialization (`simulate.py`)

```python
map_sets = {}
for sp in range(config.n_species):
    if config.movement_method[sp] == "maps":
        map_sets[sp] = MovementMapSet(config.raw_config, sp, config, grid)
```

Created once before the simulation loop. Passed to `movement()` each timestep.

### `_reset_step_variables` change

Add `is_out` reset:
```python
is_out=np.zeros(len(state), dtype=np.bool_),
```

This matches Java's `school.init()` which resets `out = false` each timestep.

### Background species

Do NOT participate in map movement. Java's `MovementProcess` only iterates focal species. Background schools are injected after movement in the simulation loop.

### No `EngineConfig` changes

Map data (`MovementMapSet` instances) are created separately in `simulate.py`. `EngineConfig` already has `movement_method` and `random_walk_range` fields.

## Scope Decisions

### CSV maps only (for now)

NetCDF map loading (`movement.netcdf.enabled = true`) is deferred. The eec_full example config uses CSV maps. NetCDF can be added later with the same `MovementMapSet` interface.

### No random patch distribution

Java's `RandomDistribution` supports `movement.distribution.ncell.sp{N}` for creating a random contiguous patch. The Python engine's existing `random_walk()` covers the full-domain random case. The partial-domain random patch is deferred — no example configs use it.

### Movement checks output deferred

`movement.checks.enabled` generates a validation CSV file. Useful for debugging but not needed for simulation correctness. Deferred.

## Java Parity Notes

### Cell index linearization

Java uses `j * nx + i` (row-major). Python's grid uses the same convention. `flat_idx // nx = j`, `flat_idx % nx = i`.

### CSV row flipping

Java reads CSV row 0 as grid row `ny - 1 - l`. Python must flip rows on load: `grid_data[ny - 1 - csv_row] = csv_values`.

### -99 values in CSV

Stored as literal -99.0 in the grid array (not converted to NaN or 0). Rejected by `proba > 0` in movement checks. This differs from NetCDF where -99 is converted to 0.0.

### `isUnlocated()` = `cell_x < 0`

Java uses `x < 0 || y < 0`. Python equivalent: `cell_x < 0` (since both are set to -1 together by `setOffGrid()`).

### `same_map` uses `(age_dt - 1, step - 1)`

Compares the map index at the school's previous age and previous simulation step to the current one. This correctly handles age-group boundary crossings.

### Accessible cells include current cell

Java's `getNeighbourCells(cell, range)` returns a rectangular window `[i-range, i+range] × [j-range, j+range]` clipped to grid bounds, including the center cell. So a school can stay in place during random walk.

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `osmose/engine/movement_maps.py` | Create | `MovementMapSet` class (loading, indexing, lookup) |
| `osmose/engine/processes/movement.py` | Modify | Add map-based movement path, split by method |
| `osmose/engine/simulate.py` | Modify | Build map_sets, pass to movement, reset is_out |
| `tests/test_engine_map_movement.py` | Create | Full test suite |

## Testing Strategy (TDD)

1. **MapSet loading:** Parse map config, build index_maps for single species
2. **MapSet loading:** Multiple maps per species (age ranges + season subsets)
3. **MapSet loading:** `"null"` file → None map → out-of-domain
4. **MapSet loading:** CSV grid loading with -99 values and row flipping
5. **MapSet loading:** max_proba trick: presence/absence map → max_proba = 0
6. **MapSet loading:** Map deduplication (same file → shared array)
7. **MapSet lookup:** `get_map(age_dt, step)` returns correct map or None
8. **MapSet validation:** Missing entries in index_maps logged as warnings
9. **Movement:** Out-of-domain: null map → is_out=True, cell_x/y=-1
10. **Movement:** Same-map detection: same index → random walk, different → rejection sampling
11. **Movement:** Rejection sampling places school on positive-probability cell
12. **Movement:** Presence/absence map: uniform placement among positive cells
13. **Movement:** Random walk stays within accessible cells (ocean, positive map value, within range)
14. **Movement:** Unlocated school (cell_x < 0) triggers new placement even if same map
14b. **Movement:** Empty accessible cells (school stranded) — school stays in place, no crash
15. **Movement:** is_out reset each timestep via _reset_step_variables
16. **Orchestrator:** Species with method="random" use existing random_walk
17. **Orchestrator:** Species with method="maps" use map_distribution
18. **Orchestrator:** Mixed species (some random, some maps) coexist
19. **Integration:** Full simulation with map movement runs without errors
20. **Backward compatibility:** Config with all "random" species works unchanged
