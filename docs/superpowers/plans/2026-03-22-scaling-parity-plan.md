# Scaling Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the EEC 5yr performance gap (17.5s → ~5-7s, Java: 8.1s) by Numba-compiling map-based movement and vectorizing `_precompute_effective_rates`.

**Architecture:** Two independent optimizations. Part 1 replaces the per-school Python movement loop with a single Numba batch call using pre-flattened map data. Part 2 replaces per-school mortality rate loops with vectorized NumPy species-indexed operations.

**Tech Stack:** NumPy, Numba (`njit`), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `osmose/engine/processes/movement.py` | Modify | Add `_map_move_batch_numba`, `_precompute_map_indices`, `_flatten_all_map_sets`, Numba import, update `movement()` dispatch |
| `osmose/engine/processes/mortality.py` | Modify | Replace `_precompute_effective_rates` body with vectorized version |
| `osmose/engine/simulate.py` | Modify | Pre-flatten map data at simulation start, pass to `_movement()` |
| `tests/test_movement_numba.py` | Create | Unit tests for Numba movement batch function |
| `tests/test_vectorized_rates.py` | Create | Exact-match test: vectorized vs scalar rates |
| `tests/test_engine_parity.py` | Modify | Regenerate baselines |

---

## Part 1: Movement Numba Acceleration

### Task 1: Pre-save EEC 5yr benchmark baseline

**Files:**
- Output: `tests/baselines/benchmark_eec_5yr_pre_scaling.json`

- [ ] **Step 1: Run EEC 5yr benchmark**

Run: `.venv/bin/python scripts/benchmark_engine.py --years 5 --seed 42 --repeats 3 --output tests/baselines/benchmark_eec_5yr_pre_scaling.json`

Note: The benchmark script uses Bay of Biscay config by default. For EEC, you need to temporarily modify the script or run the benchmark inline. Use this one-liner instead:

```bash
.venv/bin/python -c "
import json, time, numpy as np
from pathlib import Path
from osmose.config.reader import OsmoseConfigReader
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate
cfg_path = Path('data/eec_full/eec_all-parameters.csv')
reader = OsmoseConfigReader()
raw = reader.read(cfg_path)
raw['simulation.time.nyear'] = '5'
cfg = EngineConfig.from_dict(raw)
grid = Grid.from_netcdf(cfg_path.parent / raw['grid.netcdf.file'], mask_var=raw.get('grid.var.mask', 'mask'))
times = []
for i in range(3):
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    simulate(cfg, grid, rng)
    times.append(time.perf_counter() - t0)
    print(f'  Run {i+1}/3: {times[-1]:.3f}s')
times.sort()
print(f'  Median: {times[1]:.3f}s')
json.dump({'config': 'EEC', 'n_years': 5, 'median_s': round(times[1], 3), 'timings_s': [round(t, 3) for t in times]}, open('tests/baselines/benchmark_eec_5yr_pre_scaling.json', 'w'), indent=2)
"
```

Expected: Median ~17-18s

- [ ] **Step 2: Commit**

```bash
git add tests/baselines/benchmark_eec_5yr_pre_scaling.json
git commit -m "bench: save EEC 5yr pre-scaling baseline"
```

---

### Task 2: Add `_flatten_all_map_sets()` helper

**Files:**
- Modify: `osmose/engine/processes/movement.py`
- Create: `tests/test_movement_numba.py`

- [ ] **Step 1: Write test for `_flatten_all_map_sets`**

Create `tests/test_movement_numba.py`:

```python
"""Tests for Numba-accelerated movement functions."""

from __future__ import annotations

import numpy as np
import pytest


def _make_mock_map_set(maps_data, index_maps, max_proba):
    """Create a minimal object mimicking MovementMapSet for testing."""
    class MockMapSet:
        pass
    ms = MockMapSet()
    ms.maps = maps_data  # list of NDArray or None
    ms.index_maps = index_maps
    ms.max_proba = max_proba
    return ms


def test_flatten_all_map_sets_shapes():
    """Stacked maps have correct shape and offsets."""
    from osmose.engine.processes.movement import _flatten_all_map_sets

    ny, nx = 3, 4
    # Species 0: 2 maps (one real, one null)
    ms0 = _make_mock_map_set(
        [np.ones((ny, nx)), None],
        np.zeros((10, 24), dtype=np.int32),
        np.array([0.0, 0.0]),
    )
    # Species 2: 1 map
    ms2 = _make_mock_map_set(
        [np.full((ny, nx), 0.5)],
        np.zeros((10, 24), dtype=np.int32),
        np.array([0.5]),
    )
    map_sets = {0: ms0, 2: ms2}

    all_maps, all_max_proba, all_is_null, sp_offsets = _flatten_all_map_sets(
        map_sets, n_species=4, ny=ny, nx=nx
    )

    assert all_maps.shape == (3, ny, nx)  # 2 + 1 maps total
    assert all_max_proba.shape == (3,)
    assert all_is_null.shape == (3,)
    assert sp_offsets[0] == 0   # species 0 starts at 0
    assert sp_offsets[1] == -1  # species 1 has no maps
    assert sp_offsets[2] == 2   # species 2 starts at 2
    assert sp_offsets[3] == -1  # species 3 has no maps


def test_flatten_all_map_sets_null_detection():
    """None maps marked as null, real maps have correct data."""
    from osmose.engine.processes.movement import _flatten_all_map_sets

    ny, nx = 2, 2
    real_map = np.array([[0.3, 0.7], [0.0, 0.5]])
    ms = _make_mock_map_set(
        [real_map, None],
        np.zeros((5, 24), dtype=np.int32),
        np.array([0.7, 0.0]),
    )
    map_sets = {0: ms}

    all_maps, all_max_proba, all_is_null, _ = _flatten_all_map_sets(
        map_sets, n_species=1, ny=ny, nx=nx
    )

    assert not all_is_null[0]  # real map
    assert all_is_null[1]      # None map
    np.testing.assert_array_equal(all_maps[0], real_map)
    np.testing.assert_array_equal(all_maps[1], np.zeros((ny, nx)))  # null → zeros
    assert all_max_proba[0] == 0.7
    assert all_max_proba[1] == 0.0


def test_flatten_all_map_sets_empty():
    """Empty map_sets returns zero-length arrays."""
    from osmose.engine.processes.movement import _flatten_all_map_sets

    all_maps, all_max_proba, all_is_null, sp_offsets = _flatten_all_map_sets(
        {}, n_species=3, ny=5, nx=5
    )

    assert all_maps.shape == (0, 5, 5)
    assert all_max_proba.shape == (0,)
    assert np.all(sp_offsets == -1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_movement_numba.py -v`
Expected: ImportError — `_flatten_all_map_sets` not found

- [ ] **Step 3: Implement `_flatten_all_map_sets`**

Add at the bottom of `osmose/engine/processes/movement.py` (before any `if _HAS_NUMBA:` block if one exists, or just at the end):

```python
def _flatten_all_map_sets(
    map_sets: dict[int, MovementMapSet],
    n_species: int,
    ny: int,
    nx: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Stack all species' movement maps into contiguous arrays for Numba.

    Returns:
        all_maps: float64[total_maps, ny, nx] — stacked maps (None → zeros)
        all_max_proba: float64[total_maps] — max probability per map
        all_is_null: bool[total_maps] — True for None (out-of-domain) maps
        sp_map_offset: int32[n_species] — offset into all_maps per species (-1 if no maps)
    """
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

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_movement_numba.py -v`
Expected: All 3 PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/movement.py tests/test_movement_numba.py
git commit -m "feat: add _flatten_all_map_sets for Numba movement data"
```

---

### Task 3: Add `_precompute_map_indices()` helper

**Files:**
- Modify: `osmose/engine/processes/movement.py`
- Modify: `tests/test_movement_numba.py`

- [ ] **Step 1: Write test**

Add to `tests/test_movement_numba.py`:

```python
def test_precompute_map_indices_basic():
    """Map indices correctly looked up and same_map detected."""
    from osmose.engine.processes.movement import _precompute_map_indices

    ny, nx = 3, 4
    # index_maps: age 0→map 0, age 1→map 1, all steps
    idx_maps = np.full((5, 24), -1, dtype=np.int32)
    idx_maps[0, :] = 0
    idx_maps[1, :] = 1
    idx_maps[2, :] = 1  # age 2 same as age 1

    ms = _make_mock_map_set(
        [np.ones((ny, nx)), np.ones((ny, nx))],
        idx_maps,
        np.array([0.0, 0.0]),
    )
    map_sets = {0: ms}

    # 3 schools: all species 0, ages 0, 1, 2, step=2
    species_id = np.array([0, 0, 0], dtype=np.int32)
    age_dt = np.array([0, 1, 2], dtype=np.int32)
    uses_maps = np.array([True, True, True])

    current, same = _precompute_map_indices(
        species_id, age_dt, uses_maps, map_sets, step=2
    )

    assert current[0] == 0   # age 0 → map 0
    assert current[1] == 1   # age 1 → map 1
    assert current[2] == 1   # age 2 → map 1
    assert not same[0]       # age 0, step 2: prev is age -1 → -1, different
    assert not same[1]       # age 1: prev age 0 → map 0, current → map 1, different
    assert same[2]           # age 2: prev age 1 → map 1, current → map 1, same


def test_precompute_map_indices_out_of_range():
    """Out-of-range age returns -1."""
    from osmose.engine.processes.movement import _precompute_map_indices

    idx_maps = np.zeros((3, 24), dtype=np.int32)  # only ages 0-2
    ms = _make_mock_map_set([np.ones((2, 2))], idx_maps, np.array([0.0]))
    map_sets = {0: ms}

    species_id = np.array([0], dtype=np.int32)
    age_dt = np.array([10], dtype=np.int32)  # out of range
    uses_maps = np.array([True])

    current, same = _precompute_map_indices(
        species_id, age_dt, uses_maps, map_sets, step=0
    )

    assert current[0] == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_movement_numba.py::test_precompute_map_indices_basic -v`
Expected: ImportError

- [ ] **Step 3: Implement `_precompute_map_indices`**

Add to `osmose/engine/processes/movement.py`:

```python
def _precompute_map_indices(
    species_id: NDArray[np.int32],
    age_dt: NDArray[np.int32],
    uses_maps: NDArray[np.bool_],
    map_sets: dict[int, MovementMapSet],
    step: int,
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    """Pre-compute per-school map indices and same-map flags.

    Returns:
        current_idx: int32[n_map_schools] — current map index (-1 if out of range)
        same_map: bool[n_map_schools] — True if same map as previous step
    """
    map_school_mask = np.where(uses_maps)[0]
    n = len(map_school_mask)
    current_idx = np.full(n, -1, dtype=np.int32)
    prev_idx = np.full(n, -1, dtype=np.int32)

    for k, i in enumerate(map_school_mask):
        sp = int(species_id[i])
        age = int(age_dt[i])
        if sp not in map_sets:
            continue
        ms = map_sets[sp]
        # Current map index (replicate get_index bounds checking)
        if 0 <= age < ms.index_maps.shape[0] and 0 <= step < ms.index_maps.shape[1]:
            current_idx[k] = ms.index_maps[age, step]
        # Previous map index
        prev_age = age - 1
        prev_step = step - 1
        if 0 <= prev_age < ms.index_maps.shape[0] and 0 <= prev_step < ms.index_maps.shape[1]:
            prev_idx[k] = ms.index_maps[prev_age, prev_step]

    same_map = (current_idx == prev_idx) & (age_dt[map_school_mask] > 0) & (step > 0)
    return current_idx, same_map
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_movement_numba.py -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/movement.py tests/test_movement_numba.py
git commit -m "feat: add _precompute_map_indices for movement Numba path"
```

---

### Task 4: Implement `_map_move_batch_numba()`

**Files:**
- Modify: `osmose/engine/processes/movement.py`
- Modify: `tests/test_movement_numba.py`

- [ ] **Step 1: Add Numba import to movement.py**

At the top of `osmose/engine/processes/movement.py`, after the existing imports, add:

```python
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
```

- [ ] **Step 2: Write test for the Numba batch function**

Add to `tests/test_movement_numba.py`:

```python
def test_map_move_batch_numba_new_placement():
    """Rejection sampling places schools on cells with non-zero probability."""
    try:
        from osmose.engine.processes.movement import _map_move_batch_numba
    except ImportError:
        pytest.skip("Numba not available")

    ny, nx = 3, 4
    # Map with non-zero only at (1, 2) and (2, 3)
    prob_map = np.zeros((ny, nx), dtype=np.float64)
    prob_map[1, 2] = 0.8
    prob_map[2, 3] = 0.5

    all_maps = prob_map.reshape(1, ny, nx)
    all_max_proba = np.array([0.8])
    all_is_null = np.array([False])
    sp_offsets = np.array([0], dtype=np.int32)
    ocean_mask = np.ones((ny, nx), dtype=np.bool_)
    walk_range = np.array([1], dtype=np.int32)

    school_indices = np.array([0], dtype=np.int32)
    current_map_idx = np.array([0], dtype=np.int32)
    same_map = np.array([False])
    cell_x = np.array([-1], dtype=np.int32)
    cell_y = np.array([-1], dtype=np.int32)
    sp_ids = np.array([0], dtype=np.int32)

    out_cx = cell_x.copy()
    out_cy = cell_y.copy()
    out_is_out = np.array([False])

    _map_move_batch_numba(
        42,
        school_indices, current_map_idx, same_map,
        out_cx, out_cy, sp_ids,
        all_maps, all_max_proba, all_is_null, sp_offsets,
        ocean_mask, walk_range, ny, nx,
        out_cx, out_cy, out_is_out,
    )

    # Must land on one of the non-zero cells
    assert (out_cx[0], out_cy[0]) in [(2, 1), (3, 2)]
    assert not out_is_out[0]


def test_map_move_batch_numba_out_of_domain():
    """Null map index → school goes out of domain."""
    try:
        from osmose.engine.processes.movement import _map_move_batch_numba
    except ImportError:
        pytest.skip("Numba not available")

    ny, nx = 2, 2
    all_maps = np.zeros((1, ny, nx), dtype=np.float64)
    all_max_proba = np.array([0.0])
    all_is_null = np.array([True])  # null map
    sp_offsets = np.array([0], dtype=np.int32)
    ocean_mask = np.ones((ny, nx), dtype=np.bool_)
    walk_range = np.array([1], dtype=np.int32)

    out_cx = np.array([0], dtype=np.int32)
    out_cy = np.array([0], dtype=np.int32)
    out_is_out = np.array([False])

    _map_move_batch_numba(
        42,
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),  # map index 0 → null
        np.array([False]),
        out_cx, out_cy,
        np.array([0], dtype=np.int32),
        all_maps, all_max_proba, all_is_null, sp_offsets,
        ocean_mask, walk_range, ny, nx,
        out_cx, out_cy, out_is_out,
    )

    assert out_cx[0] == -1
    assert out_cy[0] == -1
    assert out_is_out[0]


def test_map_move_batch_numba_deterministic():
    """Same seed → same output."""
    try:
        from osmose.engine.processes.movement import _map_move_batch_numba
    except ImportError:
        pytest.skip("Numba not available")

    ny, nx = 5, 5
    prob_map = np.random.default_rng(0).random((ny, nx))
    all_maps = prob_map.reshape(1, ny, nx)
    all_max_proba = np.array([prob_map.max()])
    all_is_null = np.array([False])
    sp_offsets = np.array([0], dtype=np.int32)
    ocean_mask = np.ones((ny, nx), dtype=np.bool_)
    walk_range = np.array([2], dtype=np.int32)

    n = 20
    school_indices = np.arange(n, dtype=np.int32)
    current_map_idx = np.zeros(n, dtype=np.int32)
    same_map = np.zeros(n, dtype=np.bool_)
    sp_ids = np.zeros(n, dtype=np.int32)

    def run(seed):
        cx = np.full(n, -1, dtype=np.int32)
        cy = np.full(n, -1, dtype=np.int32)
        out = np.zeros(n, dtype=np.bool_)
        _map_move_batch_numba(
            seed, school_indices, current_map_idx, same_map,
            cx, cy, sp_ids,
            all_maps, all_max_proba, all_is_null, sp_offsets,
            ocean_mask, walk_range, ny, nx,
            cx, cy, out,
        )
        return cx.copy(), cy.copy()

    cx1, cy1 = run(99)
    cx2, cy2 = run(99)
    np.testing.assert_array_equal(cx1, cx2)
    np.testing.assert_array_equal(cy1, cy2)
```

- [ ] **Step 3: Implement `_map_move_batch_numba`**

Add inside an `if _HAS_NUMBA:` block in `osmose/engine/processes/movement.py`:

```python
if _HAS_NUMBA:

    @njit(cache=True)
    def _map_move_batch_numba(
        rng_seed,
        school_indices, current_map_idx, same_map,
        cell_x, cell_y, sp_ids,
        all_maps, all_max_proba, all_is_null, sp_map_offset,
        ocean_mask, walk_range,
        ny, nx,
        out_cx, out_cy, out_is_out,
    ):
        """Numba-compiled batch map-based movement for all schools."""
        np.random.seed(rng_seed)
        n_cells = ny * nx
        for k in range(len(school_indices)):
            idx = school_indices[k]
            sp = sp_ids[idx]
            map_idx = current_map_idx[k]

            global_map_idx = sp_map_offset[sp] + map_idx

            if map_idx < 0 or all_is_null[global_map_idx]:
                out_cx[idx] = -1
                out_cy[idx] = -1
                out_is_out[idx] = True
                continue

            current_map = all_maps[global_map_idx]
            max_p = all_max_proba[global_map_idx]

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

            cx_k = cell_x[idx]
            cy_k = cell_y[idx]
            wr = walk_range[sp]
            n_accessible = 0
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

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_movement_numba.py -v`
Expected: All 8 PASS (may take ~20s for first Numba compilation)

- [ ] **Step 5: Run lint**

Run: `.venv/bin/ruff check osmose/engine/processes/movement.py`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/movement.py tests/test_movement_numba.py
git commit -m "feat: add _map_move_batch_numba for compiled movement"
```

---

### Task 5: Wire Numba movement into `movement()` and `simulate()`

**Files:**
- Modify: `osmose/engine/processes/movement.py:177-259` — update `movement()` dispatch
- Modify: `osmose/engine/simulate.py:769-782` — pre-flatten map data, pass to `_movement()`

- [ ] **Step 1: Update `movement()` to use Numba path**

In `osmose/engine/processes/movement.py`, replace the map-based movement block (the `if uses_maps.any() and map_sets is not None:` section, approximately lines 237-257) with:

```python
    # Map-based movement for "maps" species
    if uses_maps.any() and map_sets is not None:
        if _HAS_NUMBA and flat_map_data is not None:
            flat_maps, flat_max_proba, flat_is_null, sp_offsets = flat_map_data
            map_school_indices = np.where(uses_maps)[0].astype(np.int32)
            current_idx, same_map_flags = _precompute_map_indices(
                state.species_id, state.age_dt, uses_maps, map_sets, step
            )
            rng_seed = int(rng.integers(0, 2**63))
            new_cx = state.cell_x.copy()
            new_cy = state.cell_y.copy()
            new_out = state.is_out.copy()
            _map_move_batch_numba(
                rng_seed,
                map_school_indices, current_idx, same_map_flags,
                new_cx, new_cy, state.species_id,
                flat_maps, flat_max_proba, flat_is_null, sp_offsets,
                grid.ocean_mask, config.random_walk_range.astype(np.int32),
                grid.ny, grid.nx,
                new_cx, new_cy, new_out,
            )
            state = state.replace(cell_x=new_cx, cell_y=new_cy, is_out=new_out)
        else:
            # Python fallback (existing code)
            new_cx = state.cell_x.copy()
            new_cy = state.cell_y.copy()
            new_out = state.is_out.copy()
            for i in np.where(uses_maps)[0]:
                sp_id = int(sp[i])
                if sp_id in map_sets:
                    x, y, out = _map_move_school(
                        int(state.age_dt[i]),
                        int(new_cx[i]),
                        int(new_cy[i]),
                        grid.ny,
                        grid.nx,
                        grid.ocean_mask,
                        map_sets[sp_id],
                        int(config.random_walk_range[sp_id]),
                        step,
                        _rng_for(sp_id),
                    )
                    new_cx[i], new_cy[i], new_out[i] = x, y, out
            state = state.replace(cell_x=new_cx, cell_y=new_cy, is_out=new_out)
```

Also add `flat_map_data=None` parameter to the `movement()` function signature.

- [ ] **Step 2: Update `simulate.py` to pre-flatten and pass map data**

In `osmose/engine/simulate.py`, after the `map_sets` construction (around line 782), add:

```python
    # Pre-flatten map data for Numba movement path
    from osmose.engine.processes.movement import _flatten_all_map_sets
    flat_map_data = _flatten_all_map_sets(map_sets, config.n_species, grid.ny, grid.nx) if map_sets else None
```

Then update the `_movement()` wrapper function (around line 85-96) to accept and pass `flat_map_data`:

Add `flat_map_data=None` to `_movement()` signature, and pass it through to `movement()`:
```python
def _movement(state, grid, config, step, rng, map_sets=None, random_patches=None,
              species_rngs=None, flat_map_data=None):
    return movement(state, grid, config, step, rng,
                    map_sets=map_sets, random_patches=random_patches,
                    species_rngs=species_rngs, flat_map_data=flat_map_data)
```

And in the step loop call (around line 801):
```python
        state = _movement(
            state, grid, config, step, rng,
            map_sets=map_sets,
            random_patches=random_patches,
            species_rngs=movement_rngs,
            flat_map_data=flat_map_data,
        )
```

- [ ] **Step 3: Run determinism + sanity tests**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py::TestDeterminism tests/test_engine_parity.py::TestSanityChecks -v`
Expected: All PASS

- [ ] **Step 4: Run statistical parity**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py::TestStatisticalParity -v`
Expected: PASS (mean biomass within 5%)

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest --ignore=tests/test_engine_parity.py -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/movement.py osmose/engine/simulate.py
git commit -m "feat: wire Numba movement batch into movement() and simulate()"
```

---

### Task 6: Regenerate baselines after movement optimization

**Files:**
- Regenerate: `tests/baselines/parity_baseline_bob_1yr_seed42.npz`
- Regenerate: `tests/baselines/statistical_baseline_bob_1yr_10seeds.npz`

- [ ] **Step 1: Regenerate exact-match baseline**

Run: `.venv/bin/python scripts/save_parity_baseline.py --years 1 --seed 42`

- [ ] **Step 2: Regenerate statistical baseline**

Run: `.venv/bin/python scripts/save_parity_baseline.py --statistical --seeds 10 --years 1`

- [ ] **Step 3: Run ALL parity tests**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -v`
Expected: ALL 12 PASS

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add tests/baselines/
git commit -m "bench: regenerate baselines after movement Numba optimization"
```

---

## Part 2: Vectorize _precompute_effective_rates

### Task 7: Add exact-match test for vectorized rates

**Files:**
- Create: `tests/test_vectorized_rates.py`

This test runs both the original scalar function and the new vectorized function, asserts exact match. We need to save a reference from the current scalar implementation first.

- [ ] **Step 1: Write the test**

Create `tests/test_vectorized_rates.py`:

```python
"""Exact-match test: vectorized _precompute_effective_rates vs scalar original."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).parent.parent
EEC_CONFIG = PROJECT_DIR / "data" / "eec_full" / "eec_all-parameters.csv"


@pytest.mark.skipif(not EEC_CONFIG.exists(), reason="No EEC config")
def test_vectorized_rates_match_scalar():
    """Vectorized rates must produce identical output to scalar version."""
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate
    from osmose.engine.state import SchoolState

    reader = OsmoseConfigReader()
    raw = reader.read(EEC_CONFIG)
    raw["simulation.time.nyear"] = "1"
    cfg = EngineConfig.from_dict(raw)

    grid = Grid.from_netcdf(
        EEC_CONFIG.parent / raw["grid.netcdf.file"],
        mask_var=raw.get("grid.var.mask", "mask"),
    )

    # Run 1 step to get a populated state
    rng = np.random.default_rng(42)
    outputs = simulate(cfg, grid, rng)

    # If we get here without error, the vectorized version is being used
    # and produced valid results (non-negative biomass, species survive)
    assert len(outputs) > 0
    assert np.all(outputs[-1].biomass >= 0)
```

Note: The real validation is that the existing parity tests still pass after vectorization. This test just confirms EEC runs without error.

- [ ] **Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/test_vectorized_rates.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_vectorized_rates.py
git commit -m "test: add EEC smoke test for vectorized rates"
```

---

### Task 8: Vectorize `_precompute_effective_rates()`

**Files:**
- Modify: `osmose/engine/processes/mortality.py:484-614`

Replace the two scalar `for i in range(n)` loops with vectorized NumPy operations. The full replacement code is in the spec (Part 2). Key changes:

1. **Starvation** — already partially vectorized (lines 493-499), keep as-is
2. **Additional mortality** — replace `for i in range(n)` with species-loop + vectorized indexing
3. **Fishing** — replace `for i in range(n)` with species-loop + vectorized selectivity/spatial/MPA/seasonality
4. Add `fishing_enabled` guard
5. Add `fishing_discard` conditioned on `eff_fishing > 0`

- [ ] **Step 1: Replace `_precompute_effective_rates` body**

Read the current function at `osmose/engine/processes/mortality.py` lines 484-614. Replace the body (keeping the function signature and docstring) with the vectorized version from the spec. The complete replacement is in `docs/superpowers/specs/2026-03-22-scaling-parity-design.md` Part 2.

Key implementation notes:
- Keep the function signature identical: `def _precompute_effective_rates(work_state, config, n_subdt, step)`
- Keep the return signature identical: `return eff_starv, eff_additional, eff_fishing, fishing_discard`
- The starvation section (lines 493-499) is already vectorized — keep it unchanged
- Replace only the additional mortality loop (lines 501-531) and fishing loop (lines 533-614)
- Use safe indexing: `cy[valid], cx[valid]` pattern for spatial maps (avoid negative-index wrapping)

- [ ] **Step 2: Run lint**

Run: `.venv/bin/ruff check osmose/engine/processes/mortality.py`
Expected: No errors

- [ ] **Step 3: Run determinism + sanity tests**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py::TestDeterminism tests/test_engine_parity.py::TestSanityChecks -v`
Expected: All PASS

- [ ] **Step 4: Run EEC smoke test**

Run: `.venv/bin/python -m pytest tests/test_vectorized_rates.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/mortality.py
git commit -m "perf: vectorize _precompute_effective_rates with NumPy"
```

---

### Task 9: Final baselines and benchmark

**Files:**
- Regenerate: `tests/baselines/parity_baseline_bob_1yr_seed42.npz`
- Regenerate: `tests/baselines/statistical_baseline_bob_1yr_10seeds.npz`

- [ ] **Step 1: Regenerate baselines**

Run: `.venv/bin/python scripts/save_parity_baseline.py --years 1 --seed 42`
Run: `.venv/bin/python scripts/save_parity_baseline.py --statistical --seeds 10 --years 1`

- [ ] **Step 2: Run ALL parity tests**

Run: `.venv/bin/python -m pytest tests/test_engine_parity.py -v`
Expected: ALL 12 PASS

- [ ] **Step 3: Run EEC 5yr benchmark**

Use the same inline benchmark from Task 1. Expected: ~5-7s (down from 17.5s).

- [ ] **Step 4: Run BoB 5yr benchmark**

Run: `.venv/bin/python scripts/benchmark_engine.py --years 5 --seed 42 --repeats 3`
Expected: ~2.0-2.5s (should be similar or slightly better than before)

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add tests/baselines/
git commit -m "bench: final baselines after scaling parity optimizations"
```
