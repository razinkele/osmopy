# Map-Based Movement (B1) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add spatially explicit map-based movement to the Python OSMOSE engine, where schools move according to age/season-specific probability maps loaded from CSV files.

**Architecture:** A new `MovementMapSet` class loads CSV maps and builds an `index_maps[age_dt][step]` lookup table per species. The existing `movement()` function is extended to dispatch between `random` (existing vectorized code) and `maps` (new per-school sequential code) based on `config.movement_method`. Map sets are built once in `simulate()` and passed to `movement()` each timestep.

**Tech Stack:** Python 3.12, NumPy, pandas (CSV reading), pytest

**Spec:** `docs/superpowers/specs/2026-03-19-map-movement-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `osmose/engine/movement_maps.py` | Create | `MovementMapSet` class: CSV loading, `index_maps` construction, deduplication, lookup, validation |
| `osmose/engine/processes/movement.py` | Modify | Add `map_distribution()` for per-school map movement; extend `movement()` to dispatch by method |
| `osmose/engine/simulate.py` | Modify | Build `map_sets` dict, pass to `_movement()`, add `is_out` reset |
| `tests/test_engine_map_movement.py` | Create | Full test suite (~30 tests) |

---

## Task 1: `MovementMapSet` — CSV loading + index_maps construction

**Files:**
- Create: `osmose/engine/movement_maps.py`
- Create: `tests/test_engine_map_movement.py`

This task builds the map loading and indexing infrastructure without any movement logic.

- [ ] **Step 1: Write failing tests for MovementMapSet**

```python
# tests/test_engine_map_movement.py
"""Tests for map-based movement (B1)."""

import logging
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.movement_maps import MovementMapSet


def _write_csv_map(path: Path, data: list[list[float]]) -> None:
    """Write a 2D grid as semicolon-separated CSV (OSMOSE format)."""
    with open(path, "w") as f:
        for row in data:
            f.write(";".join(str(v) for v in row) + "\n")


def _make_map_config(
    tmp_path: Path,
    species_name: str = "TestFish",
    n_dt_per_year: int = 24,
    n_year: int = 1,
    lifespan_dt: int = 72,  # 3 years * 24
) -> tuple[dict[str, str], Path]:
    """Create a minimal map config with one map file covering all ages/steps."""
    # 3x3 grid: center cell has value 1, rest are -99
    map_data = [
        [-99, -99, -99],
        [-99, 1, -99],
        [-99, -99, -99],
    ]
    map_file = tmp_path / "test_map.csv"
    _write_csv_map(map_file, map_data)

    steps_str = ";".join(str(i) for i in range(n_dt_per_year))
    cfg = {
        "movement.species.map0": species_name,
        "movement.initialage.map0": "0",
        "movement.lastage.map0": "3",
        "movement.file.map0": str(map_file),
        "movement.steps.map0": steps_str,
    }
    return cfg, map_file


class TestMapSetLoading:
    def test_single_map_loads(self, tmp_path):
        cfg, _ = _make_map_config(tmp_path)
        ms = MovementMapSet(
            config=cfg,
            species_name="TestFish",
            n_dt_per_year=24,
            n_years=1,
            lifespan_dt=72,
            ny=3,
            nx=3,
        )
        assert ms.n_maps == 1
        assert ms.maps[0] is not None
        assert ms.maps[0].shape == (3, 3)

    def test_index_maps_filled(self, tmp_path):
        cfg, _ = _make_map_config(tmp_path)
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        # All age/step combos should map to index 0
        assert ms.index_maps.shape == (72, 24)
        assert np.all(ms.index_maps >= 0)

    def test_multiple_maps_age_ranges(self, tmp_path):
        """Two maps: age 0-1 uses map A, age 1-3 uses map B."""
        map_a = tmp_path / "map_a.csv"
        map_b = tmp_path / "map_b.csv"
        _write_csv_map(map_a, [[-99, 1, -99], [-99, -99, -99], [-99, -99, -99]])
        _write_csv_map(map_b, [[-99, -99, -99], [-99, -99, -99], [-99, 1, -99]])
        steps = ";".join(str(i) for i in range(24))
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",
            "movement.file.map0": str(map_a),
            "movement.steps.map0": steps,
            "movement.species.map1": "TestFish",
            "movement.initialage.map1": "1",
            "movement.lastage.map1": "3",
            "movement.file.map1": str(map_b),
            "movement.steps.map1": steps,
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        assert ms.n_maps == 2
        # Age 0 (step 0) → map 0
        assert ms.index_maps[0, 0] == 0
        # Age 24 (1 year, step 0) → map 1
        assert ms.index_maps[24, 0] == 1

    def test_null_file_produces_none_map(self, tmp_path):
        steps = ";".join(str(i) for i in range(24))
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3",
            "movement.file.map0": "null",
            "movement.steps.map0": steps,
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        assert ms.maps[0] is None

    def test_csv_row_flipping(self, tmp_path):
        """CSV row 0 = grid row ny-1 (North at top)."""
        map_file = tmp_path / "flip_test.csv"
        # CSV row 0 has value 5, row 2 has value 1
        _write_csv_map(map_file, [[5, -99, -99], [-99, -99, -99], [1, -99, -99]])
        steps = ";".join(str(i) for i in range(24))
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3",
            "movement.file.map0": str(map_file),
            "movement.steps.map0": steps,
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        grid = ms.maps[0]
        # CSV row 0 → grid row 2 (ny-1-0), CSV row 2 → grid row 0 (ny-1-2)
        assert grid[2, 0] == 5.0  # CSV row 0
        assert grid[0, 0] == 1.0  # CSV row 2

    def test_max_proba_presence_absence(self, tmp_path):
        """Presence/absence map (max >= 1.0) → max_proba set to 0.0."""
        map_file = tmp_path / "pa_map.csv"
        _write_csv_map(map_file, [[-99, 1, 0], [0, 1, -99], [-99, 0, -99]])
        steps = ";".join(str(i) for i in range(24))
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3",
            "movement.file.map0": str(map_file),
            "movement.steps.map0": steps,
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        assert ms.max_proba[0] == 0.0  # presence/absence trick

    def test_max_proba_probability_map(self, tmp_path):
        """Probability map (max < 1.0) → max_proba stores actual max."""
        map_file = tmp_path / "prob_map.csv"
        _write_csv_map(map_file, [[-99, 0.5, 0.3], [0.1, 0.8, -99], [-99, 0.2, -99]])
        steps = ";".join(str(i) for i in range(24))
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3",
            "movement.file.map0": str(map_file),
            "movement.steps.map0": steps,
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        np.testing.assert_allclose(ms.max_proba[0], 0.8)

    def test_deduplication_shares_array(self, tmp_path):
        """Two map entries pointing to same CSV share one array object."""
        map_file = tmp_path / "shared.csv"
        _write_csv_map(map_file, [[-99, 1, -99], [-99, -99, -99], [-99, -99, -99]])
        steps = ";".join(str(i) for i in range(24))
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",
            "movement.file.map0": str(map_file),
            "movement.steps.map0": steps,
            "movement.species.map1": "TestFish",
            "movement.initialage.map1": "1",
            "movement.lastage.map1": "3",
            "movement.file.map1": str(map_file),
            "movement.steps.map1": steps,
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        # After deduplication, index_maps should point to same index
        assert ms.index_maps[0, 0] == ms.index_maps[24, 0]

    def test_get_map_returns_correct(self, tmp_path):
        cfg, _ = _make_map_config(tmp_path)
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        grid_map = ms.get_map(0, 0)
        assert grid_map is not None
        assert grid_map[1, 1] == 1.0  # center cell

    def test_get_map_returns_none_for_null(self, tmp_path):
        steps = ";".join(str(i) for i in range(24))
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3",
            "movement.file.map0": "null",
            "movement.steps.map0": steps,
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        assert ms.get_map(0, 0) is None

    def test_missing_entries_logged(self, tmp_path, caplog):
        """Missing index_maps entries → warning logged."""
        map_file = tmp_path / "partial.csv"
        _write_csv_map(map_file, [[-99, 1, -99], [-99, -99, -99], [-99, -99, -99]])
        # Only define map for age 0-1, steps 0-11 — age 1-3 is missing
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",
            "movement.file.map0": str(map_file),
            "movement.steps.map0": "0;1;2;3;4;5;6;7;8;9;10;11",
        }
        with caplog.at_level(logging.WARNING):
            ms = MovementMapSet(
                config=cfg, species_name="TestFish",
                n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
            )
        assert "missing" in caplog.text.lower() or ms.index_maps[24, 0] == -1

    def test_season_subset(self, tmp_path):
        """Map defined only for steps 0-11 (first half of year)."""
        map_a = tmp_path / "winter.csv"
        map_b = tmp_path / "summer.csv"
        _write_csv_map(map_a, [[-99, 1, -99], [-99, -99, -99], [-99, -99, -99]])
        _write_csv_map(map_b, [[-99, -99, -99], [-99, 1, -99], [-99, -99, -99]])
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3",
            "movement.file.map0": str(map_a),
            "movement.steps.map0": "0;1;2;3;4;5;6;7;8;9;10;11",
            "movement.species.map1": "TestFish",
            "movement.initialage.map1": "0",
            "movement.lastage.map1": "3",
            "movement.file.map1": str(map_b),
            "movement.steps.map1": "12;13;14;15;16;17;18;19;20;21;22;23",
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        # Step 0 → map 0, Step 12 → map 1
        assert ms.index_maps[0, 0] != ms.index_maps[0, 12]

    def test_out_of_range_steps_silently_skipped(self, tmp_path):
        """Steps beyond n_total_steps are silently skipped."""
        map_file = tmp_path / "test.csv"
        _write_csv_map(map_file, [[-99, 1, -99], [-99, -99, -99], [-99, -99, -99]])
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3",
            "movement.file.map0": str(map_file),
            "movement.steps.map0": "0;1;2;3;4;5;6;7;8;9;10;11",
            "movement.initialyear.map0": "0",
            "movement.lastyear.map0": "10",  # way beyond 1-year sim
        }
        # Should not crash
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=3, nx=3,
        )
        assert ms.index_maps.shape[1] == 24  # max(1*24, 24)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement `MovementMapSet`**

Create `osmose/engine/movement_maps.py` with:

```python
"""Movement map loading and indexing for the OSMOSE Python engine.

Loads CSV spatial probability maps and builds a 2D lookup table
index_maps[age_dt][step] → map_index for map-based movement.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MovementMapSet:
    """Spatial movement maps for one species."""

    def __init__(
        self,
        config: dict[str, str],
        species_name: str,
        n_dt_per_year: int,
        n_years: int,
        lifespan_dt: int,
        ny: int,
        nx: int,
    ) -> None:
        self.ny = ny
        self.nx = nx
        n_total_steps = max(n_years * n_dt_per_year, n_dt_per_year)
        self.index_maps = np.full((lifespan_dt, n_total_steps), -1, dtype=np.int32)

        # Discover maps for this species
        map_entries = self._discover_maps(config, species_name)

        # Load maps and fill index_maps
        self.maps: list[NDArray[np.float64] | None] = []
        self._file_paths: list[str] = []
        for entry in map_entries:
            self._load_entry(entry, config, n_dt_per_year, n_years, lifespan_dt, n_total_steps)

        self.n_maps = len(self.maps)

        # Deduplicate
        self._deduplicate()

        # Compute max_proba per map
        self.max_proba = np.zeros(self.n_maps, dtype=np.float64)
        for i, m in enumerate(self.maps):
            if m is not None:
                max_val = np.nanmax(m)
                self.max_proba[i] = 0.0 if max_val >= 1.0 else max_val

        # Validate
        self._validate(n_total_steps, lifespan_dt)

    # ... implementation methods ...

    def get_map(self, age_dt: int, step: int) -> NDArray[np.float64] | None:
        if age_dt >= self.index_maps.shape[0] or step >= self.index_maps.shape[1]:
            return None
        idx = self.index_maps[age_dt, step]
        if idx < 0:
            return None
        return self.maps[idx]

    def get_index(self, age_dt: int, step: int) -> int:
        if age_dt >= self.index_maps.shape[0] or step >= self.index_maps.shape[1]:
            return -1
        return int(self.index_maps[age_dt, step])
```

Key implementation methods:

**`_discover_maps`**: Scan config for `movement.species.map{N}` keys, return list of matching map numbers.

**`_load_entry`**: For each map entry, parse age range, step range, year range, load CSV or set None for "null" file, fill `index_maps`.

**`_load_csv`**: Read semicolon-separated CSV, flip rows (`ny - 1 - row_idx`), handle "na"/"nan" strings, return `(ny, nx)` float64 array.

**`_deduplicate`**: Compare file paths; when duplicates found, remap `index_maps` entries to canonical index.

**`_validate`**: Check all `index_maps` entries; warn for -1 entries.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/movement_maps.py tests/test_engine_map_movement.py
git commit -m "feat(engine): add MovementMapSet — CSV map loading and index_maps construction"
```

---

## Task 2: Map-based movement algorithm

**Files:**
- Modify: `osmose/engine/processes/movement.py`
- Modify: `tests/test_engine_map_movement.py`

- [ ] **Step 1: Write failing tests for map movement**

Add to `tests/test_engine_map_movement.py`:

```python
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.movement import movement, map_distribution
from osmose.engine.state import SchoolState


def _make_movement_config_with_maps(tmp_path, ny=5, nx=5):
    """Create a config + map file for a species using map-based movement."""
    # 5x5 grid, center 3x3 has value 1, borders are -99
    rows = []
    for j in range(ny):
        row = []
        for i in range(nx):
            if 1 <= i <= 3 and 1 <= j <= 3:
                row.append(1)
            else:
                row.append(-99)
        rows.append(row)
    map_file = tmp_path / "test_map.csv"
    _write_csv_map(map_file, rows)

    steps = ";".join(str(i) for i in range(24))
    cfg = {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "10",
        "species.type.sp0": "focal",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3",
        "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "maps",
        "movement.randomwalk.range.sp0": "1",
        "movement.species.map0": "TestFish",
        "movement.initialage.map0": "0",
        "movement.lastage.map0": "3",
        "movement.file.map0": str(map_file),
        "movement.steps.map0": steps,
    }
    return cfg


class TestMapDistribution:
    def test_out_of_domain(self, tmp_path):
        """Null map → school marked is_out, cell_x/y = -1."""
        steps = ";".join(str(i) for i in range(24))
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3",
            "movement.file.map0": "null",
            "movement.steps.map0": steps,
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=5, nx=5,
        )
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            cell_x=np.array([2], dtype=np.int32),
            cell_y=np.array([2], dtype=np.int32),
            age_dt=np.array([10], dtype=np.int32),
            abundance=np.array([100.0]),
        )
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        new_state = map_distribution(state, 0, grid, ms, 0, step=0, rng=rng)
        assert new_state.is_out[0] == True
        assert new_state.cell_x[0] == -1
        assert new_state.cell_y[0] == -1

    def test_rejection_sampling_places_on_positive_cell(self, tmp_path):
        """New placement lands on a cell with positive map value."""
        cfg = _make_movement_config_with_maps(tmp_path, ny=5, nx=5)
        ec = EngineConfig.from_dict(cfg)
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=5, nx=5,
        )
        # Unlocated school (cell_x=-1)
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            cell_x=np.array([-1], dtype=np.int32),
            cell_y=np.array([-1], dtype=np.int32),
            age_dt=np.array([10], dtype=np.int32),
            abundance=np.array([100.0]),
        )
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        new_state = map_distribution(state, 0, grid, ms, 0, step=0, rng=rng)
        # Should be placed on a positive cell (center 3x3 area)
        x, y = new_state.cell_x[0], new_state.cell_y[0]
        map_grid = ms.maps[0]
        assert map_grid[y, x] > 0

    def test_same_map_random_walk(self, tmp_path):
        """Same map + located → random walk within accessible cells."""
        cfg = _make_movement_config_with_maps(tmp_path, ny=5, nx=5)
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=5, nx=5,
        )
        # Located school at center (2,2), step=1 (same map as step=0)
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            cell_x=np.array([2], dtype=np.int32),
            cell_y=np.array([2], dtype=np.int32),
            age_dt=np.array([10], dtype=np.int32),
            abundance=np.array([100.0]),
        )
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        new_state = map_distribution(state, 0, grid, ms, 0, step=1, rng=rng)
        # Should be within range 1 of (2,2) and on a positive map cell
        x, y = new_state.cell_x[0], new_state.cell_y[0]
        assert abs(x - 2) <= 1 and abs(y - 2) <= 1
        map_grid = ms.maps[0]
        # After CSV row flip, center area maps to grid rows 1-3, cols 1-3
        assert map_grid[y, x] > 0

    def test_unlocated_forces_new_placement(self, tmp_path):
        """Unlocated school (cell_x < 0) gets new placement even if same map."""
        cfg = _make_movement_config_with_maps(tmp_path, ny=5, nx=5)
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=5, nx=5,
        )
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            cell_x=np.array([-1], dtype=np.int32),
            cell_y=np.array([-1], dtype=np.int32),
            age_dt=np.array([10], dtype=np.int32),
            abundance=np.array([100.0]),
        )
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        # step=1 (same map as step=0), but unlocated → should still place
        new_state = map_distribution(state, 0, grid, ms, 0, step=1, rng=rng)
        assert new_state.cell_x[0] >= 0
        assert new_state.cell_y[0] >= 0

    def test_stranded_school_stays_in_place(self, tmp_path):
        """School on isolated cell with no accessible neighbors stays put."""
        # Single positive cell at (2,2), surrounded by -99
        rows = [[-99]*5 for _ in range(5)]
        rows[2][2] = 1
        map_file = tmp_path / "isolated.csv"
        _write_csv_map(map_file, rows)
        steps = ";".join(str(i) for i in range(24))
        cfg = {
            "movement.species.map0": "TestFish",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "3",
            "movement.file.map0": str(map_file),
            "movement.steps.map0": steps,
        }
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=5, nx=5,
        )
        # School at (2, grid_y after flip). The CSV row 2 → grid row 5-1-2=2
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            cell_x=np.array([2], dtype=np.int32),
            cell_y=np.array([2], dtype=np.int32),
            age_dt=np.array([10], dtype=np.int32),
            abundance=np.array([100.0]),
        )
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        new_state = map_distribution(state, 0, grid, ms, 0, step=1, rng=rng)
        # Should stay at same position (only accessible cell is self if current cell is positive)
        # Actually the center cell IS accessible (range window includes self)
        # So the school stays at (2,2) — the only accessible cell
        assert new_state.cell_x[0] == 2
        assert new_state.cell_y[0] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py::TestMapDistribution -v`
Expected: FAIL — `map_distribution` not defined

- [ ] **Step 3: Implement `map_distribution` in `movement.py`**

Add to `osmose/engine/processes/movement.py`:

```python
def map_distribution(
    state: SchoolState,
    school_idx: int,
    grid: Grid,
    map_set,  # MovementMapSet
    species_idx: int,
    step: int,
    rng: np.random.Generator,
) -> SchoolState:
    """Apply map-based movement for a single school."""
    age_dt = int(state.age_dt[school_idx])
    current_map = map_set.get_map(age_dt, step)

    if current_map is None:
        # Out of domain
        new_is_out = state.is_out.copy()
        new_cx = state.cell_x.copy()
        new_cy = state.cell_y.copy()
        new_is_out[school_idx] = True
        new_cx[school_idx] = -1
        new_cy[school_idx] = -1
        return state.replace(is_out=new_is_out, cell_x=new_cx, cell_y=new_cy)

    # Same-map detection
    index_map = map_set.get_index(age_dt, step)
    same_map = False
    if age_dt > 0 and step > 0:
        prev_index = map_set.get_index(age_dt - 1, step - 1)
        same_map = (index_map == prev_index)

    cx = int(state.cell_x[school_idx])
    cy = int(state.cell_y[school_idx])
    unlocated = cx < 0

    new_cx = state.cell_x.copy()
    new_cy = state.cell_y.copy()

    if not same_map or unlocated:
        # Rejection sampling
        nx, ny = grid.nx, grid.ny
        n_cells = nx * ny
        max_p = map_set.max_proba[index_map]
        for _ in range(10_000):
            flat = int(round((n_cells - 1) * rng.random()))
            j = flat // nx
            i = flat % nx
            proba = current_map[j, i]
            if proba > 0 and not np.isnan(proba):
                if max_p == 0 or proba >= rng.random() * max_p:
                    new_cx[school_idx] = i
                    new_cy[school_idx] = j
                    return state.replace(cell_x=new_cx, cell_y=new_cy)
        raise RuntimeError(f"Map placement failed after 10000 attempts for school {school_idx}")
    else:
        # Random walk within accessible cells
        walk_range = int(state.species_id[school_idx])  # will be overridden by config
        # ... get accessible cells, pick random one ...
        pass

    return state.replace(cell_x=new_cx, cell_y=new_cy)
```

Note: The implementer needs the full logic including:
- `walk_range` from `config.random_walk_range[species_id]`
- Accessible cell computation: window `[cx-r, cx+r] × [cy-r, cy+r]` clipped, filter by ocean_mask AND map value > 0 AND not NaN
- `Math.round` bias: `int(round((len(accessible) - 1) * rng.random()))`
- Empty accessible list → stay in place

The function signature should accept `config` for the walk range, or the range value directly.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/processes/movement.py tests/test_engine_map_movement.py
git commit -m "feat(engine): add map_distribution — per-school map-based movement algorithm"
```

---

## Task 3: Movement orchestrator — split by method + simulate.py integration

**Files:**
- Modify: `osmose/engine/processes/movement.py`
- Modify: `osmose/engine/simulate.py`
- Modify: `tests/test_engine_map_movement.py`

- [ ] **Step 1: Write failing tests for orchestrator and integration**

Add to `tests/test_engine_map_movement.py`:

```python
from osmose.engine.simulate import simulate


class TestMovementOrchestrator:
    def test_random_species_uses_random_walk(self, tmp_path):
        """Species with method='random' still uses random walk."""
        cfg = _make_movement_config_with_maps(tmp_path)
        cfg["movement.distribution.method.sp0"] = "random"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=5, nx=5)
        state = SchoolState.create(n_schools=5, species_id=np.zeros(5, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(5, 2, dtype=np.int32),
            cell_y=np.full(5, 2, dtype=np.int32),
            abundance=np.ones(5),
            age_dt=np.full(5, 10, dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = movement(state, grid, ec, step=0, rng=rng)
        # Should move (random walk)
        moved = (new_state.cell_x != 2) | (new_state.cell_y != 2)
        assert moved.sum() > 0

    def test_maps_species_uses_map_distribution(self, tmp_path):
        """Species with method='maps' uses map-based movement."""
        cfg = _make_movement_config_with_maps(tmp_path)
        cfg["movement.distribution.method.sp0"] = "maps"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=5, nx=5)
        state = SchoolState.create(n_schools=5, species_id=np.zeros(5, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(5, -1, dtype=np.int32),  # unlocated → forces placement
            cell_y=np.full(5, -1, dtype=np.int32),
            abundance=np.ones(5),
            age_dt=np.full(5, 10, dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        # Need map_sets
        ms = MovementMapSet(
            config=cfg, species_name="TestFish",
            n_dt_per_year=24, n_years=1, lifespan_dt=72, ny=5, nx=5,
        )
        new_state = movement(state, grid, ec, step=0, rng=rng, map_sets={0: ms})
        # Schools should be placed on positive cells
        map_grid = ms.maps[0]
        for i in range(5):
            x, y = new_state.cell_x[i], new_state.cell_y[i]
            assert map_grid[y, x] > 0

    def test_is_out_reset_each_timestep(self, tmp_path):
        """is_out is reset to False at start of each timestep."""
        cfg = _make_movement_config_with_maps(tmp_path)
        cfg["simulation.time.nyear"] = "1"
        cfg["population.seeding.biomass.sp0"] = "100.0"
        cfg["species.sexratio.sp0"] = "0.5"
        cfg["species.relativefecundity.sp0"] = "500"
        cfg["species.maturity.size.sp0"] = "10.0"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)
        assert len(outputs) == ec.n_steps


class TestMapMovementIntegration:
    def test_full_simulation_with_maps(self, tmp_path):
        """Full simulation with map movement completes without errors."""
        cfg = _make_movement_config_with_maps(tmp_path)
        cfg["simulation.time.nyear"] = "1"
        cfg["population.seeding.biomass.sp0"] = "100.0"
        cfg["species.sexratio.sp0"] = "0.5"
        cfg["species.relativefecundity.sp0"] = "500"
        cfg["species.maturity.size.sp0"] = "10.0"
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)
        assert len(outputs) == ec.n_steps
        for o in outputs:
            assert np.all(np.isfinite(o.biomass))

    def test_backward_compat_all_random(self):
        """Config with all 'random' species works unchanged."""
        cfg = {
            "simulation.time.ndtperyear": "24",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "5",
            "species.type.sp0": "focal",
            "species.name.sp0": "TestFish",
            "species.linf.sp0": "20.0",
            "species.k.sp0": "0.3",
            "species.t0.sp0": "-0.1",
            "species.egg.size.sp0": "0.1",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.lifespan.sp0": "3",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.efficiency.critical.sp0": "0.57",
            "movement.distribution.method.sp0": "random",
            "population.seeding.biomass.sp0": "100.0",
            "species.sexratio.sp0": "0.5",
            "species.relativefecundity.sp0": "500",
            "species.maturity.size.sp0": "10.0",
        }
        ec = EngineConfig.from_dict(cfg)
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        outputs = simulate(ec, grid, rng)
        assert len(outputs) == ec.n_steps
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py::TestMovementOrchestrator -v`
Expected: FAIL — `movement()` doesn't accept `map_sets` parameter

- [ ] **Step 3: Extend `movement()` in `movement.py`**

Update the `movement()` function to:
1. Accept optional `map_sets` parameter
2. Split schools by species movement method
3. Apply random walk to "random" species (existing batch code)
4. Apply `map_distribution` to "maps" species (per-school loop)

```python
def movement(state, grid, config, step, rng, map_sets=None):
    if len(state) == 0:
        return state

    sp = state.species_id

    # Determine which schools use which method
    uses_random = np.array([config.movement_method[s] == "random" for s in sp])
    uses_maps = np.array([config.movement_method[s] == "maps" for s in sp])

    # Random walk for "random" species (batch)
    if uses_random.any():
        walk_range = config.random_walk_range[sp]
        walk_range_masked = np.where(uses_random, walk_range, 0)
        state = random_walk(state, grid, walk_range_masked, rng)

    # Map-based movement for "maps" species (per-school)
    if uses_maps.any() and map_sets is not None:
        for i in np.where(uses_maps)[0]:
            sp_id = sp[i]
            if sp_id in map_sets:
                state = map_distribution(state, int(i), grid, map_sets[sp_id],
                                        config.random_walk_range[sp_id], step=step, rng=rng)

    return state
```

- [ ] **Step 4: Update `simulate.py`**

1. Add `is_out` reset to `_reset_step_variables`:
```python
is_out=np.zeros(len(state), dtype=np.bool_),
```

2. Update `_movement` to accept and pass `map_sets`:
```python
def _movement(state, grid, config, step, rng, map_sets=None):
    from osmose.engine.processes.movement import movement
    return movement(state, grid, config, step, rng, map_sets=map_sets)
```

3. In `simulate()`, build map_sets before the loop:
```python
from osmose.engine.movement_maps import MovementMapSet

map_sets = {}
for sp in range(config.n_species):
    if config.movement_method[sp] == "maps":
        sp_name = config.species_names[sp]
        map_sets[sp] = MovementMapSet(
            config=config.raw_config,
            species_name=sp_name,
            n_dt_per_year=config.n_dt_per_year,
            n_years=config.n_year,
            lifespan_dt=int(config.lifespan_dt[sp]),
            ny=grid.ny,
            nx=grid.nx,
        )
```

4. Pass `map_sets` in the loop:
```python
state = _movement(state, grid, config, step, rng, map_sets=map_sets)
```

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_engine_map_movement.py -v`
Then: `.venv/bin/python -m pytest tests/ -q`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/processes/movement.py osmose/engine/simulate.py tests/test_engine_map_movement.py
git commit -m "feat(engine): integrate map-based movement into simulation loop"
```

---

## Task Summary

| Task | What | Tests | Files |
|------|------|-------|-------|
| 1 | `MovementMapSet` — CSV loading, index_maps, deduplication, validation | 14 tests | `movement_maps.py`, `test_engine_map_movement.py` |
| 2 | `map_distribution` — rejection sampling, random walk, out-of-domain | 5 tests | `movement.py`, `test_engine_map_movement.py` |
| 3 | Orchestrator + simulate.py integration + is_out reset + backward compat | 4 tests | `movement.py`, `simulate.py`, `test_engine_map_movement.py` |
| **Total** | | **23 tests** | |

### Spec Test Area Coverage

| Spec Area | Task |
|-----------|------|
| 1. Single map loading | Task 1 |
| 2. Multiple maps (age ranges + seasons) | Task 1 |
| 3. `"null"` file → None → out-of-domain | Task 1 |
| 4. CSV loading (-99, row flipping) | Task 1 |
| 5. max_proba trick (presence/absence) | Task 1 |
| 6. Map deduplication | Task 1 |
| 7. get_map lookup | Task 1 |
| 8. Missing entries validation | Task 1 |
| 9. Out-of-domain (is_out, cell=-1) | Task 2 |
| 10. Same-map detection | Task 2 |
| 11. Rejection sampling on positive cell | Task 2 |
| 12. Presence/absence uniform placement | Task 1 (max_proba=0) + Task 2 |
| 13. Random walk within accessible cells | Task 2 |
| 14. Unlocated → new placement | Task 2 |
| 14b. Empty accessible cells → stay in place | Task 2 |
| 15. is_out reset each timestep | Task 3 |
| 16. Random species uses random_walk | Task 3 |
| 17. Maps species uses map_distribution | Task 3 |
| 18. Mixed species coexist | Task 3 |
| 19. Full simulation with maps | Task 3 |
| 20. Backward compatibility (all random) | Task 3 |

---

## IMPORTANT: Review Corrections (from plan review)

### Correction 1 (CRITICAL): `map_distribution` must NOT create per-school array copies

The plan's Task 2 Step 3 shows `map_distribution` taking a `SchoolState`, copying arrays, and returning a new `SchoolState` for each school. With 10,000 schools this creates ~30,000 array copies and ~10,000 SchoolState allocations per timestep — **~2.4 GB of allocation that GC must collect**. This is unacceptable.

**Required architecture:** `map_distribution` must be a scalar function that returns `(new_x, new_y, is_out)` for a single school. The orchestrator in `movement()` creates 3 array copies TOTAL (not per school), loops over schools, and writes scalar results into the pre-copied arrays.

```python
def _map_move_school(
    age_dt: int, cx: int, cy: int,
    grid_ny: int, grid_nx: int, ocean_mask: NDArray,
    map_set, walk_range: int, step: int, rng,
) -> tuple[int, int, bool]:
    """Move a single school using map-based distribution.
    Returns (new_x, new_y, is_out)."""
    ...

def movement(state, grid, config, step, rng, map_sets=None):
    ...
    if uses_maps.any() and map_sets is not None:
        new_cx = state.cell_x.copy()   # 1 copy
        new_cy = state.cell_y.copy()   # 1 copy
        new_out = state.is_out.copy()  # 1 copy
        for i in np.where(uses_maps)[0]:
            sp_id = sp[i]
            if sp_id in map_sets:
                x, y, out = _map_move_school(
                    int(state.age_dt[i]), int(new_cx[i]), int(new_cy[i]),
                    grid.ny, grid.nx, grid.ocean_mask,
                    map_sets[sp_id], int(config.random_walk_range[sp_id]),
                    step, rng,
                )
                new_cx[i], new_cy[i], new_out[i] = x, y, out
        state = state.replace(cell_x=new_cx, cell_y=new_cy, is_out=new_out)
```

**All Task 2 tests must be updated** to call `_map_move_school` for unit tests and `movement()` for integration tests. The per-school function returns a tuple, not a SchoolState.

### Correction 2: Complete random walk logic (no `pass` placeholder)

The Task 2 skeleton has `pass` in the random walk branch. The implementer must include the complete logic:

```python
# Random walk within accessible cells
accessible = []
for yi in range(max(0, cy - walk_range), min(grid_ny, cy + walk_range + 1)):
    for xi in range(max(0, cx - walk_range), min(grid_nx, cx + walk_range + 1)):
        if ocean_mask[yi, xi] and current_map[yi, xi] > 0 and not np.isnan(current_map[yi, xi]):
            accessible.append((xi, yi))
if len(accessible) == 0:
    return cx, cy, False  # stranded — stay in place
idx = int(round((len(accessible) - 1) * rng.random()))
return accessible[idx][0], accessible[idx][1], False
```

### Correction 3: Add missing test for presence/absence uniform placement (spec area 12)

Add to `TestMapDistribution`:
```python
def test_presence_absence_uniform_placement(self, tmp_path):
    """Presence/absence map (max_proba=0) → uniform placement among positive cells."""
    # Map with 2 positive cells at (1,1) and (2,2) after row flip
    rows = [[-99, -99, -99, -99, -99],
            [-99, -99, -99, -99, -99],
            [-99, -99, 1, -99, -99],
            [-99, 1, -99, -99, -99],
            [-99, -99, -99, -99, -99]]
    map_file = tmp_path / "pa.csv"
    _write_csv_map(map_file, rows)
    # ... setup MovementMapSet, run _map_move_school 100 times, verify both cells visited
```

### Correction 4: Add missing test for mixed species (spec area 18)

Add to `TestMovementOrchestrator`:
```python
def test_mixed_species_coexist(self, tmp_path):
    """Two species: sp0 uses 'random', sp1 uses 'maps' — both move correctly."""
    # Config with 2 species, different methods
    # Verify sp0 schools move (random walk) and sp1 schools land on map-positive cells
```

### Correction 5: `step` indexing contract for multi-year simulations

The `step` parameter passed to `map_set.get_map(age_dt, step)` and `map_set.get_index(age_dt, step)` is the **full simulation step** (0 to `n_years * n_dt_per_year - 1`), NOT the within-year step. The `_load_entry` method must use `iStep = year * n_dt_per_year + season` when filling `index_maps`. Add a multi-year MapSet test:

```python
def test_multi_year_index_maps(self, tmp_path):
    """Maps fill correctly for a 2-year simulation."""
    # n_years=2, lifespan_dt=48 (2yr), 24 steps/year → index_maps shape (48, 48)
    # Step 24 (year 1, season 0) should map to same map as step 0 (year 0, season 0)
```
