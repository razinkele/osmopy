# Python Engine Phase 4: Movement — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spatial movement (random walk + map-based distribution), out-of-domain flagging/mortality, and incoming flux. Schools move each timestep based on species-specific distribution maps or random walks.

**Architecture:** Movement is a process function `movement(state, grid, config, step, rng) -> state`. Random walk shifts schools to neighboring cells within a configurable range. Map-based movement (future) uses probability maps per species/age/season. Out-of-domain schools get flagged and receive separate mortality.

**Tech Stack:** Python 3.12+, NumPy. Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-03-18-python-engine-design.md` (Phase 4, lines 525-532; Movement, lines 374-380; Out-of-Domain Mortality, lines 217-225)

---

## File Structure

```
osmose/engine/processes/
    movement.py             # NEW — random walk + map-based movement
osmose/engine/config.py     # MODIFY — add movement config parameters
osmose/engine/simulate.py   # MODIFY — replace _movement stub, add _incoming_flux
tests/
    test_engine_movement.py  # NEW — movement tests (Tier 1)
```

---

### Task 1: Add movement config parameters to EngineConfig

**Files:**
- Modify: `osmose/engine/config.py`
- Modify: `tests/test_engine_config.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine_config.py`:

```python
    def test_movement_method(self, minimal_config):
        minimal_config["movement.distribution.method.sp0"] = "random"
        minimal_config["movement.distribution.method.sp1"] = "maps"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.movement_method == ["random", "maps"]

    def test_random_walk_range(self, minimal_config):
        minimal_config["movement.randomwalk.range.sp0"] = "1"
        minimal_config["movement.randomwalk.range.sp1"] = "2"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.random_walk_range[0] == 1
        assert cfg.random_walk_range[1] == 2

    def test_out_mortality_rate(self, minimal_config):
        minimal_config["mortality.out.rate.sp0"] = "0.1"
        minimal_config["mortality.out.rate.sp1"] = "0.05"
        cfg = EngineConfig.from_dict(minimal_config)
        assert cfg.out_mortality_rate[0] == pytest.approx(0.1)
```

- [ ] **Step 2: Run tests — verify fail**

- [ ] **Step 3: Add fields to EngineConfig**

New dataclass fields:
```python
    # Movement
    movement_method: list[str]                  # "random" or "maps" per species
    random_walk_range: NDArray[np.int32]        # cells per timestep per species
    out_mortality_rate: NDArray[np.float64]     # mortality rate while out-of-domain
```

New extraction in `from_dict()`:
```python
            movement_method=[
                cfg.get(f"movement.distribution.method.sp{i}", "random") for i in range(n_sp)
            ],
            random_walk_range=_species_int_optional(
                cfg, "movement.randomwalk.range.sp{i}", n_sp, default=1
            ),
            out_mortality_rate=_species_float_optional(
                cfg, "mortality.out.rate.sp{i}", n_sp, default=0.0
            ),
```

Add helper `_species_int_optional`:
```python
def _species_int_optional(
    cfg: dict[str, str], pattern: str, n: int, default: int
) -> NDArray[np.int32]:
    return np.array(
        [int(cfg.get(pattern.format(i=i), str(default))) for i in range(n)], dtype=np.int32
    )
```

- [ ] **Step 4: Run tests — verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(engine): add movement parameters to EngineConfig"
```

---

### Task 2: Random walk movement function

**Files:**
- Create: `osmose/engine/processes/movement.py`
- Create: `tests/test_engine_movement.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_engine_movement.py`:

```python
"""Tests for movement process — Tier 1 verification."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.movement import random_walk, movement
from osmose.engine.state import SchoolState


def _make_movement_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "24", "simulation.time.nyear": "1",
        "simulation.nspecies": "1", "simulation.nschool.sp0": "5",
        "species.name.sp0": "TestFish", "species.linf.sp0": "20.0",
        "species.k.sp0": "0.3", "species.t0.sp0": "-0.1",
        "species.egg.size.sp0": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.lifespan.sp0": "3",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "movement.distribution.method.sp0": "random",
        "movement.randomwalk.range.sp0": "2",
    }


class TestRandomWalk:
    def test_schools_move(self):
        """Schools should move to different cells after random walk."""
        grid = Grid.from_dimensions(ny=10, nx=10)
        state = SchoolState.create(n_schools=100, species_id=np.zeros(100, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(100, 5, dtype=np.int32),
            cell_y=np.full(100, 5, dtype=np.int32),
            abundance=np.ones(100),
            age_dt=np.ones(100, dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        walk_range = np.full(100, 2, dtype=np.int32)
        new_state = random_walk(state, grid, walk_range, rng)
        # Not all schools should stay at (5, 5)
        moved = (new_state.cell_x != 5) | (new_state.cell_y != 5)
        assert moved.sum() > 0

    def test_stays_within_grid(self):
        """Schools near edges should not go out of bounds."""
        grid = Grid.from_dimensions(ny=5, nx=5)
        state = SchoolState.create(n_schools=50, species_id=np.zeros(50, dtype=np.int32))
        # Place all at corner (0, 0)
        state = state.replace(
            cell_x=np.zeros(50, dtype=np.int32),
            cell_y=np.zeros(50, dtype=np.int32),
            abundance=np.ones(50),
            age_dt=np.ones(50, dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        walk_range = np.full(50, 3, dtype=np.int32)
        new_state = random_walk(state, grid, walk_range, rng)
        assert np.all(new_state.cell_x >= 0) and np.all(new_state.cell_x < 5)
        assert np.all(new_state.cell_y >= 0) and np.all(new_state.cell_y < 5)

    def test_range_limits_displacement(self):
        """Displacement should be within walk range."""
        grid = Grid.from_dimensions(ny=20, nx=20)
        state = SchoolState.create(n_schools=200, species_id=np.zeros(200, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(200, 10, dtype=np.int32),
            cell_y=np.full(200, 10, dtype=np.int32),
            abundance=np.ones(200),
            age_dt=np.ones(200, dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        walk_range = np.full(200, 1, dtype=np.int32)  # range=1
        new_state = random_walk(state, grid, walk_range, rng)
        dx = np.abs(new_state.cell_x - 10)
        dy = np.abs(new_state.cell_y - 10)
        assert np.all(dx <= 1)
        assert np.all(dy <= 1)

    def test_avoids_land_cells(self):
        """Schools should not land on land cells."""
        mask = np.ones((5, 5), dtype=np.bool_)
        mask[2, 2] = False  # land cell
        grid = Grid(ny=5, nx=5, ocean_mask=mask)
        state = SchoolState.create(n_schools=100, species_id=np.zeros(100, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(100, 2, dtype=np.int32),
            cell_y=np.full(100, 1, dtype=np.int32),  # adjacent to land
            abundance=np.ones(100),
            age_dt=np.ones(100, dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        walk_range = np.full(100, 1, dtype=np.int32)
        new_state = random_walk(state, grid, walk_range, rng)
        # No school should be on the land cell (2, 2)
        on_land = (new_state.cell_x == 2) & (new_state.cell_y == 2)
        assert not on_land.any()
```

- [ ] **Step 2: Run tests — verify fail**
- [ ] **Step 3: Implement `random_walk` in `osmose/engine/processes/movement.py`**

```python
"""Movement process functions for the OSMOSE Python engine.

Random walk and map-based spatial distribution of schools.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.grid import Grid
from osmose.engine.state import SchoolState


def random_walk(
    state: SchoolState,
    grid: Grid,
    walk_range: NDArray[np.int32],
    rng: np.random.Generator,
) -> SchoolState:
    """Move schools by random displacement within walk_range cells.

    Each school is displaced by a random (dx, dy) where
    |dx| <= range and |dy| <= range. Displacement is clamped
    to grid bounds and land cells are avoided.
    """
    if len(state) == 0:
        return state

    n = len(state)
    # Random displacement: uniform integer in [-range, range]
    dx = np.array([rng.integers(-int(r), int(r) + 1) for r in walk_range], dtype=np.int32)
    dy = np.array([rng.integers(-int(r), int(r) + 1) for r in walk_range], dtype=np.int32)

    new_x = np.clip(state.cell_x + dx, 0, grid.nx - 1)
    new_y = np.clip(state.cell_y + dy, 0, grid.ny - 1)

    # Avoid land cells: if new position is land, stay at old position
    on_land = ~grid.ocean_mask[new_y, new_x]
    new_x = np.where(on_land, state.cell_x, new_x)
    new_y = np.where(on_land, state.cell_y, new_y)

    return state.replace(cell_x=new_x, cell_y=new_y)
```

- [ ] **Step 4: Run tests — verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(engine): add random walk movement function"
```

---

### Task 3: Out-of-domain mortality + movement orchestrator

**Files:**
- Modify: `osmose/engine/processes/movement.py`
- Modify: `osmose/engine/processes/natural.py`
- Modify: `tests/test_engine_movement.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine_movement.py`:

```python
from osmose.engine.processes.natural import out_mortality


class TestOutMortality:
    def test_kills_out_of_domain_schools(self):
        """Schools flagged is_out should receive mortality."""
        cfg = EngineConfig.from_dict({
            **_make_movement_config(),
            "mortality.out.rate.sp0": "1.0",
        })
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([5.0, 5.0]),
            is_out=np.array([True, False]),
            age_dt=np.array([10, 10], dtype=np.int32),
        )
        new_state = out_mortality(state, cfg)
        # Out school should have reduced abundance
        assert new_state.abundance[0] < 1000.0
        # In-domain school unchanged
        np.testing.assert_allclose(new_state.abundance[1], 1000.0)

    def test_zero_rate_no_mortality(self):
        cfg = EngineConfig.from_dict(_make_movement_config())
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            is_out=np.array([True]),
        )
        new_state = out_mortality(state, cfg)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)


class TestMovementOrchestrator:
    def test_movement_calls_random_walk(self):
        """Full movement function should move schools."""
        cfg = EngineConfig.from_dict(_make_movement_config())
        grid = Grid.from_dimensions(ny=10, nx=10)
        state = SchoolState.create(n_schools=50, species_id=np.zeros(50, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(50, 5, dtype=np.int32),
            cell_y=np.full(50, 5, dtype=np.int32),
            abundance=np.ones(50),
            age_dt=np.ones(50, dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = movement(state, grid, cfg, step=0, rng=rng)
        moved = (new_state.cell_x != 5) | (new_state.cell_y != 5)
        assert moved.sum() > 0
```

- [ ] **Step 2: Run tests — verify fail**

- [ ] **Step 3: Add `out_mortality` to `natural.py`**

```python
def out_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Apply mortality to out-of-domain schools.

    Java: N_dead = N * (1 - exp(-M_out / n_dt_per_year))
    Applied once per timestep after the main mortality loop.
    """
    if len(state) == 0:
        return state

    out = state.is_out
    if not out.any():
        return state

    sp = state.species_id
    d = config.out_mortality_rate[sp] / config.n_dt_per_year
    mortality_fraction = 1 - np.exp(-d)

    n_dead = np.zeros_like(state.abundance)
    n_dead[out] = state.abundance[out] * mortality_fraction[out]

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.OUT] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)
```

- [ ] **Step 4: Add `movement` orchestrator to `movement.py`**

```python
from osmose.engine.config import EngineConfig


def movement(
    state: SchoolState,
    grid: Grid,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
) -> SchoolState:
    """Move all schools according to their species' movement method.

    Supports random walk (Phase 4) and map-based (future).
    """
    if len(state) == 0:
        return state

    sp = state.species_id
    walk_range = config.random_walk_range[sp]

    # For now all species use random walk
    # Map-based distribution will be added when needed
    state = random_walk(state, grid, walk_range, rng)

    return state
```

- [ ] **Step 5: Run tests — verify pass**
- [ ] **Step 6: Commit**

```bash
git commit -m "feat(engine): add out-of-domain mortality and movement orchestrator"
```

---

### Task 4: Wire movement into simulation loop

**Files:**
- Modify: `osmose/engine/simulate.py`
- Modify: `tests/test_engine_simulate.py`

- [ ] **Step 1: Write failing integration test**

Append to `tests/test_engine_simulate.py`:

```python
    def test_schools_move_during_simulation(self):
        """After simulate, schools should not all be at their initial positions."""
        cfg_dict = {
            "simulation.time.ndtperyear": "12", "simulation.time.nyear": "1",
            "simulation.nspecies": "1", "simulation.nschool.sp0": "10",
            "species.name.sp0": "TestFish", "species.linf.sp0": "20.0",
            "species.k.sp0": "0.3", "species.t0.sp0": "-0.1",
            "species.egg.size.sp0": "0.1",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.lifespan.sp0": "3",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.efficiency.critical.sp0": "0.57",
            "movement.distribution.method.sp0": "random",
            "movement.randomwalk.range.sp0": "2",
            "population.seeding.biomass.sp0": "50000",
        }
        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=10, nx=10)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12
```

- [ ] **Step 2: Run test — verify fail (movement stub returns state unchanged)**

- [ ] **Step 3: Replace stubs in `simulate.py`**

Replace `_movement`:
```python
def _movement(
    state: SchoolState, grid: Grid, config: EngineConfig, step: int, rng: np.random.Generator,
) -> SchoolState:
    """Apply spatial movement."""
    from osmose.engine.processes.movement import movement
    return movement(state, grid, config, step, rng)
```

Replace `_incoming_flux` (simple stub that does nothing for now — will be expanded later):
```python
def _incoming_flux(
    state: SchoolState, config: EngineConfig, step: int, rng: np.random.Generator
) -> SchoolState:
    """Phase 4 stub: incoming flux (migration injection). Full implementation in Phase 7."""
    return state
```

Also wire `out_mortality` into the `_mortality` function after the main loop:
```python
def _mortality(
    state: SchoolState, resources: ResourceState, config: EngineConfig, rng: np.random.Generator,
) -> SchoolState:
    """Apply mortality sources. Phase 4: additional + larva + out-of-domain."""
    from osmose.engine.processes.natural import additional_mortality, larva_mortality, out_mortality

    n_subdt = config.mortality_subdt
    state = larva_mortality(state, config)

    for _sub in range(n_subdt):
        state = additional_mortality(state, config, n_subdt)

    # Out-of-domain mortality (after main loop)
    state = out_mortality(state, config)
    return state
```

- [ ] **Step 4: Run tests — verify all pass**
- [ ] **Step 5: Run full suite**
- [ ] **Step 6: Commit**

```bash
git commit -m "feat(engine): wire movement and out-of-domain mortality into simulation loop"
```

---

### Task 5: Lint, verify, tag

- [ ] **Step 1: Run linter**

Run: `.venv/bin/ruff check osmose/engine/ tests/test_engine_*.py`

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest -q`

- [ ] **Step 3: Commit any fixes**
- [ ] **Step 4: Tag**

```bash
git tag -a engine-phase4 -m "Python engine Phase 4: random walk movement, out-of-domain mortality"
```
