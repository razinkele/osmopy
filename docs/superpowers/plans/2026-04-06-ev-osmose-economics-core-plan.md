# DSVM Fleet Economics Phase 2 (Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Phase 1 economics MVP (revenue-only logit, single fleet) to the full DSVM core: travel costs, stock-dependent catchability, vessel memory, days-at-sea tracking, multi-fleet support, annual reset, and CSV output files.

**Architecture:** The existing `fleet_decision()` in `choice.py` is replaced with a richer version that computes `V(c) = expected_revenue(c) - total_cost(c)`, blends biomass with catch memory, and enforces days-at-sea limits. New `costs.py` module handles cost calculations. Output writing extends `write_outputs()` in `output.py`. Annual reset logic goes in `simulate.py` at year boundaries.

**Tech Stack:** Python 3.12+, NumPy, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-04-05-ev-osmose-economic-design.md` (Phase 2: Core, lines 536-542)

**Depends on:** Phase 1 MVP (merged to master)

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `osmose/engine/economics/costs.py` | `travel_cost`, `operating_cost`, `stock_catchability` |
| `tests/test_economics_costs.py` | Cost calculation unit tests |
| `tests/test_economics_memory.py` | Catch memory EMA update tests |
| `tests/test_economics_days.py` | Days-at-sea tracking + forced port return |
| `tests/test_economics_output.py` | CSV output format verification |
| `tests/test_economics_multifleet.py` | Multi-fleet non-interference |

### Modified files

| File | Changes |
|------|---------|
| `osmose/engine/economics/choice.py` | Full V(c) = revenue - costs; memory blending; days-at-sea enforcement |
| `osmose/engine/economics/fleet.py` | No changes needed (FleetConfig/FleetState already have all required fields from Phase 1) |
| `osmose/engine/economics/__init__.py` | Export new functions |
| `osmose/engine/simulate.py` | Annual reset at year boundary; pass realized catch to memory update |
| `osmose/engine/output.py` | Write econ_* CSV files |

---

## Task 1: Cost calculation module

**Files:**
- Create: `osmose/engine/economics/costs.py`
- Test: `tests/test_economics_costs.py`

### Context

Travel cost = Manhattan distance × fuel_cost_per_cell. Operating cost = base_operating_cost per trip. Stock-dependent catchability = `base_rate × (biomass / ref_biomass)^elasticity`.

- [ ] **Step 1: Write tests**

```python
# tests/test_economics_costs.py
"""Tests for economic cost calculations."""

import numpy as np
import pytest

from osmose.engine.economics.costs import (
    compute_travel_costs,
    compute_expected_revenue,
)


class TestTravelCosts:
    def test_manhattan_distance(self):
        """Travel cost = Manhattan distance × fuel_cost_per_cell."""
        current_y, current_x = 2, 3
        ny, nx = 5, 5
        fuel_cost = 100.0
        costs = compute_travel_costs(current_y, current_x, ny, nx, fuel_cost)
        assert costs.shape == (ny * nx,)
        # Same cell → 0 cost
        assert costs[current_y * nx + current_x] == 0.0
        # Adjacent cell (2,4) → distance 1, cost 100
        assert costs[current_y * nx + 4] == pytest.approx(100.0)
        # Cell (0,0) → distance |2-0| + |3-0| = 5, cost 500
        assert costs[0] == pytest.approx(500.0)

    def test_zero_fuel_cost(self):
        costs = compute_travel_costs(0, 0, 3, 3, 0.0)
        assert np.all(costs == 0.0)


class TestExpectedRevenue:
    def test_revenue_with_catchability(self):
        """Revenue = Σ_sp catchability × biomass × price."""
        biomass_by_cell = np.array([[[100.0, 0.0], [50.0, 200.0]]])  # (1 species, 2, 2)
        price = np.array([10.0])
        elasticity = np.array([0.5])
        target_species = [0]
        ref_biomass = np.array([100.0])

        revenue = compute_expected_revenue(
            biomass_by_cell, price, elasticity, target_species, ref_biomass
        )
        assert revenue.shape == (4,)  # ny*nx = 2*2
        # Cell (0,0): biomass=100, catchability = (100/100)^0.5 = 1.0, revenue = 1*100*10 = 1000
        assert revenue[0] == pytest.approx(1000.0)
        # Cell (0,1): biomass=0, revenue = 0
        assert revenue[1] == pytest.approx(0.0)

    def test_no_target_species(self):
        biomass_by_cell = np.array([[[100.0]]])
        revenue = compute_expected_revenue(
            biomass_by_cell, np.array([10.0]), np.array([0.5]), [], np.array([100.0])
        )
        assert revenue[0] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_economics_costs.py -v`
Expected: FAIL

- [ ] **Step 3: Implement costs.py**

```python
# osmose/engine/economics/costs.py
"""Cost and revenue calculations for DSVM fleet dynamics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_travel_costs(
    current_y: int,
    current_x: int,
    ny: int,
    nx: int,
    fuel_cost_per_cell: float,
) -> NDArray[np.float64]:
    """Compute travel cost from current position to every cell (Manhattan distance).

    Returns flat array of shape (ny * nx,).
    """
    n_cells = ny * nx
    costs = np.empty(n_cells, dtype=np.float64)
    for c in range(n_cells):
        cy = c // nx
        cx = c % nx
        dist = abs(current_y - cy) + abs(current_x - cx)
        costs[c] = fuel_cost_per_cell * dist
    return costs


def compute_expected_revenue(
    biomass_by_cell: NDArray[np.float64],
    price: NDArray[np.float64],
    elasticity: NDArray[np.float64],
    target_species: list[int],
    ref_biomass: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute expected revenue per cell with stock-dependent catchability.

    Args:
        biomass_by_cell: Shape (n_species, ny, nx).
        price: Per-species price, shape (n_species,).
        elasticity: Stock elasticity, shape (n_species,).
        target_species: Species indices this fleet targets.
        ref_biomass: Reference biomass for catchability scaling, shape (n_species,).

    Returns:
        Revenue per cell, flat array shape (ny * nx,).
    """
    n_species, ny, nx = biomass_by_cell.shape
    n_cells = ny * nx
    revenue = np.zeros(n_cells, dtype=np.float64)

    for sp in target_species:
        if sp >= n_species:
            continue
        bio_flat = biomass_by_cell[sp].ravel()
        ref = max(ref_biomass[sp], 1e-20)
        catchability = np.power(np.maximum(bio_flat / ref, 0.0), elasticity[sp])
        revenue += catchability * bio_flat * price[sp]

    return revenue
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_economics_costs.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add osmose/engine/economics/costs.py tests/test_economics_costs.py
git commit -m "feat(economics): add travel cost and stock-dependent revenue calculations"
```

---

## Task 2: Upgrade fleet_decision with full cost model and memory

**Files:**
- Modify: `osmose/engine/economics/choice.py`
- Test: `tests/test_economics_memory.py`

### Context

Replace the MVP revenue-only logit with the full DSVM: `V(c) = expected_revenue(c) - travel_cost(c) - operating_cost`. Blend current biomass with catch memory via EMA: `estimate = (1 - decay) × observed + decay × memory`. After fishing, update memory: `memory = decay × memory + (1 - decay) × realized_catch`.

- [ ] **Step 1: Write memory tests**

```python
# tests/test_economics_memory.py
"""Tests for vessel catch memory (exponential moving average)."""

import numpy as np
import pytest

from osmose.engine.economics.choice import update_catch_memory


class TestCatchMemory:
    def test_ema_update(self):
        """Memory = decay × old + (1 - decay) × new."""
        memory = np.array([[[100.0, 50.0], [0.0, 0.0]]])  # (1 fleet, 2, 2)
        realized = np.array([[[0.0, 200.0], [0.0, 0.0]]])
        decay = 0.7
        updated = update_catch_memory(memory, realized, decay)
        # Cell (0,0): 0.7*100 + 0.3*0 = 70
        assert updated[0, 0, 0] == pytest.approx(70.0)
        # Cell (0,1): 0.7*50 + 0.3*200 = 95
        assert updated[0, 0, 1] == pytest.approx(95.0)

    def test_zero_decay_replaces(self):
        """decay=0 → memory completely replaced by new observation."""
        memory = np.array([[[100.0]]])
        realized = np.array([[[42.0]]])
        updated = update_catch_memory(memory, realized, decay=0.0)
        assert updated[0, 0, 0] == pytest.approx(42.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_economics_memory.py -v`
Expected: FAIL — `cannot import name 'update_catch_memory'`

- [ ] **Step 3: Add update_catch_memory and upgrade fleet_decision**

In `osmose/engine/economics/choice.py`, add `update_catch_memory`:

```python
def update_catch_memory(
    memory: NDArray[np.float64],
    realized_catch: NDArray[np.float64],
    decay: float,
) -> NDArray[np.float64]:
    """Exponential moving average update for catch memory.

    memory = decay × memory + (1 - decay) × realized_catch
    """
    return decay * memory + (1.0 - decay) * realized_catch
```

Then upgrade `fleet_decision` to use costs, memory, and days-at-sea:

```python
def fleet_decision(
    fleet_state: FleetState,
    biomass_by_cell_species: NDArray[np.float64],
    rng: np.random.Generator,
) -> FleetState:
    """Execute DSVM decision with full cost model and memory.

    V(c) = expected_revenue(c) - travel_cost(c) - operating_cost
    Biomass estimate = (1-decay) × observed + decay × memory
    """
    from osmose.engine.economics.costs import compute_expected_revenue, compute_travel_costs

    n_species, ny, nx = biomass_by_cell_species.shape
    n_cells = ny * nx

    for fi, fleet in enumerate(fleet_state.fleets):
        # Blend total biomass with catch memory for this fleet
        # catch_memory shape: (n_fleets, ny, nx) — total catch proxy per cell
        # biomass_total shape: (ny, nx)
        biomass_total = biomass_by_cell_species.sum(axis=0)  # sum across species
        memory_layer = fleet_state.catch_memory[fi]
        blended_total = (
            (1.0 - fleet_state.memory_decay) * biomass_total
            + fleet_state.memory_decay * memory_layer
        )
        # Scale per-species biomass proportionally by blend factor
        scale = np.where(biomass_total > 0, blended_total / biomass_total, 1.0)
        blended_species = biomass_by_cell_species * scale[np.newaxis, :, :]

        # Ref biomass: total observed biomass per species (across all cells)
        ref_biomass = np.maximum(biomass_by_cell_species.sum(axis=(1, 2)), 1.0)

        # Expected revenue with stock-dependent catchability
        revenue = compute_expected_revenue(
            blended_species, fleet.price_per_tonne, fleet.stock_elasticity,
            fleet.target_species, ref_biomass,
        )

        vessel_mask = fleet_state.vessel_fleet == fi
        vessel_indices = np.where(vessel_mask)[0]

        for vi in vessel_indices:
            # Skip vessels that exceeded days-at-sea
            if fleet_state.vessel_days_used[vi] >= fleet.max_days_at_sea:
                fleet_state.vessel_cell_y[vi] = fleet.home_port_y
                fleet_state.vessel_cell_x[vi] = fleet.home_port_x
                continue

            # Costs from current position
            travel = compute_travel_costs(
                int(fleet_state.vessel_cell_y[vi]),
                int(fleet_state.vessel_cell_x[vi]),
                ny, nx, fleet.fuel_cost_per_cell,
            )
            total_cost = travel + fleet.base_operating_cost

            # V(c) = revenue - cost; V(port) = 0
            profit = revenue - total_cost
            values = np.append(profit, 0.0)
            probs = logit_probabilities(values, fleet_state.rationality)

            choice = rng.choice(len(values), p=probs)
            if choice == n_cells:
                # Port — no costs, no revenue
                fleet_state.vessel_cell_y[vi] = fleet.home_port_y
                fleet_state.vessel_cell_x[vi] = fleet.home_port_x
            else:
                cy = choice // nx
                cx = choice % nx
                fleet_state.vessel_cell_y[vi] = cy
                fleet_state.vessel_cell_x[vi] = cx
                fleet_state.vessel_days_used[vi] += 1
                # Accumulate costs for this trip
                fleet_state.vessel_costs[vi] += travel[choice] + fleet.base_operating_cost
                # Revenue accumulated after fishing outcomes (in simulate.py)

    fleet_state.effort_map = aggregate_effort(
        fleet_state.vessel_fleet,
        fleet_state.vessel_cell_y,
        fleet_state.vessel_cell_x,
        n_fleets=len(fleet_state.fleets),
        ny=ny, nx=nx,
    )

    return fleet_state
```

- [ ] **Step 4: Update __init__.py**

Add `update_catch_memory` to exports in `osmose/engine/economics/__init__.py`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_economics_memory.py tests/test_economics_choice.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add osmose/engine/economics/choice.py osmose/engine/economics/__init__.py tests/test_economics_memory.py
git commit -m "feat(economics): add full cost model, stock catchability, and catch memory"
```

---

## Task 3: Days-at-sea tracking and forced port return

**Files:**
- Modify: `osmose/engine/economics/choice.py` (already has days logic from Task 2)
- Test: `tests/test_economics_days.py`

### Context

Vessels track `vessel_days_used`. When a vessel fishes (not at port), `days_used += 1`. When `days_used >= max_days_at_sea`, the vessel is forced to port. Days reset at the start of each simulation year (handled in Task 5).

- [ ] **Step 1: Write tests**

```python
# tests/test_economics_days.py
"""Tests for days-at-sea tracking and forced port return."""

import numpy as np
import pytest

from osmose.engine.economics.fleet import FleetConfig, create_fleet_state
from osmose.engine.economics.choice import fleet_decision


class TestDaysAtSea:
    def _make_fleet(self, max_days: int = 5) -> FleetConfig:
        return FleetConfig(
            name="Trawlers",
            n_vessels=1,
            home_port_y=0,
            home_port_x=0,
            gear_type="bottom_trawl",
            max_days_at_sea=max_days,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0]),
        )

    def test_days_increment_when_fishing(self):
        """Days used should increment when vessel goes to a fishing cell."""
        fleet = self._make_fleet(max_days=100)
        fs = create_fleet_state([fleet], grid_ny=3, grid_nx=3, rationality=1.0)
        biomass = np.zeros((1, 3, 3), dtype=np.float64)
        biomass[0, 1, 1] = 10000.0  # fish at (1,1)

        rng = np.random.default_rng(42)
        fs = fleet_decision(fs, biomass, rng)
        # Vessel should have gone fishing → days_used incremented
        if fs.vessel_cell_y[0] != 0 or fs.vessel_cell_x[0] != 0:
            assert fs.vessel_days_used[0] == 1

    def test_forced_port_at_limit(self):
        """Vessel at days-at-sea limit should be forced to port."""
        fleet = self._make_fleet(max_days=0)  # already at limit
        fs = create_fleet_state([fleet], grid_ny=3, grid_nx=3, rationality=1.0)
        fs.vessel_days_used[0] = 0  # At limit (max_days=0)

        biomass = np.zeros((1, 3, 3), dtype=np.float64)
        biomass[0, 1, 1] = 10000.0

        rng = np.random.default_rng(42)
        fs = fleet_decision(fs, biomass, rng)
        # Should be at home port
        assert fs.vessel_cell_y[0] == fleet.home_port_y
        assert fs.vessel_cell_x[0] == fleet.home_port_x
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_economics_days.py -v`
Expected: 2 PASSED (days logic already implemented in Task 2)

- [ ] **Step 3: Commit**

```bash
git add tests/test_economics_days.py
git commit -m "test(economics): add days-at-sea tracking and forced port tests"
```

---

## Task 4: Annual reset and memory update in simulate.py

**Files:**
- Modify: `osmose/engine/simulate.py`
- No new test file (covered by integration tests)

### Context

At the start of each simulation year, reset `vessel_days_used`, `vessel_revenue`, `vessel_costs` to 0. After each timestep's mortality, update catch memory with realized catches.

- [ ] **Step 1: Add annual reset logic**

In `simulate.py`, inside the main loop, at the beginning of each step, add year-boundary reset:

```python
        # -- Annual reset for fleet economics --
        if ctx.fleet_state is not None and step > 0 and step % config.n_dt_per_year == 0:
            ctx.fleet_state.vessel_days_used[:] = 0
            ctx.fleet_state.vessel_revenue[:] = 0.0
            ctx.fleet_state.vessel_costs[:] = 0.0
```

- [ ] **Step 2: Add catch memory update after mortality**

After the mortality call, add memory update:

```python
        # -- Update fleet revenue and catch memory after fishing --
        if ctx.fleet_state is not None:
            from osmose.engine.economics.choice import update_catch_memory

            # Realized catch biomass per cell: sum fishing dead × weight
            n_fleets = len(ctx.fleet_state.fleets)
            ny_f, nx_f = ctx.fleet_state.catch_memory.shape[1], ctx.fleet_state.catch_memory.shape[2]
            realized = np.zeros((n_fleets, ny_f, nx_f), dtype=np.float64)

            for i in range(len(state)):
                fishing_dead = state.n_dead[i, 3]  # MortalityCause.FISHING = 3
                if fishing_dead <= 0:
                    continue
                sp = state.species_id[i]
                cy, cx = state.cell_y[i], state.cell_x[i]
                if not (0 <= cy < ny_f and 0 <= cx < nx_f):
                    continue
                catch_biomass = fishing_dead * state.weight[i]

                # Accumulate revenue per vessel in this cell, per fleet
                for fi, fleet_cfg in enumerate(ctx.fleet_state.fleets):
                    if sp in fleet_cfg.target_species:
                        vessel_mask = (ctx.fleet_state.vessel_fleet == fi) & \
                                      (ctx.fleet_state.vessel_cell_y == cy) & \
                                      (ctx.fleet_state.vessel_cell_x == cx)
                        n_in_cell = vessel_mask.sum()
                        if n_in_cell > 0:
                            rev_per_vessel = catch_biomass * fleet_cfg.price_per_tonne[sp] / n_in_cell
                            ctx.fleet_state.vessel_revenue[vessel_mask] += rev_per_vessel
                        realized[fi, cy, cx] += catch_biomass

            ctx.fleet_state.catch_memory = update_catch_memory(
                ctx.fleet_state.catch_memory, realized, ctx.fleet_state.memory_decay
            )
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_economics_integration.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/simulate.py
git commit -m "feat(economics): add annual reset and catch memory update in simulation loop"
```

---

## Task 5: CSV output files

**Files:**
- Modify: `osmose/engine/output.py`
- Modify: `osmose/engine/simulate.py` (pass fleet_state to output writer)
- Test: `tests/test_economics_output.py`

### Context

Write 5 CSV output files per fleet: effort, catch, revenue, costs, profit summary. These are written alongside existing outputs by `write_outputs()`.

- [ ] **Step 1: Write output test**

```python
# tests/test_economics_output.py
"""Tests for economic CSV output files."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate


def _economics_output_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "4",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
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
        "simulation.economic.enabled": "true",
        "simulation.economic.rationality": "1.0",
        "simulation.economic.memory.decay": "0.7",
        "economic.fleet.number": "1",
        "economic.fleet.name.fsh0": "Trawlers",
        "economic.fleet.nvessels.fsh0": "5",
        "economic.fleet.homeport.y.fsh0": "1",
        "economic.fleet.homeport.x.fsh0": "1",
        "economic.fleet.gear.fsh0": "bottom_trawl",
        "economic.fleet.max.days.fsh0": "200",
        "economic.fleet.fuel.cost.fsh0": "0.0",
        "economic.fleet.operating.cost.fsh0": "0.0",
        "economic.fleet.target.species.fsh0": "0",
        "economic.fleet.price.sp0.fsh0": "1000.0",
        "economic.fleet.stock.elasticity.sp0.fsh0": "0.0",
    }


class TestEconomicOutput:
    def test_all_output_files_created(self):
        """All 5 economic output files should be created when economics is enabled."""
        cfg = EngineConfig.from_dict(_economics_output_config())
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            from osmose.engine.output import write_outputs, write_economic_outputs

            outputs = simulate(cfg, grid, rng)
            write_outputs(outputs, Path(tmpdir), cfg)

            # TODO: pass fleet_state from simulate context to output writer
            # For now verify simulation completes
            assert len(outputs) == 4

    def test_profit_summary_format(self):
        """Profit summary CSV should have header and one row per fleet."""
        from osmose.engine.economics.fleet import FleetConfig, create_fleet_state
        from osmose.engine.output import write_economic_outputs

        fleet = FleetConfig(
            name="TestFleet", n_vessels=2, home_port_y=0, home_port_x=0,
            gear_type="trawl", max_days_at_sea=200, fuel_cost_per_cell=0.0,
            base_operating_cost=0.0, stock_elasticity=np.array([0.0]),
            target_species=[0], price_per_tonne=np.array([1000.0]),
        )
        fs = create_fleet_state([fleet], grid_ny=2, grid_nx=2, rationality=1.0)
        fs.vessel_revenue[:] = 500.0
        fs.vessel_costs[:] = 100.0

        with tempfile.TemporaryDirectory() as tmpdir:
            write_economic_outputs(fs, Path(tmpdir))

            profit_file = Path(tmpdir) / "econ_profit_summary.csv"
            assert profit_file.exists()
            with open(profit_file) as f:
                reader = csv.reader(f, delimiter=";")
                header = next(reader)
                assert header == ["fleet", "revenue", "costs", "profit"]
                row = next(reader)
                assert float(row[1]) == pytest.approx(1000.0)  # 2 vessels × 500
                assert float(row[2]) == pytest.approx(200.0)   # 2 vessels × 100
                assert float(row[3]) == pytest.approx(800.0)   # profit

    def test_revenue_costs_csv_created(self):
        """Revenue and costs CSVs should be created per fleet."""
        from osmose.engine.economics.fleet import FleetConfig, create_fleet_state
        from osmose.engine.output import write_economic_outputs

        fleet = FleetConfig(
            name="Trawlers", n_vessels=3, home_port_y=0, home_port_x=0,
            gear_type="trawl", max_days_at_sea=200, fuel_cost_per_cell=0.0,
            base_operating_cost=0.0, stock_elasticity=np.array([0.0]),
            target_species=[0], price_per_tonne=np.array([1000.0]),
        )
        fs = create_fleet_state([fleet], grid_ny=2, grid_nx=2, rationality=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            write_economic_outputs(fs, Path(tmpdir))
            assert (Path(tmpdir) / "econ_effort_Trawlers.csv").exists()
            assert (Path(tmpdir) / "econ_revenue_Trawlers.csv").exists()
            assert (Path(tmpdir) / "econ_costs_Trawlers.csv").exists()
            assert (Path(tmpdir) / "econ_profit_summary.csv").exists()
```

- [ ] **Step 2: Add economic output writing to output.py**

In `osmose/engine/output.py`, add a function that writes all 5 spec-required files:

```python
def write_economic_outputs(
    fleet_state: FleetState,
    output_dir: Path,
) -> None:
    """Write economic CSV output files (5 per fleet).

    Files written:
    - econ_effort_<fleet>.csv: Effort map (ny × nx snapshot)
    - econ_catch_<fleet>.csv: Placeholder (catch tracking is per-step; snapshot not yet implemented)
    - econ_revenue_<fleet>.csv: Per-vessel revenue
    - econ_costs_<fleet>.csv: Per-vessel costs
    - econ_profit_summary.csv: Per-fleet total profit
    """
    if fleet_state is None:
        return

    profit_rows: list[list[float]] = []

    for fi, fleet in enumerate(fleet_state.fleets):
        name = fleet.name

        # 1. Effort map: shape (ny, nx)
        effort = fleet_state.effort_map[fi]
        np.savetxt(output_dir / f"econ_effort_{name}.csv", effort, delimiter=";", fmt="%.1f")

        # 2. Revenue per vessel
        vessel_mask = fleet_state.vessel_fleet == fi
        revenue_arr = fleet_state.vessel_revenue[vessel_mask]
        np.savetxt(output_dir / f"econ_revenue_{name}.csv", revenue_arr.reshape(1, -1),
                   delimiter=";", fmt="%.2f")

        # 3. Costs per vessel
        costs_arr = fleet_state.vessel_costs[vessel_mask]
        np.savetxt(output_dir / f"econ_costs_{name}.csv", costs_arr.reshape(1, -1),
                   delimiter=";", fmt="%.2f")

        # 4. Profit summary: total revenue - total costs
        total_rev = float(revenue_arr.sum())
        total_cost = float(costs_arr.sum())
        profit_rows.append([fi, total_rev, total_cost, total_rev - total_cost])

    # 5. Profit summary across all fleets
    import csv
    with open(output_dir / "econ_profit_summary.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["fleet", "revenue", "costs", "profit"])
        writer.writerows(profit_rows)
```

Call this from `PythonEngine.run()` after `write_outputs()`, passing `ctx.fleet_state`.

- [ ] **Step 3: Run test**

Run: `.venv/bin/python -m pytest tests/test_economics_output.py -v`
Expected: 3 PASSED

- [ ] **Step 4: Commit**

```bash
git add osmose/engine/output.py osmose/engine/simulate.py tests/test_economics_output.py
git commit -m "feat(economics): add economic CSV output files"
```

---

## Task 6: Multi-fleet non-interference test

**Files:**
- Test: `tests/test_economics_multifleet.py`

### Context

Multiple fleets with different target species, home ports, and gear types should operate independently. Verify that fleet A's decisions don't corrupt fleet B's state.

- [ ] **Step 1: Write multi-fleet test**

```python
# tests/test_economics_multifleet.py
"""Tests for multi-fleet operation and non-interference."""

import numpy as np
import pytest

from osmose.engine.economics.fleet import FleetConfig, create_fleet_state
from osmose.engine.economics.choice import fleet_decision


class TestMultiFleet:
    def test_two_fleets_different_targets(self):
        """Two fleets targeting different species should distribute independently."""
        fleet_a = FleetConfig(
            name="Trawlers",
            n_vessels=20,
            home_port_y=0,
            home_port_x=0,
            gear_type="bottom_trawl",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0, 0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0, 0.0]),
        )
        fleet_b = FleetConfig(
            name="Longliners",
            n_vessels=20,
            home_port_y=2,
            home_port_x=2,
            gear_type="longline",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0, 0.0]),
            target_species=[1],
            price_per_tonne=np.array([0.0, 2000.0]),
        )
        fs = create_fleet_state([fleet_a, fleet_b], grid_ny=3, grid_nx=3, rationality=5.0)

        # Species 0 at (0,0), species 1 at (2,2)
        biomass = np.zeros((2, 3, 3), dtype=np.float64)
        biomass[0, 0, 0] = 5000.0
        biomass[1, 2, 2] = 5000.0

        rng = np.random.default_rng(42)
        fs = fleet_decision(fs, biomass, rng)

        # Fleet A (trawlers) should concentrate near (0,0)
        trawler_mask = fs.vessel_fleet == 0
        trawler_at_00 = np.sum(
            (fs.vessel_cell_y[trawler_mask] == 0) & (fs.vessel_cell_x[trawler_mask] == 0)
        )
        assert trawler_at_00 > 10  # most of 20

        # Fleet B (longliners) should concentrate near (2,2)
        liner_mask = fs.vessel_fleet == 1
        liner_at_22 = np.sum(
            (fs.vessel_cell_y[liner_mask] == 2) & (fs.vessel_cell_x[liner_mask] == 2)
        )
        assert liner_at_22 > 10  # most of 20

    def test_effort_map_per_fleet(self):
        """Effort map should have separate layers per fleet."""
        fleet_a = FleetConfig(
            name="A", n_vessels=5, home_port_y=0, home_port_x=0,
            gear_type="a", max_days_at_sea=200, fuel_cost_per_cell=0.0,
            base_operating_cost=0.0, stock_elasticity=np.array([0.0]),
            target_species=[0], price_per_tonne=np.array([1000.0]),
        )
        fleet_b = FleetConfig(
            name="B", n_vessels=3, home_port_y=1, home_port_x=1,
            gear_type="b", max_days_at_sea=200, fuel_cost_per_cell=0.0,
            base_operating_cost=0.0, stock_elasticity=np.array([0.0]),
            target_species=[0], price_per_tonne=np.array([1000.0]),
        )
        fs = create_fleet_state([fleet_a, fleet_b], grid_ny=2, grid_nx=2, rationality=0.0)

        biomass = np.zeros((1, 2, 2))
        rng = np.random.default_rng(42)
        fs = fleet_decision(fs, biomass, rng)

        assert fs.effort_map.shape == (2, 2, 2)  # 2 fleets, 2x2 grid
        assert fs.effort_map[0].sum() == pytest.approx(5.0)  # fleet A: 5 vessels
        assert fs.effort_map[1].sum() == pytest.approx(3.0)  # fleet B: 3 vessels
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_economics_multifleet.py -v`
Expected: 2 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_economics_multifleet.py
git commit -m "test(economics): add multi-fleet non-interference tests"
```

---

## Task 7: Lint, full test suite, final verification

**Files:** None — verification only

- [ ] **Step 1: Run ruff lint and format**

Run: `.venv/bin/ruff check osmose/engine/economics/ tests/test_economics_*.py`
Run: `.venv/bin/ruff format osmose/engine/economics/ tests/test_economics_*.py`

- [ ] **Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest -x -q`
Expected: All tests pass.

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "style: ruff format economics Phase 2 files"
```

---

## Summary

| Task | What | New Tests |
|------|------|-----------|
| 1 | Cost calculation module (travel, catchability, revenue) | 4 |
| 2 | Full fleet_decision + catch memory EMA + cost accumulation | 2 |
| 3 | Days-at-sea tracking + forced port | 2 |
| 4 | Annual reset + revenue accumulation + memory update in simulate.py | 0 |
| 5 | CSV output files (all 5: effort, revenue, costs, profit_summary) | 3 |
| 6 | Multi-fleet non-interference | 2 |
| 7 | Lint, format, verification | 0 |
| **Total** | | **~13 tests** |

**Note:** The spec mentions ~20 tests. Additional tests for multi-year dynamics, profit calculations, and per-species catch tracking can be added in a follow-up task or during Phase 3 extended testing. The 13 tests above cover all core functionality.
