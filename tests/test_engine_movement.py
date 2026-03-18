"""Tests for movement process — Tier 1 verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.movement import movement, random_walk
from osmose.engine.processes.natural import out_mortality
from osmose.engine.state import SchoolState


def _make_movement_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "24",
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
        "movement.distribution.method.sp0": "random",
        "movement.randomwalk.range.sp0": "2",
    }


class TestRandomWalk:
    def test_schools_move(self):
        grid = Grid.from_dimensions(ny=10, nx=10)
        state = SchoolState.create(n_schools=100, species_id=np.zeros(100, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(100, 5, dtype=np.int32),
            cell_y=np.full(100, 5, dtype=np.int32),
            abundance=np.ones(100),
        )
        rng = np.random.default_rng(42)
        walk_range = np.full(100, 2, dtype=np.int32)
        new_state = random_walk(state, grid, walk_range, rng)
        moved = (new_state.cell_x != 5) | (new_state.cell_y != 5)
        assert moved.sum() > 0

    def test_stays_within_grid(self):
        grid = Grid.from_dimensions(ny=5, nx=5)
        state = SchoolState.create(n_schools=50, species_id=np.zeros(50, dtype=np.int32))
        state = state.replace(
            cell_x=np.zeros(50, dtype=np.int32),
            cell_y=np.zeros(50, dtype=np.int32),
            abundance=np.ones(50),
        )
        rng = np.random.default_rng(42)
        walk_range = np.full(50, 3, dtype=np.int32)
        new_state = random_walk(state, grid, walk_range, rng)
        assert np.all(new_state.cell_x >= 0) and np.all(new_state.cell_x < 5)
        assert np.all(new_state.cell_y >= 0) and np.all(new_state.cell_y < 5)

    def test_range_limits_displacement(self):
        grid = Grid.from_dimensions(ny=20, nx=20)
        state = SchoolState.create(n_schools=200, species_id=np.zeros(200, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(200, 10, dtype=np.int32),
            cell_y=np.full(200, 10, dtype=np.int32),
            abundance=np.ones(200),
        )
        rng = np.random.default_rng(42)
        walk_range = np.full(200, 1, dtype=np.int32)
        new_state = random_walk(state, grid, walk_range, rng)
        dx = np.abs(new_state.cell_x - 10)
        dy = np.abs(new_state.cell_y - 10)
        assert np.all(dx <= 1)
        assert np.all(dy <= 1)

    def test_avoids_land_cells(self):
        mask = np.ones((5, 5), dtype=np.bool_)
        mask[2, 2] = False
        grid = Grid(ny=5, nx=5, ocean_mask=mask)
        state = SchoolState.create(n_schools=100, species_id=np.zeros(100, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(100, 2, dtype=np.int32),
            cell_y=np.full(100, 1, dtype=np.int32),
            abundance=np.ones(100),
        )
        rng = np.random.default_rng(42)
        walk_range = np.full(100, 1, dtype=np.int32)
        new_state = random_walk(state, grid, walk_range, rng)
        on_land = (new_state.cell_x == 2) & (new_state.cell_y == 2)
        assert not on_land.any()


class TestOutMortality:
    def test_kills_out_of_domain_schools(self):
        cfg = EngineConfig.from_dict({**_make_movement_config(), "mortality.out.rate.sp0": "1.0"})
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([5.0, 5.0]),
            is_out=np.array([True, False]),
            age_dt=np.array([10, 10], dtype=np.int32),
        )
        new_state = out_mortality(state, cfg)
        assert new_state.abundance[0] < 1000.0
        np.testing.assert_allclose(new_state.abundance[1], 1000.0)

    def test_zero_rate_no_mortality(self):
        cfg = EngineConfig.from_dict(_make_movement_config())
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(abundance=np.array([1000.0]), is_out=np.array([True]))
        new_state = out_mortality(state, cfg)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)


class TestMovementOrchestrator:
    def test_movement_moves_schools(self):
        cfg = EngineConfig.from_dict(_make_movement_config())
        grid = Grid.from_dimensions(ny=10, nx=10)
        state = SchoolState.create(n_schools=50, species_id=np.zeros(50, dtype=np.int32))
        state = state.replace(
            cell_x=np.full(50, 5, dtype=np.int32),
            cell_y=np.full(50, 5, dtype=np.int32),
            abundance=np.ones(50),
        )
        rng = np.random.default_rng(42)
        new_state = movement(state, grid, cfg, step=0, rng=rng)
        moved = (new_state.cell_x != 5) | (new_state.cell_y != 5)
        assert moved.sum() > 0
