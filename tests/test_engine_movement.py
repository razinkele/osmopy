"""Tests for movement process — Tier 1 verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.movement import (
    _precompute_map_indices,
    _precompute_map_indices_loop,
    movement,
    random_walk,
)
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


class _FakeMapSet:
    """Stub matching the duck-typed interface _precompute_map_indices uses."""

    def __init__(self, index_maps: np.ndarray) -> None:
        self.index_maps = index_maps


class TestPrecomputeMapIndicesVectorisedMatchesLoop:
    """A2 acceptance: the vectorised path is element-wise equal to the loop."""

    def _make_map_sets(self, rng: np.random.Generator) -> dict[int, _FakeMapSet]:
        # Two species with different (lifespan_dt, n_steps) shapes to
        # exercise per-species bounds checks. -1 sentinel for "no map".
        idx0 = rng.integers(-1, 12, size=(8, 24), dtype=np.int32)
        idx1 = rng.integers(-1, 8, size=(15, 24), dtype=np.int32)
        return {0: _FakeMapSet(idx0), 1: _FakeMapSet(idx1)}

    def _assert_equal(self, vec_out, loop_out):
        np.testing.assert_array_equal(vec_out[0], loop_out[0], err_msg="current_idx")
        np.testing.assert_array_equal(vec_out[1], loop_out[1], err_msg="same_map")

    def test_random_inputs(self):
        rng = np.random.default_rng(0)
        n = 200
        species_id = rng.integers(0, 2, size=n, dtype=np.int32)
        age_dt = rng.integers(0, 16, size=n, dtype=np.int32)
        uses_maps = rng.random(n) > 0.2  # ~80% use maps
        map_sets = self._make_map_sets(rng)
        for step in (0, 1, 5, 12, 23):
            vec = _precompute_map_indices(species_id, age_dt, uses_maps, map_sets, step)
            loop = _precompute_map_indices_loop(species_id, age_dt, uses_maps, map_sets, step)
            self._assert_equal(vec, loop)

    def test_age_zero_no_wrap_to_last_row(self):
        # age_dt == 0 → prev_age = -1. Without the pre-mask, NumPy would
        # silently return ms.index_maps[-1, prev_step] (wrap-around to last
        # row). The vectorised path must keep prev_idx[k] == -1.
        n = 5
        species_id = np.zeros(n, dtype=np.int32)
        age_dt = np.zeros(n, dtype=np.int32)  # ALL age == 0 → all prev_age == -1
        uses_maps = np.ones(n, dtype=np.bool_)
        # Build a map set whose last-row values are NOT -1, so a wrap would
        # silently put a wrong value into prev_idx instead of -1.
        idx0 = np.arange(8 * 24, dtype=np.int32).reshape(8, 24)  # all >= 0
        map_sets = {0: _FakeMapSet(idx0)}
        # step >= 1 so the cur path can return non-(-1), but prev_step = step-1
        # could still trigger the wrap if pre-mask is missing.
        vec = _precompute_map_indices(species_id, age_dt, uses_maps, map_sets, step=5)
        loop = _precompute_map_indices_loop(species_id, age_dt, uses_maps, map_sets, step=5)
        self._assert_equal(vec, loop)
        assert (vec[0] == idx0[0, 5]).all()  # current_idx → row 0, step 5
        assert not vec[1].any()  # same_map False (age == 0 disables)

    def test_step_zero_disables_prev(self):
        # step == 0 → prev_step == -1; prev_idx must be -1 for all schools.
        rng = np.random.default_rng(2)
        n = 6
        species_id = np.zeros(n, dtype=np.int32)
        age_dt = rng.integers(1, 8, size=n, dtype=np.int32)  # all > 0
        uses_maps = np.ones(n, dtype=np.bool_)
        map_sets = self._make_map_sets(rng)
        vec = _precompute_map_indices(species_id, age_dt, uses_maps, map_sets, step=0)
        loop = _precompute_map_indices_loop(species_id, age_dt, uses_maps, map_sets, step=0)
        self._assert_equal(vec, loop)

    def test_step_out_of_range(self):
        rng = np.random.default_rng(3)
        n = 6
        species_id = np.zeros(n, dtype=np.int32)
        age_dt = rng.integers(0, 8, size=n, dtype=np.int32)
        uses_maps = np.ones(n, dtype=np.bool_)
        map_sets = self._make_map_sets(rng)
        # step >= n_steps → both current and prev paths short-circuit
        vec = _precompute_map_indices(species_id, age_dt, uses_maps, map_sets, step=99)
        loop = _precompute_map_indices_loop(species_id, age_dt, uses_maps, map_sets, step=99)
        self._assert_equal(vec, loop)
        assert (vec[0] == -1).all()

    def test_unknown_species(self):
        # Schools whose species_id has no entry in map_sets must keep -1.
        rng = np.random.default_rng(4)
        n = 6
        species_id = np.full(n, 7, dtype=np.int32)  # not in map_sets
        age_dt = rng.integers(0, 8, size=n, dtype=np.int32)
        uses_maps = np.ones(n, dtype=np.bool_)
        map_sets = self._make_map_sets(rng)
        vec = _precompute_map_indices(species_id, age_dt, uses_maps, map_sets, step=5)
        loop = _precompute_map_indices_loop(species_id, age_dt, uses_maps, map_sets, step=5)
        self._assert_equal(vec, loop)
        assert (vec[0] == -1).all()

    def test_no_schools_use_maps(self):
        rng = np.random.default_rng(5)
        n = 6
        species_id = rng.integers(0, 2, size=n, dtype=np.int32)
        age_dt = rng.integers(0, 8, size=n, dtype=np.int32)
        uses_maps = np.zeros(n, dtype=np.bool_)
        map_sets = self._make_map_sets(rng)
        vec = _precompute_map_indices(species_id, age_dt, uses_maps, map_sets, step=5)
        loop = _precompute_map_indices_loop(species_id, age_dt, uses_maps, map_sets, step=5)
        self._assert_equal(vec, loop)
        assert vec[0].shape == (0,)
        assert vec[1].shape == (0,)

    def test_age_out_of_range_high(self):
        # age_dt >= n_ages for that species → current keeps -1, prev too.
        rng = np.random.default_rng(6)
        n = 4
        species_id = np.zeros(n, dtype=np.int32)
        age_dt = np.array([100, 200, 100, 200], dtype=np.int32)  # all > 8
        uses_maps = np.ones(n, dtype=np.bool_)
        map_sets = self._make_map_sets(rng)
        vec = _precompute_map_indices(species_id, age_dt, uses_maps, map_sets, step=5)
        loop = _precompute_map_indices_loop(species_id, age_dt, uses_maps, map_sets, step=5)
        self._assert_equal(vec, loop)


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
