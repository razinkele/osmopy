"""Tests for movement process — Tier 1 verification."""

from collections import Counter

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

    def test_unlocated_school_is_not_walked(self):
        # An unlocated school has no position to displace. Clipping (-1 + dx) into the grid
        # is what put every new "random"-method cohort in the corner (review S1, 2026-11-07).
        # The last school is located so the walk really draws displacements (an all-unlocated
        # state could pass by never drawing at all).
        grid = Grid.from_dimensions(ny=5, nx=5)
        state = SchoolState.create(n_schools=51, species_id=np.zeros(51, dtype=np.int32))
        state = state.replace(
            cell_x=np.append(np.full(50, -1), 2).astype(np.int32),
            cell_y=np.append(np.full(50, -1), 2).astype(np.int32),
            abundance=np.ones(51),
        )
        rng = np.random.default_rng(3)
        walk_range = np.full(51, 2, dtype=np.int32)
        new_state = random_walk(state, grid, walk_range, rng)
        assert (new_state.cell_x[:50] == -1).all()
        assert (new_state.cell_y[:50] == -1).all()


def _walk(grid: Grid, cells_x, cells_y, ranges, seed: int = 0):
    """Walk one school per entry once; return the new (cell_x, cell_y) arrays."""
    n = len(cells_x)
    state = SchoolState.create(n_schools=n, species_id=np.zeros(n, dtype=np.int32))
    state = state.replace(
        cell_x=np.asarray(cells_x, dtype=np.int32),
        cell_y=np.asarray(cells_y, dtype=np.int32),
        abundance=np.ones(n),
    )
    rng = np.random.default_rng(seed)
    out = random_walk(state, grid, np.asarray(ranges, dtype=np.int32), rng)
    return out.cell_x, out.cell_y


class TestRandomWalkKernel:
    """A located school moves to a uniformly random ACCESSIBLE cell, as in Java.

    Java's RandomDistribution (4.3.3 ``getAccessibleCells`` + ``randomDeal``, :185-209): the
    candidates are the in-grid cells of the (2r+1) x (2r+1) square centred on the school, its own
    cell included, minus land; one of them is dealt. The frequencies below are checked against
    those hand-derived probabilities with a band of about 4 binomial standard deviations.
    """

    def test_range_1_school_is_uniform_over_its_9_cells_beside_a_longer_range(self):
        # A range-3 school makes 3 the largest range. The old kernel drew every step from
        # [-3, 3] and clipped it to the school's own range, so a range-1 school stayed put with
        # probability 1/49 and took each diagonal with 9/49. Java: 1/9 for each of the 9 cells.
        grid = Grid.from_dimensions(ny=21, nx=21)
        n = 9000
        xs, ys = _walk(grid, [10] * n + [3], [10] * n + [3], [1] * n + [3])
        cells = Counter(zip((xs[:n] - 10).tolist(), (ys[:n] - 10).tolist(), strict=True))
        assert set(cells) == {(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        for cell, count in cells.items():
            assert abs(count / n - 1 / 9) < 0.014, (cell, count / n)

    def test_corner_school_is_uniform_over_the_4_cells_inside_the_grid(self):
        # The old kernel clipped steps that left the grid back onto the edge, so the corner kept
        # 4/9 of the draws and its diagonal neighbour got 1/9. Java deals only in-grid cells.
        grid = Grid.from_dimensions(ny=5, nx=5)
        n = 4000
        xs, ys = _walk(grid, [0] * n, [0] * n, [1] * n)
        cells = Counter(zip(xs.tolist(), ys.tolist(), strict=True))
        assert set(cells) == {(0, 0), (1, 0), (0, 1), (1, 1)}
        for cell, count in cells.items():
            assert abs(count / n - 1 / 4) < 0.03, (cell, count / n)

    def test_school_beside_land_is_uniform_over_the_sea_cells_only(self):
        # Only the school's cell and three neighbours are sea. The old kernel drew among all 9
        # cells and stayed put on a land draw, so it stayed with probability 6/9. Java deals only
        # the 4 sea cells, 1/4 each.
        mask = np.zeros((3, 3), dtype=np.bool_)
        mask[1, 1] = mask[1, 2] = mask[2, 1] = mask[2, 2] = True  # (x, y) = (1,1) (2,1) (1,2) (2,2)
        grid = Grid(ny=3, nx=3, ocean_mask=mask)
        n = 4000
        xs, ys = _walk(grid, [1] * n, [1] * n, [1] * n)
        cells = Counter(zip(xs.tolist(), ys.tolist(), strict=True))
        assert set(cells) == {(1, 1), (2, 1), (1, 2), (2, 2)}
        for cell, count in cells.items():
            assert abs(count / n - 1 / 4) < 0.03, (cell, count / n)

    def test_school_with_no_sea_in_range_stays_put(self):
        # Defensive: a school on land with no sea cell in range has nothing to be dealt (Java
        # would throw on the empty list). It stays where it is instead of raising. The second
        # school, on the only sea cell, keeps the draw path exercised.
        mask = np.zeros((4, 4), dtype=np.bool_)
        mask[3, 3] = True
        grid = Grid(ny=4, nx=4, ocean_mask=mask)
        xs, ys = _walk(grid, [0, 3], [0, 3], [1, 1])
        assert (int(xs[0]), int(ys[0])) == (0, 0)
        assert (int(xs[1]), int(ys[1])) == (3, 3)


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
