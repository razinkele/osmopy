"""Tests for MovementMapSet and _map_move_school (Task B1)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from osmose.engine.grid import Grid
from osmose.engine.movement_maps import MovementMapSet
from osmose.engine.processes.movement import _map_move_school


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv_map(path, data):
    """Write a 2D list as a semicolon-delimited CSV (no header)."""
    with open(path, "w") as f:
        for row in data:
            f.write(";".join(str(v) for v in row) + "\n")


def _base_config(tmp_path, map_file="map0.csv"):
    """Minimal config dict for a single map covering all ages/steps."""
    return {
        "movement.species.map0": "Anchovy",
        "movement.initialage.map0": "0",
        "movement.lastage.map0": "3",
        "movement.file.map0": str(tmp_path / map_file),
    }


# ---------------------------------------------------------------------------
# Test 1 — Single map loads correctly (shape, non-None)
# ---------------------------------------------------------------------------


class TestSingleMapLoad:
    def test_map_shape(self, tmp_path):
        ny, nx = 4, 5
        data = [[float(i * nx + j) / (ny * nx) for j in range(nx)] for i in range(ny)]
        _write_csv_map(tmp_path / "map0.csv", data)
        cfg = _base_config(tmp_path)
        mms = MovementMapSet(cfg, "Anchovy", n_dt_per_year=4, n_years=1,
                             lifespan_dt=12, ny=ny, nx=nx)
        assert mms.n_maps == 1
        assert mms.maps[0] is not None
        assert mms.maps[0].shape == (ny, nx)

    def test_map_is_float64(self, tmp_path):
        ny, nx = 3, 3
        data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        _write_csv_map(tmp_path / "map0.csv", data)
        cfg = _base_config(tmp_path)
        mms = MovementMapSet(cfg, "Anchovy", n_dt_per_year=4, n_years=1,
                             lifespan_dt=12, ny=ny, nx=nx)
        assert mms.maps[0].dtype == np.float64


# ---------------------------------------------------------------------------
# Test 2 — index_maps filled for all age/step combos
# ---------------------------------------------------------------------------


class TestIndexMapsFilled:
    def test_all_covered(self, tmp_path):
        ny, nx = 3, 3
        data = [[0.1, 0.2, 0.3]] * ny
        _write_csv_map(tmp_path / "map0.csv", data)
        n_dt, n_yr, lifespan_dt = 4, 2, 8
        cfg = {
            "movement.species.map0": "Anchovy",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "2",
            "movement.file.map0": str(tmp_path / "map0.csv"),
        }
        mms = MovementMapSet(cfg, "Anchovy", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=ny, nx=nx)
        # Every age_dt 0..7 (lifespan_dt-1=7, lastage=2*4=8 -> clamped to 7)
        # and every step 0..7 should be covered
        for age_dt in range(lifespan_dt):
            for step in range(n_dt * n_yr):
                assert mms.index_maps[age_dt, step] != -1, \
                    f"Expected coverage at age_dt={age_dt} step={step}"


# ---------------------------------------------------------------------------
# Test 3 — Multiple maps with different age ranges
# ---------------------------------------------------------------------------


class TestMultipleAgeMaps:
    def test_age_range_separation(self, tmp_path):
        ny, nx = 3, 3
        data_a = [[0.1, 0.2, 0.3]] * ny
        data_b = [[0.4, 0.5, 0.6]] * ny
        _write_csv_map(tmp_path / "map0.csv", data_a)
        _write_csv_map(tmp_path / "map1.csv", data_b)
        n_dt, n_yr = 4, 1
        lifespan_dt = 8
        cfg = {
            "movement.species.map0": "Tuna",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",  # 0..3 dt
            "movement.file.map0": str(tmp_path / "map0.csv"),
            "movement.species.map1": "Tuna",
            "movement.initialage.map1": "1",
            "movement.lastage.map1": "3",  # 4..8 dt (clamped to 7)
            "movement.file.map1": str(tmp_path / "map1.csv"),
        }
        mms = MovementMapSet(cfg, "Tuna", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=ny, nx=nx)
        # age_dt=0 -> map0 index
        idx_young = mms.get_index(0, 0)
        # age_dt=4 -> map1 index
        idx_old = mms.get_index(4, 0)
        assert idx_young != idx_old
        assert mms.maps[idx_young] is not None
        assert mms.maps[idx_old] is not None
        # Check actual values differ
        assert not np.allclose(mms.maps[idx_young], mms.maps[idx_old])


# ---------------------------------------------------------------------------
# Test 4 — "null" file -> None map
# ---------------------------------------------------------------------------


class TestNullMap:
    def test_null_file_returns_none_map(self, tmp_path):
        n_dt, n_yr, lifespan_dt = 4, 1, 8
        cfg = {
            "movement.species.map0": "Sardine",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "2",
            "movement.file.map0": "null",
        }
        mms = MovementMapSet(cfg, "Sardine", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=3, nx=3)
        assert mms.maps[0] is None

    def test_null_get_map_returns_none(self, tmp_path):
        n_dt, n_yr, lifespan_dt = 4, 1, 8
        cfg = {
            "movement.species.map0": "Sardine",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "2",
            "movement.file.map0": "null",
        }
        mms = MovementMapSet(cfg, "Sardine", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=3, nx=3)
        result = mms.get_map(0, 0)
        assert result is None


# ---------------------------------------------------------------------------
# Test 5 — CSV row flipping (row 0 in CSV -> grid row ny-1)
# ---------------------------------------------------------------------------


class TestCsvRowFlipping:
    def test_north_row_at_top_of_csv_becomes_last_row_in_grid(self, tmp_path):
        ny, nx = 4, 3
        # CSV row 0 = all 1.0; CSV row 3 = all 0.0
        csv_data = [
            [1.0, 1.0, 1.0],  # CSV row 0 -> grid row ny-1 = 3
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],  # CSV row 3 -> grid row 0
        ]
        _write_csv_map(tmp_path / "map0.csv", csv_data)
        cfg = _base_config(tmp_path)
        mms = MovementMapSet(cfg, "Anchovy", n_dt_per_year=4, n_years=1,
                             lifespan_dt=12, ny=ny, nx=nx)
        grid = mms.maps[0]
        # grid[ny-1] should have values from CSV row 0 (1.0)
        np.testing.assert_allclose(grid[ny - 1, :], [1.0, 1.0, 1.0])
        # grid[0] should have values from CSV row ny-1 (0.0)
        np.testing.assert_allclose(grid[0, :], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Test 6 — max_proba for presence/absence map (max >= 1.0 -> 0.0)
# ---------------------------------------------------------------------------


class TestMaxProbaPresenceAbsence:
    def test_max_ge_1_gives_zero_max_proba(self, tmp_path):
        ny, nx = 3, 3
        # Presence/absence: values are 0 or 1
        data = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
        _write_csv_map(tmp_path / "map0.csv", data)
        cfg = _base_config(tmp_path)
        mms = MovementMapSet(cfg, "Anchovy", n_dt_per_year=4, n_years=1,
                             lifespan_dt=12, ny=ny, nx=nx)
        assert mms.max_proba[0] == 0.0


# ---------------------------------------------------------------------------
# Test 7 — max_proba for probability map (max < 1.0 -> actual max)
# ---------------------------------------------------------------------------


class TestMaxProbaProbabilityMap:
    def test_max_lt_1_stores_actual_max(self, tmp_path):
        ny, nx = 3, 3
        data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        _write_csv_map(tmp_path / "map0.csv", data)
        cfg = _base_config(tmp_path)
        mms = MovementMapSet(cfg, "Anchovy", n_dt_per_year=4, n_years=1,
                             lifespan_dt=12, ny=ny, nx=nx)
        # max value in data is 0.9 (stored in grid row 0 after flip, but doesn't matter)
        assert pytest.approx(mms.max_proba[0], abs=1e-9) == 0.9


# ---------------------------------------------------------------------------
# Test 8 — Deduplication: same file -> same index in index_maps
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_file_same_map_index(self, tmp_path):
        ny, nx = 3, 3
        data = [[0.1, 0.2, 0.3]] * ny
        _write_csv_map(tmp_path / "shared.csv", data)
        n_dt, n_yr, lifespan_dt = 4, 1, 8
        cfg = {
            "movement.species.map0": "Cod",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",   # age_dt 0..3
            "movement.file.map0": str(tmp_path / "shared.csv"),
            "movement.species.map1": "Cod",
            "movement.initialage.map1": "1",
            "movement.lastage.map1": "2",   # age_dt 4..7
            "movement.file.map1": str(tmp_path / "shared.csv"),  # same file
        }
        mms = MovementMapSet(cfg, "Cod", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=ny, nx=nx)
        # Both age ranges should point to same canonical map index
        idx_young = mms.get_index(0, 0)
        idx_old = mms.get_index(4, 0)
        assert idx_young == idx_old
        # Only one unique map stored
        assert mms.n_maps == 1

    def test_different_files_different_indices(self, tmp_path):
        ny, nx = 3, 3
        _write_csv_map(tmp_path / "map0.csv", [[0.1, 0.2, 0.3]] * ny)
        _write_csv_map(tmp_path / "map1.csv", [[0.4, 0.5, 0.6]] * ny)
        n_dt, n_yr, lifespan_dt = 4, 1, 8
        cfg = {
            "movement.species.map0": "Cod",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",
            "movement.file.map0": str(tmp_path / "map0.csv"),
            "movement.species.map1": "Cod",
            "movement.initialage.map1": "1",
            "movement.lastage.map1": "2",
            "movement.file.map1": str(tmp_path / "map1.csv"),
        }
        mms = MovementMapSet(cfg, "Cod", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=ny, nx=nx)
        idx_young = mms.get_index(0, 0)
        idx_old = mms.get_index(4, 0)
        assert idx_young != idx_old
        assert mms.n_maps == 2


# ---------------------------------------------------------------------------
# Test 9 — get_map returns correct map
# ---------------------------------------------------------------------------


class TestGetMap:
    def test_returns_correct_grid(self, tmp_path):
        ny, nx = 3, 3
        expected_val = 0.42
        data = [[expected_val] * nx] * ny
        _write_csv_map(tmp_path / "map0.csv", data)
        cfg = _base_config(tmp_path)
        mms = MovementMapSet(cfg, "Anchovy", n_dt_per_year=4, n_years=1,
                             lifespan_dt=12, ny=ny, nx=nx)
        grid = mms.get_map(0, 0)
        assert grid is not None
        np.testing.assert_allclose(grid, expected_val)

    def test_out_of_bounds_returns_none(self, tmp_path):
        ny, nx = 3, 3
        data = [[0.1, 0.2, 0.3]] * ny
        _write_csv_map(tmp_path / "map0.csv", data)
        cfg = _base_config(tmp_path)
        mms = MovementMapSet(cfg, "Anchovy", n_dt_per_year=4, n_years=1,
                             lifespan_dt=12, ny=ny, nx=nx)
        assert mms.get_map(-1, 0) is None
        assert mms.get_map(0, -1) is None
        assert mms.get_map(999, 0) is None
        assert mms.get_map(0, 999) is None


# ---------------------------------------------------------------------------
# Test 10 — get_map returns None for null file
# ---------------------------------------------------------------------------


class TestGetMapNullFile:
    def test_get_map_null(self, tmp_path):
        n_dt, n_yr, lifespan_dt = 4, 1, 8
        cfg = {
            "movement.species.map0": "Herring",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "2",
            "movement.file.map0": "null",
        }
        mms = MovementMapSet(cfg, "Herring", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=3, nx=3)
        result = mms.get_map(0, 0)
        assert result is None


# ---------------------------------------------------------------------------
# Test 11 — Missing entries logged as warnings (partial config)
# ---------------------------------------------------------------------------


class TestMissingEntriesWarned:
    def test_partial_coverage_logs_warnings(self, tmp_path, caplog):
        ny, nx = 3, 3
        data = [[0.1, 0.2, 0.3]] * ny
        _write_csv_map(tmp_path / "map0.csv", data)
        n_dt, n_yr, lifespan_dt = 4, 1, 8
        # Only covers age_dt 0..3 — age_dt 4..7 will be -1 -> warnings
        cfg = {
            "movement.species.map0": "Mackerel",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",   # age_dt 0..4 (4*1=4 dt -> clamped to min(4,7)=4)
            "movement.file.map0": str(tmp_path / "map0.csv"),
        }
        with caplog.at_level(logging.WARNING, logger="osmose.engine.movement_maps"):
            MovementMapSet(cfg, "Mackerel", n_dt_per_year=n_dt, n_years=n_yr,
                           lifespan_dt=lifespan_dt, ny=ny, nx=nx)
        # There should be at least one warning about missing coverage
        assert any("No movement map" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Test 12 — Season subset (steps 0-11 vs 12-23 -> different maps)
# ---------------------------------------------------------------------------


class TestSeasonSubset:
    def test_season_subsets_map_to_different_grids(self, tmp_path):
        ny, nx = 3, 3
        data_a = [[0.1, 0.2, 0.3]] * ny
        data_b = [[0.7, 0.8, 0.9]] * ny
        _write_csv_map(tmp_path / "map0.csv", data_a)
        _write_csv_map(tmp_path / "map1.csv", data_b)
        n_dt, n_yr, lifespan_dt = 24, 1, 5
        cfg = {
            "movement.species.map0": "Tuna",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",
            "movement.steps.map0": ";".join(str(s) for s in range(12)),  # steps 0-11
            "movement.file.map0": str(tmp_path / "map0.csv"),
            "movement.species.map1": "Tuna",
            "movement.initialage.map1": "0",
            "movement.lastage.map1": "1",
            "movement.steps.map1": ";".join(str(s) for s in range(12, 24)),  # steps 12-23
            "movement.file.map1": str(tmp_path / "map1.csv"),
        }
        mms = MovementMapSet(cfg, "Tuna", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=ny, nx=nx)
        idx_first_half = mms.get_index(0, 0)   # step 0 -> map0
        idx_second_half = mms.get_index(0, 12)  # step 12 -> map1
        assert idx_first_half != idx_second_half
        assert not np.allclose(mms.maps[idx_first_half], mms.maps[idx_second_half])


# ---------------------------------------------------------------------------
# Test 13 — Out-of-range steps silently skipped (lastYear beyond simulation)
# ---------------------------------------------------------------------------


class TestOutOfRangeStepsSkipped:
    def test_out_of_range_years_silently_skipped(self, tmp_path):
        ny, nx = 3, 3
        data = [[0.3, 0.3, 0.3]] * ny
        _write_csv_map(tmp_path / "map0.csv", data)
        n_dt, n_yr, lifespan_dt = 4, 2, 8
        # lastyear=5 is beyond n_years=2 — should not raise
        cfg = {
            "movement.species.map0": "Cod",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "2",
            "movement.initialyear.map0": "0",
            "movement.lastyear.map0": "5",  # beyond n_years=2
            "movement.file.map0": str(tmp_path / "map0.csv"),
        }
        # Should complete without error
        mms = MovementMapSet(cfg, "Cod", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=ny, nx=nx)
        # Years 0 and 1 should be covered, no crash for out-of-range year 2-5
        assert mms.get_index(0, 0) != -1
        assert mms.get_index(0, n_dt) != -1  # year 1, step 0


# ---------------------------------------------------------------------------
# Test 14 — Multi-year: 2-year sim, step 24 (year 1 season 0) maps correctly
# ---------------------------------------------------------------------------


class TestMultiYearMapping:
    def test_step_n_dt_per_year_is_year1_season0(self, tmp_path):
        ny, nx = 3, 3
        data = [[0.5, 0.5, 0.5]] * ny
        _write_csv_map(tmp_path / "map0.csv", data)
        n_dt, n_yr, lifespan_dt = 24, 2, 10
        cfg = {
            "movement.species.map0": "Anchovy",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "2",
            "movement.initialyear.map0": "0",
            "movement.lastyear.map0": "1",  # both years
            "movement.file.map0": str(tmp_path / "map0.csv"),
        }
        mms = MovementMapSet(cfg, "Anchovy", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=ny, nx=nx)
        # step 24 = year 1, season 0
        idx = mms.get_index(0, n_dt)
        assert idx != -1
        grid = mms.get_map(0, n_dt)
        assert grid is not None
        np.testing.assert_allclose(grid, 0.5)

    def test_both_years_covered(self, tmp_path):
        ny, nx = 3, 3
        data = [[0.2, 0.3, 0.4]] * ny
        _write_csv_map(tmp_path / "map0.csv", data)
        n_dt, n_yr, lifespan_dt = 12, 2, 8
        cfg = {
            "movement.species.map0": "Sardine",
            "movement.initialage.map0": "0",
            "movement.lastage.map0": "1",
            "movement.initialyear.map0": "0",
            "movement.lastyear.map0": "1",
            "movement.file.map0": str(tmp_path / "map0.csv"),
        }
        mms = MovementMapSet(cfg, "Sardine", n_dt_per_year=n_dt, n_years=n_yr,
                             lifespan_dt=lifespan_dt, ny=ny, nx=nx)
        # Year 0, step 0
        assert mms.get_index(0, 0) != -1
        # Year 1, step 0 (global step = n_dt * 1 = 12)
        assert mms.get_index(0, n_dt) != -1
        # Year 1, last step (global step = n_dt * 1 + n_dt - 1 = 23)
        assert mms.get_index(0, n_dt + n_dt - 1) != -1


# ---------------------------------------------------------------------------
# Tests for _map_move_school — per-school map-based movement (Task 2)
# ---------------------------------------------------------------------------


class TestMapMoveSchool:
    def test_out_of_domain(self, tmp_path):
        """Null map -> returns (-1, -1, True)."""
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
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        x, y, out = _map_move_school(10, 2, 2, 5, 5, grid.ocean_mask, ms, 1, 0, rng)
        assert out is True
        assert x == -1 and y == -1

    def test_rejection_sampling_positive_cell(self, tmp_path):
        """Unlocated school placed on positive-probability cell."""
        rows = [[-99] * 5 for _ in range(5)]
        for r in range(1, 4):
            for c in range(1, 4):
                rows[r][c] = 1
        map_file = tmp_path / "test.csv"
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
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        # Unlocated school (cx=-1) -> rejection sampling
        x, y, out = _map_move_school(10, -1, -1, 5, 5, grid.ocean_mask, ms, 1, 0, rng)
        assert out is False
        assert ms.maps[0][y, x] > 0

    def test_same_map_random_walk(self, tmp_path):
        """Same map + located -> random walk within range."""
        rows = [[-99] * 5 for _ in range(5)]
        for r in range(1, 4):
            for c in range(1, 4):
                rows[r][c] = 1
        map_file = tmp_path / "test.csv"
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
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        # Located at (2,2), step=1 -> same map as step=0 -> random walk
        x, y, out = _map_move_school(10, 2, 2, 5, 5, grid.ocean_mask, ms, 1, 1, rng)
        assert out is False
        assert abs(x - 2) <= 1 and abs(y - 2) <= 1
        assert ms.maps[0][y, x] > 0

    def test_unlocated_forces_new_placement(self, tmp_path):
        """Unlocated (cx<0) forces rejection sampling even if same map."""
        rows = [[-99] * 5 for _ in range(5)]
        for r in range(1, 4):
            for c in range(1, 4):
                rows[r][c] = 1
        map_file = tmp_path / "test.csv"
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
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        x, y, out = _map_move_school(10, -1, -1, 5, 5, grid.ocean_mask, ms, 1, 1, rng)
        assert out is False
        assert x >= 0 and y >= 0

    def test_stranded_stays_in_place(self, tmp_path):
        """School on isolated positive cell with no positive neighbors stays put."""
        rows = [[-99] * 5 for _ in range(5)]
        rows[2][2] = 1  # only this cell is positive
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
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        # Find where the positive cell is in the flipped grid
        grid_map = ms.maps[0]
        pos_ys, pos_xs = np.where(grid_map > 0)
        assert len(pos_ys) == 1
        gy, gx = int(pos_ys[0]), int(pos_xs[0])
        # Located at that cell, step=1 (same map) -> random walk, only accessible cell is self
        x, y, out = _map_move_school(10, gx, gy, 5, 5, grid.ocean_mask, ms, 1, 1, rng)
        assert x == gx and y == gy  # stays in place

    def test_presence_absence_uniform(self, tmp_path):
        """Presence/absence map (max_proba=0) -> multiple cells reachable."""
        rows = [[-99] * 5 for _ in range(5)]
        rows[1][1] = 1
        rows[1][3] = 1
        rows[3][1] = 1
        rows[3][3] = 1
        map_file = tmp_path / "pa.csv"
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
        assert ms.max_proba[0] == 0.0  # presence/absence trick
        grid = Grid.from_dimensions(ny=5, nx=5)
        # Run 50 placements, should visit at least 2 distinct cells
        cells_seen: set[tuple[int, int]] = set()
        for seed in range(50):
            rng = np.random.default_rng(seed)
            x, y, out = _map_move_school(10, -1, -1, 5, 5, grid.ocean_mask, ms, 1, 0, rng)
            assert out is False
            assert ms.maps[0][y, x] > 0
            cells_seen.add((x, y))
        assert len(cells_seen) >= 2  # at least 2 distinct cells visited
