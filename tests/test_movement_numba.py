"""Tests for Numba-accelerated movement functions."""

from __future__ import annotations

import numpy as np
import pytest


def _make_mock_map_set(maps_data, index_maps, max_proba):
    """Create a minimal object mimicking MovementMapSet for testing."""
    class MockMapSet:
        pass
    ms = MockMapSet()
    ms.maps = maps_data
    ms.index_maps = index_maps
    ms.max_proba = max_proba
    return ms


def test_flatten_all_map_sets_shapes():
    from osmose.engine.processes.movement import _flatten_all_map_sets
    ny, nx = 3, 4
    ms0 = _make_mock_map_set(
        [np.ones((ny, nx)), None],
        np.zeros((10, 24), dtype=np.int32),
        np.array([0.0, 0.0]),
    )
    ms2 = _make_mock_map_set(
        [np.full((ny, nx), 0.5)],
        np.zeros((10, 24), dtype=np.int32),
        np.array([0.5]),
    )
    map_sets = {0: ms0, 2: ms2}
    all_maps, all_max_proba, all_is_null, sp_offsets = _flatten_all_map_sets(
        map_sets, n_species=4, ny=ny, nx=nx
    )
    assert all_maps.shape == (3, ny, nx)
    assert all_max_proba.shape == (3,)
    assert all_is_null.shape == (3,)
    assert sp_offsets[0] == 0
    assert sp_offsets[1] == -1
    assert sp_offsets[2] == 2
    assert sp_offsets[3] == -1


def test_flatten_all_map_sets_null_detection():
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
    assert not all_is_null[0]
    assert all_is_null[1]
    np.testing.assert_array_equal(all_maps[0], real_map)
    np.testing.assert_array_equal(all_maps[1], np.zeros((ny, nx)))
    assert all_max_proba[0] == 0.7
    assert all_max_proba[1] == 0.0


def test_flatten_all_map_sets_empty():
    from osmose.engine.processes.movement import _flatten_all_map_sets
    all_maps, all_max_proba, all_is_null, sp_offsets = _flatten_all_map_sets(
        {}, n_species=3, ny=5, nx=5
    )
    assert all_maps.shape == (0, 5, 5)
    assert all_max_proba.shape == (0,)
    assert np.all(sp_offsets == -1)


def test_precompute_map_indices_basic():
    from osmose.engine.processes.movement import _precompute_map_indices

    ny, nx = 3, 4
    idx_maps = np.full((5, 24), -1, dtype=np.int32)
    idx_maps[0, :] = 0
    idx_maps[1, :] = 1
    idx_maps[2, :] = 1

    ms = _make_mock_map_set(
        [np.ones((ny, nx)), np.ones((ny, nx))],
        idx_maps,
        np.array([0.0, 0.0]),
    )
    map_sets = {0: ms}

    species_id = np.array([0, 0, 0], dtype=np.int32)
    age_dt = np.array([0, 1, 2], dtype=np.int32)
    uses_maps = np.array([True, True, True])

    current, same = _precompute_map_indices(
        species_id, age_dt, uses_maps, map_sets, step=2
    )

    assert current[0] == 0
    assert current[1] == 1
    assert current[2] == 1
    assert not same[0]
    assert not same[1]
    assert same[2]


def test_precompute_map_indices_out_of_range():
    from osmose.engine.processes.movement import _precompute_map_indices

    idx_maps = np.zeros((3, 24), dtype=np.int32)
    ms = _make_mock_map_set([np.ones((2, 2))], idx_maps, np.array([0.0]))
    map_sets = {0: ms}

    species_id = np.array([0], dtype=np.int32)
    age_dt = np.array([10], dtype=np.int32)
    uses_maps = np.array([True])

    current, same = _precompute_map_indices(
        species_id, age_dt, uses_maps, map_sets, step=0
    )

    assert current[0] == -1


def test_map_move_batch_numba_new_placement():
    """Rejection sampling places schools on cells with non-zero probability."""
    try:
        from osmose.engine.processes.movement import _map_move_batch_numba
    except ImportError:
        pytest.skip("Numba not available")

    ny, nx = 3, 4
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
    sp_ids = np.array([0], dtype=np.int32)

    out_cx = np.array([-1], dtype=np.int32)
    out_cy = np.array([-1], dtype=np.int32)
    out_is_out = np.array([False])

    _map_move_batch_numba(
        42,
        school_indices, current_map_idx, same_map,
        out_cx, out_cy, sp_ids,
        all_maps, all_max_proba, all_is_null, sp_offsets,
        ocean_mask, walk_range, ny, nx,
        out_cx, out_cy, out_is_out,
    )

    assert (out_cx[0], out_cy[0]) in [(2, 1), (3, 2)]
    assert not out_is_out[0]


def test_map_move_batch_numba_out_of_domain():
    """Null map index -> school goes out of domain."""
    try:
        from osmose.engine.processes.movement import _map_move_batch_numba
    except ImportError:
        pytest.skip("Numba not available")

    ny, nx = 2, 2
    all_maps = np.zeros((1, ny, nx), dtype=np.float64)
    all_max_proba = np.array([0.0])
    all_is_null = np.array([True])
    sp_offsets = np.array([0], dtype=np.int32)
    ocean_mask = np.ones((ny, nx), dtype=np.bool_)
    walk_range = np.array([1], dtype=np.int32)

    out_cx = np.array([0], dtype=np.int32)
    out_cy = np.array([0], dtype=np.int32)
    out_is_out = np.array([False])

    _map_move_batch_numba(
        42,
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
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
    """Same seed -> same output."""
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
    same_map_arr = np.zeros(n, dtype=np.bool_)
    sp_ids = np.zeros(n, dtype=np.int32)

    def run(seed):
        cx = np.full(n, -1, dtype=np.int32)
        cy = np.full(n, -1, dtype=np.int32)
        out = np.zeros(n, dtype=np.bool_)
        _map_move_batch_numba(
            seed, school_indices, current_map_idx, same_map_arr,
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


def test_map_move_batch_numba_random_walk():
    """Same-map path: school walks to accessible neighbor cell."""
    try:
        from osmose.engine.processes.movement import _map_move_batch_numba
    except ImportError:
        pytest.skip("Numba not available")

    ny, nx = 5, 5
    # Map with positive values in a 3x3 block centered at (2,2)
    prob_map = np.zeros((ny, nx), dtype=np.float64)
    prob_map[1:4, 1:4] = 0.5
    # Set (2,2) = 0 to force walk away from center
    prob_map[2, 2] = 0.0

    all_maps = prob_map.reshape(1, ny, nx)
    all_max_proba = np.array([0.5])
    all_is_null = np.array([False])
    sp_offsets = np.array([0], dtype=np.int32)
    ocean_mask = np.ones((ny, nx), dtype=np.bool_)
    walk_range = np.array([1], dtype=np.int32)

    # School at (2,2), same_map=True → random walk
    out_cx = np.array([2], dtype=np.int32)
    out_cy = np.array([2], dtype=np.int32)
    out_is_out = np.array([False])

    _map_move_batch_numba(
        42,
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([True]),  # same_map = True → random walk path
        out_cx, out_cy,
        np.array([0], dtype=np.int32),
        all_maps, all_max_proba, all_is_null, sp_offsets,
        ocean_mask, walk_range, ny, nx,
        out_cx, out_cy, out_is_out,
    )

    # Must move to one of the accessible neighbors (not (2,2) since prob=0 there)
    accessible = [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (1, 3), (2, 3), (3, 3)]
    assert (out_cx[0], out_cy[0]) in accessible
    assert not out_is_out[0]


def test_map_move_batch_numba_stranded():
    """Same-map path with no accessible neighbors: school stays in place."""
    try:
        from osmose.engine.processes.movement import _map_move_batch_numba
    except ImportError:
        pytest.skip("Numba not available")

    ny, nx = 3, 3
    # Map with positive only at center (1,1) — all neighbors are 0
    prob_map = np.zeros((ny, nx), dtype=np.float64)
    prob_map[1, 1] = 0.5  # school is here but walk_range neighbors are all 0

    all_maps = prob_map.reshape(1, ny, nx)
    all_max_proba = np.array([0.5])
    all_is_null = np.array([False])
    sp_offsets = np.array([0], dtype=np.int32)
    ocean_mask = np.ones((ny, nx), dtype=np.bool_)
    walk_range = np.array([1], dtype=np.int32)

    out_cx = np.array([1], dtype=np.int32)
    out_cy = np.array([1], dtype=np.int32)
    out_is_out = np.array([False])

    _map_move_batch_numba(
        42,
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([True]),
        out_cx, out_cy,
        np.array([0], dtype=np.int32),
        all_maps, all_max_proba, all_is_null, sp_offsets,
        ocean_mask, walk_range, ny, nx,
        out_cx, out_cy, out_is_out,
    )

    # Stranded: should stay at (1,1)
    assert out_cx[0] == 1
    assert out_cy[0] == 1
    assert not out_is_out[0]
