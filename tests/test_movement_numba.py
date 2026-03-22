"""Tests for Numba-accelerated movement functions."""

from __future__ import annotations

import numpy as np


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
