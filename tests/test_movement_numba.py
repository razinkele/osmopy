"""Tests for Numba-accelerated movement functions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from osmose.engine.processes.movement import (
    _flatten_all_map_sets,
    _precompute_map_indices,
)


def _mock_map_set(maps_data, index_maps, max_proba):
    """Create a minimal object mimicking MovementMapSet."""
    return SimpleNamespace(maps=maps_data, index_maps=index_maps, max_proba=max_proba)


# ---------------------------------------------------------------------------
# _flatten_all_map_sets tests
# ---------------------------------------------------------------------------


def test_flatten_all_map_sets_shapes():
    ny, nx = 3, 4
    ms0 = _mock_map_set(
        [np.ones((ny, nx)), None],
        np.zeros((10, 24), dtype=np.int32),
        np.array([0.0, 0.0]),
    )
    ms2 = _mock_map_set(
        [np.full((ny, nx), 0.5)],
        np.zeros((10, 24), dtype=np.int32),
        np.array([0.5]),
    )
    all_maps, all_max_proba, all_is_null, sp_offsets = _flatten_all_map_sets(
        {0: ms0, 2: ms2}, n_species=4, ny=ny, nx=nx
    )
    assert all_maps.shape == (3, ny, nx)
    assert all_max_proba.shape == (3,)
    assert all_is_null.shape == (3,)
    assert sp_offsets[0] == 0
    assert sp_offsets[1] == -1
    assert sp_offsets[2] == 2
    assert sp_offsets[3] == -1


def test_flatten_all_map_sets_null_detection():
    ny, nx = 2, 2
    real_map = np.array([[0.3, 0.7], [0.0, 0.5]])
    ms = _mock_map_set([real_map, None], np.zeros((5, 24), dtype=np.int32), np.array([0.7, 0.0]))
    all_maps, all_max_proba, all_is_null, _ = _flatten_all_map_sets(
        {0: ms}, n_species=1, ny=ny, nx=nx
    )
    assert not all_is_null[0]
    assert all_is_null[1]
    np.testing.assert_array_equal(all_maps[0], real_map)
    np.testing.assert_array_equal(all_maps[1], np.zeros((ny, nx)))
    assert all_max_proba[0] == 0.7
    assert all_max_proba[1] == 0.0


def test_flatten_all_map_sets_empty():
    all_maps, all_max_proba, all_is_null, sp_offsets = _flatten_all_map_sets(
        {}, n_species=3, ny=5, nx=5
    )
    assert all_maps.shape == (0, 5, 5)
    assert all_max_proba.shape == (0,)
    assert np.all(sp_offsets == -1)


def test_flatten_all_map_sets_bounds_check():
    """Species ID >= n_species raises ValueError."""
    ms = _mock_map_set([np.ones((2, 2))], np.zeros((3, 24), dtype=np.int32), np.array([0.0]))
    with pytest.raises(ValueError, match="species ID 5 exceeds"):
        _flatten_all_map_sets({5: ms}, n_species=3, ny=2, nx=2)


# ---------------------------------------------------------------------------
# _precompute_map_indices tests
# ---------------------------------------------------------------------------


def test_precompute_map_indices_basic():
    ny, nx = 3, 4
    idx_maps = np.full((5, 24), -1, dtype=np.int32)
    idx_maps[0, :] = 0
    idx_maps[1, :] = 1
    idx_maps[2, :] = 1
    ms = _mock_map_set([np.ones((ny, nx)), np.ones((ny, nx))], idx_maps, np.array([0.0, 0.0]))

    species_id = np.array([0, 0, 0], dtype=np.int32)
    age_dt = np.array([0, 1, 2], dtype=np.int32)
    uses_maps = np.array([True, True, True])

    current, same = _precompute_map_indices(species_id, age_dt, uses_maps, {0: ms}, step=2)

    assert current[0] == 0
    assert current[1] == 1
    assert current[2] == 1
    assert not same[0]
    assert not same[1]
    assert same[2]


def test_precompute_map_indices_out_of_range():
    idx_maps = np.zeros((3, 24), dtype=np.int32)
    ms = _mock_map_set([np.ones((2, 2))], idx_maps, np.array([0.0]))

    current, same = _precompute_map_indices(
        np.array([0], dtype=np.int32),
        np.array([10], dtype=np.int32),
        np.array([True]),
        {0: ms},
        step=0,
    )
    assert current[0] == -1


# ---------------------------------------------------------------------------
# _map_move_batch_numba tests (skip if Numba unavailable)
# ---------------------------------------------------------------------------

_numba_move = pytest.importorskip("osmose.engine.processes.movement", reason="Numba not available")
_map_move_batch_numba = getattr(_numba_move, "_map_move_batch_numba", None)
pytestmark = pytest.mark.skipif(
    _map_move_batch_numba is None, reason="Numba not available or function missing"
)


@pytest.fixture
def numba_base():
    """Common arrays for Numba movement tests."""
    ny, nx = 5, 5
    return SimpleNamespace(
        ny=ny,
        nx=nx,
        ocean_mask=np.ones((ny, nx), dtype=np.bool_),
        walk_range=np.array([1], dtype=np.int32),
        sp_offsets=np.array([0], dtype=np.int32),
    )


def _call_numba(
    seed,
    school_indices,
    map_idx,
    same_map,
    cx,
    cy,
    sp_ids,
    all_maps,
    all_max_proba,
    all_is_null,
    sp_offsets,
    ocean_mask,
    walk_range,
    ny,
    nx,
):
    """Helper to call _map_move_batch_numba with in-place output arrays."""
    out_cx = cx.copy()
    out_cy = cy.copy()
    out_is_out = np.zeros(len(cx), dtype=np.bool_)
    _map_move_batch_numba(
        seed,
        school_indices,
        map_idx,
        same_map,
        out_cx,
        out_cy,
        sp_ids,
        all_maps,
        all_max_proba,
        all_is_null,
        sp_offsets,
        ocean_mask,
        walk_range,
        ny,
        nx,
        out_cx,
        out_cy,
        out_is_out,
    )
    return out_cx, out_cy, out_is_out


def test_numba_new_placement(numba_base):
    """Rejection sampling places schools on cells with non-zero probability."""
    ny, nx = 3, 4
    prob_map = np.zeros((ny, nx), dtype=np.float64)
    prob_map[1, 2] = 0.8
    prob_map[2, 3] = 0.5

    cx, cy, is_out = _call_numba(
        42,
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([False]),
        np.array([-1], dtype=np.int32),
        np.array([-1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        prob_map.reshape(1, ny, nx),
        np.array([0.8]),
        np.array([False]),
        np.array([0], dtype=np.int32),
        np.ones((ny, nx), dtype=np.bool_),
        np.array([1], dtype=np.int32),
        ny,
        nx,
    )
    assert (cx[0], cy[0]) in [(2, 1), (3, 2)]
    assert not is_out[0]


def test_numba_out_of_domain(numba_base):
    """Null map → school goes out of domain."""
    ny, nx = 2, 2
    cx, cy, is_out = _call_numba(
        42,
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([False]),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.zeros((1, ny, nx), dtype=np.float64),
        np.array([0.0]),
        np.array([True]),
        np.array([0], dtype=np.int32),
        np.ones((ny, nx), dtype=np.bool_),
        np.array([1], dtype=np.int32),
        ny,
        nx,
    )
    assert cx[0] == -1
    assert cy[0] == -1
    assert is_out[0]


def test_numba_deterministic(numba_base):
    """Same seed produces same output."""
    b = numba_base
    prob_map = np.random.default_rng(0).random((b.ny, b.nx))
    n = 20
    school_indices = np.arange(n, dtype=np.int32)
    map_idx = np.zeros(n, dtype=np.int32)
    same_map_arr = np.zeros(n, dtype=np.bool_)
    sp_ids = np.zeros(n, dtype=np.int32)

    def run(seed):
        return _call_numba(
            seed,
            school_indices,
            map_idx,
            same_map_arr,
            np.full(n, -1, dtype=np.int32),
            np.full(n, -1, dtype=np.int32),
            sp_ids,
            prob_map.reshape(1, b.ny, b.nx),
            np.array([prob_map.max()]),
            np.array([False]),
            b.sp_offsets,
            b.ocean_mask,
            np.array([2], dtype=np.int32),
            b.ny,
            b.nx,
        )

    cx1, cy1, _ = run(99)
    cx2, cy2, _ = run(99)
    np.testing.assert_array_equal(cx1, cx2)
    np.testing.assert_array_equal(cy1, cy2)


def test_numba_random_walk(numba_base):
    """Same-map path: school walks to accessible neighbor cell."""
    b = numba_base
    prob_map = np.zeros((b.ny, b.nx), dtype=np.float64)
    prob_map[1:4, 1:4] = 0.5
    prob_map[2, 2] = 0.0  # force walk away from center

    cx, cy, is_out = _call_numba(
        42,
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([True]),
        np.array([2], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([0], dtype=np.int32),
        prob_map.reshape(1, b.ny, b.nx),
        np.array([0.5]),
        np.array([False]),
        b.sp_offsets,
        b.ocean_mask,
        b.walk_range,
        b.ny,
        b.nx,
    )
    accessible = [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (1, 3), (2, 3), (3, 3)]
    assert (cx[0], cy[0]) in accessible
    assert not is_out[0]


def test_numba_stranded(numba_base):
    """Same-map path with no accessible neighbors: school stays in place."""
    ny, nx = 3, 3
    prob_map = np.zeros((ny, nx), dtype=np.float64)
    prob_map[1, 1] = 0.5  # only center has positive probability

    cx, cy, is_out = _call_numba(
        42,
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([True]),
        np.array([1], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        prob_map.reshape(1, ny, nx),
        np.array([0.5]),
        np.array([False]),
        np.array([0], dtype=np.int32),
        np.ones((ny, nx), dtype=np.bool_),
        np.array([1], dtype=np.int32),
        ny,
        nx,
    )
    assert cx[0] == 1
    assert cy[0] == 1
    assert not is_out[0]


def test_round_based_sampling_has_boundary_bias():
    """Verify that int(round((n-1)*random())) is biased and our fix removes it.

    The old pattern int(round((n-1)*rand)) gives boundary cells half the
    probability of interior cells. rng.integers(0, n) is uniform.
    """
    n = 5
    rng = np.random.default_rng(42)

    # Old pattern — biased
    old_counts = np.zeros(n, dtype=np.int64)
    for _ in range(100_000):
        idx = int(round((n - 1) * rng.random()))
        old_counts[idx] += 1
    # Boundary cells (0 and n-1) should have ~half the hits of interior cells
    assert old_counts[0] < old_counts[2] * 0.7, "Old pattern should show boundary bias"

    # New pattern — uniform
    new_counts = np.zeros(n, dtype=np.int64)
    rng2 = np.random.default_rng(42)
    for _ in range(100_000):
        idx = rng2.integers(0, n)
        new_counts[idx] += 1
    expected = 100_000 / n
    for i in range(n):
        assert abs(new_counts[i] - expected) < expected * 0.1, (
            f"Cell {i} got {new_counts[i]} hits, expected ~{expected}"
        )
