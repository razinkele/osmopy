"""Tests for mortality RNG pre-generation."""

from __future__ import annotations

import numpy as np


def test_pre_generate_cell_rng_shapes():
    """Verify output shapes match cell structure."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    # 3 cells with 2, 0, 3 schools respectively
    boundaries = np.array([0, 2, 2, 5], dtype=np.int64)
    n_cells = 3

    seq_bufs, cause_orders_buf = _pre_generate_cell_rng(rng, boundaries, n_cells)

    assert len(seq_bufs) == 4
    for buf in seq_bufs:
        assert buf.shape == (5,)
        assert buf.dtype == np.int32
    assert cause_orders_buf.shape == (5, 4)
    assert cause_orders_buf.dtype == np.int32


def test_pre_generate_cell_rng_local_indices():
    """Verify permutation values are in [0, n_local) for each cell."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    boundaries = np.array([0, 4, 4, 7], dtype=np.int64)
    n_cells = 3

    seq_bufs, _ = _pre_generate_cell_rng(rng, boundaries, n_cells)

    # Cell 0: 4 schools → values in [0, 4)
    for buf in seq_bufs:
        cell0 = buf[0:4]
        assert np.all(cell0 >= 0) and np.all(cell0 < 4)
        assert sorted(cell0) == [0, 1, 2, 3]  # permutation

    # Cell 1: 0 schools → nothing to check

    # Cell 2: 3 schools → values in [0, 3)
    for buf in seq_bufs:
        cell2 = buf[4:7]
        assert np.all(cell2 >= 0) and np.all(cell2 < 3)
        assert sorted(cell2) == [0, 1, 2]


def test_pre_generate_cell_rng_cause_orders_valid():
    """Each cause_orders row must be a permutation of [0,1,2,3]."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    boundaries = np.array([0, 3, 6], dtype=np.int64)
    n_cells = 2

    _, cause_orders_buf = _pre_generate_cell_rng(rng, boundaries, n_cells)

    for i in range(6):
        assert sorted(cause_orders_buf[i]) == [0, 1, 2, 3]


def test_pre_generate_cell_rng_deterministic():
    """Same seed must produce identical output."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    boundaries = np.array([0, 3, 5], dtype=np.int64)

    rng1 = np.random.default_rng(99)
    s1, c1 = _pre_generate_cell_rng(rng1, boundaries, 2)

    rng2 = np.random.default_rng(99)
    s2, c2 = _pre_generate_cell_rng(rng2, boundaries, 2)

    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(c1, c2)
