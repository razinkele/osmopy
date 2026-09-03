"""Tests for mortality RNG pre-generation.

``_pre_generate_cell_rng`` must consume randomness identically to the reference
per-cell loop, ``_mortality_in_cell`` (mortality.py:~1729-1758): FIVE independent
permutations per cell (``seq_pred, seq_starv, seq_fish, seq_nat, seq_for`` — drawn
unconditionally, even when bioen is off and ``seq_for`` goes unused), followed by
``n_local`` calls to ``rng.shuffle`` on a cause list obtained fresh from
``_get_mortality_causes(config)`` (4 causes without bioen, 5 -- FORAGING included --
with it). Before 2026-09, this helper drew only 4 permutations and shuffled a
hard-coded ``[PRED, STARV, ADDITIONAL, FISHING]`` literal, which was already out of
step with the reference's 5-draw pattern even before bioen existed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

_NON_BIOEN = SimpleNamespace(bioen_enabled=False)
_BIOEN = SimpleNamespace(bioen_enabled=True)


def test_pre_generate_cell_rng_shapes():
    """Verify output shapes match cell structure (non-bioen: 4 causes)."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    # 3 cells with 2, 0, 3 schools respectively
    boundaries = np.array([0, 2, 2, 5], dtype=np.int64)
    n_cells = 3

    seq_bufs, cause_orders_buf = _pre_generate_cell_rng(rng, boundaries, n_cells, _NON_BIOEN)

    assert len(seq_bufs) == 5
    for buf in seq_bufs:
        assert buf.shape == (5,)
        assert buf.dtype == np.int32
    assert cause_orders_buf.shape == (5, 4)
    assert cause_orders_buf.dtype == np.int32


def test_pre_generate_cell_rng_shapes_bioen():
    """Under bioen, the cause-order buffer widens to 5 (FORAGING included)."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    boundaries = np.array([0, 2, 2, 5], dtype=np.int64)
    n_cells = 3

    seq_bufs, cause_orders_buf = _pre_generate_cell_rng(rng, boundaries, n_cells, _BIOEN)

    assert len(seq_bufs) == 5  # seq_bufs count is unconditional, bioen or not
    assert cause_orders_buf.shape == (5, 5)
    assert cause_orders_buf.dtype == np.int32


def test_pre_generate_cell_rng_local_indices():
    """Verify permutation values are in [0, n_local) for each cell."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    boundaries = np.array([0, 4, 4, 7], dtype=np.int64)
    n_cells = 3

    seq_bufs, _ = _pre_generate_cell_rng(rng, boundaries, n_cells, _NON_BIOEN)

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
    """Each cause_orders row must be a permutation of [0,1,2,3] (non-bioen)."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    rng = np.random.default_rng(42)
    boundaries = np.array([0, 3, 6], dtype=np.int64)
    n_cells = 2

    _, cause_orders_buf = _pre_generate_cell_rng(rng, boundaries, n_cells, _NON_BIOEN)

    for i in range(6):
        assert sorted(cause_orders_buf[i]) == [0, 1, 2, 3]


def test_pre_generate_cell_rng_cause_orders_valid_bioen():
    """Each cause_orders row must be a permutation of the bioen cause set (FORAGING=5)."""
    from osmose.engine.processes.mortality import _FORAGING, _pre_generate_cell_rng

    assert _FORAGING == 5  # MortalityCause.FORAGING; OUT=4 is handled post-loop, not in this set
    rng = np.random.default_rng(42)
    boundaries = np.array([0, 3, 6], dtype=np.int64)
    n_cells = 2

    _, cause_orders_buf = _pre_generate_cell_rng(rng, boundaries, n_cells, _BIOEN)

    for i in range(6):
        assert sorted(cause_orders_buf[i]) == [0, 1, 2, 3, 5]


def test_pre_generate_cell_rng_deterministic():
    """Same seed must produce identical output."""
    from osmose.engine.processes.mortality import _pre_generate_cell_rng

    boundaries = np.array([0, 3, 5], dtype=np.int64)

    rng1 = np.random.default_rng(99)
    s1, c1 = _pre_generate_cell_rng(rng1, boundaries, 2, _NON_BIOEN)

    rng2 = np.random.default_rng(99)
    s2, c2 = _pre_generate_cell_rng(rng2, boundaries, 2, _NON_BIOEN)

    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(c1, c2)


def _reference_draw(rng, n_local, causes_template):
    """Replicate `_mortality_in_cell`'s per-cell RNG consumption by hand.

    Five permutations (seq_pred, seq_starv, seq_fish, seq_nat, seq_for), THEN
    `n_local` calls to `rng.shuffle` on a cause list -- exactly the sequence
    `_mortality_in_cell` performs for one cell (mortality.py:~1729-1758).
    """
    seqs = [rng.permutation(n_local).astype(np.int32) for _ in range(5)]
    causes = list(causes_template)
    orders = np.empty((n_local, len(causes)), dtype=np.int32)
    for i in range(n_local):
        rng.shuffle(causes)
        orders[i, :] = causes
    return seqs, orders


def test_pre_generate_cell_rng_matches_reference_stream_non_bioen():
    """The helper must consume EXACTLY the randomness `_mortality_in_cell` would.

    This is the equivalence test the docstring's "tested reference implementation"
    claim depends on: not just matching shapes, but matching the RNG stream draw
    for draw, cell for cell -- proven by requiring a SUBSEQUENT draw from both
    generators to still agree.
    """
    from osmose.engine.processes.mortality import (
        _ADDITIONAL,
        _FISHING,
        _PREDATION,
        _STARVATION,
        _pre_generate_cell_rng,
    )

    boundaries = np.array([0, 3, 3, 7], dtype=np.int64)  # cells of 3, 0, 4
    n_cells = 3
    causes_template = [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING]

    rng_helper = np.random.default_rng(2024)
    seq_bufs, cause_orders_buf = _pre_generate_cell_rng(rng_helper, boundaries, n_cells, _NON_BIOEN)

    rng_ref = np.random.default_rng(2024)
    expected_seqs = [np.empty(7, dtype=np.int32) for _ in range(5)]
    expected_orders = np.empty((7, 4), dtype=np.int32)
    for cell, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        n_local = int(end - start)
        if n_local == 0:
            continue
        seqs, orders = _reference_draw(rng_ref, n_local, causes_template)
        for k in range(5):
            expected_seqs[k][start:end] = seqs[k]
        expected_orders[start:end] = orders

    for got, want in zip(seq_bufs, expected_seqs):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(cause_orders_buf, expected_orders)

    # Equal consumption: the next draw from each generator must still agree.
    assert rng_helper.integers(0, 2**31) == rng_ref.integers(0, 2**31)


def test_pre_generate_cell_rng_matches_reference_stream_bioen():
    """Same stream-equivalence proof, under bioen (5 causes, FORAGING included)."""
    from osmose.engine.processes.mortality import (
        _ADDITIONAL,
        _FISHING,
        _FORAGING,
        _PREDATION,
        _STARVATION,
        _pre_generate_cell_rng,
    )

    boundaries = np.array([0, 2, 5], dtype=np.int64)  # cells of 2, 3
    n_cells = 2
    causes_template = [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING, _FORAGING]

    rng_helper = np.random.default_rng(7)
    seq_bufs, cause_orders_buf = _pre_generate_cell_rng(rng_helper, boundaries, n_cells, _BIOEN)

    rng_ref = np.random.default_rng(7)
    expected_seqs = [np.empty(5, dtype=np.int32) for _ in range(5)]
    expected_orders = np.empty((5, 5), dtype=np.int32)
    for cell, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        n_local = int(end - start)
        if n_local == 0:
            continue
        seqs, orders = _reference_draw(rng_ref, n_local, causes_template)
        for k in range(5):
            expected_seqs[k][start:end] = seqs[k]
        expected_orders[start:end] = orders

    for got, want in zip(seq_bufs, expected_seqs):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(cause_orders_buf, expected_orders)

    assert rng_helper.integers(0, 2**31) == rng_ref.integers(0, 2**31)
