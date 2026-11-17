"""The Phase 1a candidate lists must be a superset of the schools the kernel accepts, in cell order.

``_cell_prey_candidates`` gives each predator the cell positions of the schools whose length can
pass its size window; ``_apply_predation_numba`` then runs its exact tests on that list only
(perf plan item 6). The kernel stays bit-identical only if every list (a) contains every school
the exact ratio test ``r_min <= pred_len / prey_len < r_max`` accepts, and (b) is in ascending cell
order, because ``total_available`` is a running sum. The whole-run parity baselines catch a list
missing a prey or out of order, but no baseline cell has a list longer than 32, the ``np.sort``
branch; these cells do.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.processes import mortality as M

pytestmark = pytest.mark.skipif(not M._HAS_NUMBA, reason="the candidate builder is a Numba kernel")


def _exact_accepts(pred_len: float, prey_len: float, r_min: float, r_max: float) -> bool:
    """The kernel's own size test (the abundance and self tests are applied after the list)."""
    if prey_len <= 0:
        return False
    ratio = pred_len / prey_len
    return not (ratio < r_min or ratio >= r_max)


def _build(lengths_per_cell, r_min, r_max, young=()):
    n_species = 1
    lengths = np.concatenate([np.asarray(c, dtype=np.float64) for c in lengths_per_cell])
    n = len(lengths)
    sizes = [len(c) for c in lengths_per_cell]
    boundaries = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    sorted_indices = np.arange(n, dtype=np.int32)
    species_id = np.zeros(n, dtype=np.int32)
    feeding_stage = np.zeros(n, dtype=np.int32)
    size_ratio_min = np.full((n_species, 1), r_min, dtype=np.float64)
    size_ratio_max = np.full((n_species, 1), r_max, dtype=np.float64)
    age_dt = np.full(n, 10, dtype=np.int32)
    first_feeding = np.ones(n, dtype=np.int32)
    for i in young:
        age_dt[i] = 0
    ptr, pos = M._cell_prey_candidates(
        sorted_indices,
        boundaries,
        len(sizes),
        lengths,
        species_id,
        feeding_stage,
        size_ratio_min,
        size_ratio_max,
        age_dt,
        first_feeding,
    )
    return lengths, boundaries, ptr, pos


def _check(lengths, boundaries, ptr, pos, r_min, r_max, young=()):
    n_long = 0
    for cell in range(len(boundaries) - 1):
        start, end = int(boundaries[cell]), int(boundaries[cell + 1])
        for p in range(end - start):
            j = start + p
            got = pos[ptr[j] : ptr[j + 1]].tolist()
            if j in young:
                assert got == [], "a predator too young to feed needs no list"
                continue
            assert got == sorted(set(got)), f"list of {j} not in ascending cell order: {got}"
            assert all(0 <= q < end - start for q in got)
            want = {
                q
                for q in range(end - start)
                if _exact_accepts(lengths[j], lengths[start + q], r_min, r_max)
            }
            assert want <= set(got), f"list of {j} misses accepted prey {sorted(want - set(got))}"
            n_long += len(got) > 32
    return n_long


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lists_are_supersets_in_cell_order_including_long_ones(seed):
    rng = np.random.default_rng(seed)
    cells = [rng.uniform(0.5, 60.0, size=k) for k in (1, 7, 40, 120, 300)]
    lengths, boundaries, ptr, pos = _build(cells, 1.0, 3.5)
    assert _check(lengths, boundaries, ptr, pos, 1.0, 3.5) > 0, "no list exercised np.sort"


def test_the_window_filters():
    """The lists must actually be short: a builder that listed every school would pass the
    superset test and save nothing."""
    rng = np.random.default_rng(3)
    _, _, ptr, _ = _build([rng.uniform(1.0, 100.0, size=200)], 2.0, 2.2)
    assert ptr[-1] < 0.2 * 200 * 200


def test_boundary_lengths_are_kept():
    """Prey exactly at the window's edges, where pred_len / prey_len rounds onto r_min or
    r_max, are decided by the exact test, so they must be on the list whenever it accepts them."""
    pred = 35.0
    r_min, r_max = 1.0, 3.5
    edge = [
        pred / r_max,
        pred / r_min,
        np.nextafter(pred / r_max, 0),
        np.nextafter(pred / r_min, 99),
    ]
    lengths, boundaries, ptr, pos = _build([[pred, *edge, 5.0, 50.0]], r_min, r_max)
    _check(lengths, boundaries, ptr, pos, r_min, r_max)


@pytest.mark.parametrize(
    ("r_min", "r_max"),
    [(0.0, 3.5), (-1.0, 3.5), (1.0, 0.0), (1.0, -2.0), (float("nan"), 3.5), (1.0, float("nan"))],
)
def test_degenerate_windows(r_min, r_max):
    rng = np.random.default_rng(4)
    cells = [np.concatenate([[0.0], rng.uniform(0.5, 60.0, size=50)])]  # a zero-length school too
    lengths, boundaries, ptr, pos = _build(cells, r_min, r_max, young=(3,))
    _check(lengths, boundaries, ptr, pos, r_min, r_max, young=(3,))


# Found by search: the exact test ACCEPTS a prey one ULP above fl(pred / r_min), because
# pred / prey rounds onto r_min. An unwidened binary-search bound would drop it.
_ROUNDING_CASES = [
    (88.63505378844101, 2.8864762342268433, 12.934402926611018, 30.707009722594286),
    (160.57423516494327, 2.808825399458674, 8.204817818075295, 57.16775246901772),
    (112.88413660346295, 1.1471614829273307, 6.056508193404794, 98.40300453202528),
]


@pytest.mark.parametrize(("pred", "r_min", "r_max", "prey"), _ROUNDING_CASES)
def test_a_prey_the_rounding_admits_is_on_the_list(pred, r_min, r_max, prey):
    assert prey > pred / r_min and _exact_accepts(pred, prey, r_min, r_max)  # the case holds
    _, _, ptr, pos = _build([[pred, prey]], r_min, r_max)
    assert 1 in pos[ptr[0] : ptr[1]].tolist()
