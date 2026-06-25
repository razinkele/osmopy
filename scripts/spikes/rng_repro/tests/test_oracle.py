import numpy as np

from scripts.spikes.rng_repro.oracle import cpython_reference, oracle_cell_rng


def test_oracle_matches_cpython_legacy_randomstate():
    # The premise (spec 0a): the @njit oracle == CPython legacy RandomState, bit-identical.
    # Seeds MUST span up to the large value the Task 3 parity grid uses (2**62+...): the
    # parity gate compares C-vs-oracle and ASSUMES oracle==RandomState there, so the premise
    # must be proven at that seed too — else a real Numba-vs-NumPy gap (a STOP/beta-signal)
    # would be misread as a C bug.
    for seed in (0, 1, 7919, 12345, 2**40 + 7919 * 3, 2**62 + 12345 + 990 * 7919):
        for n in (1, 2, 4, 12, 24, 33, 100):
            got = oracle_cell_rng(seed, n)
            ref = cpython_reference(seed, n)
            for g, r in zip(got, ref):
                assert np.array_equal(g, r), f"oracle != RandomState at seed={seed} n={n}"


def test_oracle_shuffle_carries_over_not_reset():
    # cause_orders rows must come from shuffling-in-place (carry-over), not re-shuffling
    # a fresh [0,1,2,3] each row. With carry-over, consecutive rows are (almost surely)
    # different permutations; this pins the in-place semantics.
    _, _, _, _, orders = oracle_cell_rng(12345, 20)
    assert orders.shape == (20, 4)
    assert {tuple(r) for r in orders}.issubset({(a, b, c, d) for a in range(4)
            for b in range(4) for c in range(4) for d in range(4)
            if len({a, b, c, d}) == 4})  # every row is a permutation of 0..3
