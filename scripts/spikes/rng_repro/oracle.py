"""Ground-truth RNG oracle: mirrors the cell-loop's per-cell draws (mortality.py:1479-1497).

The @njit oracle is what production actually draws; the CPython RandomState reference is
the documented NumPy-legacy MT19937 algorithm the C port targets (spec 0a proved they agree).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _njit_cell_rng(seed, n):
    np.random.seed(seed)
    seq_pred = np.random.permutation(n).astype(np.int32)
    seq_starv = np.random.permutation(n).astype(np.int32)
    seq_fish = np.random.permutation(n).astype(np.int32)
    seq_nat = np.random.permutation(n).astype(np.int32)
    causes = np.array([0, 1, 2, 3], dtype=np.int32)  # created ONCE
    cause_orders = np.empty((n, 4), dtype=np.int32)
    for i in range(n):
        np.random.shuffle(causes)  # in place — carries over from the previous row
        cause_orders[i, 0] = causes[0]
        cause_orders[i, 1] = causes[1]
        cause_orders[i, 2] = causes[2]
        cause_orders[i, 3] = causes[3]
    return seq_pred, seq_starv, seq_fish, seq_nat, cause_orders


def oracle_cell_rng(seed: int, n: int):
    return _njit_cell_rng(np.int64(seed), int(n))


def cpython_reference(seed: int, n: int):
    rs = np.random.RandomState(np.uint32(seed & 0xFFFFFFFF))  # legacy MT19937, uint32 seed
    a = rs.permutation(n).astype(np.int32)
    b = rs.permutation(n).astype(np.int32)
    c = rs.permutation(n).astype(np.int32)
    d = rs.permutation(n).astype(np.int32)
    causes = np.array([0, 1, 2, 3], dtype=np.int32)
    orders = np.empty((n, 4), dtype=np.int32)
    for i in range(n):
        rs.shuffle(causes)
        orders[i] = causes
    return a, b, c, d, orders
