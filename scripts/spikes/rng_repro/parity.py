"""Bit-exact parity: C cell_rng vs the @njit oracle, across an n x seed grid."""
from __future__ import annotations

import numpy as np

from .oracle import oracle_cell_rng

GRID_N = [1, 2, 3, 4, 5, 8, 12, 24, 33, 50, 100]
GRID_SEEDS = [0, 1, 7919, 12345] + [
    rs + c * 7919 for (rs, c) in [(0, 0), (0, 1), (987654321, 5), (2**62 + 12345, 990)]
]


def c_cell_rng(seed: int, n: int, lib, ffi):
    """Marshal C cell_rng call with int32 out-buffers."""
    a = np.empty(n, dtype=np.int32)
    b = np.empty(n, dtype=np.int32)
    c = np.empty(n, dtype=np.int32)
    d = np.empty(n, dtype=np.int32)
    orders = np.empty(n * 4, dtype=np.int32)
    cast = lambda arr: ffi.cast("int32_t *", arr.ctypes.data)  # noqa: E731
    # seed is int64 in the C signature; pass the Python int directly (cffi coerces).
    lib.cell_rng(int(seed), n, cast(a), cast(b), cast(c), cast(d), cast(orders))
    return a, b, c, d, orders.reshape(n, 4)


def compare_cell(seed: int, n: int, lib, ffi) -> dict[str, bool]:
    """Compare C cell_rng vs oracle cell_rng, bit-exact per array."""
    got = c_cell_rng(seed, n, lib, ffi)
    ref = oracle_cell_rng(seed, n)
    names = ("seq_pred", "seq_starv", "seq_fish", "seq_nat", "cause_orders")
    return {name: bool(np.array_equal(g, r)) for name, g, r in zip(names, got, ref)}
