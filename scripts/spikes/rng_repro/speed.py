# scripts/spikes/rng_repro/speed.py
"""Boundary-free per-cell RNG-gen timing: C cell_rng_bench vs an @njit driver."""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from .oracle import _njit_cell_rng


@njit(cache=True)
def _numba_bench(seed, n, n_iter):
    acc = np.int64(0)
    for _ in range(n_iter):
        a, b, c, d, orders = _njit_cell_rng(seed, n)
        acc += a[0] + b[0] + c[0] + d[0] + orders[0, 0]  # defeat dead-code elimination
    return acc


def bench_rng(n: int, n_iter: int, n_samples: int, lib, ffi) -> dict:
    seed = np.int64(12345)
    out = [np.empty(n if k < 4 else n * 4, dtype=np.int32) for k in range(5)]
    cast = lambda a: ffi.cast("int32_t *", a.ctypes.data)  # noqa: E731
    cargs = (int(seed), n, int(n_iter), cast(out[0]), cast(out[1]),
             cast(out[2]), cast(out[3]), cast(out[4]))

    _numba_bench(seed, n, 1)             # warm JIT
    lib.cell_rng_bench(*cargs)           # warm C

    numba_ns, c_ns = [], []
    for _ in range(n_samples):           # interleaved A/B to cancel drift
        t = time.perf_counter_ns(); _numba_bench(seed, n, n_iter)
        numba_ns.append((time.perf_counter_ns() - t) / n_iter)
        t = time.perf_counter_ns(); lib.cell_rng_bench(*cargs)
        c_ns.append((time.perf_counter_ns() - t) / n_iter)

    def med_iqr(xs):
        s = sorted(xs)
        med = s[len(s) // 2]
        iqr = s[int(len(s) * 0.75)] - s[int(len(s) * 0.25)]
        return med, iqr

    nm, ni = med_iqr(numba_ns)
    cm, ci = med_iqr(c_ns)
    return {"numba_med_ns": nm, "numba_iqr_ns": ni, "c_med_ns": cm,
            "c_iqr_ns": ci, "ratio": (nm / cm if cm else float("inf")), "n": n}
