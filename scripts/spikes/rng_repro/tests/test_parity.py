import numpy as np
import pytest

from scripts.spikes.rng_repro.parity import GRID_N, GRID_SEEDS, compare_cell

try:
    from scripts.spikes.rng_repro import _rng_portable as R
    _HAVE_SO = True
except ImportError:
    _HAVE_SO = False

pytestmark = pytest.mark.skipif(not _HAVE_SO, reason="run build_ffi.py first")


def test_c_matches_numba_oracle_bit_exact_across_grid():
    for seed in GRID_SEEDS:
        for n in GRID_N:
            report = compare_cell(int(seed), n, R.lib, R.ffi)
            assert all(report.values()), f"C != oracle at seed={seed} n={n}: {report}"
