"""Parity gate: C kernel vs Numba oracle must agree to <=1e-12 (op-order rounding)."""
from pathlib import Path

import pytest

from scripts.spikes.native_predation.leaf_args import load_capture, select_cells
from scripts.spikes.native_predation.parity import assert_parity, parity_for_cell

FIX = Path(__file__).resolve().parents[1] / "_fixtures" / "cellloop.npz"
pytestmark = pytest.mark.skipif(not FIX.exists(), reason="run capture.py + build_ffi.py first")


def test_c_matches_numba_to_op_order_rounding():
    arrays, meta = load_capture(FIX)
    sel = select_cells(arrays)
    for key in ("small", "p10", "p50", "p95"):
        report = parity_for_cell(arrays, meta, sel[key])
        assert_parity(report, bar=1e-12)
