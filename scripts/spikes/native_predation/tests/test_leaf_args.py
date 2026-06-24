from pathlib import Path

import numpy as np
import pytest

from scripts.spikes.native_predation.leaf_args import (
    build_leaf_args, load_capture, select_cells,
)

FIX = Path(__file__).resolve().parents[1] / "_fixtures" / "cellloop.npz"

pytestmark = pytest.mark.skipif(not FIX.exists(), reason="run capture.py first")


def test_select_cells_returns_four_valid_indices():
    arrays, meta = load_capture(FIX)
    sel = select_cells(arrays)
    assert set(sel) == {"p10", "p50", "p95", "small"}
    n_cells = len(arrays["boundaries"]) - 1
    for c in sel.values():
        assert 0 <= c < n_cells


def test_build_leaf_args_isolates_one_call_and_does_not_mutate_capture():
    arrays, meta = load_capture(FIX)
    sel = select_cells(arrays)
    before = np.copy(arrays["inst_abd"])
    args, p_idx = build_leaf_args(arrays, meta, sel["p50"])
    assert len(args) == 41  # full leaf signature
    # building args must not touch the captured arrays (fresh copies)
    assert np.array_equal(arrays["inst_abd"], before)
    # p_idx is a real live predator
    assert arrays["inst_abd"][p_idx] > 0
