"""Property-based tests: size-spectrum pure helpers."""

import math

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from osmose.size_spectrum import (
    _infer_bin_width,
    _large_fish_indicator,
    _mean_size,
    _window_by_time,
)
from tests.strategies import edges_and_values, shuffled_bin_edges, time_value_frames


@given(ev=edges_and_values())
def test_mean_size_convexity(ev):
    edges, values = ev  # use edges as the midpoint positions
    m = _mean_size(edges, values)
    if sum(values) > 0:
        assert min(edges) - 1e-9 <= m <= max(edges) + 1e-9
    else:
        assert math.isnan(m)


@given(ev=edges_and_values(), data=st.data())
def test_lfi_threshold_boundary(ev, data):
    edges, values = ev
    if sum(values) <= 0:
        assert _large_fish_indicator(edges, values, edges[0]) == 0.0
        return
    # Draw the threshold from an edge whose bin has POSITIVE value: the `edge >=
    # threshold` comparator counts that bin at thr==edge, so dropping it (thr just
    # above) strictly lowers LFI. (Same total denominator in both calls.)
    positive_edges = [e for e, v in zip(edges, values) if v > 0]
    thr = data.draw(st.sampled_from(positive_edges))
    incl = _large_fish_indicator(edges, values, thr)
    excl = _large_fish_indicator(edges, values, thr + 1e-6)
    assert incl > excl


@given(se=shuffled_bin_edges())
def test_bin_width_order_invariant(se):
    shuffled, canonical = se
    assert _infer_bin_width(shuffled) == _infer_bin_width(canonical)


@given(tv=time_value_frames(), w=st.integers(min_value=1, max_value=10))
def test_window_keeps_in_range(tv, w):
    out = _window_by_time(tv, "time", w)
    tmax = tv["time"].max()
    assert (out["time"] > tmax - w).all()
    assert len(out) <= len(tv)
