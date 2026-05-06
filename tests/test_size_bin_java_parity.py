"""M3 (verified): Python size-bin count matches Java's `OutputDistribution`.

Java OSMOSE `OutputDistribution.initialize` (verified 2026-05-06 against
osmose-model/osmose master, java/.../output/distribution/OutputDistribution.java):

    int nClass = (int) Math.ceil((max - min) / incr) + 1;

i.e. Java's bin count is `ceil((max - min) / incr) + 1`, with the trailing
+1 reserved as an overflow bin for values >= max. Python's
implementation in `simulate._collect_outputs`:

    edges = np.arange(min, max + incr, incr)
    n_bins = len(edges)

These tests pin the equivalence so a future refactor doesn't accidentally
break parity. The plan's r1 sketch (`int((max - min) / incr)` — floor)
was wrong-direction; floor would under-count by 1 vs Java's ceil + 1.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def java_nclass(mn: float, mx: float, incr: float) -> int:
    """Direct port of Java's OutputDistribution.initialize line."""
    return int(math.ceil((mx - mn) / incr)) + 1


def python_n_bins(mn: float, mx: float, incr: float) -> int:
    """How simulate._collect_outputs computes bin count."""
    return len(np.arange(mn, mx + incr, incr))


# Cases covering: clean integer division, non-integer division, float-precision
# edges, off-by-one boundaries, large-grid case from EEC-style configs.
_CASES = [
    # (min, max, incr, expected — pre-computed from Java formula)
    (0.0, 30.0, 5.0, 7),
    (5.0, 30.0, 5.0, 6),
    (5.0, 30.0, 7.0, 5),
    (5.0, 30.0, 10.0, 4),
    (0.0, 30.0, 10.0, 4),
    (0.0, 0.3, 0.1, 4),  # float-precision: 0.3/0.1 == 2.9999... in IEEE 754
    (0.0, 1.0, 0.1, 11),  # classic float-step accumulation case
    (0.0, 10.0, 1.0, 11),
    (0.0, 100.0, 10.0, 11),
    (0.0, 100.0, 7.5, 15),
    (0.5, 50.5, 5.0, 11),  # offset start: (50.5-0.5)/5 = 10, ceil + 1 = 11
    (1.0, 100.0, 1.0, 100),  # large grid
]


@pytest.mark.parametrize("mn,mx,incr,expected", _CASES)
def test_java_formula_matches_expected(
    mn: float, mx: float, incr: float, expected: int
) -> None:
    """Sanity: the Java port produces the documented expected count."""
    assert java_nclass(mn, mx, incr) == expected


@pytest.mark.parametrize("mn,mx,incr,expected", _CASES)
def test_python_n_bins_matches_java(
    mn: float, mx: float, incr: float, expected: int
) -> None:
    """The production Python expression produces the same count as Java."""
    py = python_n_bins(mn, mx, incr)
    j = java_nclass(mn, mx, incr)
    assert py == j == expected, (
        f"min={mn}, max={mx}, incr={incr}: Python={py}, Java={j}, expected={expected}"
    )


def test_python_floor_formula_would_undercount() -> None:
    """Documents WHY we use np.arange-len rather than the plan's r1 floor sketch.

    The plan originally proposed `int((max - min) / incr)` — floor. For
    most inputs this UNDER-counts by exactly 1 vs Java's ceil + 1.
    """
    # 0..30 step 5: floor=6, Java=7 -> diverge by 1
    assert int((30.0 - 0.0) / 5.0) == 6
    assert java_nclass(0.0, 30.0, 5.0) == 7

    # 0..100 step 10: floor=10, Java=11
    assert int((100.0 - 0.0) / 10.0) == 10
    assert java_nclass(0.0, 100.0, 10.0) == 11
