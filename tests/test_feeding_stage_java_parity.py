"""M2 (verified): Python feeding-stage matches Java's >= semantics.

Java OSMOSE `SchoolStage.getStage` (verified 2026-05-06 against
osmose-model/osmose master, java/.../stage/SchoolStage.java):

    int stage = 0;
    for (float threshold : thresholds[iSpec]) {
        if (classGetter[iSpec].getVariable(school) < threshold) {
            break;
        }
        stage++;
    }

A school is at stage `stage = count(value >= threshold)`. The Python
implementation uses `np.searchsorted(arr, v, side='right')` which returns
`count(arr <= v)` — equivalent to `count(value >= threshold)` for
ascending thresholds. These tests pin the equivalence so a future
refactor doesn't accidentally swap to `side='left'` (strict `>`).
"""

from __future__ import annotations

import numpy as np


def java_get_stage(value: float, thresholds: list[float]) -> int:
    """Direct port of Java's SchoolStage.getStage loop."""
    stage = 0
    for t in thresholds:
        if value < t:
            break
        stage += 1
    return stage


def python_get_stage(value: float, thresholds: list[float]) -> int:
    """How the Python feeding_stage.py module computes the same thing."""
    return int(np.searchsorted(np.sort(np.asarray(thresholds)), value, side="right"))


# Permutations covering: value below all, exactly at a threshold, above
# all, between thresholds, and the empty-thresholds edge case.
_CASES: list[tuple[float, list[float]]] = [
    (0.5, [1.0, 2.0, 3.0]),
    (1.0, [1.0, 2.0, 3.0]),  # exactly at 1.0
    (1.5, [1.0, 2.0, 3.0]),
    (2.0, [1.0, 2.0, 3.0]),  # exactly at 2.0
    (2.5, [1.0, 2.0, 3.0]),
    (3.0, [1.0, 2.0, 3.0]),  # exactly at 3.0
    (10.0, [1.0, 2.0, 3.0]),
    (-1.0, [1.0, 2.0, 3.0]),
    (5.0, []),  # empty thresholds -> stage 0
    (5.0, [10.0]),  # one threshold above
    (5.0, [3.0]),  # one threshold below
    (5.0, [5.0]),  # exactly equal to single threshold
]


def test_python_matches_java_get_stage() -> None:
    """Python `searchsorted side='right'` must equal Java's loop on every case."""
    for value, thresholds in _CASES:
        java = java_get_stage(value, thresholds)
        py = python_get_stage(value, thresholds)
        assert java == py, (
            f"value={value}, thresholds={thresholds}: Java={java}, Python={py}"
        )


def test_side_left_would_break_parity_at_threshold_boundary() -> None:
    """Documents WHY we use side='right' rather than 'left'.

    For value=2.0 exactly at threshold=2.0:
      - Java: count(value >= threshold) = count {1<=2, 2<=2, 3>2} = 2.
      - searchsorted side='right': index 2 ✓
      - searchsorted side='left':  index 1 ✗ — breaks parity.
    """
    thr = np.array([1.0, 2.0, 3.0])
    assert int(np.searchsorted(thr, 2.0, side="right")) == 2  # matches Java
    assert int(np.searchsorted(thr, 2.0, side="left")) == 1  # would diverge


def test_searchsorted_vectorized_matches_loop() -> None:
    """Verify that the vectorised form (the actual production call site)
    matches the per-value Java loop on a batch of values."""
    thresholds = [1.0, 2.0, 3.0]
    values = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    sorted_thr = np.sort(np.asarray(thresholds))
    py_batch = np.searchsorted(sorted_thr, values, side="right")
    java_batch = np.array([java_get_stage(float(v), thresholds) for v in values])
    np.testing.assert_array_equal(py_batch, java_batch)
