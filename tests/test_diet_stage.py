"""Java's diet stages (osmose/engine/diet_stage.py; SchoolStage, DietOutput, PredatorPressureOutput,
BiomassDietStageOutput)."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine import diet_stage as m
from osmose.engine.diet_stage import DietStages, parse_thresholds


def _ds(thresholds, structure, n_rsc=1):
    return DietStages(
        [np.asarray(t, dtype=np.float32) for t in thresholds],
        np.asarray(structure, dtype=np.int32),
        n_rsc,
    )


def test_stage_counts_leading_thresholds_at_or_below_in_float32():
    ds = _ds([[10.0, 30.0]], [1])
    v = np.array([0.0, 9.99, 10.0, 29.999, 30.0, 99.0], dtype=np.float32)
    assert ds.stage(np.zeros(6, int), v).tolist() == [0, 0, 1, 1, 2, 2]
    # a float64 length just below 10 that rounds to 10.0 in float32 is stage 1, as in Java
    just_below = np.array([np.nextafter(10.0, 0.0)], dtype=np.float64).astype(np.float32)
    assert ds.stage(np.zeros(1, int), just_below).tolist() == [1]


def test_structures_and_columns():
    ds = _ds([[1.0], [10.0], [5.0], []], [0, 1, 2, 3], n_rsc=2)
    assert ds.n_stages.tolist() == [2, 2, 2, 1] and ds.width == 7 + 2
    v = ds.values(
        np.array([0, 1, 2, 3]),
        age_dt=np.array([48, 0, 0, 0]),  # 2 years at 24 steps
        length=np.array([0.0, 12.0, 0.0, 0.0]),
        weight=np.array([0.0, 0.0, 6e-6, 0.0]),  # 6 g
        trophic_level=np.array([0.0, 0.0, 0.0, 3.2]),
        n_dt_per_year=24,
    )
    assert v.dtype == np.float32 and v.tolist() == pytest.approx([2.0, 12.0, 6.0, 3.2], rel=1e-6)
    assert ds.prey_columns(np.array([0, 1, 2, 3]), v).tolist() == [1, 3, 5, 6]


@pytest.mark.parametrize("bad", ["30;10", "10;10", "nan"])
def test_thresholds_must_be_increasing_and_finite(bad):
    with pytest.raises(ValueError, match="threshold"):
        parse_thresholds(bad, "output.diet.stage.threshold.sp0")


def test_an_unknown_structure_is_refused():
    from osmose.engine.config import _parse_diet_stages

    with pytest.raises(ValueError, match="output.diet.stage.structure"):
        _parse_diet_stages({"output.diet.stage.structure": "volume"}, 1, [])


def test_numba_and_numpy_accumulate_identically():
    rng = np.random.default_rng(3)
    n, w, rows = 200, 9, 5
    r = rng.integers(0, rows, n).astype(np.int64)
    a = rng.uniform(0, 10, n)
    x = rng.uniform(0, 1, (n, w)) * (rng.random((n, w)) < 0.4)
    d_in, p_in = rng.random(n) < 0.8, rng.random(n) < 0.9
    outs = []
    for fn in (m._accumulate_numpy, m._numba_kernel()):
        num, den, pp = np.zeros((rows, w)), np.zeros(rows), np.zeros((rows, w))
        fn(r, a, x, d_in, p_in, num, den, pp)
        outs.append((num.tobytes(), den.tobytes(), pp.tobytes()))
    assert outs[0] == outs[1]


def test_java_weighting_is_abundance_weighted_fractions():
    """Two predator schools of one stage: diet = sum(A * prey/preyed) / sum(A), not biomass."""
    ds = _ds([[], []], [1, 1], n_rsc=0)
    num, den, pp = np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2))
    rows = np.array([[1.0, 0.0], [0.0, 100.0]])  # school 1 ate prey 0, school 2 ate prey 1
    m._accumulate(ds, np.array([0, 1]), np.array([0, 0]), np.array([0, 0]), np.array([3.0, 1.0]),
                  rows, np.ones(2, bool), np.ones(2, bool), num, den, pp)  # fmt: skip
    pct = 100 * num[0] / den[0]
    assert pct.tolist() == [75.0, 25.0]  # by abundance (3:1), whatever the biomass eaten
    assert pp[0].tolist() == [1.0, 100.0]  # pressure is the biomass
