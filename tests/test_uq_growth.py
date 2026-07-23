"""Tests for the bounded design-growth loop + synthetic acceptance."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.uq.design import (
    DesignResult,
    _merge_designs,
    grow_until_calibrated,
)
from osmose.calibration.uq.gate import GateReport

# The misspecified acceptance case drives the GP to its length-scale bounds by
# design; silence the expected ConvergenceWarning so output stays pristine.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _fp2():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", 0.0, 1.0, Transform.LINEAR),
    ]


def _report(passed):
    return GateReport(
        n=10,
        coverage=0.95,
        mssr=1.0,
        pit_pvalue=0.5,
        r2=0.9,
        r2_ceiling=0.95,
        passed=passed,
        reasons=[],
    )


def test_merge_designs_concatenates_x_and_per_key_arrays():
    a = DesignResult(
        X=np.zeros((2, 2)),
        keys=["k"],
        Y={"k": np.array([1.0, 2.0])},
        alpha={"k": np.array([0.1, 0.2])},
    )
    b = DesignResult(
        X=np.ones((3, 2)),
        keys=["k"],
        Y={"k": np.array([3.0, 4.0, 5.0])},
        alpha={"k": np.array([0.3, 0.4, 0.5])},
    )
    m = _merge_designs(a, b)
    assert m.X.shape == (5, 2)
    assert np.array_equal(m.Y["k"], np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.array_equal(m.alpha["k"], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))


def test_growth_aborts_at_n_max_when_gate_always_fails():
    ev = lambda x, seed: {"k": 10.0}  # noqa: E731
    gate_fn = lambda X, Y, alpha, **kw: _report(False)  # noqa: E731
    result = grow_until_calibrated(
        ev, _fp2(), ["k"], n_seeds=2, n0=10, increment=10, n_max=25, seed=0, gate_fn=gate_fn
    )
    assert result.status == "aborted_n_max"
    assert len(result.design.X) <= 25
    assert result.rounds >= 1


def test_growth_returns_calibrated_after_one_append():
    ev = lambda x, seed: {"k": 10.0}  # noqa: E731
    calls = {"n": 0}

    def gate_fn(X, Y, alpha, **kw):
        calls["n"] += 1
        return _report(calls["n"] >= 2)  # fail first gate, pass the second

    result = grow_until_calibrated(
        ev, _fp2(), ["k"], n_seeds=2, n0=10, increment=10, n_max=100, seed=0, gate_fn=gate_fn
    )
    assert result.status == "calibrated"
    assert result.rounds == 1  # one append happened
    assert len(result.design.X) == 20


def _synthetic_evaluator(misspecified):
    def ev(x, seed):
        rng = np.random.default_rng(int(seed))
        if misspecified:
            mean_log = 5.0 if x[0] > 0.5 else 1.0
        else:
            mean_log = 2.0 + np.sin(3.0 * x[0]) + 0.5 * np.cos(4.0 * x[1])
        return {"cod_biomass_mean": float(np.exp(mean_log + rng.normal(0.0, 0.15)))}

    return ev


def test_growth_well_specified_synthetic_calibrates():
    result = grow_until_calibrated(
        _synthetic_evaluator(misspecified=False),
        _fp2(),
        ["cod_biomass_mean"],
        n_seeds=8,
        n0=60,
        increment=30,
        n_max=120,
        seed=0,
    )
    assert result.status == "calibrated"
    assert result.reports["cod_biomass_mean"].passed


def test_growth_misspecified_synthetic_aborts_loudly():
    result = grow_until_calibrated(
        _synthetic_evaluator(misspecified=True),
        _fp2(),
        ["cod_biomass_mean"],
        n_seeds=8,
        n0=60,
        increment=30,
        n_max=60,
        seed=0,
    )  # n_max == n0: one gate, then abort
    assert result.status == "aborted_n_max"
    assert not result.reports["cod_biomass_mean"].passed


def test_growth_rejects_nonpositive_increment():
    ev = lambda x, seed: {"k": 10.0}  # noqa: E731
    with pytest.raises(ValueError, match="positive"):
        grow_until_calibrated(ev, _fp2(), ["k"], n_seeds=2, n0=10, increment=0, n_max=100)


def test_growth_rejects_n0_exceeding_n_max():
    ev = lambda x, seed: {"k": 10.0}  # noqa: E731
    with pytest.raises(ValueError, match="n_max"):
        grow_until_calibrated(ev, _fp2(), ["k"], n_seeds=2, n0=50, increment=10, n_max=25)


def test_growth_rejects_empty_target_keys():
    ev = lambda x, seed: {"k": 10.0}  # noqa: E731
    with pytest.raises(ValueError, match="target_keys"):
        grow_until_calibrated(ev, _fp2(), [], n_seeds=2, n0=10, increment=10, n_max=100)
