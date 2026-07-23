"""Tests for the UQ design executor (helpers, seed reduction, engine evaluator)."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.uq.design import lhs_design, point_to_overrides


def _params():
    return [
        FreeParameter("mortality.fishing.rate.sp0", 0.0, 2.0, Transform.LINEAR),
        FreeParameter("species.larva.mortality.rate.sp0", -3.0, 0.0, Transform.LOG),
    ]


def test_point_to_overrides_linear_passthrough():
    ov = point_to_overrides(np.array([1.5, -1.0]), _params())
    assert ov["mortality.fishing.rate.sp0"] == "1.5"


def test_point_to_overrides_log_is_base10():
    ov = point_to_overrides(np.array([1.5, -2.0]), _params())
    # LOG param: 10**(-2.0) == 0.01
    assert float(ov["species.larva.mortality.rate.sp0"]) == pytest.approx(0.01)


def test_point_to_overrides_all_keys_stringified():
    ov = point_to_overrides(np.array([0.3, -1.0]), _params())
    assert set(ov) == {"mortality.fishing.rate.sp0", "species.larva.mortality.rate.sp0"}
    assert all(isinstance(v, str) for v in ov.values())


def test_lhs_design_shape_and_bounds():
    X = lhs_design(_params(), n_points=25, seed=0)
    assert X.shape == (25, 2)
    assert np.all(X[:, 0] >= 0.0) and np.all(X[:, 0] <= 2.0)
    assert np.all(X[:, 1] >= -3.0) and np.all(X[:, 1] <= 0.0)


def test_lhs_design_deterministic():
    a = lhs_design(_params(), n_points=25, seed=7)
    b = lhs_design(_params(), n_points=25, seed=7)
    assert np.array_equal(a, b)
    c = lhs_design(_params(), n_points=25, seed=8)
    assert not np.array_equal(a, c)
