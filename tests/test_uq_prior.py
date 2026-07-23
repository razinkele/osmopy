"""Tests for the uniform sampling-space prior."""

from __future__ import annotations

import math

import numpy as np

from osmose.calibration.problem import FreeParameter, Transform
from osmose.calibration.uq.prior import log_prior


def _params():
    return [
        FreeParameter("a.sp0", 0.0, 1.0, Transform.LINEAR),
        FreeParameter("b.sp0", -2.0, 2.0, Transform.LOG),
    ]


def test_log_prior_inside_box_is_zero():
    assert log_prior(np.array([0.5, 0.0]), _params()) == 0.0


def test_log_prior_outside_box_is_neg_inf():
    assert log_prior(np.array([1.5, 0.0]), _params()) == -math.inf
    assert log_prior(np.array([0.5, -3.0]), _params()) == -math.inf


def test_log_prior_on_boundary_is_included():
    assert log_prior(np.array([0.0, 2.0]), _params()) == 0.0
    assert log_prior(np.array([1.0, -2.0]), _params()) == 0.0


def test_log_prior_nan_is_neg_inf():
    assert log_prior(np.array([np.nan, 0.0]), _params()) == -math.inf
