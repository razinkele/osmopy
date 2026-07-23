"""Uniform prior over the FreeParameter sampling-space box.

Uniform in SAMPLING space (where the emulator trains and the posterior is
evaluated). The base-10 simulator transform lives only at the simulator-input
boundary (Phase 1's point_to_overrides), so prior and posterior share one
measure — there is no Jacobian term.
"""

from __future__ import annotations

import math

import numpy as np

from osmose.calibration.problem import FreeParameter


def log_prior(theta: np.ndarray, free_params: list[FreeParameter]) -> float:
    """0.0 inside the box (bounds inclusive), -inf outside."""
    for j, fp in enumerate(free_params):
        if not (fp.lower_bound <= theta[j] <= fp.upper_bound):
            return -math.inf
    return 0.0
