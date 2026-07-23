"""Seeded LHS design executed through the Python engine, reduced for the UQ emulator.

The design runs through an INJECTABLE evaluator — a ``(point, seed) -> stat-dict``
callable — so the whole pipeline is testable without OSMOSE runs. The real
evaluator (``make_engine_evaluator``) runs the Python engine with SSB output
enabled; tests pass a synthetic function.

Two transforms live here and must not be conflated: base-10 ``10**val`` at the
simulator-input boundary (``point_to_overrides``, mirroring problem.py:263) and
natural ``np.log`` of the linear stat when forming the GP target Y (``run_design``).
"""

from __future__ import annotations

import numpy as np
from scipy.stats.qmc import LatinHypercube

from osmose.calibration.problem import FreeParameter, Transform


def point_to_overrides(x: np.ndarray, free_params: list[FreeParameter]) -> dict[str, str]:
    """One sampling-space point -> OSMOSE override dict.

    Applies base-10 ``10**val`` for ``Transform.LOG`` params (the simulator-input
    transform, matching osmose/calibration/problem.py:263) and stringifies every
    value. NOT the natural-log GP-target transform, which is separate.
    """
    overrides: dict[str, str] = {}
    for j, fp in enumerate(free_params):
        val = float(x[j])
        if fp.transform == Transform.LOG:
            val = 10.0**val
        overrides[fp.key] = str(val)
    return overrides


def lhs_design(free_params: list[FreeParameter], n_points: int, seed: int) -> np.ndarray:
    """Seeded Latin-hypercube design, ``(n_points, d)``, scaled to sampling-space bounds.

    No transform is applied — the design lives in sampling space; the emulator
    trains on sampling-space X and ``point_to_overrides`` transforms only at the
    simulator-input boundary.
    """
    d = len(free_params)
    unit = LatinHypercube(d=d, seed=seed).random(n=n_points)
    lower = np.array([fp.lower_bound for fp in free_params])
    upper = np.array([fp.upper_bound for fp in free_params])
    return unit * (upper - lower) + lower
