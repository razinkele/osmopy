"""Dissolved oxygen dose-response function."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def f_o2(o2: NDArray[np.float64], c1: float, c2: float) -> NDArray[np.float64]:
    """Oxygen dose-response: f_O2 = C1 * O2 / (O2 + C2)."""
    denom = o2 + c2
    # Guard: when both o2 and c2 are zero, dose-response is zero
    guard = np.abs(denom) < 1e-30
    return np.where(
        guard,
        0.0,
        c1 * o2 / np.where(guard, 1.0, denom),
    )
