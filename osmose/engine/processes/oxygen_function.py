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


def f_o2_hill(o2: NDArray[np.float64], c50: float, n: float) -> NDArray[np.float64]:
    """Normalized Hill oxygen dose-response: f = O2^n / (O2^n + C50^n).

    Unlike ``f_o2`` (unnormalized Michaelis-Menten, used by bioenergetics), this
    asymptotes to 1.0 as O2 grows large, so a fully-oxygenated cell is not
    artificially discounted. f(C50) == 0.5 by construction; increasing `n`
    sharpens the transition into a threshold-like response.
    """
    o2n = np.asarray(o2, dtype=np.float64) ** n
    c50n = c50**n
    denom = o2n + c50n
    # Guard: when both o2 and c50 are zero, dose-response is zero (mirrors f_o2's style)
    guard = np.abs(denom) < 1e-30
    return np.where(
        guard,
        0.0,
        o2n / np.where(guard, 1.0, denom),
    )
