"""Salinity-dependent occupancy weighting for movement (prototype spike).

Pure helpers: a salinity -> [0,1] occupancy weight and its application to a
2D movement map with an all-zero guard. See
docs/superpowers/specs/2026-07-04-salinity-gated-cod-occupancy-design.md.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def salinity_weight(
    salinity: NDArray[np.float64] | float, s_low: float, s_high: float
) -> NDArray[np.float64] | float:
    """Per-cell occupancy weight in [0,1]: clip((S - s_low)/(s_high - s_low), 0, 1).

    Weight 0 at/below s_low (predator excluded), 1 at/above s_high (full), linear
    between. Accepts a scalar or an ndarray. Raises ValueError if s_high <= s_low.
    """
    if s_high <= s_low:
        raise ValueError(f"salinity_weight: s_high ({s_high}) must be > s_low ({s_low})")
    return np.clip((np.asarray(salinity, dtype=np.float64) - s_low) / (s_high - s_low), 0.0, 1.0)


def salinity_weighted_map(
    map2d: NDArray[np.float64], weight_grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply a precomputed per-cell weight to a movement map.

    Returns ``map2d * weight_grid``. If the product has no positive finite cell,
    returns the ORIGINAL ``map2d`` object unchanged (all-zero guard) so a predator
    is never left with zero valid cells; callers detect the fallback by identity
    (``result is map2d``).
    """
    wmap = map2d * weight_grid
    finite_pos = np.isfinite(wmap) & (wmap > 0.0)
    if not finite_pos.any():
        return map2d
    return wmap
