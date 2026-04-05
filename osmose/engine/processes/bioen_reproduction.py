"""Gonad-weight-based reproduction for bioenergetic mode.
Matches Java BioenReproductionProcess.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def bioen_egg_production(
    gonad_weight: NDArray[np.float64],
    length: NDArray[np.float64],
    age_dt: NDArray[np.int32],
    m0: float,
    m1: float,
    egg_weight: float,
    n_dt_per_year: int,
) -> NDArray[np.float64]:
    """Compute number of eggs from gonad weight.

    Maturity by Linear Maturation Reaction Norm (LMRN):
        L_mature = m0 + m1 * age_years
    Fish is mature if length >= L_mature.
    Eggs = gonad_weight / egg_weight for mature fish with gonad > 0.
    """
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    l_mature = m0 + m1 * age_years
    is_mature = length >= l_mature
    safe_egg_weight = max(egg_weight, 1e-20)
    return np.where(is_mature & (gonad_weight > 0), gonad_weight / safe_egg_weight, 0.0)
