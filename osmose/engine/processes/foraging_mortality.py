"""Foraging mortality for bioenergetic OSMOSE simulations.

Matches Java ForagingMortality.getRate():
- Without genetics: F = k_for / ndt_per_year (constant)
- With genetics: F = k1_for * exp(k2_for * (imax_trait - I_max)) / ndt_per_year

Config keys:
- species.bioen.forage.k_for.sp{i} (without genetics)
- species.bioen.forage.k1_for.sp{i} (with genetics)
- species.bioen.forage.k2_for.sp{i} (with genetics)
- predation.ingestion.rate.max.bioen.sp{i} (I_max reference)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def foraging_rate(
    k_for: NDArray[np.float64] | None,
    ndt_per_year: int,
    k1_for: NDArray[np.float64] | None = None,
    k2_for: NDArray[np.float64] | None = None,
    imax_trait: NDArray[np.float64] | None = None,
    I_max: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute foraging mortality rate per school.

    Two modes (matching Java ForagingMortality):
    - Constant (no genetics): k_for / ndt_per_year
    - Genetic: k1_for * exp(k2_for * (imax_trait - I_max)) / ndt_per_year

    Returns rate array, clamped to >= 0.
    """
    if k1_for is not None and k2_for is not None and imax_trait is not None and I_max is not None:
        # Genetic mode
        rate = k1_for * np.exp(k2_for * (imax_trait - I_max)) / ndt_per_year
    elif k_for is not None:
        # Constant mode
        rate = k_for / ndt_per_year
    else:
        raise ValueError(
            "Must provide either k_for (constant) or k1_for+k2_for+imax_trait+I_max (genetic)"
        )

    return np.maximum(rate, 0.0)
