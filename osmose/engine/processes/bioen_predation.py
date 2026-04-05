"""Bioenergetic predation — allometric ingestion cap.
Matches Java BioenPredationMortality.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def bioen_ingestion_cap(
    weight: NDArray[np.float64],
    i_max: float,
    beta: float,
    n_dt_per_year: int,
    n_subdt: int,
    is_larvae: NDArray[np.bool_],
    theta: float = 1.0,
    c_rate: float = 0.0,
) -> NDArray[np.float64]:
    """Compute max ingestion per sub-timestep for bioen mode.

    Adults: I_max * w_g^beta / (n_dt * subdt)
    Larvae: (I_max + (theta-1)*c_rate) * w_g^beta / (n_dt * subdt)

    Weight is converted from tonnes to grams internally.
    """
    w_grams = weight * 1e6
    i_eff = np.where(is_larvae, i_max + (theta - 1.0) * c_rate, i_max)
    return i_eff * np.power(w_grams, beta) / (n_dt_per_year * n_subdt)
