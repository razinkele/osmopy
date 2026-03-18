"""Growth process functions for the OSMOSE Python engine.

Von Bertalanffy and Gompertz growth, with predation-success gating.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState


def expected_length_vb(
    age_dt: NDArray[np.int32],
    linf: NDArray[np.float64],
    k: NDArray[np.float64],
    t0: NDArray[np.float64],
    egg_size: NDArray[np.float64],
    vb_threshold_age: NDArray[np.float64],
    n_dt_per_year: int,
) -> NDArray[np.float64]:
    """Compute Von Bertalanffy expected length at a given age.

    Three phases:
      age == 0:             L_egg
      0 < age < a_thres:    linear interpolation from L_egg to L_thres
      age >= a_thres:        L_inf * (1 - exp(-K * (age - t0)))
    """
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    threshold_years = vb_threshold_age

    # Standard VB formula (used for age >= threshold AND for computing L_thres)
    l_vb = linf * (1 - np.exp(-k * (age_years - t0)))
    l_thres = linf * (1 - np.exp(-k * (threshold_years - t0)))

    # Linear phase for young-of-year
    frac = np.where(threshold_years > 0, age_years / threshold_years, 1.0)
    l_linear = egg_size + (l_thres - egg_size) * frac

    # Select phase
    result = np.where(age_dt == 0, egg_size, np.where(age_years < threshold_years, l_linear, l_vb))
    return result


def growth(state: SchoolState, config: EngineConfig, rng: np.random.Generator) -> SchoolState:
    """Apply Von Bertalanffy growth gated by predation success.

    Special cases:
    - Eggs (age_dt == 0): always get mean delta_L (bypass gating)
    - Out-of-domain schools: always get mean delta_L
    """
    if len(state) == 0:
        return state

    sp = state.species_id
    n_dt = config.n_dt_per_year

    # Expected length at current and next age
    l_current = expected_length_vb(
        state.age_dt,
        config.linf[sp],
        config.k[sp],
        config.t0[sp],
        config.egg_size[sp],
        config.vb_threshold_age[sp],
        n_dt,
    )
    l_next = expected_length_vb(
        state.age_dt + 1,
        config.linf[sp],
        config.k[sp],
        config.t0[sp],
        config.egg_size[sp],
        config.vb_threshold_age[sp],
        n_dt,
    )
    delta_l = l_next - l_current  # mean length increment

    # Growth factor gated by predation success
    csr = config.critical_success_rate[sp]
    sr = state.pred_success_rate
    max_delta = config.delta_lmax_factor[sp] * delta_l

    # Gated growth: 0 below critical, linear scaling above
    growth_factor = np.where(
        sr >= csr,
        max_delta * (sr - csr) / np.where(csr < 1.0, 1.0 - csr, 1.0),
        0.0,
    )

    # Special cases: eggs and out-of-domain always get mean delta_L
    bypass = (state.age_dt == 0) | state.is_out
    growth_factor = np.where(bypass, delta_l, growth_factor)

    # Apply growth, cap at L_inf. Design decision: L_inf is used as L_max because
    # OSMOSE does not define a separate lmax parameter distinct from linf.
    new_length = np.minimum(state.length + growth_factor, config.linf[sp])

    # Update weight from new length: W = c * L^b
    new_weight = config.condition_factor[sp] * new_length ** config.allometric_power[sp]

    # Update biomass
    new_biomass = state.abundance * new_weight

    return state.replace(length=new_length, weight=new_weight, biomass=new_biomass)
