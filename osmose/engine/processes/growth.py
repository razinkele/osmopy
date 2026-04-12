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
    safe_threshold = np.where(threshold_years > 0, threshold_years, 1.0)
    frac = np.where(threshold_years > 0, age_years / safe_threshold, 1.0)
    l_linear = egg_size + (l_thres - egg_size) * frac

    # Select phase
    result = np.where(age_dt == 0, egg_size, np.where(age_years < threshold_years, l_linear, l_vb))
    return result


def growth(state: SchoolState, config: EngineConfig, rng: np.random.Generator) -> SchoolState:
    """Apply growth gated by predation success (dispatches VB or Gompertz by species class).

    Special cases:
    - Eggs (age_dt == 0): always get mean delta_L (bypass gating)
    - Out-of-domain schools: always get mean delta_L
    """
    if len(state) == 0:
        return state

    sp = state.species_id
    n_dt = config.n_dt_per_year

    # Expected length at current and next age (dispatches by growth class)
    l_current = _expected_length(state.age_dt, sp, config, n_dt)
    l_next = _expected_length(state.age_dt + 1, sp, config, n_dt)
    delta_l = l_next - l_current  # mean length increment

    # Growth factor gated by predation success
    csr = config.critical_success_rate[sp]
    sr = state.pred_success_rate
    max_delta = config.delta_lmax_factor[sp] * delta_l

    # Gated growth: 0 below critical, linear scaling above
    denom = np.where(csr < 1.0, 1.0 - csr, 1.0)
    raw_factor = np.where(sr >= csr, max_delta * (sr - csr) / denom, 0.0)
    # When csr == 1.0, only sr == 1.0 passes the gate; give max_delta directly
    growth_factor = np.where((csr >= 1.0) & (sr >= 1.0), max_delta, raw_factor)

    # Special cases: eggs and out-of-domain always get mean delta_L
    bypass = (state.age_dt == 0) | state.is_out
    growth_factor = np.where(bypass, delta_l, growth_factor)

    # Apply growth, cap at lmax (which defaults to linf if not separately configured)
    new_length = np.minimum(state.length + growth_factor, config.lmax[sp])

    # W_tonnes = c * L^b * 1e-6 (allometric grams -> tonnes)
    new_weight = config.condition_factor[sp] * new_length ** config.allometric_power[sp] * 1e-6

    # Update biomass
    new_biomass = state.abundance * new_weight

    return state.replace(length=new_length, weight=new_weight, biomass=new_biomass)


def expected_length_gompertz(
    age_dt: NDArray[np.int32],
    linf: NDArray[np.float64],
    k_gom: NDArray[np.float64],
    t_gom: NDArray[np.float64],
    k_exp: NDArray[np.float64],
    a_exp_dt: NDArray[np.int32],
    a_gom_dt: NDArray[np.int32],
    lstart: NDArray[np.float64],
    n_dt_per_year: int,
) -> NDArray[np.float64]:
    """Compute Gompertz expected length at a given age.

    Four phases:
      age == 0:                lstart
      0 < age < a_exp:         lstart * exp(K_exp * age)  (exponential)
      a_exp <= age < a_gom:    linear transition between exponential and Gompertz
      age >= a_gom:            L_inf * exp(-exp(-K_gom * (age - t_gom)))
    """
    age_years = age_dt.astype(np.float64) / n_dt_per_year
    a_exp_years = a_exp_dt.astype(np.float64) / n_dt_per_year
    a_gom_years = a_gom_dt.astype(np.float64) / n_dt_per_year

    # Gompertz formula
    l_gom = linf * np.exp(-np.exp(-k_gom * (age_years - t_gom)))
    l_gom_at_boundary = linf * np.exp(-np.exp(-k_gom * (a_gom_years - t_gom)))

    # Exponential phase
    l_exp = lstart * np.exp(k_exp * age_years)
    l_exp_at_boundary = lstart * np.exp(k_exp * a_exp_years)

    # Linear transition between exponential and Gompertz boundaries
    frac_linear = np.where(
        a_gom_years > a_exp_years,
        (age_years - a_exp_years) / (a_gom_years - a_exp_years),
        1.0,
    )
    l_linear = l_exp_at_boundary + (l_gom_at_boundary - l_exp_at_boundary) * frac_linear

    # Select phase
    result = np.where(
        age_dt == 0,
        lstart,
        np.where(
            age_years < a_exp_years,
            l_exp,
            np.where(age_years < a_gom_years, l_linear, l_gom),
        ),
    )
    return result


def _expected_length(
    age_dt: NDArray[np.int32],
    sp: NDArray[np.int32],
    config: EngineConfig,
    n_dt: int,
) -> NDArray[np.float64]:
    """Dispatch expected-length computation by growth class per school."""
    result = np.zeros(len(age_dt), dtype=np.float64)
    vb_mask = np.isin(sp, np.array(sorted(config.vb_species_ids), dtype=np.int32))
    if vb_mask.any():
        result[vb_mask] = expected_length_vb(
            age_dt[vb_mask],
            config.linf[sp[vb_mask]],
            config.k[sp[vb_mask]],
            config.t0[sp[vb_mask]],
            config.egg_size[sp[vb_mask]],
            config.vb_threshold_age[sp[vb_mask]],
            n_dt,
        )
    gom_mask = ~vb_mask
    if gom_mask.any() and config.gompertz_ke is not None:
        if config.gompertz_linf is None or config.gompertz_kg is None or config.gompertz_tg is None or config.gompertz_thr_age_exp_dt is None or config.gompertz_thr_age_gom_dt is None or config.gompertz_lstart is None:
            raise RuntimeError("Gompertz config arrays must not be None when gompertz_ke is not None")
        result[gom_mask] = expected_length_gompertz(
            age_dt[gom_mask],
            config.gompertz_linf[sp[gom_mask]],
            config.gompertz_kg[sp[gom_mask]],
            config.gompertz_tg[sp[gom_mask]],
            config.gompertz_ke[sp[gom_mask]],
            config.gompertz_thr_age_exp_dt[sp[gom_mask]],
            config.gompertz_thr_age_gom_dt[sp[gom_mask]],
            config.gompertz_lstart[sp[gom_mask]],
            n_dt,
        )
    return result
