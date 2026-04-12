"""Starvation mortality for the OSMOSE Python engine.

Starvation is a lagged process: the rate applied THIS step was computed
from the PREVIOUS step's predation success rate.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.state import MortalityCause, SchoolState


def starvation_mortality(state: SchoolState, config: EngineConfig, n_subdt: int) -> SchoolState:
    """Apply starvation mortality using the lagged starvation_rate.

    The starvation_rate was computed at the end of the previous timestep.
    Java: D = starvation_rate / subdt; N_dead = N * (1 - exp(-D))
    """
    if len(state) == 0:
        return state

    # Use the stored (lagged) starvation rate, divide by subdt
    d = state.starvation_rate / (config.n_dt_per_year * n_subdt)
    mortality_fraction = 1 - np.exp(-d)
    n_dead = state.abundance * mortality_fraction
    n_dead[state.is_background] = 0.0

    # Skip pre-feeding schools (eggs/larvae before first feeding age)
    n_dead[state.age_dt < state.first_feeding_age_dt] = 0.0

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.STARVATION] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)


def update_starvation_rate(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Compute new starvation rate from this step's predation success.

    Called at the END of the mortality phase. The computed rate will be
    APPLIED in the NEXT timestep (lagged variable).

    M_starv = M_max * (1 - S_R / C_SR)    if S_R <= C_SR
    M_starv = 0                             if S_R > C_SR
    """
    if len(state) == 0:
        return state

    sp = state.species_id
    m_max = config.starvation_rate_max[sp]
    csr = config.critical_success_rate[sp]
    sr = state.pred_success_rate

    new_rate = np.where(
        sr <= csr,
        m_max * (1 - sr / np.where(csr > 0, csr, 1.0)),
        0.0,
    )
    # Clamp to non-negative: float precision can push sr slightly above 1.0
    # when csr == 0, producing a small negative rate.
    new_rate = np.maximum(0.0, new_rate)
    new_rate[state.is_background] = 0.0

    return state.replace(starvation_rate=new_rate)
