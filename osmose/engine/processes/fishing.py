"""Fishing mortality for the OSMOSE Python engine.

Simple by-rate fishing: F_annual / (n_dt_per_year * n_subdt) per sub-step.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.state import MortalityCause, SchoolState


def fishing_mortality(state: SchoolState, config: EngineConfig, n_subdt: int) -> SchoolState:
    """Apply fishing mortality per sub-timestep.

    D = F_annual / (n_dt_per_year * n_subdt)
    N_dead = N * (1 - exp(-D))
    """
    if len(state) == 0 or not config.fishing_enabled:
        return state

    sp = state.species_id
    f_rate = config.fishing_rate[sp]
    d = f_rate / (config.n_dt_per_year * n_subdt)
    mortality_fraction = 1 - np.exp(-d)

    # Apply selectivity (knife-edge at L50 if configured)
    l50 = config.fishing_selectivity_l50[sp]
    selectivity = np.where(l50 > 0, np.where(state.length >= l50, 1.0, 0.0), 1.0)
    n_dead = state.abundance * mortality_fraction * selectivity

    # Skip eggs (age_dt == 0)
    n_dead[state.age_dt == 0] = 0.0

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.FISHING] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)
