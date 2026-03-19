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

    # Apply selectivity: age-based (type 0) or length-based (type 1)
    sel_type = config.fishing_selectivity_type[sp]
    a50 = config.fishing_selectivity_a50[sp]
    l50 = config.fishing_selectivity_l50[sp]

    # Age-based knife-edge: selected if age_years >= a50
    age_years = state.age_dt.astype(np.float64) / config.n_dt_per_year
    age_select = np.where(sel_type == 0, np.where(age_years >= a50, 1.0, 0.0), 0.0)

    # Length-based knife-edge: selected if length >= l50
    len_select = np.where(l50 > 0, np.where(state.length >= l50, 1.0, 0.0), 1.0)

    # Combine: use age selectivity for type==0, length selectivity for type==1,
    # default (no type / type==-1) uses length selectivity for backward compat
    selectivity = np.where(sel_type == 0, age_select, len_select)
    n_dead = state.abundance * mortality_fraction * selectivity
    n_dead[state.is_background] = 0.0

    # Skip eggs (age_dt == 0)
    n_dead[state.age_dt == 0] = 0.0

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.FISHING] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)
