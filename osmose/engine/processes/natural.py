"""Natural mortality processes: additional mortality and aging.

Additional mortality is a fixed annual rate applied per sub-timestep.
Aging mortality kills schools that have reached their species' lifespan.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.state import MortalityCause, SchoolState


def additional_mortality(state: SchoolState, config: EngineConfig, n_subdt: int) -> SchoolState:
    """Apply additional (background) mortality per sub-timestep.

    Java converts: D = (M_annual / n_dt_per_year) / n_subdt
    Then applies: N_dead = N * (1 - exp(-D))

    This gives the correct annual decay: after n_dt_per_year * n_subdt
    applications, total mortality ≈ 1 - exp(-M_annual).
    """
    if len(state) == 0:
        return state

    sp = state.species_id
    m_rate = config.additional_mortality_rate[sp]
    # Match Java: rate per sub-step = M_annual / (n_dt_per_year * n_subdt)
    d = m_rate / (config.n_dt_per_year * n_subdt)
    mortality_fraction = 1 - np.exp(-d)
    n_dead = state.abundance * mortality_fraction
    # Java skips age_dt == 0 (eggs handled by larva_mortality pre-pass)
    n_dead[state.age_dt == 0] = 0.0

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.ADDITIONAL] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)


def larva_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Apply additional mortality to eggs/larvae before the main mortality loop.

    Only affects schools where is_egg is True. Applied ONCE per main
    timestep (not per sub-step), matching Java's egg mortality handling.
    Java rate: M_larva / n_dt_per_year (per-timestep rate).
    """
    if len(state) == 0:
        return state

    eggs = state.is_egg
    if not eggs.any():
        return state

    sp = state.species_id
    m_rate = config.larva_mortality_rate[sp]
    # Match Java: rate per timestep = M_larva / n_dt_per_year
    d = m_rate / config.n_dt_per_year
    mortality_fraction = 1 - np.exp(-d)

    n_dead = np.zeros_like(state.abundance)
    n_dead[eggs] = state.abundance[eggs] * mortality_fraction[eggs]

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.ADDITIONAL] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)


def aging_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Kill schools that have reached their species' lifespan.

    Uses age_dt >= lifespan_dt - 1 because aging runs BEFORE
    reproduction (where age_dt is incremented).
    """
    if len(state) == 0:
        return state

    lifespan_dt = config.lifespan_dt[state.species_id]
    expired = state.age_dt >= lifespan_dt - 1

    if not expired.any():
        return state

    new_n_dead = state.n_dead.copy()
    new_n_dead[expired, MortalityCause.AGING] += state.abundance[expired]

    new_abundance = state.abundance.copy()
    new_abundance[expired] = 0.0
    new_biomass = new_abundance * state.weight

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)
