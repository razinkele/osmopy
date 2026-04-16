"""Natural mortality processes: additional mortality and aging.

Additional mortality is a fixed annual rate applied per sub-timestep.
Aging mortality kills schools that have reached their species' lifespan.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.state import MortalityCause, SchoolState


def additional_mortality(
    state: SchoolState, config: EngineConfig, n_subdt: int, step: int = 0
) -> SchoolState:
    """Apply additional (background) mortality per sub-timestep.

    Java converts: D = (M_annual / n_dt_per_year) / n_subdt
    Then applies: N_dead = N * (1 - exp(-D))

    This gives the correct annual decay: after n_dt_per_year * n_subdt
    applications, total mortality ≈ 1 - exp(-M_annual).
    """
    if len(state) == 0:
        return state

    sp = state.species_id
    m_rate = config.additional_mortality_rate[sp].copy()

    # Override with time-varying BY_DT rates where available
    if config.additional_mortality_by_dt is not None:
        for i in range(len(state)):
            sp_i = sp[i]
            if (
                sp_i < len(config.additional_mortality_by_dt)
                and config.additional_mortality_by_dt[sp_i] is not None
            ):
                arr = config.additional_mortality_by_dt[sp_i]
                m_rate[i] = arr[step % len(arr)]

    # Override with by-dt-by-class rates (ByClassTimeSeries)
    if config.additional_mortality_by_dt_by_class is not None:
        for sp_i in range(config.n_species):
            ts = config.additional_mortality_by_dt_by_class[sp_i]
            if ts is None:
                continue
            sp_mask = sp == sp_i
            if not sp_mask.any():
                continue
            ages_dt = state.age_dt[sp_mask].astype(float)
            step_idx = min(step, len(ts.values) - 1)
            for j in range(len(ages_dt)):
                class_idx = ts.class_of(ages_dt[j])
                school_idx = np.where(sp_mask)[0][j]
                if class_idx >= 0:
                    m_rate[school_idx] = ts.get_by_class(step_idx, class_idx)
                else:
                    m_rate[school_idx] = 0.0  # below first threshold

    # Match Java: rate per sub-step = M_annual / (n_dt_per_year * n_subdt)
    d = m_rate / (config.n_dt_per_year * n_subdt)
    mortality_fraction = 1 - np.exp(-d)

    # Apply spatial factor (per-cell multiplier)
    spatial_factor = np.ones(len(state), dtype=np.float64)
    if config.additional_mortality_spatial is not None:
        cy = state.cell_y.astype(np.intp)
        cx = state.cell_x.astype(np.intp)
        for sp_i in range(config.n_species):
            sp_map = config.additional_mortality_spatial[sp_i]
            if sp_map is None:
                continue
            sp_mask = sp == sp_i
            if not sp_mask.any():
                continue
            sy, sx = cy[sp_mask], cx[sp_mask]
            valid = (sy >= 0) & (sy < sp_map.shape[0]) & (sx >= 0) & (sx < sp_map.shape[1])
            vals = np.zeros(sp_mask.sum(), dtype=np.float64)
            vals[valid] = sp_map[sy[valid], sx[valid]]
            spatial_factor[sp_mask] = vals

    n_dead = state.abundance * mortality_fraction * spatial_factor
    n_dead[state.is_background] = 0.0
    # Java skips pre-feeding schools (eggs handled by larva_mortality pre-pass)
    n_dead[state.age_dt < state.first_feeding_age_dt] = 0.0

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.ADDITIONAL] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)


def larva_mortality(state: SchoolState, config: EngineConfig, step: int = 0) -> SchoolState:
    """Apply additional mortality to eggs/larvae before the main mortality loop.

    Only affects schools where is_egg is True. Applied ONCE per egg cohort
    (each cohort is age-0 for exactly one timestep before age increment).
    Java: AnnualLarvaMortality applies the FULL configured rate in one step
    (D = rate, NOT rate / n_dt_per_year). This is correct because each egg
    cohort receives this mortality exactly once in its lifetime, so the rate
    represents the total larval mortality, not a rate to be spread over time.

    Supports constant (Annual) and time-varying (ByDt) rates.
    """
    if len(state) == 0:
        return state

    eggs = state.is_egg
    if not eggs.any():
        return state

    sp = state.species_id
    m_rate = config.larva_mortality_rate[sp].copy()

    # Override with time-varying BY_DT larval rates
    if config.larva_mortality_by_dt is not None:
        for i in range(len(state)):
            if not eggs[i]:
                continue
            sp_i = sp[i]
            if (
                sp_i < len(config.larva_mortality_by_dt)
                and config.larva_mortality_by_dt[sp_i] is not None
            ):
                ts = config.larva_mortality_by_dt[sp_i]
                m_rate[i] = ts.get(step)

    # Full rate applied once per cohort (matching Java AnnualLarvaMortality)
    d = m_rate
    mortality_fraction = 1 - np.exp(-d)

    n_dead = np.zeros_like(state.abundance)
    n_dead[eggs] = state.abundance[eggs] * mortality_fraction[eggs]

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.ADDITIONAL] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)


def out_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Apply mortality to out-of-domain schools.

    Java: D = M_out / n_dt_per_year, applied once per timestep.
    """
    if len(state) == 0:
        return state

    out = state.is_out
    if not out.any():
        return state

    sp = state.species_id
    d = config.out_mortality_rate[sp] / config.n_dt_per_year
    mortality_fraction = 1 - np.exp(-d)

    n_dead = np.zeros_like(state.abundance)
    n_dead[out] = state.abundance[out] * mortality_fraction[out]

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.OUT] += n_dead

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
