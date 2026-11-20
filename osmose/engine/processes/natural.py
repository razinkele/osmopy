"""Natural mortality processes: additional mortality and aging.

Additional mortality is a fixed annual rate applied per sub-timestep.
Aging mortality kills schools that have reached their species' lifespan.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.processes.maturity import mature_mask
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
    # OSMOSE 4.4.x multiplier and seasonality of the constant rate; a by-step rate below
    # replaces it, as Java's scenarios are exclusive.
    factor = getattr(config, "additional_mortality_factor", None)
    if factor is not None:
        focal = sp < factor.shape[0]
        m_rate[focal] *= factor[sp[focal], min(step, factor.shape[1] - 1)]

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
    # M5: clamp so spatial_factor > 1 cannot drive abundance negative.
    # mortality_fraction is in [0, 1] but spatial_factor is unbounded above
    # in the Java engine — without this clamp a multiplier > 1 would let
    # `state.abundance - n_dead` go negative and propagate sentinel values
    # into downstream biomass / mortality bookkeeping.
    np.minimum(n_dead, state.abundance, out=n_dead)

    new_abundance = state.abundance - n_dead
    new_biomass = new_abundance * state.weight
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, MortalityCause.ADDITIONAL] += n_dead

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)


def _density_dependent_larva_rate(
    m_rate: NDArray[np.float64], state: SchoolState, config: EngineConfig
) -> NDArray[np.float64]:
    """The larval rate of each egg school, scaled by its species' spawning stock.

    Java (``AnnualLarvaMortality.getRate``, ``ByDtLarvaMortality.getRate``, v4.4.1):
    ``M * (1 + a * SSB / reference)`` with ``a = ln 2 / M``, so larval survival halves when
    the stock equals the reference. Java applies larval mortality to eggs before the sub-step
    loop, so SSB is the total taken at the start of the step after movement
    (``SimulationStep.step``): the biomass of every mature focal school, out-of-domain schools
    included, which is the state handed to this pre-pass. Where M is 0, Java's formula is
    0 x infinity (not a number); the limit, ``ln 2 * SSB / reference``, is used instead.

    Java totals ``school.isMature()``, a flag set during the previous step's reproduction,
    before ages were incremented; this tests maturity on the current state. With a maturity
    age, a school crossing it counts one step later in Java. A species that does not reproduce
    (sex ratio or fecundity 0) never gets the flag in Java, so its stock is 0 there. Under
    bioenergetics the flag follows the m0/m1 line, which this engine does not latch, so the
    key is refused there (``EngineConfig.from_dict``).
    """
    ref = config.larva_reference_density
    assert ref is not None
    n_sp = config.n_species
    sp = state.species_id
    # The engine's one maturity test (thresholds, or the drawn flag of a stochastic species),
    # the same reproduction and the SSB output use; Java reads school.isMature() here.
    mature = (sp < n_sp) & (state.abundance > 0) & mature_mask(state, config)
    ssb = np.zeros(n_sp, dtype=np.float64)
    np.add.at(ssb, sp[mature], state.abundance[mature] * state.weight[mature])
    out = m_rate.copy()
    ln2 = np.log(2.0)
    for i in np.flatnonzero(state.is_egg & (sp < n_sp)):
        r = ref[sp[i]]
        if not r > 0:
            continue
        x = ssb[sp[i]] / r
        m = out[i]
        out[i] = m * (1.0 + (ln2 / m) * x) if m > 0 else ln2 * x
    return out


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
    # OSMOSE 4.4.x multiplier and seasonality of the constant larval rate; a by-step rate below
    # replaces it.
    factor = getattr(config, "larva_mortality_factor", None)
    if factor is not None:
        focal = sp < factor.shape[0]
        m_rate[focal] *= factor[sp[focal], min(step, factor.shape[1] - 1)]

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

    # Density dependence (OSMOSE 4.4.x, mortality.additional.larva.referencedensity).
    if config.larva_reference_density is not None:
        m_rate = _density_dependent_larva_rate(m_rate, state, config)

    # Full rate applied once per cohort (matching Java AnnualLarvaMortality)
    d = m_rate
    mortality_fraction = 1 - np.exp(-d)

    n_dead = np.zeros_like(state.abundance)
    n_dead[eggs] = state.abundance[eggs] * mortality_fraction[eggs]

    # Spatial reproductive-volume egg survival (Baltic cod). Inert unless enabled;
    # multiplies egg survival by clip(RV(cell)/RV_ref, 0, 1) for enabled, non-seeded,
    # located eggs. Increases n_dead by the extra kill (survival deficit).
    field = config.rv_spatial_field
    if field is not None:
        assert config.rv_spatial_enabled is not None  # set together in _load_rv_spatial
        rv_grid = field.get_grid(step)  # (ny, nx)
        rv_ref = field.rv_ref or 1.0
        for i in range(len(state)):
            if not eggs[i]:
                continue
            spi = int(sp[i])
            if spi >= len(config.rv_spatial_enabled) or not config.rv_spatial_enabled[spi]:
                continue
            if state.is_out[i]:
                continue
            fs = state.from_seeding
            if fs is not None and fs[i]:
                continue
            cy = int(state.cell_y[i])
            cx = int(state.cell_x[i])
            if cy < 0 or cx < 0:
                continue
            rv = rv_grid[cy, cx]
            if not np.isfinite(rv):
                continue
            s = min(1.0, max(0.0, rv / rv_ref))  # clip
            survivors_after = state.abundance[i] * (1.0 - mortality_fraction[i]) * s
            extra_dead = state.abundance[i] * (1.0 - mortality_fraction[i]) - survivors_after
            n_dead[i] += extra_dead

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


def spawning_mortality(state: SchoolState, config: EngineConfig, step: int) -> SchoolState:
    """OSMOSE 4.4.x post-reproduction mortality (Java ``SchoolSet.updateSpawningMortality``).

    Runs after growth and aging, before reproduction. A school of a semelparous species that has
    spawned dies, cause SPAWNING, once ``step - spawn_step`` reaches the post-spawning survival
    time (``config.postspawning_survival_dt``): with the default 0, at the step after it spawned.
    """
    semel = config.reproduction_semelparous
    if semel is None or state.has_spawned is None or state.spawn_step is None or not len(state):
        return state
    sp = state.species_id
    focal = sp < config.n_species
    die = (
        focal
        & semel[np.where(focal, sp, 0)]
        & state.has_spawned
        & (step - state.spawn_step >= config.postspawning_survival_dt)
        & (state.abundance > 0)
    )
    if not die.any():
        return state
    new_n_dead = state.n_dead.copy()
    new_n_dead[die, MortalityCause.SPAWNING] += state.abundance[die]
    new_abundance = np.where(die, 0.0, state.abundance)
    return state.replace(
        abundance=new_abundance, biomass=new_abundance * state.weight, n_dead=new_n_dead
    )


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
