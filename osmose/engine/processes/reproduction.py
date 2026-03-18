"""Reproduction process: egg production from spawning stock biomass.

Also handles age increment for all schools (Java side-effect of reproduction).
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState


def reproduction(
    state: SchoolState, config: EngineConfig, step: int, rng: np.random.Generator
) -> SchoolState:
    """Produce eggs from mature schools and increment age for all schools.

    Maturity condition: age_dt >= maturity_age AND length >= maturity_size.
    Since maturity_age is not separately configured, we use maturity_size only.

    N_eggs = sex_ratio * relative_fecundity * SSB * season_factor
    """
    n_sp = config.n_species

    # --- Egg production ---
    # Maturity: length >= maturity_size (and abundance > 0)
    mature = (state.length >= config.maturity_size[state.species_id]) & (state.abundance > 0)

    # Spawning stock biomass per species
    ssb = np.zeros(n_sp, dtype=np.float64)
    if mature.any():
        np.add.at(ssb, state.species_id[mature], state.abundance[mature] * state.weight[mature])

    # Season factor from loaded CSV or uniform
    step_in_year = step % config.n_dt_per_year
    if config.spawning_season is not None:
        season_factor = config.spawning_season[:, step_in_year]
    else:
        season_factor = np.full(n_sp, 1.0 / config.n_dt_per_year)

    # Seeding: if SSB is zero and within seeding period, use seeding biomass
    # Seeding period is typically the first `lifespan` years
    step_year = step / config.n_dt_per_year
    for sp in range(n_sp):
        if ssb[sp] == 0.0:
            max_lifespan_years = config.lifespan_dt[sp] / config.n_dt_per_year
            if step_year < max_lifespan_years:
                ssb[sp] = config.seeding_biomass[sp]

    # Egg count per species
    # Java: nEgg = sexRatio * beta * season * SSB * 1_000_000
    # The 1e6 converts SSB from tonnes to grams (fecundity is eggs per gram)
    TONNES_TO_GRAMS = 1_000_000.0
    n_eggs = config.sex_ratio * config.relative_fecundity * ssb * season_factor * TONNES_TO_GRAMS

    # Create new schools from eggs
    new_schools_list = []
    for sp in range(n_sp):
        if n_eggs[sp] <= 0:
            continue
        n_new = int(config.n_schools[sp])
        if n_new <= 0:
            continue
        eggs_per_school = n_eggs[sp] / n_new
        egg_len = config.egg_size[sp]
        egg_weight = config.condition_factor[sp] * egg_len ** config.allometric_power[sp]

        new = SchoolState.create(n_schools=n_new, species_id=np.full(n_new, sp, dtype=np.int32))
        new = new.replace(
            abundance=np.full(n_new, eggs_per_school, dtype=np.float64),
            length=np.full(n_new, egg_len, dtype=np.float64),
            weight=np.full(n_new, egg_weight, dtype=np.float64),
            biomass=np.full(n_new, eggs_per_school * egg_weight, dtype=np.float64),
            is_egg=np.ones(n_new, dtype=np.bool_),
        )
        # Place eggs randomly on the grid (will be replaced by proper movement in Phase 4)
        new = new.replace(
            cell_x=rng.integers(0, 10, size=n_new).astype(np.int32),
            cell_y=rng.integers(0, 10, size=n_new).astype(np.int32),
        )
        new_schools_list.append(new)

    # --- Age increment for ALL schools ---
    new_age_dt = state.age_dt + 1
    new_is_egg = new_age_dt < state.first_feeding_age_dt
    state = state.replace(age_dt=new_age_dt, is_egg=new_is_egg)

    # Append new egg schools
    for new in new_schools_list:
        state = state.append(new)

    return state
