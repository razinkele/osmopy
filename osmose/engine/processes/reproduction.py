"""Reproduction process: egg production from spawning stock biomass.

Also handles age increment for all schools (Java side-effect of reproduction).
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState


def reproduction(
    state: SchoolState,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
    grid_ny: int = 10,
    grid_nx: int = 10,
) -> SchoolState:
    """Produce eggs from mature schools and increment age for all schools.

    Maturity condition: age_dt >= maturity_age AND length >= maturity_size.

    N_eggs = sex_ratio * relative_fecundity * SSB * season_factor
    """
    n_sp = config.n_species

    # --- Egg production ---
    # Maturity: length >= maturity_size AND age_dt >= maturity_age_dt (and abundance > 0)
    mature = (
        (state.length >= config.maturity_size[state.species_id])
        & (state.age_dt >= config.maturity_age_dt[state.species_id])
        & (state.abundance > 0)
    )

    # Spawning stock biomass per species
    ssb = np.zeros(n_sp, dtype=np.float64)
    if mature.any():
        np.add.at(ssb, state.species_id[mature], state.abundance[mature] * state.weight[mature])

    # Season factor from loaded CSV or uniform
    # Multi-year CSVs: wrap by column count (handles both single-year and multi-year)
    if config.spawning_season is not None:
        n_cols = config.spawning_season.shape[1]
        season_idx = step % n_cols
        season_factor = config.spawning_season[:, season_idx]
    else:
        season_factor = np.full(n_sp, 1.0 / config.n_dt_per_year)

    # Seeding: if SSB is zero and within seeding period, use seeding biomass
    for sp in range(n_sp):
        if ssb[sp] == 0.0:
            if step < config.seeding_max_step[sp]:
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
        # Edge case: fewer eggs than schools -> create just 1 school
        if n_eggs[sp] < n_new:
            n_new = 1
        eggs_per_school = n_eggs[sp] / n_new
        egg_len = config.egg_size[sp]
        egg_weight = config.condition_factor[sp] * egg_len ** config.allometric_power[sp] * 1e-6
        # Use egg weight override if available (already in tonnes)
        if config.egg_weight_override is not None and not np.isnan(config.egg_weight_override[sp]):
            egg_weight = config.egg_weight_override[sp]

        new = SchoolState.create(n_schools=n_new, species_id=np.full(n_new, sp, dtype=np.int32))
        new = new.replace(
            abundance=np.full(n_new, eggs_per_school, dtype=np.float64),
            length=np.full(n_new, egg_len, dtype=np.float64),
            weight=np.full(n_new, egg_weight, dtype=np.float64),
            biomass=np.full(n_new, eggs_per_school * egg_weight, dtype=np.float64),
            is_egg=np.ones(n_new, dtype=np.bool_),
            # Eggs cannot feed for their first timestep (Java convention: first_feeding_age_dt=1)
            first_feeding_age_dt=np.ones(n_new, dtype=np.int32),
        )
        # Eggs are created unlocated; movement places them on the next step
        new = new.replace(
            cell_x=np.full(n_new, -1, dtype=np.int32),
            cell_y=np.full(n_new, -1, dtype=np.int32),
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
