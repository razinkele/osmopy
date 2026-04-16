"""Fishing mortality for the OSMOSE Python engine.

Simple by-rate fishing: F_annual / (n_dt_per_year * n_subdt) per sub-step.
Supports seasonality, time-varying rates, sigmoid selectivity, MPA, and discards.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.state import MortalityCause, SchoolState


def fishing_mortality(
    state: SchoolState, config: EngineConfig, n_subdt: int, step: int = 0
) -> SchoolState:
    """Apply fishing mortality per sub-timestep.

    D = F_annual * season_factor / n_subdt
    N_dead = N * (1 - exp(-D))

    Parameters
    ----------
    state : SchoolState
    config : EngineConfig
    n_subdt : int
        Number of sub-timesteps within the current mortality loop.
    step : int
        Current simulation step (0-based), used for seasonality and rate-by-year.
    """
    if len(state) == 0 or not config.fishing_enabled:
        return state

    sp = state.species_id
    f_rate = config.fishing_rate[sp].copy()

    # 2.3: Rate by year — override base rate with time-varying annual rate
    if config.fishing_rate_by_year is not None:
        year = step // config.n_dt_per_year
        for sp_i in range(config.n_species):
            arr = config.fishing_rate_by_year[sp_i]
            if arr is not None and year < len(arr):
                mask = sp == sp_i
                f_rate[mask] = arr[year]

    # SP-1: Rate by dt by class — overrides base rate with per-class rate for this step
    if config.fishing_rate_by_dt_by_class is not None:
        step_idx = step % (config.n_dt_per_year * max(1, config.n_year))
        for sp_i in range(config.n_species):
            ts = config.fishing_rate_by_dt_by_class[sp_i]
            if ts is None:
                continue
            sp_mask = sp == sp_i
            if not sp_mask.any():
                continue
            ages_dt = state.age_dt[sp_mask].astype(float)
            sp_indices = np.where(sp_mask)[0]
            for j in range(len(ages_dt)):
                class_idx = ts.class_of(ages_dt[j])
                if class_idx >= 0:
                    f_rate[sp_indices[j]] = ts.get_by_class(step_idx, class_idx)
                else:
                    f_rate[sp_indices[j]] = 0.0

    # 2.1: Seasonality — multiply by normalized season weight for this step
    step_in_year = step % config.n_dt_per_year
    if config.fishing_seasonality is not None:
        season_weight = config.fishing_seasonality[sp, step_in_year]
        # seasonality replaces the uniform 1/n_dt_per_year division
        d = f_rate * season_weight / n_subdt
    else:
        d = f_rate / (config.n_dt_per_year * n_subdt)

    mortality_fraction = 1 - np.exp(-d)

    # Apply selectivity: type 0=age, 1=sigmoid, 2=Gaussian, 3=log-normal, -1=default
    from osmose.engine.processes.selectivity import (
        gaussian,
        knife_edge,
        log_normal,
        sigmoid,
        sigmoid_slope,
    )

    selectivity = np.ones(len(state), dtype=np.float64)
    for sp_i in range(config.n_species):
        sp_mask = sp == sp_i
        if not sp_mask.any():
            continue
        t = int(config.fishing_selectivity_type[sp_i])
        lengths = state.length[sp_mask]
        l50_val = float(config.fishing_selectivity_l50[sp_i])
        l75_val = float(config.fishing_selectivity_l75[sp_i])

        if t == 0:
            # Age-based knife-edge
            a50_val = float(config.fishing_selectivity_a50[sp_i])
            age_years = state.age_dt[sp_mask].astype(np.float64) / config.n_dt_per_year
            selectivity[sp_mask] = np.where(age_years >= a50_val, 1.0, 0.0)
        elif t == 1:
            # Sigmoid: prefer L50/L75 formula, fall back to slope-based
            if l75_val > 0:
                selectivity[sp_mask] = sigmoid(lengths, l50_val, l75_val)
            else:
                slope_val = float(config.fishing_selectivity_slope[sp_i])
                selectivity[sp_mask] = sigmoid_slope(lengths, l50_val, slope_val)
        elif t == 2:
            # Gaussian
            selectivity[sp_mask] = gaussian(lengths, l50_val, l75_val)
        elif t == 3:
            # Log-normal
            selectivity[sp_mask] = log_normal(lengths, l50_val, l75_val)
        else:
            # Default: length knife-edge (if l50 > 0)
            if l50_val > 0:
                selectivity[sp_mask] = knife_edge(lengths, l50_val)
            # else: selectivity stays 1.0

    # SP-1: Catch-based fishing -- proportional allocation (Java CatchesBySeason* variants)
    if config.fishing_catches is not None:
        return _catch_based_fishing(state, config, sp, selectivity, n_subdt, step)

    # Spatial fishing distribution: multiply by cell-specific factor
    spatial_factor = np.ones(len(state), dtype=np.float64)
    cy = state.cell_y.astype(np.intp)
    cx = state.cell_x.astype(np.intp)
    for sp_idx in range(len(config.fishing_spatial_maps)):
        sp_map = config.fishing_spatial_maps[sp_idx]
        if sp_map is None:
            continue
        sp_mask = sp == sp_idx
        if not sp_mask.any():
            continue
        sy, sx = cy[sp_mask], cx[sp_mask]
        valid = (sy >= 0) & (sy < sp_map.shape[0]) & (sx >= 0) & (sx < sp_map.shape[1])
        vals = np.zeros(sp_mask.sum(), dtype=np.float64)
        vals[valid] = sp_map[sy[valid], sx[valid]]
        vals[(vals <= 0) | np.isnan(vals)] = 0.0
        spatial_factor[sp_mask] = vals

    # 2.5: MPA — reduce fishing in protected cells
    mpa_factor = np.ones(len(state), dtype=np.float64)
    if config.mpa_zones is not None:
        year = step // config.n_dt_per_year
        for mpa in config.mpa_zones:
            if not (mpa.start_year <= year < mpa.end_year):
                continue
            valid = (cy >= 0) & (cy < mpa.grid.shape[0]) & (cx >= 0) & (cx < mpa.grid.shape[1])
            in_mpa = np.zeros(len(state), dtype=bool)
            in_mpa[valid] = mpa.grid[cy[valid], cx[valid]] > 0
            mpa_factor[in_mpa] *= 1.0 - mpa.percentage

    n_dead_total = state.abundance * mortality_fraction * selectivity * spatial_factor * mpa_factor
    n_dead_total[state.is_background] = 0.0

    # Skip pre-feeding schools (eggs/larvae before first feeding age)
    n_dead_total[state.age_dt < state.first_feeding_age_dt] = 0.0

    # 2.6: Discards — split total fishing deaths into landed and discarded
    new_n_dead = state.n_dead.copy()
    if config.fishing_discard_rate is not None:
        discard_rate = config.fishing_discard_rate[sp]
        n_discarded = n_dead_total * discard_rate
        n_landed = n_dead_total - n_discarded
        new_n_dead[:, MortalityCause.FISHING] += n_landed
        new_n_dead[:, MortalityCause.DISCARDS] += n_discarded
    else:
        new_n_dead[:, MortalityCause.FISHING] += n_dead_total

    new_abundance = state.abundance - n_dead_total
    new_biomass = new_abundance * state.weight

    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)


def _catch_based_fishing(
    state: SchoolState,
    config: EngineConfig,
    sp: np.ndarray,
    selectivity: np.ndarray,
    n_subdt: int,
    step: int,
) -> SchoolState:
    """Catch-based proportional allocation (Java CatchesBySeason* variants).

    catch_per_school = (school_biomass / total_fishable) * annual_catch * season_weight / n_subdt
    """
    step_in_year = step % config.n_dt_per_year
    year = step // config.n_dt_per_year
    n_dead_total = np.zeros(len(state), dtype=np.float64)

    for sp_i in range(config.n_species):
        sp_mask = sp == sp_i

        # Determine annual catch target
        annual_catch = config.fishing_catches[sp_i]
        if config.fishing_catches_by_year is not None:
            arr = config.fishing_catches_by_year[sp_i]
            if arr is not None and year < len(arr):
                annual_catch = arr[year]

        if annual_catch <= 0:
            continue

        # Season weight (default uniform)
        if config.fishing_catches_season is not None:
            season_w = config.fishing_catches_season[sp_i, step_in_year]
        else:
            season_w = 1.0 / config.n_dt_per_year

        # Compute fishable biomass (selectivity-weighted)
        fishable = (state.abundance * state.weight * selectivity)[sp_mask]
        total_fishable = fishable.sum()

        if total_fishable <= 0:
            continue

        # Proportional allocation per school
        catch_this_step = annual_catch * season_w / n_subdt
        school_catch = (fishable / total_fishable) * catch_this_step

        # Convert biomass catch to abundance
        school_weights = state.weight[sp_mask]
        n_dead_catch = np.where(school_weights > 0, school_catch / school_weights, 0.0)
        # Cap at available abundance
        n_dead_catch = np.minimum(n_dead_catch, state.abundance[sp_mask])

        n_dead_total[sp_mask] = n_dead_catch

    # Skip background and pre-feeding schools
    n_dead_total[state.is_background] = 0.0
    n_dead_total[state.age_dt < state.first_feeding_age_dt] = 0.0

    new_n_dead = state.n_dead.copy()
    if config.fishing_discard_rate is not None:
        discard_rate = config.fishing_discard_rate[sp]
        n_discarded = n_dead_total * discard_rate
        n_landed = n_dead_total - n_discarded
        new_n_dead[:, MortalityCause.FISHING] += n_landed
        new_n_dead[:, MortalityCause.DISCARDS] += n_discarded
    else:
        new_n_dead[:, MortalityCause.FISHING] += n_dead_total

    new_abundance = state.abundance - n_dead_total
    new_biomass = new_abundance * state.weight
    return state.replace(abundance=new_abundance, biomass=new_biomass, n_dead=new_n_dead)
