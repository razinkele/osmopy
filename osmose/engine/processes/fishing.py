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

    # 2.1: Seasonality — multiply by normalized season weight for this step
    step_in_year = step % config.n_dt_per_year
    if config.fishing_seasonality is not None:
        season_weight = config.fishing_seasonality[sp, step_in_year]
        # seasonality replaces the uniform 1/n_dt_per_year division
        d = f_rate * season_weight / n_subdt
    else:
        d = f_rate / (config.n_dt_per_year * n_subdt)

    mortality_fraction = 1 - np.exp(-d)

    # Apply selectivity: age-based (type 0), sigmoidal (type 1), or length knife-edge
    sel_type = config.fishing_selectivity_type[sp]
    a50 = config.fishing_selectivity_a50[sp]
    l50 = config.fishing_selectivity_l50[sp]

    # Age-based knife-edge: selected if age_years >= a50
    age_years = state.age_dt.astype(np.float64) / config.n_dt_per_year
    age_select = np.where(sel_type == 0, np.where(age_years >= a50, 1.0, 0.0), 0.0)

    # 2.4: Sigmoidal size selectivity (type 1)
    slope = config.fishing_selectivity_slope[sp]
    sigmoid_select = np.where(
        sel_type == 1,
        1.0 / (1.0 + np.exp(-slope * (state.length - l50))),
        0.0,
    )

    # Length-based knife-edge: selected if length >= l50
    len_select = np.where(l50 > 0, np.where(state.length >= l50, 1.0, 0.0), 1.0)

    # Combine: type 0 = age, type 1 = sigmoid, default = length knife-edge
    selectivity = np.where(
        sel_type == 0,
        age_select,
        np.where(sel_type == 1, sigmoid_select, len_select),
    )

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
