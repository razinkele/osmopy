"""Predation process for the OSMOSE Python engine.

Size-based opportunistic predation within grid cells. Predators are
processed sequentially in random order with asynchronous prey biomass updates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState


def predation_in_cell(
    indices: NDArray[np.int32],
    state: SchoolState,
    config: EngineConfig,
    rng: np.random.Generator,
    n_subdt: int,
) -> None:
    """Apply predation within a single cell. Modifies state arrays IN-PLACE.

    Predators are processed in random order. Prey biomass is decremented
    immediately after each predator eats (asynchronous update).

    Args:
        indices: Indices into state arrays for schools in this cell.
        state: SchoolState (modified in place for performance).
        config: Engine configuration.
        rng: Random number generator.
        n_subdt: Number of mortality sub-timesteps.
    """
    n_local = len(indices)
    if n_local < 2:
        return  # Need at least 2 schools for predation

    # Random predator order
    order = rng.permutation(n_local)

    for p_pos in order:
        p_idx = indices[p_pos]

        # Skip non-feeding schools (eggs)
        if state.age_dt[p_idx] < state.first_feeding_age_dt[p_idx]:
            continue

        # Skip dead schools
        if state.abundance[p_idx] <= 0:
            continue

        pred_len = state.length[p_idx]
        sp_pred = state.species_id[p_idx]

        # Size ratio thresholds for this predator species
        r_min = config.size_ratio_min[sp_pred]  # upper bound of ratio
        r_max = config.size_ratio_max[sp_pred]  # lower bound of ratio

        # Maximum eatable biomass this sub-step
        max_eatable = state.biomass[p_idx] * config.ingestion_rate[sp_pred] / n_subdt
        if max_eatable <= 0:
            continue

        # Scan all potential prey in this cell
        available = 0.0
        prey_eligible = np.zeros(n_local, dtype=np.float64)

        for q_pos in range(n_local):
            q_idx = indices[q_pos]

            # Skip self
            if q_idx == p_idx:
                continue

            # Skip dead prey
            if state.abundance[q_idx] <= 0:
                continue

            prey_len = state.length[q_idx]
            if prey_len <= 0:
                continue

            # Size ratio check: r_max < pred/prey <= r_min
            ratio = pred_len / prey_len
            if ratio <= r_max or ratio > r_min:
                continue

            # Prey is eligible -- use its current biomass (asynchronous)
            prey_bio = state.abundance[q_idx] * state.weight[q_idx]
            if prey_bio <= 0:
                continue

            prey_eligible[q_pos] = prey_bio
            available += prey_bio

        if available <= 0:
            continue

        # How much the predator eats
        eaten_total = min(available, max_eatable)

        # Distribute eaten biomass proportionally among eligible prey
        for q_pos in range(n_local):
            if prey_eligible[q_pos] <= 0:
                continue

            q_idx = indices[q_pos]
            share = prey_eligible[q_pos] / available
            eaten_from_prey = eaten_total * share

            # Update prey -- IMMEDIATE (asynchronous)
            if state.weight[q_idx] > 0:
                n_dead = eaten_from_prey / state.weight[q_idx]
                state.abundance[q_idx] = max(0.0, state.abundance[q_idx] - n_dead)

        # Update predator success rate
        state.pred_success_rate[p_idx] += eaten_total / max_eatable
        state.preyed_biomass[p_idx] += eaten_total


def predation(
    state: SchoolState,
    config: EngineConfig,
    rng: np.random.Generator,
    n_subdt: int,
    grid_ny: int,
    grid_nx: int,
) -> SchoolState:
    """Apply predation across all grid cells.

    Groups schools by cell, then processes predation within each
    occupied cell independently.
    """
    if len(state) == 0:
        return state

    # Make working copies for in-place modification
    abundance = state.abundance.copy()
    pred_success_rate = state.pred_success_rate.copy()
    preyed_biomass = state.preyed_biomass.copy()

    # Create a temporary mutable view
    work_state = state.replace(
        abundance=abundance,
        pred_success_rate=pred_success_rate,
        preyed_biomass=preyed_biomass,
    )

    # Group schools by cell
    cell_ids = work_state.cell_y * grid_nx + work_state.cell_x
    order = np.argsort(cell_ids)
    sorted_cells = cell_ids[order]

    # Find boundaries of each cell group
    unique_cells = np.unique(sorted_cells)
    for cell in unique_cells:
        mask = sorted_cells == cell
        cell_indices = order[mask]
        if len(cell_indices) >= 2:
            predation_in_cell(cell_indices, work_state, config, rng, n_subdt)

    # Update biomass from new abundance
    new_biomass = work_state.abundance * work_state.weight

    return state.replace(
        abundance=work_state.abundance,
        biomass=new_biomass,
        pred_success_rate=work_state.pred_success_rate,
        preyed_biomass=work_state.preyed_biomass,
    )
