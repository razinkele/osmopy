"""Predation process for the OSMOSE Python engine.

Size-based opportunistic predation within grid cells. Predators are
processed sequentially in random order with asynchronous prey biomass updates.

Uses Numba JIT compilation for the inner cell loop when available,
falling back to pure Python otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.processes.feeding_stage import compute_feeding_stages
from osmose.engine.resources import ResourceState
from osmose.engine.state import SchoolState

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Diet tracking (module-level state)
# ---------------------------------------------------------------------------

_diet_tracking_enabled: bool = False
_diet_matrix: NDArray[np.float64] | None = None


def enable_diet_tracking(n_schools: int, n_species: int) -> None:
    """Enable per-school diet tracking with a (n_schools, n_species) matrix."""
    global _diet_tracking_enabled, _diet_matrix
    _diet_tracking_enabled = True
    _diet_matrix = np.zeros((n_schools, n_species), dtype=np.float64)


def disable_diet_tracking() -> None:
    """Disable diet tracking and clear the matrix."""
    global _diet_tracking_enabled, _diet_matrix
    _diet_tracking_enabled = False
    _diet_matrix = None


def get_diet_matrix() -> NDArray[np.float64] | None:
    """Return the current diet matrix, or None if tracking is disabled."""
    return _diet_matrix


# ---------------------------------------------------------------------------
# Numba-accelerated inner loop
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True)
    def _predation_in_cell_numba(
        indices: NDArray[np.int32],
        pred_order: NDArray[np.int32],
        abundance: NDArray[np.float64],
        length: NDArray[np.float64],
        weight: NDArray[np.float64],
        age_dt: NDArray[np.int32],
        first_feeding_age_dt: NDArray[np.int32],
        species_id: NDArray[np.int32],
        pred_success_rate: NDArray[np.float64],
        preyed_biomass: NDArray[np.float64],
        size_ratio_min: NDArray[np.float64],
        size_ratio_max: NDArray[np.float64],
        ingestion_rate: NDArray[np.float64],
        access_matrix: NDArray[np.float64],
        has_access: bool,
        n_subdt: int,
        feeding_stage: NDArray[np.int32],
    ) -> None:
        """Numba-compiled predation within a single cell."""
        n_local = len(indices)

        for p_pos_idx in range(len(pred_order)):
            p_pos = pred_order[p_pos_idx]
            p_idx = indices[p_pos]

            if age_dt[p_idx] < first_feeding_age_dt[p_idx]:
                continue
            if abundance[p_idx] <= 0:
                continue

            pred_len = length[p_idx]
            sp_pred = species_id[p_idx]
            stage = feeding_stage[p_idx]
            r_min = size_ratio_min[sp_pred, stage]
            r_max = size_ratio_max[sp_pred, stage]

            biomass_p = abundance[p_idx] * weight[p_idx]
            max_eatable = biomass_p * ingestion_rate[sp_pred] / n_subdt
            if max_eatable <= 0:
                continue

            available = 0.0
            # Use a fixed-size array for eligible prey biomass
            prey_bio_eligible = np.zeros(n_local, dtype=np.float64)

            for q_pos in range(n_local):
                q_idx = indices[q_pos]
                if q_idx == p_idx:
                    continue
                if abundance[q_idx] <= 0:
                    continue
                prey_len = length[q_idx]
                if prey_len <= 0:
                    continue

                ratio = pred_len / prey_len
                if ratio < r_min or ratio >= r_max:
                    continue

                sp_prey = species_id[q_idx]
                access_coeff = 1.0
                if has_access:
                    if sp_pred < access_matrix.shape[0] and sp_prey < access_matrix.shape[1]:
                        access_coeff = access_matrix[sp_pred, sp_prey]
                        if access_coeff <= 0:
                            continue

                prey_bio = abundance[q_idx] * weight[q_idx]
                if prey_bio <= 0:
                    continue

                val = prey_bio * access_coeff
                prey_bio_eligible[q_pos] = val
                available += val

            if available <= 0:
                continue

            eaten_total = min(available, max_eatable)

            for q_pos in range(n_local):
                if prey_bio_eligible[q_pos] <= 0:
                    continue
                q_idx = indices[q_pos]
                share = prey_bio_eligible[q_pos] / available
                eaten_from_prey = eaten_total * share
                if weight[q_idx] > 0:
                    n_dead = eaten_from_prey / weight[q_idx]
                    new_abd = abundance[q_idx] - n_dead
                    abundance[q_idx] = max(0.0, new_abd)

            pred_success_rate[p_idx] += eaten_total / max_eatable
            preyed_biomass[p_idx] += eaten_total


# ---------------------------------------------------------------------------
# Pure Python fallback
# ---------------------------------------------------------------------------


def _predation_in_cell_python(
    indices: NDArray[np.int32],
    state: SchoolState,
    config: EngineConfig,
    rng: np.random.Generator,
    n_subdt: int,
) -> None:
    """Pure Python fallback for predation within a single cell."""
    n_local = len(indices)
    if n_local < 2:
        return

    order = rng.permutation(n_local)

    for p_pos in order:
        p_idx = indices[p_pos]
        if state.age_dt[p_idx] < state.first_feeding_age_dt[p_idx]:
            continue
        if state.abundance[p_idx] <= 0:
            continue

        pred_len = state.length[p_idx]
        sp_pred = state.species_id[p_idx]
        r_min = config.size_ratio_min[sp_pred, state.feeding_stage[p_idx]]
        r_max = config.size_ratio_max[sp_pred, state.feeding_stage[p_idx]]

        max_eatable = state.biomass[p_idx] * config.ingestion_rate[sp_pred] / n_subdt
        if max_eatable <= 0:
            continue

        available = 0.0
        prey_eligible = np.zeros(n_local, dtype=np.float64)

        for q_pos in range(n_local):
            q_idx = indices[q_pos]
            if q_idx == p_idx:
                continue
            if state.abundance[q_idx] <= 0:
                continue
            prey_len = state.length[q_idx]
            if prey_len <= 0:
                continue

            ratio = pred_len / prey_len
            if ratio < r_min or ratio >= r_max:
                continue

            sp_prey = state.species_id[q_idx]
            access_coeff = 1.0
            if config.accessibility_matrix is not None:
                if (
                    sp_pred < config.accessibility_matrix.shape[0]
                    and sp_prey < config.accessibility_matrix.shape[1]
                ):
                    access_coeff = config.accessibility_matrix[sp_pred, sp_prey]
                    if access_coeff <= 0:
                        continue

            prey_bio = state.abundance[q_idx] * state.weight[q_idx]
            if prey_bio <= 0:
                continue

            prey_eligible[q_pos] = prey_bio * access_coeff
            available += prey_bio * access_coeff

        if available <= 0:
            continue

        eaten_total = min(available, max_eatable)

        for q_pos in range(n_local):
            if prey_eligible[q_pos] <= 0:
                continue
            q_idx = indices[q_pos]
            share = prey_eligible[q_pos] / available
            eaten_from_prey = eaten_total * share
            if state.weight[q_idx] > 0:
                n_dead = eaten_from_prey / state.weight[q_idx]
                state.abundance[q_idx] = max(0.0, state.abundance[q_idx] - n_dead)

            # Diet tracking: record biomass eaten per prey species
            if _diet_tracking_enabled and _diet_matrix is not None:
                prey_sp = state.species_id[q_idx]
                if p_idx < _diet_matrix.shape[0] and prey_sp < _diet_matrix.shape[1]:
                    _diet_matrix[p_idx, prey_sp] += eaten_from_prey

        state.pred_success_rate[p_idx] += eaten_total / max_eatable
        state.preyed_biomass[p_idx] += eaten_total


# ---------------------------------------------------------------------------
# Resource predation (focal species eating LTL plankton/detritus)
# ---------------------------------------------------------------------------


def _predation_on_resources(
    cell_indices: NDArray[np.int32],
    state: SchoolState,
    config: EngineConfig,
    resources: ResourceState,
    cell_y: int,
    cell_x: int,
    rng: np.random.Generator,
    n_subdt: int,
) -> None:
    """Let focal schools in this cell eat resource species.

    Called after school-to-school predation so that pred_success_rate
    already reflects food obtained from other schools. Resources fill
    the remaining appetite.
    """
    if resources.n_resources == 0:
        return

    n_local = len(cell_indices)
    order = rng.permutation(n_local)

    for p_pos_idx in range(len(order)):
        p_idx = cell_indices[order[p_pos_idx]]

        if state.age_dt[p_idx] < state.first_feeding_age_dt[p_idx]:
            continue
        if state.abundance[p_idx] <= 0:
            continue

        pred_len = state.length[p_idx]
        sp_pred = state.species_id[p_idx]
        stage = state.feeding_stage[p_idx]
        r_min_val = config.size_ratio_min[sp_pred, stage]
        r_max_val = config.size_ratio_max[sp_pred, stage]

        biomass_p = state.abundance[p_idx] * state.weight[p_idx]
        max_eatable = biomass_p * config.ingestion_rate[sp_pred] / n_subdt
        if max_eatable <= 0:
            continue

        # Check each resource species
        available = 0.0
        rsc_eligible = np.zeros(resources.n_resources, dtype=np.float64)

        for r in range(resources.n_resources):
            rsc = resources.species[r]
            rsc_bio = resources.get_cell_biomass(r, cell_y, cell_x)
            if rsc_bio <= 0:
                continue

            # Size overlap: what fraction of the resource size range
            # falls within the predator's prey window?
            # Prey window: [L/r_max, L/r_min] (r_max > r_min by convention)
            prey_size_min = pred_len / r_max_val  # smallest prey this predator eats
            prey_size_max = pred_len / r_min_val  # largest prey this predator eats

            overlap_min = max(rsc.size_min, prey_size_min)
            overlap_max = min(rsc.size_max, prey_size_max)
            if overlap_max <= overlap_min:
                continue
            rsc_range = rsc.size_max - rsc.size_min
            if rsc_range <= 0:
                continue
            percent_resource = (overlap_max - overlap_min) / rsc_range

            # Accessibility from matrix (resource index = n_species + r)
            rsc_sp_idx = config.n_species + r
            access_coeff = 1.0
            if config.accessibility_matrix is not None:
                if (
                    sp_pred < config.accessibility_matrix.shape[0]
                    and rsc_sp_idx < config.accessibility_matrix.shape[1]
                ):
                    access_coeff = config.accessibility_matrix[sp_pred, rsc_sp_idx]
                    if access_coeff <= 0:
                        continue

            eligible_bio = rsc_bio * percent_resource * access_coeff
            rsc_eligible[r] = eligible_bio
            available += eligible_bio

        if available <= 0:
            continue

        # Remaining appetite after school-to-school predation
        remaining = max_eatable * (1.0 - min(1.0, state.pred_success_rate[p_idx]))
        if remaining <= 0:
            continue

        eaten_total = min(available, remaining)

        # Deduct from resources proportionally
        cell_id = cell_y * resources.grid.nx + cell_x
        for r in range(resources.n_resources):
            if rsc_eligible[r] <= 0:
                continue
            share = rsc_eligible[r] / available
            eaten_from_rsc = eaten_total * share
            resources.biomass[r, cell_id] = max(0.0, resources.biomass[r, cell_id] - eaten_from_rsc)

            # Diet tracking: resource species index = n_species + r
            if _diet_tracking_enabled and _diet_matrix is not None:
                rsc_col = config.n_species + r
                if p_idx < _diet_matrix.shape[0] and rsc_col < _diet_matrix.shape[1]:
                    _diet_matrix[p_idx, rsc_col] += eaten_from_rsc

        # Update predator success rate
        if max_eatable > 0:
            state.pred_success_rate[p_idx] += eaten_total / max_eatable
            state.preyed_biomass[p_idx] += eaten_total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Dummy accessibility matrix for Numba when none is loaded
_DUMMY_ACCESS = np.zeros((1, 1), dtype=np.float64)


def predation(
    state: SchoolState,
    config: EngineConfig,
    rng: np.random.Generator,
    n_subdt: int,
    grid_ny: int,
    grid_nx: int,
    resources: ResourceState | None = None,
) -> SchoolState:
    """Apply predation across all grid cells.

    Groups schools by cell via argsort+searchsorted, then processes
    predation within each occupied cell. Uses Numba if available.
    """
    if len(state) == 0:
        return state

    # Make working copies for in-place modification
    abundance = state.abundance.copy()
    pred_success_rate = state.pred_success_rate.copy()
    preyed_biomass = state.preyed_biomass.copy()

    work_state = state.replace(
        abundance=abundance,
        pred_success_rate=pred_success_rate,
        preyed_biomass=preyed_biomass,
    )

    # Compute feeding stages for 2D size ratio lookup
    feeding_stage = compute_feeding_stages(work_state, config)
    work_state = work_state.replace(feeding_stage=feeding_stage)

    # Group schools by cell using searchsorted (fast boundary detection)
    cell_ids = work_state.cell_y * grid_nx + work_state.cell_x
    order = np.argsort(cell_ids, kind="mergesort")
    sorted_cells = cell_ids[order]

    # Find group boundaries with searchsorted
    n_cells = grid_ny * grid_nx
    boundaries = np.searchsorted(sorted_cells, np.arange(n_cells + 1))

    # Precompute accessibility info
    has_access = config.accessibility_matrix is not None
    access_matrix = config.accessibility_matrix if has_access else _DUMMY_ACCESS

    for cell in range(n_cells):
        start = boundaries[cell]
        end = boundaries[cell + 1]
        if end - start < 2:
            continue

        cell_indices = order[start:end].astype(np.int32)

        if _HAS_NUMBA and not _diet_tracking_enabled:
            pred_order = rng.permutation(len(cell_indices)).astype(np.int32)
            _predation_in_cell_numba(
                cell_indices,
                pred_order,
                work_state.abundance,
                work_state.length,
                work_state.weight,
                work_state.age_dt,
                work_state.first_feeding_age_dt,
                work_state.species_id,
                work_state.pred_success_rate,
                work_state.preyed_biomass,
                config.size_ratio_min,
                config.size_ratio_max,
                config.ingestion_rate,
                access_matrix,
                has_access,
                n_subdt,
                work_state.feeding_stage,
            )
        else:
            _predation_in_cell_python(cell_indices, work_state, config, rng, n_subdt)

        # Resource predation: focal schools eat LTL plankton/detritus
        if resources is not None and resources.n_resources > 0:
            cell_y_val = cell // grid_nx
            cell_x_val = cell % grid_nx
            _predation_on_resources(
                cell_indices,
                work_state,
                config,
                resources,
                cell_y_val,
                cell_x_val,
                rng,
                n_subdt,
            )

    new_biomass = work_state.abundance * work_state.weight

    return state.replace(
        abundance=work_state.abundance,
        biomass=new_biomass,
        pred_success_rate=work_state.pred_success_rate,
        preyed_biomass=work_state.preyed_biomass,
    )
