"""Mortality orchestrator for the OSMOSE Python engine.

Implements Java's per-cell per-school interleaved mortality loop:
for each sub-timestep, for each cell, for each school slot,
shuffle mortality causes and apply ONE cause to ONE school
(from that cause's own shuffled sequence), updating n_dead in-place
so that subsequent causes see reduced instantaneous abundance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.feeding_stage import compute_feeding_stages
from osmose.engine.processes.fishing import fishing_mortality  # noqa: F401 — used by tests
from osmose.engine.processes.natural import (
    additional_mortality,  # noqa: F401 — used by tests
    larva_mortality,
    out_mortality,
)
from osmose.engine.processes.predation import (
    _diet_matrix,
    _diet_tracking_enabled,
)
from osmose.engine.processes.starvation import (
    starvation_mortality,  # noqa: F401 — used by tests
    update_starvation_rate,
)
from osmose.engine.resources import ResourceState
from osmose.engine.state import MortalityCause, SchoolState

# Cause indices matching MortalityCause enum
_PREDATION = int(MortalityCause.PREDATION)
_STARVATION = int(MortalityCause.STARVATION)
_ADDITIONAL = int(MortalityCause.ADDITIONAL)
_FISHING = int(MortalityCause.FISHING)

# Module-level TL tracking accumulator (set by mortality(), used by _apply_predation_for_school)
_tl_weighted_sum: NDArray[np.float64] | None = None


# ---------------------------------------------------------------------------
# Per-school mortality helpers (operate in-place on state arrays)
# ---------------------------------------------------------------------------


def _inst_abundance(state: SchoolState, idx: int) -> float:
    """Instantaneous abundance: original abundance minus all deaths so far."""
    return float(state.abundance[idx] - state.n_dead[idx].sum())


def _apply_starvation_for_school(
    idx: int, state: SchoolState, config: EngineConfig, n_subdt: int
) -> None:
    """Apply starvation mortality to a single school (in-place on n_dead)."""
    if state.is_background[idx]:
        return
    if state.age_dt[idx] == 0:
        return
    M = state.starvation_rate[idx] / (config.n_dt_per_year * n_subdt)
    if M <= 0:
        return
    inst_abd = _inst_abundance(state, idx)
    if inst_abd <= 0:
        return
    n_dead = inst_abd * (1.0 - np.exp(-M))
    state.n_dead[idx, _STARVATION] += n_dead


def _apply_additional_for_school(
    idx: int, state: SchoolState, config: EngineConfig, n_subdt: int
) -> None:
    """Apply additional (natural) mortality to a single school (in-place on n_dead)."""
    if state.is_background[idx]:
        return
    if state.age_dt[idx] == 0:
        return
    sp = state.species_id[idx]
    D = config.additional_mortality_rate[sp] / (config.n_dt_per_year * n_subdt)
    if D <= 0:
        return
    inst_abd = _inst_abundance(state, idx)
    if inst_abd <= 0:
        return
    n_dead = inst_abd * (1.0 - np.exp(-D))
    state.n_dead[idx, _ADDITIONAL] += n_dead


def _apply_fishing_for_school(
    idx: int, state: SchoolState, config: EngineConfig, n_subdt: int
) -> None:
    """Apply fishing mortality to a single school (in-place on n_dead)."""
    if state.is_background[idx]:
        return
    if state.age_dt[idx] == 0:
        return
    if not config.fishing_enabled:
        return
    sp = state.species_id[idx]
    f_rate = config.fishing_rate[sp]
    if f_rate <= 0:
        return

    # Selectivity check
    sel_type = config.fishing_selectivity_type[sp]
    if sel_type == 0:
        # Age-based knife-edge
        age_years = state.age_dt[idx] / config.n_dt_per_year
        a50 = config.fishing_selectivity_a50[sp]
        if age_years < a50:
            return
    else:
        # Length-based knife-edge (sel_type == 1 or -1 for legacy)
        l50 = config.fishing_selectivity_l50[sp]
        if l50 > 0 and state.length[idx] < l50:
            return

    # Spatial fishing distribution: multiply rate by cell-specific factor
    sp_map = config.fishing_spatial_maps[sp] if sp < len(config.fishing_spatial_maps) else None
    if sp_map is not None:
        cy = int(state.cell_y[idx])
        cx = int(state.cell_x[idx])
        if 0 <= cy < sp_map.shape[0] and 0 <= cx < sp_map.shape[1]:
            cell_factor = sp_map[cy, cx]
            if cell_factor <= 0 or np.isnan(cell_factor):
                return
            f_rate = f_rate * cell_factor
        else:
            return  # out of map bounds

    F = f_rate / (config.n_dt_per_year * n_subdt)
    inst_abd = _inst_abundance(state, idx)
    if inst_abd <= 0:
        return
    n_dead = inst_abd * (1.0 - np.exp(-F))
    state.n_dead[idx, _FISHING] += n_dead


def _apply_predation_for_school(
    p_idx: int,
    cell_indices: NDArray[np.int32],
    state: SchoolState,
    config: EngineConfig,
    resources: ResourceState | None,
    cell_y: int,
    cell_x: int,
    rng: np.random.Generator,
    n_subdt: int,
    access_matrix: NDArray[np.float64] | None,
    has_access: bool,
    use_stage_access: bool,
    prey_access_idx: NDArray[np.int32] | None,
    pred_access_idx: NDArray[np.int32] | None,
) -> None:
    """Apply predation for a single predator against all preys in the cell.

    Deaths are tracked via n_dead (not direct abundance subtraction),
    so subsequent causes see reduced instantaneous abundance.
    Also handles resource predation for remaining appetite.
    """
    if state.age_dt[p_idx] < state.first_feeding_age_dt[p_idx]:
        return

    inst_abd_p = _inst_abundance(state, p_idx)
    if inst_abd_p <= 0:
        return

    sp_pred = state.species_id[p_idx]
    pred_len = state.length[p_idx]
    stage = state.feeding_stage[p_idx]
    r_min = config.size_ratio_min[sp_pred, stage]
    r_max = config.size_ratio_max[sp_pred, stage]

    biomass_p = inst_abd_p * state.weight[p_idx]
    max_eatable = biomass_p * config.ingestion_rate[sp_pred] / (config.n_dt_per_year * n_subdt)
    if max_eatable <= 0:
        return

    # Scan cell for eligible prey
    available = 0.0
    prey_eligible: list[tuple[int, float]] = []

    for q_idx_val in cell_indices:
        q_idx = int(q_idx_val)
        if q_idx == p_idx:
            continue
        inst_abd_q = _inst_abundance(state, q_idx)
        if inst_abd_q <= 0:
            continue
        prey_len = state.length[q_idx]
        if prey_len <= 0:
            continue
        ratio = pred_len / prey_len
        if ratio < r_min or ratio >= r_max:
            continue

        # Accessibility check
        access_coeff = 1.0
        if has_access and access_matrix is not None:
            if use_stage_access and pred_access_idx is not None and prey_access_idx is not None:
                p_acc = pred_access_idx[p_idx]
                q_acc = prey_access_idx[q_idx]
                if p_acc >= 0 and q_acc >= 0:
                    if q_acc < access_matrix.shape[0] and p_acc < access_matrix.shape[1]:
                        access_coeff = access_matrix[q_acc, p_acc]
                    if access_coeff <= 0:
                        continue
            else:
                sp_prey = state.species_id[q_idx]
                if sp_pred < access_matrix.shape[0] and sp_prey < access_matrix.shape[1]:
                    access_coeff = access_matrix[sp_pred, sp_prey]
                    if access_coeff <= 0:
                        continue

        prey_bio = inst_abd_q * state.weight[q_idx]
        if prey_bio <= 0:
            continue

        eligible = prey_bio * access_coeff
        prey_eligible.append((q_idx, eligible))
        available += eligible

    # School-to-school predation
    eaten_total = 0.0
    if available > 0:
        eaten_total = min(available, max_eatable)
        for q_idx, eligible in prey_eligible:
            share = eligible / available
            eaten_from_prey = eaten_total * share
            if state.weight[q_idx] > 0:
                n_dead_prey = eaten_from_prey / state.weight[q_idx]
                state.n_dead[q_idx, _PREDATION] += n_dead_prey

            # TL tracking: accumulate prey_tl * eaten_biomass for predator
            if _tl_weighted_sum is not None:
                prey_tl = state.trophic_level[q_idx]
                if prey_tl <= 0:
                    prey_tl = 1.0  # default for uninitialized prey
                _tl_weighted_sum[p_idx] += prey_tl * eaten_from_prey

            # Diet tracking
            if _diet_tracking_enabled and _diet_matrix is not None:
                prey_sp = state.species_id[q_idx]
                if p_idx < _diet_matrix.shape[0] and prey_sp < _diet_matrix.shape[1]:
                    _diet_matrix[p_idx, prey_sp] += eaten_from_prey

    # Update predator success rate from school-to-school
    if max_eatable > 0 and eaten_total > 0:
        success = min(eaten_total / max_eatable, 1.0)
        state.pred_success_rate[p_idx] += success / n_subdt
        state.preyed_biomass[p_idx] += eaten_total

    # Resource predation: fill remaining appetite from LTL plankton/detritus
    if resources is not None and resources.n_resources > 0:
        _predation_on_resources_for_school(
            p_idx,
            state,
            config,
            resources,
            cell_y,
            cell_x,
            n_subdt,
            max_eatable,
            access_matrix,
            use_stage_access,
            pred_access_idx,
        )


def _predation_on_resources_for_school(
    p_idx: int,
    state: SchoolState,
    config: EngineConfig,
    resources: ResourceState,
    cell_y: int,
    cell_x: int,
    n_subdt: int,
    max_eatable: float,
    access_matrix: NDArray[np.float64] | None,
    use_stage_access: bool,
    pred_access_idx: NDArray[np.int32] | None,
) -> None:
    """Let one predator eat resources to fill remaining appetite."""
    remaining = max_eatable * (1.0 - min(1.0, state.pred_success_rate[p_idx]))
    if remaining <= 0:
        return

    sp_pred = state.species_id[p_idx]
    pred_len = state.length[p_idx]
    stage = state.feeding_stage[p_idx]
    r_min_val = config.size_ratio_min[sp_pred, stage]
    r_max_val = config.size_ratio_max[sp_pred, stage]

    available = 0.0
    rsc_eligible = np.zeros(resources.n_resources, dtype=np.float64)

    for r in range(resources.n_resources):
        rsc = resources.species[r]
        rsc_bio = resources.get_cell_biomass(r, cell_y, cell_x)
        if rsc_bio <= 0:
            continue

        prey_size_min = pred_len / r_max_val
        prey_size_max = pred_len / r_min_val
        overlap_min = max(rsc.size_min, prey_size_min)
        overlap_max = min(rsc.size_max, prey_size_max)
        if overlap_max <= overlap_min:
            continue
        rsc_range = rsc.size_max - rsc.size_min
        if rsc_range <= 0:
            continue
        percent_resource = (overlap_max - overlap_min) / rsc_range

        access_coeff = 1.0
        if (
            use_stage_access
            and access_matrix is not None
            and pred_access_idx is not None
            and config.stage_accessibility is not None
        ):
            sa = config.stage_accessibility
            rsc_name = rsc.name
            csv_name = sa.resolve_name(rsc_name)
            if csv_name is not None:
                rsc_row = sa.get_index(csv_name, 0.0, role="prey")
                p_acc = pred_access_idx[p_idx]
                if (
                    rsc_row >= 0
                    and p_acc >= 0
                    and rsc_row < access_matrix.shape[0]
                    and p_acc < access_matrix.shape[1]
                ):
                    access_coeff = access_matrix[rsc_row, p_acc]
                    if access_coeff <= 0:
                        continue
        elif not use_stage_access and access_matrix is not None:
            rsc_sp_idx = config.n_species + r
            if sp_pred < access_matrix.shape[0] and rsc_sp_idx < access_matrix.shape[1]:
                access_coeff = access_matrix[sp_pred, rsc_sp_idx]
                if access_coeff <= 0:
                    continue

        eligible_bio = rsc_bio * percent_resource * access_coeff
        rsc_eligible[r] = eligible_bio
        available += eligible_bio

    if available <= 0:
        return

    eaten_total = min(available, remaining)

    cell_id = cell_y * resources.grid.nx + cell_x
    for r in range(resources.n_resources):
        if rsc_eligible[r] <= 0:
            continue
        share = rsc_eligible[r] / available
        eaten_from_rsc = eaten_total * share
        resources.biomass[r, cell_id] = max(0.0, resources.biomass[r, cell_id] - eaten_from_rsc)

        # TL tracking for resource predation
        if _tl_weighted_sum is not None:
            rsc_tl = resources.species[r].trophic_level
            if rsc_tl <= 0:
                rsc_tl = 1.0
            _tl_weighted_sum[p_idx] += rsc_tl * eaten_from_rsc

        if _diet_tracking_enabled and _diet_matrix is not None:
            rsc_col = config.n_species + r
            if p_idx < _diet_matrix.shape[0] and rsc_col < _diet_matrix.shape[1]:
                _diet_matrix[p_idx, rsc_col] += eaten_from_rsc

    if max_eatable > 0:
        success = min(eaten_total / max_eatable, 1.0)
        state.pred_success_rate[p_idx] += success / n_subdt
        state.preyed_biomass[p_idx] += eaten_total


# ---------------------------------------------------------------------------
# Per-cell interleaved mortality
# ---------------------------------------------------------------------------

_DUMMY_ACCESS = np.zeros((1, 1), dtype=np.float64)


def _mortality_in_cell(
    cell_indices: NDArray[np.int32],
    state: SchoolState,
    config: EngineConfig,
    resources: ResourceState | None,
    cell_y: int,
    cell_x: int,
    rng: np.random.Generator,
    n_subdt: int,
    access_matrix: NDArray[np.float64] | None,
    has_access: bool,
    use_stage_access: bool,
    prey_access_idx: NDArray[np.int32] | None,
    pred_access_idx: NDArray[np.int32] | None,
) -> None:
    """Apply interleaved mortality within one cell, matching Java's computeMortality().

    For each school slot i:
      - Shuffle mortality cause order
      - For each cause, use that cause's own shuffled school sequence
      - Apply one cause to one school, updating n_dead in-place

    Modifies state arrays in-place (n_dead, pred_success_rate, preyed_biomass,
    and resources.biomass for resource predation).
    """
    n_local = len(cell_indices)
    if n_local == 0:
        return

    # Create independent shuffled sequences for each cause
    seq_pred = rng.permutation(n_local).astype(np.int32)
    seq_starv = rng.permutation(n_local).astype(np.int32)
    seq_fish = rng.permutation(n_local).astype(np.int32)
    seq_nat = rng.permutation(n_local).astype(np.int32)

    causes = [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING]

    for i in range(n_local):
        rng.shuffle(causes)

        for cause in causes:
            if cause == _PREDATION:
                p_local = seq_pred[i]
                p_idx = int(cell_indices[p_local])
                _apply_predation_for_school(
                    p_idx,
                    cell_indices,
                    state,
                    config,
                    resources,
                    cell_y,
                    cell_x,
                    rng,
                    n_subdt,
                    access_matrix,
                    has_access,
                    use_stage_access,
                    prey_access_idx,
                    pred_access_idx,
                )
            elif cause == _STARVATION:
                s_local = seq_starv[i]
                s_idx = int(cell_indices[s_local])
                _apply_starvation_for_school(s_idx, state, config, n_subdt)
            elif cause == _ADDITIONAL:
                a_local = seq_nat[i]
                a_idx = int(cell_indices[a_local])
                _apply_additional_for_school(a_idx, state, config, n_subdt)
            elif cause == _FISHING:
                f_local = seq_fish[i]
                f_idx = int(cell_indices[f_local])
                _apply_fishing_for_school(f_idx, state, config, n_subdt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
    grid: Grid,
) -> SchoolState:
    """Apply all mortality sources with per-cell per-school interleaved ordering.

    Matches Java's MortalityProcess.computeMortality() structure:
    1. Pre-pass: larva mortality on eggs
    2. Retain eggs (withheld from prey pool)
    3. Per sub-timestep, per cell, per school slot:
       - Shuffle cause order
       - Apply each cause to its own shuffled school
       - n_dead accumulates in-place; instantaneous abundance decreases
    4. Post-loop: abundance = original - total_dead
    5. Out-of-domain mortality, starvation rate update
    """
    global _tl_weighted_sum

    n_subdt = config.mortality_subdt

    # Initialize TL tracking accumulator
    _tl_weighted_sum = np.zeros(len(state), dtype=np.float64)

    # Pre-pass: larva mortality on eggs
    state = larva_mortality(state, config)

    # Retain eggs: withheld from prey pool
    egg_retained = np.where(state.is_egg, state.abundance, 0.0)
    state = state.replace(egg_retained=egg_retained)

    # Make working copies for in-place modification
    n_dead = state.n_dead.copy()
    pred_success_rate = state.pred_success_rate.copy()
    preyed_biomass = state.preyed_biomass.copy()

    work_state = state.replace(
        n_dead=n_dead,
        pred_success_rate=pred_success_rate,
        preyed_biomass=preyed_biomass,
    )

    # Compute feeding stages for predation
    feeding_stage = compute_feeding_stages(work_state, config)
    work_state = work_state.replace(feeding_stage=feeding_stage)

    # Precompute accessibility info
    if config.stage_accessibility is not None:
        sa = config.stage_accessibility
        prey_access_idx = sa.compute_school_indices(
            work_state.species_id,
            work_state.age_dt,
            config.n_dt_per_year,
            config.all_species_names,
            role="prey",
        )
        pred_access_idx = sa.compute_school_indices(
            work_state.species_id,
            work_state.age_dt,
            config.n_dt_per_year,
            config.all_species_names,
            role="pred",
        )
        access_matrix = sa.raw_matrix
        has_access = True
        use_stage_access = True
    else:
        prey_access_idx = np.zeros(len(work_state), dtype=np.int32)
        pred_access_idx = np.zeros(len(work_state), dtype=np.int32)
        has_access = config.accessibility_matrix is not None
        access_matrix = config.accessibility_matrix if has_access else _DUMMY_ACCESS
        use_stage_access = False

    # Group schools by cell
    cell_ids = work_state.cell_y * grid.nx + work_state.cell_x
    valid = (work_state.cell_x >= 0) & (work_state.cell_y >= 0)

    # Precompute cell groupings using argsort+searchsorted for efficiency
    # Only consider valid (located) schools
    valid_indices = np.where(valid)[0].astype(np.int32)
    if len(valid_indices) > 0:
        valid_cell_ids = cell_ids[valid_indices]
        order = np.argsort(valid_cell_ids, kind="mergesort")
        sorted_cells = valid_cell_ids[order]
        sorted_indices = valid_indices[order]

        n_cells = grid.ny * grid.nx
        boundaries = np.searchsorted(sorted_cells, np.arange(n_cells + 1))
    else:
        n_cells = 0
        boundaries = np.array([0, 0])

    for _sub in range(n_subdt):
        # Release fraction of eggs into prey pool
        release = np.where(
            work_state.is_egg & (work_state.egg_retained > 0),
            work_state.abundance / n_subdt,
            0.0,
        )
        new_retained = np.maximum(0, work_state.egg_retained - release)
        work_state = work_state.replace(egg_retained=new_retained)

        # Per-cell mortality
        for cell in range(n_cells):
            start = boundaries[cell]
            end = boundaries[cell + 1]
            if end <= start:
                continue
            cell_indices = sorted_indices[start:end]
            cy = cell // grid.nx
            cx = cell % grid.nx

            _mortality_in_cell(
                cell_indices,
                work_state,
                config,
                resources,
                cy,
                cx,
                rng,
                n_subdt,
                access_matrix,
                has_access,
                use_stage_access,
                prey_access_idx,
                pred_access_idx,
            )

    # Update abundance from accumulated n_dead
    total_dead = work_state.n_dead.sum(axis=1)
    new_abundance = np.maximum(0.0, work_state.abundance - total_dead)
    new_biomass = new_abundance * work_state.weight

    state = state.replace(
        abundance=new_abundance,
        biomass=new_biomass,
        n_dead=work_state.n_dead,
        pred_success_rate=work_state.pred_success_rate,
        preyed_biomass=work_state.preyed_biomass,
        egg_retained=work_state.egg_retained,
    )

    # Post-loop: out-of-domain mortality
    state = out_mortality(state, config)

    # Compute new starvation rate for NEXT step (lagged)
    state = update_starvation_rate(state, config)

    # Update trophic level from predation: TL = 1 + sum(prey_TL * eaten) / total_preyed
    mask = state.preyed_biomass > 0
    if mask.any() and _tl_weighted_sum is not None:
        new_tl = state.trophic_level.copy()
        # Handle schools that may have been appended after _tl_weighted_sum was created
        tl_ws = (
            _tl_weighted_sum[: len(state)]
            if len(_tl_weighted_sum) >= len(state)
            else np.pad(_tl_weighted_sum, (0, len(state) - len(_tl_weighted_sum)))
        )
        valid = mask & (tl_ws > 0)
        if valid.any():
            new_tl[valid] = 1.0 + tl_ws[valid] / state.preyed_biomass[valid]
        state = state.replace(trophic_level=new_tl)

    _tl_weighted_sum = None
    return state
