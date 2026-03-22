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

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# Cause indices matching MortalityCause enum
_PREDATION = int(MortalityCause.PREDATION)
_STARVATION = int(MortalityCause.STARVATION)
_ADDITIONAL = int(MortalityCause.ADDITIONAL)
_FISHING = int(MortalityCause.FISHING)
_DISCARDS = int(MortalityCause.DISCARDS)

# Module-level TL tracking accumulator (set by mortality(), used by _apply_predation_for_school)
_tl_weighted_sum: NDArray[np.float64] | None = None


# ---------------------------------------------------------------------------
# Per-school mortality helpers (operate in-place on state arrays)
# ---------------------------------------------------------------------------


def _apply_starvation_for_school(
    idx: int,
    state: SchoolState,
    config: EngineConfig,
    n_subdt: int,
    inst_abd: NDArray[np.float64],
) -> None:
    """Apply starvation mortality to a single school (in-place on n_dead)."""
    if state.is_background[idx]:
        return
    if state.age_dt[idx] == 0:
        return
    M = state.starvation_rate[idx] / (config.n_dt_per_year * n_subdt)
    if M <= 0:
        return
    abd = inst_abd[idx]
    if abd <= 0:
        return
    n_dead = abd * (1.0 - np.exp(-M))
    state.n_dead[idx, _STARVATION] += n_dead
    inst_abd[idx] -= n_dead


def _apply_additional_for_school(
    idx: int,
    state: SchoolState,
    config: EngineConfig,
    n_subdt: int,
    inst_abd: NDArray[np.float64],
    step: int = 0,
) -> None:
    """Apply additional (natural) mortality to a single school (in-place on n_dead)."""
    if state.is_background[idx]:
        return
    if state.age_dt[idx] == 0:
        return
    sp = state.species_id[idx]

    # Base rate: constant or time-varying (BY_DT)
    rate = config.additional_mortality_rate[sp]
    if config.additional_mortality_by_dt is not None and config.additional_mortality_by_dt[sp] is not None:
        arr = config.additional_mortality_by_dt[sp]
        rate = arr[step % len(arr)]

    # Spatial multiplier
    if config.additional_mortality_spatial is not None and config.additional_mortality_spatial[sp] is not None:
        sp_map = config.additional_mortality_spatial[sp]
        cy = int(state.cell_y[idx])
        cx = int(state.cell_x[idx])
        if 0 <= cy < sp_map.shape[0] and 0 <= cx < sp_map.shape[1]:
            spatial_factor = sp_map[cy, cx]
            if spatial_factor <= 0 or np.isnan(spatial_factor):
                return
            rate = rate * spatial_factor
        else:
            return  # out of map bounds

    D = rate / (config.n_dt_per_year * n_subdt)
    if D <= 0:
        return
    abd = inst_abd[idx]
    if abd <= 0:
        return
    n_dead = abd * (1.0 - np.exp(-D))
    state.n_dead[idx, _ADDITIONAL] += n_dead
    inst_abd[idx] -= n_dead


def _apply_fishing_for_school(
    idx: int,
    state: SchoolState,
    config: EngineConfig,
    n_subdt: int,
    inst_abd: NDArray[np.float64],
    step: int = 0,
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

    # Rate by year override
    if config.fishing_rate_by_year is not None:
        year = step // config.n_dt_per_year
        arr = config.fishing_rate_by_year[sp] if sp < len(config.fishing_rate_by_year) else None
        if arr is not None and year < len(arr):
            f_rate = arr[year]

    if f_rate <= 0:
        return

    # Selectivity check
    sel_type = config.fishing_selectivity_type[sp]
    selectivity = 1.0
    if sel_type == 0:
        # Age-based knife-edge
        age_years = state.age_dt[idx] / config.n_dt_per_year
        a50 = config.fishing_selectivity_a50[sp]
        if age_years < a50:
            return
    elif sel_type == 1:
        # Sigmoidal size selectivity
        l50 = config.fishing_selectivity_l50[sp]
        slope = config.fishing_selectivity_slope[sp]
        selectivity = 1.0 / (1.0 + np.exp(-slope * (state.length[idx] - l50)))
    else:
        # Length-based knife-edge (sel_type == -1 for legacy)
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

    # MPA reduction
    if config.mpa_zones is not None:
        year = step // config.n_dt_per_year
        cy = int(state.cell_y[idx])
        cx = int(state.cell_x[idx])
        for mpa in config.mpa_zones:
            if not (mpa.start_year <= year < mpa.end_year):
                continue
            if 0 <= cy < mpa.grid.shape[0] and 0 <= cx < mpa.grid.shape[1] and mpa.grid[cy, cx] > 0:
                f_rate *= 1.0 - mpa.percentage

    # Seasonality
    if config.fishing_seasonality is not None:
        step_in_year = step % config.n_dt_per_year
        season_weight = config.fishing_seasonality[sp, step_in_year]
        F = f_rate * season_weight * selectivity / n_subdt
    else:
        F = f_rate * selectivity / (config.n_dt_per_year * n_subdt)

    abd = inst_abd[idx]
    if abd <= 0:
        return
    n_dead = abd * (1.0 - np.exp(-F))

    # Discards split
    if config.fishing_discard_rate is not None:
        discard_r = config.fishing_discard_rate[sp]
        state.n_dead[idx, _FISHING] += n_dead * (1.0 - discard_r)
        state.n_dead[idx, _DISCARDS] += n_dead * discard_r
    else:
        state.n_dead[idx, _FISHING] += n_dead
    inst_abd[idx] -= n_dead


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
    inst_abd: NDArray[np.float64] | None = None,
) -> None:
    """Apply predation for a single predator against ALL preys in the cell.

    Matches Java's computePredation(): schools and resources are combined
    into a single accessible-biomass pool and eating is distributed
    proportionally across ALL prey types simultaneously.

    Deaths are tracked via n_dead (not direct abundance subtraction),
    so subsequent causes see reduced instantaneous abundance.
    """
    if state.age_dt[p_idx] < state.first_feeding_age_dt[p_idx]:
        return

    inst_abd_p = inst_abd[p_idx]
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

    # --- Phase 1: Scan ALL prey (schools + resources) into unified pool ---
    # Each entry: ("school", q_idx, accessible_biomass) or ("rsc", r_idx, accessible_biomass)
    all_prey: list[tuple[str, int, float]] = []
    total_available = 0.0

    # 1a. Scan cell schools as prey
    for q_idx_val in cell_indices:
        q_idx = int(q_idx_val)
        if q_idx == p_idx:
            continue
        inst_abd_q = inst_abd[q_idx]
        if inst_abd_q <= 0:
            continue
        prey_len = state.length[q_idx]
        if prey_len <= 0:
            continue
        ratio = pred_len / prey_len
        if ratio < r_min or ratio >= r_max:
            continue

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
        all_prey.append(("school", q_idx, eligible))
        total_available += eligible

    # 1b. Scan resources as prey (matching Java: resources in same preys list)
    if resources is not None and resources.n_resources > 0:
        for r in range(resources.n_resources):
            rsc = resources.species[r]
            rsc_bio = resources.get_cell_biomass(r, cell_y, cell_x)
            if rsc_bio <= 0:
                continue

            prey_size_min = pred_len / r_max
            prey_size_max = pred_len / r_min
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
                csv_name = sa.resolve_name(rsc.name)
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
            all_prey.append(("rsc", r, eligible_bio))
            total_available += eligible_bio

    if total_available <= 0:
        return

    # --- Phase 2: Distribute eating proportionally (matching Java) ---
    eaten_total = min(total_available, max_eatable)

    cell_id = cell_y * (resources.grid.nx if resources else 0) + cell_x

    for prey_type, prey_id, eligible in all_prey:
        share = eligible / total_available
        eaten_from_prey = eaten_total * share

        if prey_type == "school":
            q_idx = prey_id
            if state.weight[q_idx] > 0:
                n_dead_prey = eaten_from_prey / state.weight[q_idx]
                state.n_dead[q_idx, _PREDATION] += n_dead_prey
                inst_abd[q_idx] -= n_dead_prey

            if _tl_weighted_sum is not None:
                prey_tl = state.trophic_level[q_idx]
                if prey_tl <= 0:
                    prey_tl = 1.0
                _tl_weighted_sum[p_idx] += prey_tl * eaten_from_prey

            if _diet_tracking_enabled and _diet_matrix is not None:
                prey_sp = state.species_id[q_idx]
                if p_idx < _diet_matrix.shape[0] and prey_sp < _diet_matrix.shape[1]:
                    _diet_matrix[p_idx, prey_sp] += eaten_from_prey
        else:
            r = prey_id
            if resources is not None:
                resources.biomass[r, cell_id] = max(
                    0.0, resources.biomass[r, cell_id] - eaten_from_prey
                )

            if _tl_weighted_sum is not None:
                rsc_tl = resources.species[r].trophic_level if resources else 1.0
                if rsc_tl <= 0:
                    rsc_tl = 1.0
                _tl_weighted_sum[p_idx] += rsc_tl * eaten_from_prey

            if _diet_tracking_enabled and _diet_matrix is not None and resources:
                rsc_col = config.n_species + r
                if p_idx < _diet_matrix.shape[0] and rsc_col < _diet_matrix.shape[1]:
                    _diet_matrix[p_idx, rsc_col] += eaten_from_prey

    # --- Phase 3: Update predation success rate ONCE (matching Java) ---
    success = min(eaten_total / max_eatable, 1.0)
    state.pred_success_rate[p_idx] += success / n_subdt
    state.preyed_biomass[p_idx] += eaten_total


# ---------------------------------------------------------------------------
# Numba-accelerated predation for the interleaved mortality path (Tier 2)
# ---------------------------------------------------------------------------

_DUMMY_RSC_1D = np.zeros(0, dtype=np.float64)
_DUMMY_RSC_I1D = np.zeros(0, dtype=np.int32)
_DUMMY_RSC_2D = np.zeros((0, 0), dtype=np.float64)
_DUMMY_DIET = np.zeros((1, 1), dtype=np.float64)


def _pre_generate_cell_rng(
    rng: np.random.Generator,
    boundaries: NDArray[np.int64],
    n_cells: int,
) -> tuple[list[NDArray[np.int32]], NDArray[np.int32]]:
    """Pre-generate all random data for all cells in one Python pass.

    Note: Not used by the batch Numba path (which generates RNG inline).
    Retained as a tested reference implementation and for potential future use.

    Returns:
        seq_bufs: list of 4 int32 arrays, each of length boundaries[n_cells].
            seq_bufs[k][start:end] is rng.permutation(n_local) for cell k.
        cause_orders_buf: int32 array of shape (total, 4).
            cause_orders_buf[start+i] is a shuffled [PRED, STARV, ADDITIONAL, FISHING].
    """
    total = int(boundaries[n_cells])
    seq_bufs = [np.empty(total, dtype=np.int32) for _ in range(4)]
    cause_orders_buf = np.empty((total, 4), dtype=np.int32)
    causes = [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING]

    for cell in range(n_cells):
        start = int(boundaries[cell])
        end = int(boundaries[cell + 1])
        n_local = end - start
        if n_local == 0:
            continue
        for k in range(4):
            seq_bufs[k][start:end] = rng.permutation(n_local).astype(np.int32)
        for i in range(n_local):
            rng.shuffle(causes)
            cause_orders_buf[start + i, 0] = causes[0]
            cause_orders_buf[start + i, 1] = causes[1]
            cause_orders_buf[start + i, 2] = causes[2]
            cause_orders_buf[start + i, 3] = causes[3]

    return seq_bufs, cause_orders_buf


def _precompute_resource_arrays(config, resources):
    """Extract resource metadata into flat Numba-compatible arrays."""
    if resources is None or resources.n_resources == 0:
        return _DUMMY_RSC_1D, _DUMMY_RSC_1D, _DUMMY_RSC_1D, _DUMMY_RSC_I1D, 0

    n_rsc = resources.n_resources
    rsc_size_min = np.array([s.size_min for s in resources.species], dtype=np.float64)
    rsc_size_max = np.array([s.size_max for s in resources.species], dtype=np.float64)
    rsc_tl = np.array([s.trophic_level for s in resources.species], dtype=np.float64)

    rsc_access_rows = np.full(n_rsc, -1, dtype=np.int32)
    if config.stage_accessibility is not None:
        sa = config.stage_accessibility
        for r in range(n_rsc):
            csv_name = sa.resolve_name(resources.species[r].name)
            if csv_name is not None:
                rsc_access_rows[r] = sa.get_index(csv_name, 0.0, role="prey")

    return rsc_size_min, rsc_size_max, rsc_tl, rsc_access_rows, n_rsc


def _precompute_effective_rates(work_state, config, n_subdt, step):
    """Pre-compute per-school effective mortality rates for the Numba path.

    Returns (eff_starv, eff_additional, eff_fishing, fishing_discard) arrays,
    each of shape (n_schools,). The Numba cell loop applies:
        dead = inst_abd[idx] * (1 - exp(-rate))
    """
    n = len(work_state)

    # Starvation: D = starvation_rate / (n_dt * n_subdt)
    denom = config.n_dt_per_year * n_subdt
    eff_starv = work_state.starvation_rate / denom
    eff_starv = eff_starv.copy()  # don't modify state array
    eff_starv[work_state.is_background] = 0.0
    eff_starv[work_state.age_dt == 0] = 0.0
    eff_starv[eff_starv < 0] = 0.0

    # Additional mortality (vectorized over species)
    sp = work_state.species_id
    rates = config.additional_mortality_rate[sp].copy()

    if config.additional_mortality_by_dt is not None:
        for sp_id in range(config.n_species):
            arr = config.additional_mortality_by_dt[sp_id]
            if arr is not None:
                mask = sp == sp_id
                rates[mask] = arr[step % len(arr)]

    if config.additional_mortality_spatial is not None:
        for sp_id in range(config.n_species):
            sp_map = config.additional_mortality_spatial[sp_id]
            if sp_map is not None:
                mask = sp == sp_id
                cy = work_state.cell_y[mask]
                cx = work_state.cell_x[mask]
                valid = (
                    (cy >= 0)
                    & (cy < sp_map.shape[0])
                    & (cx >= 0)
                    & (cx < sp_map.shape[1])
                )
                factors = np.zeros(mask.sum(), dtype=np.float64)
                if valid.any():
                    f_vals = sp_map[cy[valid], cx[valid]]
                    f_vals = np.where(np.isnan(f_vals) | (f_vals <= 0), 0.0, f_vals)
                    factors[valid] = f_vals
                rates[mask] *= factors

    rates[work_state.is_background] = 0.0
    rates[work_state.age_dt == 0] = 0.0
    rates[rates < 0] = 0.0
    eff_additional = rates / denom
    np.nan_to_num(eff_additional, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Fishing (vectorized over species)
    eff_fishing = np.zeros(n, dtype=np.float64)
    fishing_discard = np.zeros(n, dtype=np.float64)

    if config.fishing_enabled:
        f_rates = config.fishing_rate[sp].copy()

        if config.fishing_rate_by_year is not None:
            year = step // config.n_dt_per_year
            for sp_id in range(config.n_species):
                arr = (
                    config.fishing_rate_by_year[sp_id]
                    if sp_id < len(config.fishing_rate_by_year)
                    else None
                )
                if arr is not None and year < len(arr):
                    f_rates[sp == sp_id] = arr[year]

        selectivity = np.ones(n, dtype=np.float64)
        for sp_id in range(config.n_species):
            mask = sp == sp_id
            if not mask.any():
                continue
            sel_type = config.fishing_selectivity_type[sp_id]
            if sel_type == 0:  # age-based
                age_years = work_state.age_dt[mask] / config.n_dt_per_year
                a50 = config.fishing_selectivity_a50[sp_id]
                selectivity[mask] = np.where(age_years < a50, 0.0, 1.0)
            elif sel_type == 1:  # logistic
                l50 = config.fishing_selectivity_l50[sp_id]
                slope = config.fishing_selectivity_slope[sp_id]
                selectivity[mask] = 1.0 / (
                    1.0 + np.exp(-slope * (work_state.length[mask] - l50))
                )
            else:  # length cutoff
                l50 = config.fishing_selectivity_l50[sp_id]
                selectivity[mask] = np.where(
                    (l50 > 0) & (work_state.length[mask] < l50), 0.0, 1.0
                )

        spatial_factor = np.ones(n, dtype=np.float64)
        for sp_id in range(config.n_species):
            sp_map = (
                config.fishing_spatial_maps[sp_id]
                if sp_id < len(config.fishing_spatial_maps)
                else None
            )
            if sp_map is None:
                continue
            mask = sp == sp_id
            cy = work_state.cell_y[mask]
            cx = work_state.cell_x[mask]
            valid = (
                (cy >= 0)
                & (cy < sp_map.shape[0])
                & (cx >= 0)
                & (cx < sp_map.shape[1])
            )
            factors = np.zeros(mask.sum(), dtype=np.float64)
            if valid.any():
                f_vals = sp_map[cy[valid], cx[valid]]
                f_vals = np.where(np.isnan(f_vals) | (f_vals <= 0), 0.0, f_vals)
                factors[valid] = f_vals
            spatial_factor[mask] = factors

        mpa_factor = np.ones(n, dtype=np.float64)
        if config.mpa_zones is not None:
            year = step // config.n_dt_per_year
            for mpa in config.mpa_zones:
                if not (mpa.start_year <= year < mpa.end_year):
                    continue
                cy = work_state.cell_y
                cx = work_state.cell_x
                valid = (
                    (cy >= 0)
                    & (cy < mpa.grid.shape[0])
                    & (cx >= 0)
                    & (cx < mpa.grid.shape[1])
                )
                in_mpa = np.zeros(n, dtype=np.bool_)
                in_mpa[valid] = mpa.grid[cy[valid], cx[valid]] > 0
                mpa_factor *= np.where(in_mpa, 1.0 - mpa.percentage, 1.0)

        # Combine — denominator differs based on seasonality
        if config.fishing_seasonality is not None:
            step_in_year = step % config.n_dt_per_year
            season = config.fishing_seasonality[sp, step_in_year]
            eff_fishing = (
                f_rates * selectivity * spatial_factor * mpa_factor * season / n_subdt
            )
        else:
            eff_fishing = f_rates * selectivity * spatial_factor * mpa_factor / denom

        eff_fishing[work_state.is_background] = 0.0
        eff_fishing[work_state.age_dt == 0] = 0.0
        eff_fishing[eff_fishing < 0] = 0.0
        np.nan_to_num(eff_fishing, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if config.fishing_discard_rate is not None:
            fishing_discard = np.where(
                eff_fishing > 0, config.fishing_discard_rate[sp], 0.0
            )

    return eff_starv, eff_additional, eff_fishing, fishing_discard


if _HAS_NUMBA:

    @njit(cache=True)
    def _apply_predation_numba(
        p_idx,
        cell_indices,
        inst_abd,
        n_dead,
        species_id,
        length,
        weight,
        age_dt,
        first_feeding_age_dt,
        feeding_stage,
        pred_success_rate,
        preyed_biomass,
        trophic_level,
        size_ratio_min,
        size_ratio_max,
        ingestion_rate,
        n_dt_per_year,
        n_subdt,
        access_matrix,
        has_access,
        use_stage_access,
        prey_access_idx,
        pred_access_idx,
        rsc_biomass,
        rsc_size_min,
        rsc_size_max,
        rsc_tl,
        rsc_access_rows,
        n_resources,
        n_species,
        cell_id,
        tl_weighted_sum,
        tl_tracking,
        diet_matrix,
        diet_enabled,
    ):
        """Numba-compiled single-predator predation (schools + resources)."""
        if age_dt[p_idx] < first_feeding_age_dt[p_idx]:
            return

        abd_p = inst_abd[p_idx]
        if abd_p <= 0:
            return

        sp_pred = species_id[p_idx]
        pred_len = length[p_idx]
        stage = feeding_stage[p_idx]
        r_min = size_ratio_min[sp_pred, stage]
        r_max = size_ratio_max[sp_pred, stage]

        biomass_p = abd_p * weight[p_idx]
        max_eatable = biomass_p * ingestion_rate[sp_pred] / (n_dt_per_year * n_subdt)
        if max_eatable <= 0:
            return

        n_local = len(cell_indices)
        max_prey = n_local + n_resources

        # Phase 1: Scan all prey into fixed-size arrays
        prey_type = np.zeros(max_prey, dtype=np.int32)
        prey_id = np.zeros(max_prey, dtype=np.int32)
        prey_eligible = np.zeros(max_prey, dtype=np.float64)
        total_available = 0.0
        n_prey = 0

        # 1a: School prey
        for q_pos in range(n_local):
            q_idx = cell_indices[q_pos]
            if q_idx == p_idx:
                continue
            abd_q = inst_abd[q_idx]
            if abd_q <= 0:
                continue
            prey_len = length[q_idx]
            if prey_len <= 0:
                continue
            ratio = pred_len / prey_len
            if ratio < r_min or ratio >= r_max:
                continue

            access_coeff = 1.0
            if has_access:
                if use_stage_access:
                    p_acc = pred_access_idx[p_idx]
                    q_acc = prey_access_idx[q_idx]
                    if p_acc >= 0 and q_acc >= 0:
                        if (
                            q_acc < access_matrix.shape[0]
                            and p_acc < access_matrix.shape[1]
                        ):
                            access_coeff = access_matrix[q_acc, p_acc]
                        if access_coeff <= 0:
                            continue
                else:
                    sp_prey = species_id[q_idx]
                    if (
                        sp_pred < access_matrix.shape[0]
                        and sp_prey < access_matrix.shape[1]
                    ):
                        access_coeff = access_matrix[sp_pred, sp_prey]
                        if access_coeff <= 0:
                            continue

            prey_bio = abd_q * weight[q_idx]
            if prey_bio <= 0:
                continue

            eligible = prey_bio * access_coeff
            prey_type[n_prey] = 0
            prey_id[n_prey] = q_idx
            prey_eligible[n_prey] = eligible
            total_available += eligible
            n_prey += 1

        # 1b: Resource prey
        for r in range(n_resources):
            rsc_bio = rsc_biomass[r, cell_id]
            if rsc_bio <= 0:
                continue

            prey_size_min = pred_len / r_max
            prey_size_max = pred_len / r_min
            overlap_min = max(rsc_size_min[r], prey_size_min)
            overlap_max = min(rsc_size_max[r], prey_size_max)
            if overlap_max <= overlap_min:
                continue
            rsc_range = rsc_size_max[r] - rsc_size_min[r]
            if rsc_range <= 0:
                continue
            percent_resource = (overlap_max - overlap_min) / rsc_range

            access_coeff = 1.0
            if use_stage_access:
                rsc_row = rsc_access_rows[r]
                p_acc = pred_access_idx[p_idx]
                if rsc_row >= 0 and p_acc >= 0:
                    if (
                        rsc_row < access_matrix.shape[0]
                        and p_acc < access_matrix.shape[1]
                    ):
                        access_coeff = access_matrix[rsc_row, p_acc]
                        if access_coeff <= 0:
                            continue
            elif has_access:
                rsc_sp_idx = n_species + r
                if (
                    sp_pred < access_matrix.shape[0]
                    and rsc_sp_idx < access_matrix.shape[1]
                ):
                    access_coeff = access_matrix[sp_pred, rsc_sp_idx]
                    if access_coeff <= 0:
                        continue

            eligible_bio = rsc_bio * percent_resource * access_coeff
            prey_type[n_prey] = 1
            prey_id[n_prey] = r
            prey_eligible[n_prey] = eligible_bio
            total_available += eligible_bio
            n_prey += 1

        if total_available <= 0:
            return

        # Phase 2: Distribute eating proportionally
        eaten_total = min(total_available, max_eatable)

        for k in range(n_prey):
            share = prey_eligible[k] / total_available
            eaten_from_prey = eaten_total * share

            if prey_type[k] == 0:  # school
                q_idx = prey_id[k]
                if weight[q_idx] > 0:
                    n_dead_prey = eaten_from_prey / weight[q_idx]
                    n_dead[q_idx, 0] += n_dead_prey
                    inst_abd[q_idx] -= n_dead_prey

                if tl_tracking:
                    prey_tl = trophic_level[q_idx]
                    if prey_tl <= 0:
                        prey_tl = 1.0
                    tl_weighted_sum[p_idx] += prey_tl * eaten_from_prey

                if diet_enabled:
                    prey_sp = species_id[q_idx]
                    if (
                        p_idx < diet_matrix.shape[0]
                        and prey_sp < diet_matrix.shape[1]
                    ):
                        diet_matrix[p_idx, prey_sp] += eaten_from_prey
            else:  # resource
                r_idx = prey_id[k]
                rsc_biomass[r_idx, cell_id] = max(
                    0.0, rsc_biomass[r_idx, cell_id] - eaten_from_prey
                )

                if tl_tracking:
                    r_tl = rsc_tl[r_idx]
                    if r_tl <= 0:
                        r_tl = 1.0
                    tl_weighted_sum[p_idx] += r_tl * eaten_from_prey

                if diet_enabled:
                    rsc_col = n_species + r_idx
                    if (
                        p_idx < diet_matrix.shape[0]
                        and rsc_col < diet_matrix.shape[1]
                    ):
                        diet_matrix[p_idx, rsc_col] += eaten_from_prey

        # Phase 3: Update predation success rate
        success = min(eaten_total / max_eatable, 1.0)
        pred_success_rate[p_idx] += success / n_subdt
        preyed_biomass[p_idx] += eaten_total

    @njit(cache=True)
    def _mortality_in_cell_numba(
        cell_indices,
        seq_pred,
        seq_starv,
        seq_fish,
        seq_nat,
        cause_orders,
        inst_abd,
        n_dead,
        eff_starv,
        eff_additional,
        eff_fishing,
        fishing_discard,
        species_id,
        length,
        weight,
        age_dt,
        first_feeding_age_dt,
        feeding_stage,
        pred_success_rate,
        preyed_biomass,
        trophic_level,
        size_ratio_min,
        size_ratio_max,
        ingestion_rate,
        n_dt_per_year,
        n_subdt,
        access_matrix,
        has_access,
        use_stage_access,
        prey_access_idx,
        pred_access_idx,
        rsc_biomass,
        rsc_size_min,
        rsc_size_max,
        rsc_tl,
        rsc_access_rows,
        n_resources,
        n_species,
        cell_id,
        tl_weighted_sum,
        tl_tracking,
        diet_matrix,
        diet_enabled,
    ):
        """Numba-compiled full interleaved mortality for all 4 causes.

        SYNC: Inner loop logic duplicated in _mortality_all_cells_numba and
        _mortality_all_cells_parallel. Changes here must be mirrored there.
        """
        n_local = len(cell_indices)
        for i in range(n_local):
            for c in range(4):
                cause = cause_orders[i, c]
                if cause == 0:  # PREDATION
                    p_idx = cell_indices[seq_pred[i]]
                    _apply_predation_numba(
                        p_idx,
                        cell_indices,
                        inst_abd,
                        n_dead,
                        species_id,
                        length,
                        weight,
                        age_dt,
                        first_feeding_age_dt,
                        feeding_stage,
                        pred_success_rate,
                        preyed_biomass,
                        trophic_level,
                        size_ratio_min,
                        size_ratio_max,
                        ingestion_rate,
                        n_dt_per_year,
                        n_subdt,
                        access_matrix,
                        has_access,
                        use_stage_access,
                        prey_access_idx,
                        pred_access_idx,
                        rsc_biomass,
                        rsc_size_min,
                        rsc_size_max,
                        rsc_tl,
                        rsc_access_rows,
                        n_resources,
                        n_species,
                        cell_id,
                        tl_weighted_sum,
                        tl_tracking,
                        diet_matrix,
                        diet_enabled,
                    )
                elif cause == 1:  # STARVATION
                    idx = cell_indices[seq_starv[i]]
                    D = eff_starv[idx]
                    if D > 0:
                        abd = inst_abd[idx]
                        if abd > 0:
                            dead = abd * (1.0 - np.exp(-D))
                            n_dead[idx, 1] += dead
                            inst_abd[idx] -= dead
                elif cause == 2:  # ADDITIONAL
                    idx = cell_indices[seq_nat[i]]
                    D = eff_additional[idx]
                    if D > 0:
                        abd = inst_abd[idx]
                        if abd > 0:
                            dead = abd * (1.0 - np.exp(-D))
                            n_dead[idx, 2] += dead
                            inst_abd[idx] -= dead
                elif cause == 3:  # FISHING
                    idx = cell_indices[seq_fish[i]]
                    F = eff_fishing[idx]
                    if F > 0:
                        abd = inst_abd[idx]
                        if abd > 0:
                            dead = abd * (1.0 - np.exp(-F))
                            discard_r = fishing_discard[idx]
                            if discard_r > 0:
                                n_dead[idx, 3] += dead * (1.0 - discard_r)
                                n_dead[idx, 6] += dead * discard_r
                            else:
                                n_dead[idx, 3] += dead
                            inst_abd[idx] -= dead

    @njit(cache=True)
    def _mortality_all_cells_numba(
        rng_seed,
        sorted_indices, boundaries, n_cells,
        inst_abd, n_dead,
        eff_starv, eff_additional, eff_fishing, fishing_discard,
        species_id, length, weight, age_dt,
        first_feeding_age_dt, feeding_stage, pred_success_rate,
        preyed_biomass, trophic_level,
        size_ratio_min, size_ratio_max, ingestion_rate,
        n_dt_per_year, n_subdt,
        access_matrix, has_access, use_stage_access,
        prey_access_idx, pred_access_idx,
        rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl,
        rsc_access_rows, n_resources, n_species,
        tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled,
    ):
        """Numba-compiled batch mortality for ALL cells in one call.

        RNG is generated inline using Numba's np.random (seeded from Python).
        This avoids the Python loop overhead of pre-generating RNG data.

        SYNC: Inner loop logic duplicated from _mortality_in_cell_numba.
        Also duplicated in _mortality_all_cells_parallel. Keep all three in sync.
        """
        np.random.seed(rng_seed)
        for cell in range(n_cells):
            start = boundaries[cell]
            end = boundaries[cell + 1]
            if end <= start:
                continue

            cell_indices = sorted_indices[start:end]
            n_local = end - start
            cell_id = cell  # flat row-major index

            # Generate RNG inline (compiled, no Python overhead)
            seq_pred = np.random.permutation(n_local).astype(np.int32)
            seq_starv = np.random.permutation(n_local).astype(np.int32)
            seq_fish = np.random.permutation(n_local).astype(np.int32)
            seq_nat = np.random.permutation(n_local).astype(np.int32)
            causes = np.array([0, 1, 2, 3], dtype=np.int32)
            cause_orders = np.empty((n_local, 4), dtype=np.int32)
            for ii in range(n_local):
                np.random.shuffle(causes)
                cause_orders[ii, 0] = causes[0]
                cause_orders[ii, 1] = causes[1]
                cause_orders[ii, 2] = causes[2]
                cause_orders[ii, 3] = causes[3]

            for i in range(n_local):
                for c in range(4):
                    cause = cause_orders[i, c]
                    if cause == 0:  # PREDATION
                        p_idx = cell_indices[seq_pred[i]]
                        _apply_predation_numba(
                            p_idx, cell_indices,
                            inst_abd, n_dead, species_id, length, weight,
                            age_dt, first_feeding_age_dt, feeding_stage,
                            pred_success_rate, preyed_biomass, trophic_level,
                            size_ratio_min, size_ratio_max, ingestion_rate,
                            n_dt_per_year, n_subdt,
                            access_matrix, has_access, use_stage_access,
                            prey_access_idx, pred_access_idx,
                            rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl,
                            rsc_access_rows, n_resources, n_species, cell_id,
                            tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled,
                        )
                    elif cause == 1:  # STARVATION
                        idx = cell_indices[seq_starv[i]]
                        D = eff_starv[idx]
                        if D > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-D))
                                n_dead[idx, 1] += dead
                                inst_abd[idx] -= dead
                    elif cause == 2:  # ADDITIONAL
                        idx = cell_indices[seq_nat[i]]
                        D = eff_additional[idx]
                        if D > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-D))
                                n_dead[idx, 2] += dead
                                inst_abd[idx] -= dead
                    elif cause == 3:  # FISHING
                        idx = cell_indices[seq_fish[i]]
                        F = eff_fishing[idx]
                        if F > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-F))
                                discard_r = fishing_discard[idx]
                                if discard_r > 0:
                                    n_dead[idx, 3] += dead * (1.0 - discard_r)
                                    n_dead[idx, 6] += dead * discard_r
                                else:
                                    n_dead[idx, 3] += dead
                                inst_abd[idx] -= dead

    @njit(cache=True, parallel=True)
    def _mortality_all_cells_parallel(
        rng_seed,
        sorted_indices, boundaries, n_cells,
        inst_abd, n_dead,
        eff_starv, eff_additional, eff_fishing, fishing_discard,
        species_id, length, weight, age_dt,
        first_feeding_age_dt, feeding_stage, pred_success_rate,
        preyed_biomass, trophic_level,
        size_ratio_min, size_ratio_max, ingestion_rate,
        n_dt_per_year, n_subdt,
        access_matrix, has_access, use_stage_access,
        prey_access_idx, pred_access_idx,
        rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl,
        rsc_access_rows, n_resources, n_species,
        tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled,
    ):
        """Parallel batch mortality — prange over cells for multi-core execution.

        Each cell gets a deterministic seed derived from rng_seed + cell index.
        RNG is generated inline per cell (same as sequential version) to avoid
        the overhead of a separate pre-generation loop. Deterministic because
        np.random.seed() resets the thread-local PRNG at each iteration.

        Safe because all school-level mutations are cell-local: each cell's
        index range [start, end) is disjoint.

        SYNC: Inner loop logic duplicated from _mortality_in_cell_numba.
        Also duplicated in _mortality_all_cells_numba. Keep all three in sync.
        """
        for cell in prange(n_cells):
            start = boundaries[cell]
            end = boundaries[cell + 1]
            if end <= start:
                continue

            # Per-cell deterministic seed
            np.random.seed(rng_seed + np.int64(cell) * np.int64(7919))

            cell_indices = sorted_indices[start:end]
            n_local = end - start
            cell_id = cell

            # Generate RNG inline (compiled, no Python overhead)
            seq_pred = np.random.permutation(n_local).astype(np.int32)
            seq_starv = np.random.permutation(n_local).astype(np.int32)
            seq_fish = np.random.permutation(n_local).astype(np.int32)
            seq_nat = np.random.permutation(n_local).astype(np.int32)
            causes = np.array([0, 1, 2, 3], dtype=np.int32)
            cause_orders = np.empty((n_local, 4), dtype=np.int32)
            for ii in range(n_local):
                np.random.shuffle(causes)
                cause_orders[ii, 0] = causes[0]
                cause_orders[ii, 1] = causes[1]
                cause_orders[ii, 2] = causes[2]
                cause_orders[ii, 3] = causes[3]

            for i in range(n_local):
                for c in range(4):
                    cause = cause_orders[i, c]
                    if cause == 0:  # PREDATION
                        p_idx = cell_indices[seq_pred[i]]
                        _apply_predation_numba(
                            p_idx, cell_indices,
                            inst_abd, n_dead, species_id, length, weight,
                            age_dt, first_feeding_age_dt, feeding_stage,
                            pred_success_rate, preyed_biomass, trophic_level,
                            size_ratio_min, size_ratio_max, ingestion_rate,
                            n_dt_per_year, n_subdt,
                            access_matrix, has_access, use_stage_access,
                            prey_access_idx, pred_access_idx,
                            rsc_biomass, rsc_size_min, rsc_size_max, rsc_tl,
                            rsc_access_rows, n_resources, n_species, cell_id,
                            tl_weighted_sum, tl_tracking, diet_matrix, diet_enabled,
                        )
                    elif cause == 1:  # STARVATION
                        idx = cell_indices[seq_starv[i]]
                        D = eff_starv[idx]
                        if D > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-D))
                                n_dead[idx, 1] += dead
                                inst_abd[idx] -= dead
                    elif cause == 2:  # ADDITIONAL
                        idx = cell_indices[seq_nat[i]]
                        D = eff_additional[idx]
                        if D > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-D))
                                n_dead[idx, 2] += dead
                                inst_abd[idx] -= dead
                    elif cause == 3:  # FISHING
                        idx = cell_indices[seq_fish[i]]
                        F = eff_fishing[idx]
                        if F > 0:
                            abd = inst_abd[idx]
                            if abd > 0:
                                dead = abd * (1.0 - np.exp(-F))
                                discard_r = fishing_discard[idx]
                                if discard_r > 0:
                                    n_dead[idx, 3] += dead * (1.0 - discard_r)
                                    n_dead[idx, 6] += dead * discard_r
                                else:
                                    n_dead[idx, 3] += dead
                                inst_abd[idx] -= dead


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
    inst_abd: NDArray[np.float64] | None = None,
    step: int = 0,
    rsc_size_min: NDArray[np.float64] | None = None,
    rsc_size_max: NDArray[np.float64] | None = None,
    rsc_tl: NDArray[np.float64] | None = None,
    rsc_access_rows: NDArray[np.int32] | None = None,
    n_rsc: int = 0,
    grid_nx: int = 1,
    eff_starv: NDArray[np.float64] | None = None,
    eff_additional: NDArray[np.float64] | None = None,
    eff_fishing: NDArray[np.float64] | None = None,
    fishing_discard: NDArray[np.float64] | None = None,
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

    # Full Numba path: all 4 causes compiled (Tier 3)
    use_full_numba = (
        _HAS_NUMBA
        and inst_abd is not None
        and rsc_size_min is not None
        and eff_starv is not None
    )
    if use_full_numba:
        rsc_bio = resources.biomass if resources is not None else _DUMMY_RSC_2D
        cell_id = cell_y * grid_nx + cell_x
        tl_ws = _tl_weighted_sum if _tl_weighted_sum is not None else _DUMMY_RSC_1D
        tl_track = _tl_weighted_sum is not None
        d_mat = (
            _diet_matrix
            if _diet_tracking_enabled and _diet_matrix is not None
            else _DUMMY_DIET
        )
        d_en = _diet_tracking_enabled and _diet_matrix is not None

        # Pre-generate cause orders (must use same RNG sequence as Python path)
        cause_orders = np.zeros((n_local, 4), dtype=np.int32)
        causes = [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING]
        for i in range(n_local):
            rng.shuffle(causes)
            cause_orders[i, 0] = causes[0]
            cause_orders[i, 1] = causes[1]
            cause_orders[i, 2] = causes[2]
            cause_orders[i, 3] = causes[3]

        _mortality_in_cell_numba(
            cell_indices,
            seq_pred,
            seq_starv,
            seq_fish,
            seq_nat,
            cause_orders,
            inst_abd,
            state.n_dead,
            eff_starv,
            eff_additional,
            eff_fishing,
            fishing_discard,
            state.species_id,
            state.length,
            state.weight,
            state.age_dt,
            state.first_feeding_age_dt,
            state.feeding_stage,
            state.pred_success_rate,
            state.preyed_biomass,
            state.trophic_level,
            config.size_ratio_min,
            config.size_ratio_max,
            config.ingestion_rate,
            config.n_dt_per_year,
            n_subdt,
            access_matrix,
            has_access,
            use_stage_access,
            prey_access_idx,
            pred_access_idx,
            rsc_bio,
            rsc_size_min,
            rsc_size_max,
            rsc_tl,
            rsc_access_rows,
            n_rsc,
            config.n_species,
            cell_id,
            tl_ws,
            tl_track,
            d_mat,
            d_en,
        )
        return

    # Python fallback path
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
                    inst_abd=inst_abd,
                )
            elif cause == _STARVATION:
                s_local = seq_starv[i]
                s_idx = int(cell_indices[s_local])
                _apply_starvation_for_school(s_idx, state, config, n_subdt, inst_abd)
            elif cause == _ADDITIONAL:
                a_local = seq_nat[i]
                a_idx = int(cell_indices[a_local])
                _apply_additional_for_school(
                    a_idx, state, config, n_subdt, inst_abd, step=step
                )
            elif cause == _FISHING:
                f_local = seq_fish[i]
                f_idx = int(cell_indices[f_local])
                _apply_fishing_for_school(
                    f_idx, state, config, n_subdt, inst_abd, step=step
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
    grid: Grid,
    step: int = 0,
    species_rngs: list[np.random.Generator] | None = None,
    parallel: bool = True,
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
    # larva_mortality both reduces abundance AND records in n_dead[:, ADDITIONAL]
    state = larva_mortality(state, config)

    # Save larva deaths for output, then reset n_dead so the interleaved loop
    # doesn't double-count them (abundance is already reduced)
    larva_deaths = state.n_dead.copy()
    state = state.replace(n_dead=np.zeros_like(state.n_dead))

    # Retain eggs: withheld from prey pool
    egg_retained = np.where(state.is_egg, state.abundance, 0.0)
    state = state.replace(egg_retained=egg_retained)

    # Make working copies for in-place modification
    n_dead = state.n_dead.copy()  # now zeros (larva deaths already in abundance)
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

    # Cached instantaneous abundance: avoids recomputing abundance - n_dead.sum()
    # on every access (Tier 1 optimization). n_dead is zero here, so inst_abd
    # starts as a copy of abundance. Each _apply_* function decrements it in-place
    # when recording deaths, keeping it in sync without calling .sum().
    inst_abd = work_state.abundance.copy()

    # Pre-compute resource metadata for Numba predation (Tier 2)
    rsc_sm, rsc_sx, rsc_tl, rsc_ar, n_rsc = _precompute_resource_arrays(config, resources)

    # Pre-compute effective mortality rates for Numba cell loop (Tier 3)
    eff_s, eff_a, eff_f, f_disc = _precompute_effective_rates(
        work_state, config, n_subdt, step
    )

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
        if _HAS_NUMBA and len(valid_indices) > 0:
            # Generate a seed from Python RNG for Numba's internal PRNG
            rng_seed = int(rng.integers(0, 2**63))

            # Extract tracking arrays from module globals BEFORE Numba call
            rsc_bio = resources.biomass if resources is not None else _DUMMY_RSC_2D
            tl_ws = _tl_weighted_sum if _tl_weighted_sum is not None else _DUMMY_RSC_1D
            tl_track = _tl_weighted_sum is not None
            d_mat = (
                _diet_matrix
                if _diet_tracking_enabled and _diet_matrix is not None
                else _DUMMY_DIET
            )
            d_en = _diet_tracking_enabled and _diet_matrix is not None

            # Single Numba call for all cells (RNG generated inside)
            _batch_fn = (
                _mortality_all_cells_parallel
                if parallel
                else _mortality_all_cells_numba
            )
            _batch_fn(
                rng_seed,
                sorted_indices, boundaries, n_cells,
                inst_abd, work_state.n_dead,
                eff_s, eff_a, eff_f, f_disc,
                work_state.species_id, work_state.length, work_state.weight,
                work_state.age_dt, work_state.first_feeding_age_dt,
                work_state.feeding_stage, work_state.pred_success_rate,
                work_state.preyed_biomass, work_state.trophic_level,
                config.size_ratio_min, config.size_ratio_max, config.ingestion_rate,
                config.n_dt_per_year, n_subdt,
                access_matrix, has_access, use_stage_access,
                prey_access_idx, pred_access_idx,
                rsc_bio, rsc_sm, rsc_sx, rsc_tl, rsc_ar,
                n_rsc, config.n_species,
                tl_ws, tl_track, d_mat, d_en,
            )
        else:
            # Python fallback: per-cell dispatch (unchanged)
            for cell in range(n_cells):
                start = boundaries[cell]
                end = boundaries[cell + 1]
                if end <= start:
                    continue
                cell_indices = sorted_indices[start:end]
                cy = cell // grid.nx
                cx = cell % grid.nx
                _mortality_in_cell(
                    cell_indices, work_state, config, resources,
                    cy, cx, rng, n_subdt,
                    access_matrix, has_access, use_stage_access,
                    prey_access_idx, pred_access_idx,
                    inst_abd=inst_abd, step=step,
                    rsc_size_min=rsc_sm, rsc_size_max=rsc_sx,
                    rsc_tl=rsc_tl, rsc_access_rows=rsc_ar, n_rsc=n_rsc,
                    eff_starv=eff_s, eff_additional=eff_a,
                    eff_fishing=eff_f, fishing_discard=f_disc,
                    grid_nx=grid.nx,
                )

    # Update abundance from accumulated n_dead
    total_dead = work_state.n_dead.sum(axis=1)
    new_abundance = np.maximum(0.0, work_state.abundance - total_dead)
    new_biomass = new_abundance * work_state.weight

    # Merge larva deaths (pre-pass) back into n_dead for output tracking
    combined_n_dead = work_state.n_dead + larva_deaths

    state = state.replace(
        abundance=new_abundance,
        biomass=new_biomass,
        n_dead=combined_n_dead,
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
