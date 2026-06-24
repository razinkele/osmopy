"""Numba @njit driver for the predation-kernel benchmark (Task 6).

make_driver() -> (driver_fn, reset_only_fn)

Both functions have an EXPLICIT positional signature:
  the 41 leaf params (in LEAF_ARG_ORDER), then n_iter (int64),
  then 7 pristine snapshots for the MUTATED arrays
  (snap_inst_abd, snap_n_dead, snap_pred_success_rate, snap_preyed_biomass,
   snap_rsc_biomass, snap_tl_weighted_sum, snap_diet_matrix).

Total = 41 + 1 + 7 = 49 parameters.

driver_fn: per-iteration reset from snapshots + leaf call.
reset_only_fn: per-iteration reset only, no leaf call.

MUTATED arrays (indices in LEAF_ARG_ORDER, also passed as snapshot args):
  idx 2  inst_abd         (1D float64)
  idx 3  n_dead           (2D float64)
  idx 10 pred_success_rate(1D float64)
  idx 11 preyed_biomass   (1D float64)
  idx 25 rsc_biomass      (2D float64)
  idx 33 tl_weighted_sum  (1D float64)
  idx 35 diet_matrix      (2D float64)
"""
from __future__ import annotations

from numba import njit  # type: ignore[import-untyped]

from osmose.engine.processes import mortality as _mortality_mod

# Grab the already-compiled @njit leaf.
_apply_predation_numba = _mortality_mod._apply_predation_numba


@njit
def _driver_with_leaf(
    # --- 41 leaf args in LEAF_ARG_ORDER ---
    p_idx, cell_indices,
    inst_abd, n_dead,
    species_id, length, weight,
    age_dt, first_feeding_age_dt, feeding_stage,
    pred_success_rate, preyed_biomass, trophic_level,
    size_ratio_min, size_ratio_max,
    ingestion_rate, fr_shape, fr_halfsat,
    n_dt_per_year, n_subdt,
    access_matrix,
    has_access, use_stage_access,
    prey_access_idx, pred_access_idx,
    rsc_biomass,
    rsc_size_min, rsc_size_max, rsc_tl, rsc_access_rows,
    n_resources, n_species, cell_id,
    tl_weighted_sum, tl_tracking,
    diet_matrix, diet_enabled,
    prey_type_buf, prey_id_buf, prey_eligible_buf,
    egg_retained,
    # --- bench args ---
    n_iter,
    # --- 7 pristine snapshots (in MUTATED order) ---
    snap_inst_abd, snap_n_dead,
    snap_pred_success_rate, snap_preyed_biomass,
    snap_rsc_biomass,
    snap_tl_weighted_sum,
    snap_diet_matrix,
):
    """Loop n_iter times: reset 7 mutated arrays from snapshots, then call leaf."""
    for _ in range(n_iter):
        inst_abd[:] = snap_inst_abd
        n_dead[:, :] = snap_n_dead
        pred_success_rate[:] = snap_pred_success_rate
        preyed_biomass[:] = snap_preyed_biomass
        rsc_biomass[:, :] = snap_rsc_biomass
        tl_weighted_sum[:] = snap_tl_weighted_sum
        diet_matrix[:, :] = snap_diet_matrix

        _apply_predation_numba(
            p_idx, cell_indices,
            inst_abd, n_dead,
            species_id, length, weight,
            age_dt, first_feeding_age_dt, feeding_stage,
            pred_success_rate, preyed_biomass, trophic_level,
            size_ratio_min, size_ratio_max,
            ingestion_rate, fr_shape, fr_halfsat,
            n_dt_per_year, n_subdt,
            access_matrix,
            has_access, use_stage_access,
            prey_access_idx, pred_access_idx,
            rsc_biomass,
            rsc_size_min, rsc_size_max, rsc_tl, rsc_access_rows,
            n_resources, n_species, cell_id,
            tl_weighted_sum, tl_tracking,
            diet_matrix, diet_enabled,
            prey_type_buf, prey_id_buf, prey_eligible_buf,
            egg_retained,
        )


@njit
def _reset_only(
    # --- 41 leaf args in LEAF_ARG_ORDER ---
    p_idx, cell_indices,
    inst_abd, n_dead,
    species_id, length, weight,
    age_dt, first_feeding_age_dt, feeding_stage,
    pred_success_rate, preyed_biomass, trophic_level,
    size_ratio_min, size_ratio_max,
    ingestion_rate, fr_shape, fr_halfsat,
    n_dt_per_year, n_subdt,
    access_matrix,
    has_access, use_stage_access,
    prey_access_idx, pred_access_idx,
    rsc_biomass,
    rsc_size_min, rsc_size_max, rsc_tl, rsc_access_rows,
    n_resources, n_species, cell_id,
    tl_weighted_sum, tl_tracking,
    diet_matrix, diet_enabled,
    prey_type_buf, prey_id_buf, prey_eligible_buf,
    egg_retained,
    # --- bench args ---
    n_iter,
    # --- 7 pristine snapshots (in MUTATED order) ---
    snap_inst_abd, snap_n_dead,
    snap_pred_success_rate, snap_preyed_biomass,
    snap_rsc_biomass,
    snap_tl_weighted_sum,
    snap_diet_matrix,
):
    """Loop n_iter times: reset 7 mutated arrays from snapshots only (no leaf call)."""
    for _ in range(n_iter):
        inst_abd[:] = snap_inst_abd
        n_dead[:, :] = snap_n_dead
        pred_success_rate[:] = snap_pred_success_rate
        preyed_biomass[:] = snap_preyed_biomass
        rsc_biomass[:, :] = snap_rsc_biomass
        tl_weighted_sum[:] = snap_tl_weighted_sum
        diet_matrix[:, :] = snap_diet_matrix


def make_driver():
    """Return the (driver_with_leaf, reset_only) @njit function pair.

    Both functions share the same explicit 49-param signature.
    Call once to trigger JIT compilation; time with separate warm calls.
    """
    return _driver_with_leaf, _reset_only
