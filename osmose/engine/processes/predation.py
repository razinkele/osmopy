"""Predation process.

Public API:
    predation_for_cell(cell_indices, state, config, rng, n_subdt, ...)
        Apply predation within a single cell. In-place on state.
    enable_diet_tracking / disable_diet_tracking / get_diet_matrix
        Diet-tracking helpers.
    compute_size_overlap, compute_appetite, compute_feeding_stages
        Utility predicates.

Test-exposed private helpers (leading underscore — not stable API):
    _predation_in_cell_python, _predation_in_cell_numba
    _predation_on_resources
        Used by targeted tests that need to exercise a specific backend
        or the resource-predation path in isolation. Tests that import
        these take on the maintenance burden if signatures change.

Production code uses mortality.mortality() rather than this module
directly; predation_for_cell is exposed for predation-isolated
testing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.processes.feeding_stage import compute_feeding_stages
from osmose.engine.resources import ResourceState
from osmose.engine.simulate import SimulationContext
from osmose.engine.state import SchoolState
from osmose.logging import setup_logging

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

_log = setup_logging("osmose.engine.processes.predation")

if not _HAS_NUMBA:
    _log.warning(
        "Numba is not installed. Predation will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance."
    )


# ---------------------------------------------------------------------------
# Diet tracking (context-based state)
# ---------------------------------------------------------------------------


def enable_diet_tracking(
    n_schools: int,
    n_species: int,
    ctx: SimulationContext | None = None,
) -> None:
    """Enable per-school diet tracking with a (n_schools, n_species) matrix.

    If the context already has a diet matrix with sufficient capacity,
    reuses it (zeroing active rows) to avoid repeated allocation.
    """
    if ctx is None:
        return
    ctx.diet_tracking_enabled = True
    if ctx.diet_matrix is not None and ctx.diet_matrix.shape[0] >= n_schools and ctx.diet_matrix.shape[1] == n_species:
        ctx.diet_matrix[:n_schools] = 0.0
    else:
        ctx.diet_matrix = np.zeros((n_schools, n_species), dtype=np.float64)
    ctx._diet_active_rows = n_schools


def disable_diet_tracking(ctx: SimulationContext | None = None) -> None:
    """Disable diet tracking (keeps buffer for reuse)."""
    if ctx is None:
        return
    ctx.diet_tracking_enabled = False
    ctx._diet_active_rows = 0


def get_diet_matrix(ctx: SimulationContext | None = None) -> NDArray[np.float64] | None:
    """Return the active diet matrix, or None if tracking is disabled."""
    if ctx is None or not ctx.diet_tracking_enabled:
        return None
    return ctx.diet_matrix


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

# These functions are extracted as testable utilities to validate predation logic
# in isolation (see test_engine_predation_helpers.py). The inner loops keep
# inline logic for Numba performance reasons — extracting calls into the Numba-
# accelerated loop would add overhead from repeated function dispatch.


def compute_size_overlap(
    pred_length: float,
    prey_length: float,
    ratio_min: float,
    ratio_max: float,
) -> bool:
    """Return True if the predator/prey size ratio falls within [ratio_min, ratio_max)."""
    if prey_length <= 0:
        return False
    ratio = pred_length / prey_length
    return ratio_min <= ratio < ratio_max


def compute_appetite(
    biomass: float,
    ingestion_rate: float,
    n_dt_per_year: int,
    n_subdt: int,
) -> float:
    """Return the maximum biomass a predator can eat in one sub-timestep."""
    return biomass * ingestion_rate / (n_dt_per_year * n_subdt)


if _HAS_NUMBA:
    _compute_size_overlap_numba = njit(cache=True)(compute_size_overlap)
    _compute_appetite_numba = njit(cache=True)(compute_appetite)


# ---------------------------------------------------------------------------
# Numba-accelerated inner loop
# ---------------------------------------------------------------------------

# Dummy diet matrix used when diet tracking is disabled (avoids None in Numba)
_DUMMY_DIET = np.zeros((1, 1), dtype=np.float64)

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
        n_dt_per_year: int,
        feeding_stage: NDArray[np.int32],
        prey_access_idx: NDArray[np.int32],
        pred_access_idx: NDArray[np.int32],
        use_stage_access: bool,
        diet_matrix: NDArray[np.float64],
        diet_enabled: bool,
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
            max_eatable = biomass_p * ingestion_rate[sp_pred] / (n_dt_per_year * n_subdt)
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

                access_coeff = 1.0
                if has_access:
                    if use_stage_access:
                        # Stage-indexed: use pre-computed indices
                        p_acc = pred_access_idx[p_idx]
                        q_acc = prey_access_idx[q_idx]
                        if p_acc >= 0 and q_acc >= 0:
                            if q_acc < access_matrix.shape[0] and p_acc < access_matrix.shape[1]:
                                access_coeff = access_matrix[q_acc, p_acc]
                            if access_coeff <= 0.0 or access_coeff != access_coeff:
                                continue
                        # If index is -1 (not found), keep default 1.0
                    else:
                        sp_prey = species_id[q_idx]
                        if sp_pred < access_matrix.shape[0] and sp_prey < access_matrix.shape[1]:
                            access_coeff = access_matrix[sp_pred, sp_prey]
                            if access_coeff <= 0.0 or access_coeff != access_coeff:
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

                # Diet tracking: accumulate eaten biomass per prey species
                if diet_enabled:
                    prey_sp = species_id[q_idx]
                    if p_idx < diet_matrix.shape[0] and prey_sp < diet_matrix.shape[1]:
                        diet_matrix[p_idx, prey_sp] += eaten_from_prey

            success = min(eaten_total / max_eatable, 1.0)
            pred_success_rate[p_idx] += success / n_subdt
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
    prey_access_idx: NDArray[np.int32] | None = None,
    pred_access_idx: NDArray[np.int32] | None = None,
    stage_access_matrix: NDArray[np.float64] | None = None,
    ctx: SimulationContext | None = None,
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

        max_eatable = (
            state.abundance[p_idx] * state.weight[p_idx]
            * config.ingestion_rate[sp_pred] / (config.n_dt_per_year * n_subdt)
        )
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

            access_coeff = 1.0
            if stage_access_matrix is not None and prey_access_idx is not None and pred_access_idx is not None:
                p_acc = pred_access_idx[p_idx]
                q_acc = prey_access_idx[q_idx]
                if p_acc >= 0 and q_acc >= 0:
                    if (
                        q_acc < stage_access_matrix.shape[0]
                        and p_acc < stage_access_matrix.shape[1]
                    ):
                        access_coeff = stage_access_matrix[q_acc, p_acc]
                    if access_coeff <= 0.0 or access_coeff != access_coeff:
                        continue
            elif config.accessibility_matrix is not None:
                sp_prey = state.species_id[q_idx]
                if (
                    sp_pred < config.accessibility_matrix.shape[0]
                    and sp_prey < config.accessibility_matrix.shape[1]
                ):
                    access_coeff = config.accessibility_matrix[sp_pred, sp_prey]
                    if access_coeff <= 0.0 or access_coeff != access_coeff:
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
            _diet_en = ctx.diet_tracking_enabled if ctx else False
            _diet_mat = ctx.diet_matrix if ctx else None
            if _diet_en and _diet_mat is not None:
                prey_sp = state.species_id[q_idx]
                if p_idx < _diet_mat.shape[0] and prey_sp < _diet_mat.shape[1]:
                    _diet_mat[p_idx, prey_sp] += eaten_from_prey

        success = min(eaten_total / max_eatable, 1.0)
        state.pred_success_rate[p_idx] += success / n_subdt
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
    pred_access_idx: NDArray[np.int32] | None = None,
    stage_access_matrix: NDArray[np.float64] | None = None,
    ctx: SimulationContext | None = None,
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
        max_eatable = biomass_p * config.ingestion_rate[sp_pred] / (config.n_dt_per_year * n_subdt)
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
            if r_min_val <= 0 or r_max_val <= 0:
                continue
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

            # Accessibility from matrix
            access_coeff = 1.0
            if (
                stage_access_matrix is not None
                and pred_access_idx is not None
                and config.stage_accessibility is not None
            ):
                # Stage-indexed: resolve predator column, resource row
                sa = config.stage_accessibility
                rsc_name = rsc.name
                csv_name = sa.resolve_name(rsc_name)
                if csv_name is not None:
                    rsc_row = sa.get_index(csv_name, 0.0, role="prey")
                    p_acc = pred_access_idx[p_idx]
                    if (
                        rsc_row >= 0
                        and p_acc >= 0
                        and rsc_row < stage_access_matrix.shape[0]
                        and p_acc < stage_access_matrix.shape[1]
                    ):
                        access_coeff = stage_access_matrix[rsc_row, p_acc]
                        if access_coeff <= 0.0 or access_coeff != access_coeff:
                            continue
            elif config.accessibility_matrix is not None:
                rsc_sp_idx = config.n_species + r
                if (
                    sp_pred < config.accessibility_matrix.shape[0]
                    and rsc_sp_idx < config.accessibility_matrix.shape[1]
                ):
                    access_coeff = config.accessibility_matrix[sp_pred, rsc_sp_idx]
                    if access_coeff <= 0.0 or access_coeff != access_coeff:
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
            _diet_en = ctx.diet_tracking_enabled if ctx else False
            _diet_mat = ctx.diet_matrix if ctx else None
            if _diet_en and _diet_mat is not None:
                rsc_col = config.n_species + r
                if p_idx < _diet_mat.shape[0] and rsc_col < _diet_mat.shape[1]:
                    _diet_mat[p_idx, rsc_col] += eaten_from_rsc

        # Update predator success rate
        if max_eatable > 0:
            success = min(eaten_total / max_eatable, 1.0)
            state.pred_success_rate[p_idx] += success / n_subdt
            state.preyed_biomass[p_idx] += eaten_total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Dummy accessibility matrix for Numba when none is loaded
_DUMMY_ACCESS = np.zeros((1, 1), dtype=np.float64)


def predation_for_cell(
    cell_indices: NDArray[np.int32],
    state: SchoolState,
    config: EngineConfig,
    rng: np.random.Generator,
    n_subdt: int,
    *,
    use_numba: bool = _HAS_NUMBA,
    ctx: SimulationContext | None = None,
    species_rngs: list[np.random.Generator] | None = None,
    resources: ResourceState | None = None,
    cell_y: int = 0,
    cell_x: int = 0,
) -> None:
    """Apply predation (+ optional resource predation) within a single cell.

    In-place modification of state arrays. Public contract for test harnesses
    and any caller that needs predation isolation without running the full
    mortality pipeline. Production code uses mortality.mortality() instead.

    Fields mutated in place on ``state``:
      - abundance       (absolute assignment)
      - pred_success_rate  (accumulating +=)
      - preyed_biomass     (accumulating +=)
      - feeding_stage   (fresh overwrite via [:])

    Caller owns cell_indices correctness (unique, in-range, all in cell
    (cell_y, cell_x)). No internal validation.
    """
    if len(cell_indices) < 2:
        return

    # Ensure feeding_stage is current. compute_feeding_stages returns a fresh
    # np.int32 array of length n_schools; frozen dataclass permits in-place
    # buffer mutation even though attribute reassignment is blocked.
    state.feeding_stage[:] = compute_feeding_stages(state, config)

    # Precompute accessibility info for this call — scoped to a single cell.
    # Per-call recomputation is correct because stages/access-indices are
    # invariant within a timestep (they derive from species_id, age_dt,
    # length, weight, and trophic_level — none of which predation mutates).
    if config.stage_accessibility is not None:
        sa = config.stage_accessibility
        prey_access_idx = sa.compute_school_indices(
            state.species_id,
            state.age_dt,
            config.n_dt_per_year,
            config.all_species_names,
            role="prey",
        )
        pred_access_idx = sa.compute_school_indices(
            state.species_id,
            state.age_dt,
            config.n_dt_per_year,
            config.all_species_names,
            role="pred",
        )
        access_matrix = sa.raw_matrix
        has_access = True
        use_stage_access = True
    else:
        prey_access_idx = np.zeros(len(state), dtype=np.int32)
        pred_access_idx = np.zeros(len(state), dtype=np.int32)
        has_access = config.accessibility_matrix is not None
        access_matrix = config.accessibility_matrix if has_access else _DUMMY_ACCESS
        use_stage_access = False

    cell_indices_i32 = cell_indices.astype(np.int32, copy=False)

    # Dispatch to Numba or Python backend. Silent fallback if Numba was
    # requested but is unavailable (use_numba=True without _HAS_NUMBA
    # drops into the Python path).
    if use_numba and _HAS_NUMBA:
        if species_rngs is not None and len(cell_indices_i32) > 0:
            first_pred_sp = int(state.species_id[cell_indices_i32[0]])
            _cell_rng = (
                species_rngs[first_pred_sp]
                if first_pred_sp < len(species_rngs)
                else rng
            )
        else:
            _cell_rng = rng
        pred_order = _cell_rng.permutation(len(cell_indices_i32)).astype(np.int32)
        _diet_en = ctx.diet_tracking_enabled if ctx else False
        _diet_mat = ctx.diet_matrix if ctx else None
        diet_mat = _diet_mat if _diet_en and _diet_mat is not None else _DUMMY_DIET
        if access_matrix is None:
            access_matrix = _DUMMY_ACCESS
        _predation_in_cell_numba(
            cell_indices_i32,
            pred_order,
            state.abundance,
            state.length,
            state.weight,
            state.age_dt,
            state.first_feeding_age_dt,
            state.species_id,
            state.pred_success_rate,
            state.preyed_biomass,
            config.size_ratio_min,
            config.size_ratio_max,
            config.ingestion_rate,
            access_matrix,
            has_access,
            n_subdt,
            config.n_dt_per_year,
            state.feeding_stage,
            prey_access_idx,
            pred_access_idx,
            use_stage_access,
            diet_mat,
            _diet_en,
        )
    else:
        _predation_in_cell_python(
            cell_indices_i32,
            state,
            config,
            rng,
            n_subdt,
            prey_access_idx=prey_access_idx if use_stage_access else None,
            pred_access_idx=pred_access_idx if use_stage_access else None,
            stage_access_matrix=access_matrix if use_stage_access else None,
            ctx=ctx,
        )

    # Resource predation: focal schools eat LTL plankton/detritus.
    if resources is not None and resources.n_resources > 0:
        _predation_on_resources(
            cell_indices_i32,
            state,
            config,
            resources,
            cell_y,
            cell_x,
            rng,
            n_subdt,
            pred_access_idx=pred_access_idx if use_stage_access else None,
            stage_access_matrix=access_matrix if use_stage_access else None,
            ctx=ctx,
        )


