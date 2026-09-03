"""Mortality orchestrator for the OSMOSE Python engine.

Implements Java's per-cell per-school interleaved mortality loop:
for each sub-timestep, for each cell, all four mortality causes
(predation, starvation, fishing, additional+out-of-domain) are applied
in a shuffled interleaved sequence — schools and causes are both shuffled
so that each cause is applied to one school at a time, updating n_dead
in-place so subsequent causes see the reduced instantaneous abundance.
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
from osmose.engine.simulate import SimulationContext
from osmose.engine.processes.starvation import (
    starvation_mortality,  # noqa: F401 — used by tests
    update_starvation_rate,
)
from osmose.engine.resources import ResourceState
from osmose.engine.state import MortalityCause, SchoolState
from osmose.logging import setup_logging

try:
    from numba import njit, prange  # type: ignore[import-not-found]

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

_log = setup_logging("osmose.engine.processes.mortality")

if not _HAS_NUMBA:
    _log.warning(
        "Numba is not installed. Mortality will use pure Python fallback, "
        "which may be 10-100x slower. Install numba for optimal performance."
    )

# Cause indices matching MortalityCause enum
_PREDATION = int(MortalityCause.PREDATION)
_STARVATION = int(MortalityCause.STARVATION)
_ADDITIONAL = int(MortalityCause.ADDITIONAL)
_FISHING = int(MortalityCause.FISHING)
_DISCARDS = int(MortalityCause.DISCARDS)
_FORAGING = int(MortalityCause.FORAGING)


def _get_mortality_causes(config: EngineConfig) -> list[int]:
    """Get the list of mortality causes for the interleaved loop.

    Without bioen: [PREDATION, STARVATION, ADDITIONAL, FISHING]
    With bioen:    [PREDATION, STARVATION, ADDITIONAL, FISHING, FORAGING]

    Bioen starvation runs INSIDE the interleaved loop, competing with the other
    causes, and consumes the PREVIOUS step's `e_net` — Java's step order is
    mortality -> EnergyBudget -> reproduction (`SimulationStep.java:190-198`), so
    when `BioenStarvationMortality.computeStarvation` reads `school.getENet()` the
    only value there is last step's. `_bioen_step` therefore no longer applies
    starvation at all; it would otherwise be applied twice.

    Java builds the set as "all MortalityCause values minus DISCARDS and AGING,
    minus FORAGING when bioen is off" (`MortalityProcess.java:506-517`). OUT is
    handled post-loop by `out_mortality`, matching Java's own out-school pass.
    """
    if config.bioen_enabled:
        return [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING, _FORAGING]
    return [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING]


def _consume(
    state: SchoolState,
    idx: int,
    n_dead: float,
    inst_abd: NDArray[np.float64],
    bioen: bool,
) -> None:
    """Decrement instantaneous abundance and apply Java's survivor rescaling.

    Java's `School.setNdead`/`incrementNdead` (`School.java:371-402`) multiply the
    school's accumulated `ingestion` and its stored `e_net` by
    `(N_inst - nDead) / N_inst` at EVERY death, so a school that eats in sub-step 1
    and then loses half its fish carries only half that ingestion into the energy
    budget. `AbstractSchool.incrementNdead` — the background-school override — does
    NOT rescale (background schools have no bioen budget), hence the
    `is_background` guard.

    The denominator is the PRE-death instantaneous abundance. Java reaches the same
    value indirectly: every cause reads `getInstantaneousAbundance()` to size its
    own `nDead`, which clears `abundanceHasChanged`, so the read inside
    `incrementNdead` returns the cached pre-death number. (The bioen STARVATION
    branch does not read it first, so Java can there hit a stale-flag double
    subtraction — order-dependent and plainly unintended; not replicated.)

    The `max(..., 0.0)` clamp is a deliberate departure: Java has no clamp, so when
    `nDead > instantaneousAbundance` its factor goes NEGATIVE and it multiplies
    `ingestion`/`e_net` by it (`School.java:394-399`). That is reachable on the bioen
    STARVATION path, where the toll is `deficit / weight` and nothing bounds it by the
    abundance. Clamping to [0, 1] keeps the rescaling a survivor fraction.

    For `bioen=False` this is exactly the `inst_abd[idx] -= n_dead` it replaced.
    """
    before = inst_abd[idx]
    inst_abd[idx] = before - n_dead
    if bioen and before > 0.0 and not state.is_background[idx]:
        factor = max(inst_abd[idx], 0.0) / before
        state.preyed_biomass[idx] *= factor
        state.e_net[idx] *= factor


def _kill(
    state: SchoolState,
    idx: int,
    cause: int,
    n_dead: float,
    inst_abd: NDArray[np.float64],
    bioen: bool,
) -> None:
    """Record `n_dead` deaths of one school for `cause` and consume the abundance.

    Under bioen this also rescales the school's ingestion and stored E_net — see
    `_consume`. Callers that must split one death event across two causes (fishing
    vs discards) record the split themselves and call `_consume` once with the FULL
    count, so the abundance decrement stays a single subtraction.
    """
    if n_dead <= 0.0:
        return
    state.n_dead[idx, cause] += n_dead
    _consume(state, idx, n_dead, inst_abd, bioen)


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
    """Apply starvation mortality to a single school (in-place on n_dead).

    When bioen_enabled=True: BioenStarvationMortality.computeStarvation for ONE
    sub-step, on the PREVIOUS step's `e_net` (Java step order mortality ->
    EnergyBudget -> reproduction), with strict `ageDt > firstFeedingAgeDt`
    eligibility (`Species.isStarvationEnabledBioen`).
    When bioen_enabled=False: standard starvation rate, eligibility `>=`
    (`Species.isStarvationEnabledNoBioen`).
    Matches Java MortalityProcess lines 604-626.
    """
    if state.is_background[idx]:
        return

    if config.bioen_enabled:
        # Java Species.isStarvationEnabledBioen: ageDt > firstFeedingAgeDt (strict).
        if state.age_dt[idx] <= state.first_feeding_age_dt[idx]:
            return
        if inst_abd[idx] <= 0:
            return

        from osmose.engine.processes.bioen_starvation import bioen_starvation_substep

        sp_i = state.species_id[idx]
        eta = float(config.bioen_eta[sp_i]) if config.bioen_eta is not None else 1.0
        n_dead, new_gonad, new_enet = bioen_starvation_substep(
            float(state.e_net[idx]),
            float(state.gonad_weight[idx]),
            float(state.weight[idx]),
            eta,
            n_subdt,
        )
        state.gonad_weight[idx] = new_gonad
        state.e_net[idx] = new_enet
        _kill(state, idx, _STARVATION, n_dead, inst_abd, bioen=True)
        return

    if state.age_dt[idx] < state.first_feeding_age_dt[idx]:
        return
    abd = inst_abd[idx]
    if abd <= 0:
        return

    # Standard starvation
    M = state.starvation_rate[idx] / (config.n_dt_per_year * n_subdt)
    if M <= 0:
        return
    n_dead = abd * (1.0 - np.exp(-M))
    _kill(state, idx, _STARVATION, n_dead, inst_abd, bioen=False)


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
    if state.age_dt[idx] < state.first_feeding_age_dt[idx]:
        return
    sp = state.species_id[idx]

    # Base rate: constant or time-varying (BY_DT)
    rate = config.additional_mortality_rate[sp]
    if (
        config.additional_mortality_by_dt is not None
        and config.additional_mortality_by_dt[sp] is not None
    ):
        arr = config.additional_mortality_by_dt[sp]
        rate = arr[step % len(arr)]

    # Spatial multiplier
    if (
        config.additional_mortality_spatial is not None
        and config.additional_mortality_spatial[sp] is not None
    ):
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
    _kill(state, idx, _ADDITIONAL, n_dead, inst_abd, config.bioen_enabled)


def _apply_fishing_for_school(
    idx: int,
    state: SchoolState,
    config: EngineConfig,
    n_subdt: int,
    inst_abd: NDArray[np.float64],
    step: int = 0,
    fleet_state=None,
) -> None:
    """Apply fishing mortality to a single school (in-place on n_dead)."""
    if state.is_background[idx]:
        return
    if state.age_dt[idx] < state.first_feeding_age_dt[idx]:
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

    # Fleet-effort (DSVM economics) — parity with the Numba path's
    # _precompute_effective_rates. 1.0 when no fleet_state / species not targeted.
    f_rate = f_rate * _fleet_effort_factor(
        sp, int(state.cell_y[idx]), int(state.cell_x[idx]), fleet_state
    )

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

    # Discards split. Java issues TWO incrementNdead calls — FISHING with nFished
    # then DISCARDS with nDiscarded (`MortalityProcess.java:673-674`) — whose survivor
    # factors compose to ((I - nFished) / I) * ((I - nFished - nDiscarded) / (I - nFished))
    # = (I - nDead) / I. So recording the split here and consuming the FULL n_dead once
    # is Java-exact for bioen AND leaves the bioen-off arithmetic a single subtraction,
    # exactly as before.
    if config.fishing_discard_rate is not None:
        discard_r = config.fishing_discard_rate[sp]
        state.n_dead[idx, _FISHING] += n_dead * (1.0 - discard_r)
        state.n_dead[idx, _DISCARDS] += n_dead * discard_r
    else:
        state.n_dead[idx, _FISHING] += n_dead
    _consume(state, idx, n_dead, inst_abd, config.bioen_enabled)


def _apply_foraging_for_school(
    idx: int,
    state: SchoolState,
    config: EngineConfig,
    n_subdt: int,
    inst_abd: NDArray[np.float64],
) -> None:
    """Apply foraging mortality to a single school (bioen only, in-place)."""
    if state.is_background[idx]:
        return
    if state.age_dt[idx] < state.first_feeding_age_dt[idx]:
        return

    abd = inst_abd[idx]
    if abd <= 0:
        return

    from osmose.engine.processes.foraging_mortality import foraging_rate

    sp_i = state.species_id[idx]

    # Java checks isGeneticEnabled() to decide mode
    genetic = (
        config.foraging_k1_for is not None
        and config.foraging_k2_for is not None
        and config.foraging_I_max is not None
        and state.imax_trait is not None
    )
    if genetic:
        # Narrow Optional fields to concrete arrays for the type-checker; the
        # `genetic` guard above already established all four are non-None.
        k1_arr = config.foraging_k1_for
        k2_arr = config.foraging_k2_for
        i_max_arr = config.foraging_I_max
        imax_trait = state.imax_trait
        assert (
            k1_arr is not None
            and k2_arr is not None
            and i_max_arr is not None
            and imax_trait is not None
        )
        rate = foraging_rate(
            k_for=None,
            ndt_per_year=config.n_dt_per_year,
            k1_for=np.array([k1_arr[sp_i]]),
            k2_for=np.array([k2_arr[sp_i]]),
            imax_trait=np.array([imax_trait[idx]]),
            I_max=np.array([i_max_arr[sp_i]]),
        )
    else:
        k_for = (
            np.array([config.bioen_k_for[sp_i]])
            if config.bioen_k_for is not None
            else np.array([0.0])
        )
        rate = foraging_rate(k_for=k_for, ndt_per_year=config.n_dt_per_year)

    M = float(rate[0]) / n_subdt
    if M <= 0:
        return
    n_dead = abd * (1.0 - np.exp(-M))
    _kill(state, idx, _FORAGING, n_dead, inst_abd, config.bioen_enabled)


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
    ctx: SimulationContext | None = None,
    cap_fish: NDArray[np.float64] | None = None,
    raw_preyed: NDArray[np.float64] | None = None,
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

    if inst_abd is None:
        raise RuntimeError("inst_abd must not be None in _apply_predation_for_school")
    inst_abd_p = inst_abd[p_idx]
    if inst_abd_p <= 0:
        return

    sp_pred = state.species_id[p_idx]
    pred_len = state.length[p_idx]
    stage = state.feeding_stage[p_idx]
    r_min = config.size_ratio_min[sp_pred, stage]
    r_max = config.size_ratio_max[sp_pred, stage]

    if cap_fish is None:
        biomass_p = inst_abd_p * state.weight[p_idx]
        max_eatable = biomass_p * config.ingestion_rate[sp_pred] / (config.n_dt_per_year * n_subdt)
    else:
        # Java BioenPredationMortality: the per-fish allometric cap is multiplied by the
        # INSTANTANEOUS abundance at every predator visit (`:140-145`).
        max_eatable = cap_fish[p_idx] * inst_abd_p
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
        inst_abd_q = inst_abd[q_idx] - state.egg_retained[q_idx]
        if inst_abd_q < 0.0:
            inst_abd_q = 0.0
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

            if r_min <= 0 or r_max <= 0:
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
    if config.fr_shape[sp_pred] == 1:
        eaten_total = min(total_available, max_eatable)  # verbatim type-I (bit-exact)
    else:
        r = total_available / max_eatable
        k_fr = config.fr_halfsat[sp_pred]
        if config.fr_shape[sp_pred] == 2:
            g_form = r / (r + k_fr)
        else:  # type-III
            g_form = (r * r) / (r * r + k_fr * k_fr)
        cap = r if r < 1.0 else 1.0  # min(r, 1)
        g = g_form if g_form < cap else cap  # conservation clamp
        eaten_total = max_eatable * g

    # cell_id is only meaningful when resources are present (used inside the
    # `if resources is not None:` branch below). When resources is None we
    # assign a sentinel to satisfy the variable-always-defined rule, but the
    # value is never read. Deep review v3 M-6.
    if resources is not None:
        cell_id = cell_y * resources.grid.nx + cell_x
    else:
        cell_id = -1

    for prey_type, prey_id, eligible in all_prey:
        share = eligible / total_available
        eaten_from_prey = eaten_total * share

        if prey_type == "school":
            q_idx = prey_id
            if state.weight[q_idx] > 0:
                n_dead_prey = eaten_from_prey / state.weight[q_idx]
                _kill(state, q_idx, _PREDATION, n_dead_prey, inst_abd, config.bioen_enabled)

            _tl_ws = ctx.tl_weighted_sum if ctx else None
            if _tl_ws is not None:
                prey_tl = state.trophic_level[q_idx]
                if prey_tl <= 0:
                    prey_tl = 1.0
                _tl_ws[p_idx] += prey_tl * eaten_from_prey

            _diet_en = ctx.diet_tracking_enabled if ctx else False
            _d_mat = ctx.diet_matrix if ctx else None
            if _diet_en and _d_mat is not None:
                prey_sp = state.species_id[q_idx]
                if p_idx < _d_mat.shape[0] and prey_sp < _d_mat.shape[1]:
                    _d_mat[p_idx, prey_sp] += eaten_from_prey
        else:
            r = prey_id
            if resources is not None:
                resources.biomass[r, cell_id] = max(
                    0.0, resources.biomass[r, cell_id] - eaten_from_prey
                )

            _tl_ws = ctx.tl_weighted_sum if ctx else None
            if _tl_ws is not None:
                rsc_tl = resources.species[r].trophic_level if resources else 1.0
                if rsc_tl <= 0:
                    rsc_tl = 1.0
                _tl_ws[p_idx] += rsc_tl * eaten_from_prey

            _diet_en = ctx.diet_tracking_enabled if ctx else False
            _d_mat = ctx.diet_matrix if ctx else None
            if _diet_en and _d_mat is not None and resources:
                rsc_col = config.n_species + r
                if p_idx < _d_mat.shape[0] and rsc_col < _d_mat.shape[1]:
                    _d_mat[p_idx, rsc_col] += eaten_from_prey

    # --- Phase 3: Update predation success rate ONCE (matching Java) ---
    success = min(eaten_total / max_eatable, 1.0)
    state.pred_success_rate[p_idx] += success / n_subdt
    state.preyed_biomass[p_idx] += eaten_total
    if raw_preyed is not None:
        # Java keeps TWO accumulators: `ingestion` (rescaled by the survivor fraction at
        # every death, feeds the energy budget) and `preyedBiomass` (raw, feeds the
        # trophic-level update at MortalityProcess:396-401 and the diet outputs). The port
        # conflates them in `preyed_biomass`; under bioen this array carries the raw total
        # so the TL denominator stays Java's.
        raw_preyed[p_idx] += eaten_total


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
    config: EngineConfig,
) -> tuple[list[NDArray[np.int32]], NDArray[np.int32]]:
    """Pre-generate all random data for all cells in one Python pass.

    Mirrors ``_mortality_in_cell``'s per-cell RNG consumption exactly
    (mortality.py's ``seq_pred, seq_starv, seq_fish, seq_nat, seq_for`` draw,
    followed by ``n_local`` calls to ``rng.shuffle`` on a fresh
    ``_get_mortality_causes(config)`` list): FIVE permutations are drawn per cell
    UNCONDITIONALLY -- even without bioen, where the fifth (``seq_for``) goes
    unused by the caller, because the reference draws it unconditionally too and a
    stream that skips it would desync from step one. The cause-order buffer's
    width instead follows ``_get_mortality_causes(config)``: 4 without bioen, 5
    (FORAGING included) with it, matching Java's own cause set
    (``MortalityProcess.java:506-517``).

    This "pre-generate everything up front" pattern is only valid because NOTHING drawn
    from ``rng`` inside the reference's cause-application step is left unaccounted for:
    ``_apply_predation_for_school`` receives an ``rng`` parameter but never reads it
    (predation is distributed deterministically, proportional to accessible biomass), and
    ``_apply_starvation_for_school``/``_apply_additional_for_school``/
    ``_apply_fishing_for_school``/``_apply_foraging_for_school`` do not take ``rng`` at
    all. So the per-cell stream this function reproduces -- 5 permutations then
    ``n_local`` shuffles -- is the reference's ENTIRE per-cell RNG consumption, for both
    the Numba-dispatch branch (which pre-draws for the same reason) and the Python
    fallback branch (verified directly against the fallback loop's cause-application
    calls, not inferred from the dispatch branch alone).

    Note: Not used by the batch Numba path (which generates RNG inline).
    Retained as a tested reference implementation of the per-cell RNG stream, and
    for potential future use.

    Returns:
        seq_bufs: list of 5 int32 arrays, each of length boundaries[n_cells], in
            DRAW order -- ``[seq_pred, seq_starv, seq_fish, seq_nat, seq_for]``,
            i.e. NOT cause-code order (the reference's third draw is FISHING, its
            fourth ADDITIONAL). seq_bufs[k][start:end] is rng.permutation(n_local)
            for cell k.
        cause_orders_buf: int32 array of shape (total, len(_get_mortality_causes(config))).
            cause_orders_buf[start+i] is one shuffled cause order (cause codes from
            the ``MortalityCause`` enum), for school slot i.
    """
    total = int(boundaries[n_cells])
    seq_bufs = [np.empty(total, dtype=np.int32) for _ in range(5)]
    n_causes = len(_get_mortality_causes(config))
    cause_orders_buf = np.empty((total, n_causes), dtype=np.int32)

    for cell in range(n_cells):
        start = int(boundaries[cell])
        end = int(boundaries[cell + 1])
        n_local = end - start
        if n_local == 0:
            continue
        for k in range(5):
            seq_bufs[k][start:end] = rng.permutation(n_local).astype(np.int32)
        causes = _get_mortality_causes(config)
        for i in range(n_local):
            rng.shuffle(causes)
            cause_orders_buf[start + i, :] = causes

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


def _lookup_spatial_factors(
    sp_map: NDArray[np.float64],
    cell_y: NDArray[np.int32],
    cell_x: NDArray[np.int32],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Look up spatial map factors for a species mask, with bounds/NaN safety.

    Returns float64 array of length mask.sum(). Out-of-bounds or NaN/negative
    cells get factor 0.0.
    """
    cy = cell_y[mask]
    cx = cell_x[mask]
    valid = (cy >= 0) & (cy < sp_map.shape[0]) & (cx >= 0) & (cx < sp_map.shape[1])
    factors = np.zeros(mask.sum(), dtype=np.float64)
    if valid.any():
        f_vals = sp_map[cy[valid], cx[valid]]
        f_vals = np.where(np.isnan(f_vals) | (f_vals <= 0), 0.0, f_vals)
        factors[valid] = f_vals
    return factors


def _zero_exempt(arr: NDArray[np.float64], work_state) -> None:
    """Zero out rates for background species, pre-feeding schools, and negatives."""
    arr[work_state.is_background] = 0.0
    arr[work_state.age_dt < work_state.first_feeding_age_dt] = 0.0
    arr[arr < 0] = 0.0


def _fleet_effort_factor(sp_id, cell_y, cell_x, fleet_state) -> float:
    """Multiplicative fishing-effort factor for a school. Returns 1.0 when
    fleet_state is None OR sp_id is not targeted by any fleet (base F unchanged).
    For a TARGETED species: the sum of effort_map across fleets at (cell_y,
    cell_x), or 0.0 if that cell is out of bounds. Mirrors mortality.py:786-802."""
    if fleet_state is None:
        return 1.0
    targeted: set[int] = set()
    for f in fleet_state.fleets:
        targeted.update(f.target_species)
    if int(sp_id) not in targeted:
        return 1.0
    ny, nx = fleet_state.effort_map.shape[1], fleet_state.effort_map.shape[2]
    if 0 <= cell_y < ny and 0 <= cell_x < nx:
        return float(fleet_state.effort_map[:, cell_y, cell_x].sum())
    return 0.0


def _precompute_effective_rates(work_state, config, n_subdt, step, fleet_state=None):
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
    _zero_exempt(eff_starv, work_state)
    if config.bioen_enabled:
        # Bioen starvation is the gonad-depletion formula, applied inside the
        # interleaved loop (Python: _apply_starvation_for_school; Numba:
        # _apply_single_cause's cause==1 bioen branch). The standard (pred-success)
        # rate must never be applied on top of it. Since bioen-Numba-kernel plan Task 3
        # flipped the dispatch, the batched kernels DO run under bioen and read
        # eff_starv directly — this is no longer defence in depth, it is the ONLY
        # guard against double-counting: `_apply_single_cause`'s bioen branch always
        # returns before reaching the `D = eff_starv[idx]` tail, so this line is what
        # keeps that tail dead code under bioen rather than a live second application.
        eff_starv[:] = 0.0

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
                rates[mask] *= _lookup_spatial_factors(
                    sp_map, work_state.cell_y, work_state.cell_x, mask
                )

    _zero_exempt(rates, work_state)
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
                selectivity[mask] = 1.0 / (1.0 + np.exp(-slope * (work_state.length[mask] - l50)))
            else:  # length cutoff
                l50 = config.fishing_selectivity_l50[sp_id]
                selectivity[mask] = np.where((l50 > 0) & (work_state.length[mask] < l50), 0.0, 1.0)

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
            spatial_factor[mask] = _lookup_spatial_factors(
                sp_map, work_state.cell_y, work_state.cell_x, mask
            )

        mpa_factor = np.ones(n, dtype=np.float64)
        if config.mpa_zones is not None:
            year = step // config.n_dt_per_year
            for mpa in config.mpa_zones:
                if not (mpa.start_year <= year < mpa.end_year):
                    continue
                cy = work_state.cell_y
                cx = work_state.cell_x
                valid = (cy >= 0) & (cy < mpa.grid.shape[0]) & (cx >= 0) & (cx < mpa.grid.shape[1])
                in_mpa = np.zeros(n, dtype=np.bool_)
                in_mpa[valid] = mpa.grid[cy[valid], cx[valid]] > 0
                mpa_factor *= np.where(in_mpa, 1.0 - mpa.percentage, 1.0)

        # Combine — denominator differs based on seasonality
        if config.fishing_seasonality is not None:
            step_in_year = step % config.n_dt_per_year
            season = config.fishing_seasonality[sp, step_in_year]
            eff_fishing = f_rates * selectivity * spatial_factor * mpa_factor * season / n_subdt
        else:
            eff_fishing = f_rates * selectivity * spatial_factor * mpa_factor / denom

        _zero_exempt(eff_fishing, work_state)
        np.nan_to_num(eff_fishing, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if config.fishing_discard_rate is not None:
            fishing_discard = np.where(eff_fishing > 0, config.fishing_discard_rate[sp], 0.0)

        # Scale fishing by fleet effort when economic module is active
        if fleet_state is not None:
            for i in range(n):
                eff_fishing[i] *= _fleet_effort_factor(
                    work_state.species_id[i],
                    work_state.cell_y[i],
                    work_state.cell_x[i],
                    fleet_state,
                )

    return eff_starv, eff_additional, eff_fishing, fishing_discard


def _precompute_foraging_rates(work_state, config, n_subdt) -> NDArray[np.float64]:
    """Per-school FORAGING sub-step rate for the Numba kernels (bioen behaviour 4).

    This is `_apply_foraging_for_school`'s `M`, precomputed the way `eff_additional` /
    `eff_fishing` already are, so the kernel only has to evaluate `1 - exp(-D)`.

    Mirrors the reference exactly:

    * the division ORDER is `foraging_rate`'s `k / ndt_per_year` and then `/ n_subdt` —
      two separate divisions. Folding them into `/(ndt * n_subdt)` is a different
      floating-point expression and would break bit-exact agreement with the reference.
    * `foraging_rate`'s `np.maximum(rate, 0.0)` clamp, and `_zero_exempt`, together
      reproduce the reference's three ways of killing nobody: `is_background`,
      `age_dt < first_feeding_age_dt`, and `M <= 0`.
    * the genetic variant (`k1_for * exp(k2_for * (imax_trait - I_max))`) is evaluated
      HERE, in NumPy, rather than inlined in the kernel. That keeps the exponential on
      NumPy's implementation for BOTH paths (Numba lowers `np.exp` to libm — see the
      kernel gate's EXP LIBRARY HAZARD), and it means the genetic branch is *supported*
      rather than dropped or shunted back to Python. The four-way predicate is copied
      verbatim from `_apply_foraging_for_school`, `config.foraging_I_max` included (which
      is a different array from `config.bioen_i_max_all`).

    Background schools carry `species_id >= n_species`, so every per-species lookup is
    made through a clamped index and then zeroed by `_zero_exempt`; the reference reaches
    the same place by returning before it reads the species array at all.

    Returns all-zeros when bioen is off, so the kernels can take a real `float64[:]`
    unconditionally — Numba needs stable argument types and must never be handed `None`.
    """
    n = len(work_state)
    if not config.bioen_enabled:
        return np.zeros(n, dtype=np.float64)

    from osmose.engine.processes.foraging_mortality import foraging_rate

    sp = work_state.species_id
    sp_f = np.where(sp < config.n_species, sp, 0)
    genetic = (
        config.foraging_k1_for is not None
        and config.foraging_k2_for is not None
        and config.foraging_I_max is not None
        and work_state.imax_trait is not None
    )
    if genetic:
        rate = foraging_rate(
            k_for=None,
            ndt_per_year=config.n_dt_per_year,
            k1_for=config.foraging_k1_for[sp_f],
            k2_for=config.foraging_k2_for[sp_f],
            imax_trait=work_state.imax_trait,
            I_max=config.foraging_I_max[sp_f],
        )
    else:
        k_for = (
            config.bioen_k_for[sp_f]
            if config.bioen_k_for is not None
            else np.zeros(n, dtype=np.float64)
        )
        rate = foraging_rate(k_for=k_for, ndt_per_year=config.n_dt_per_year)

    eff_foraging = rate / n_subdt
    _zero_exempt(eff_foraging, work_state)
    return eff_foraging


def _precompute_bioen_eta(work_state, config) -> NDArray[np.float64]:
    """Per-school `eta` (`species.maturity.eta.sp{i}`) for the kernels' bioen starvation.

    Per SCHOOL rather than per species so the kernel never indexes an `n_species`-long
    array with a background school's `species_id` (which is `>= n_species`). Background
    schools are skipped by the starvation branch anyway and get 1.0 here, which is also
    the reference's fallback when `config.bioen_eta is None`
    (`_apply_starvation_for_school`).
    """
    n = len(work_state)
    eta = np.ones(n, dtype=np.float64)
    if config.bioen_eta is None:
        return eta
    sp = work_state.species_id
    focal = sp < config.n_species
    eta[focal] = config.bioen_eta[sp[focal]]
    return eta


if _HAS_NUMBA:

    @njit(cache=True)
    def _consume_numba(idx, dead, inst_abd, preyed_biomass, e_net, is_background, bioen):
        """Numba twin of `_consume` — Java's survivor rescale, at EVERY death site.

        `School.incrementNdead` has no switch on cause, so all five causes rescale the
        school's accumulated ingestion and its stored `E_net` by the survivor fraction.
        The four properties of `_consume` are carried verbatim:

        * the denominator is the PRE-death instantaneous abundance;
        * `before > 0.0` guards the 0/0 that would produce NaN;
        * `max(inst_abd[idx], 0.0)` clamps the factor into [0, 1] — Java has no clamp and
          its factor goes negative when `nDead > instantaneousAbundance`, which is
          reachable on the bioen STARVATION path where the toll is `deficit / weight`;
        * background schools are skipped (`AbstractSchool.incrementNdead` does not
          rescale — background schools have no bioen budget).

        With `bioen` false this is exactly the `inst_abd[idx] -= dead` it replaced, so the
        bioen-OFF arithmetic (and `tests/test_engine_parity.py`) is untouched.
        """
        before = inst_abd[idx]
        inst_abd[idx] = before - dead
        if bioen and before > 0.0 and not is_background[idx]:
            factor = max(inst_abd[idx], 0.0) / before
            preyed_biomass[idx] *= factor
            e_net[idx] *= factor

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
        fr_shape,
        fr_halfsat,
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
        prey_type_buf,
        prey_id_buf,
        prey_eligible_buf,
        egg_retained,
        bioen,
        cap_fish,
        raw_preyed,
        e_net,
        is_background,
    ):
        """Numba-compiled single-predator predation (schools + resources).

        The five trailing parameters are the bioen extension (plan behaviours 1, 2 and 5).
        They are ALWAYS real arrays — zero-filled when bioen is off, never `None`, because
        Numba needs stable argument types — and every use of them is gated on `bioen`.

        K1 (2026-05-08): the three scratch buffers `prey_type_buf`,
        `prey_id_buf`, `prey_eligible_buf` are allocated per-cell by the
        caller and reused across every predator in that cell — replaces
        the prior per-call `np.zeros(max_prey, ...)` allocation triple.
        Write-then-read pattern: each Phase-1 scan loop iteration
        writes to `..._buf[n_prey]` before incrementing, and Phase 2
        only reads `..._buf[:n_prey]`, so no zero-init is required.
        """
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

        if bioen:
            # Behaviour 1. Java `BioenPredationMortality` multiplies the per-fish
            # allometric cap by the INSTANTANEOUS abundance at every predator visit
            # (`:140-145`), so the school's biomass never enters the expression.
            max_eatable = cap_fish[p_idx] * abd_p
        else:
            biomass_p = abd_p * weight[p_idx]
            max_eatable = biomass_p * ingestion_rate[sp_pred] / (n_dt_per_year * n_subdt)
        if max_eatable <= 0:
            return

        n_local = len(cell_indices)

        # Phase 1: Scan all prey using caller-provided scratch buffers.
        prey_type = prey_type_buf
        prey_id = prey_id_buf
        prey_eligible = prey_eligible_buf
        total_available = 0.0
        n_prey = 0

        # 1a: School prey
        for q_pos in range(n_local):
            q_idx = cell_indices[q_pos]
            if q_idx == p_idx:
                continue
            abd_q = inst_abd[q_idx] - egg_retained[q_idx]
            if abd_q < 0.0:
                abd_q = 0.0
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
                        if q_acc < access_matrix.shape[0] and p_acc < access_matrix.shape[1]:
                            access_coeff = access_matrix[q_acc, p_acc]
                        if access_coeff <= 0:
                            continue
                else:
                    sp_prey = species_id[q_idx]
                    if sp_pred < access_matrix.shape[0] and sp_prey < access_matrix.shape[1]:
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

            if r_min <= 0 or r_max <= 0:
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
                    if rsc_row < access_matrix.shape[0] and p_acc < access_matrix.shape[1]:
                        access_coeff = access_matrix[rsc_row, p_acc]
                        if access_coeff <= 0:
                            continue
            elif has_access:
                rsc_sp_idx = n_species + r
                if sp_pred < access_matrix.shape[0] and rsc_sp_idx < access_matrix.shape[1]:
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
        if fr_shape[sp_pred] == 1:
            eaten_total = min(total_available, max_eatable)  # verbatim type-I (bit-exact)
        else:
            r = total_available / max_eatable
            k_fr = fr_halfsat[sp_pred]
            if fr_shape[sp_pred] == 2:
                g_form = r / (r + k_fr)
            else:  # type-III
                g_form = (r * r) / (r * r + k_fr * k_fr)
            cap = r if r < 1.0 else 1.0  # min(r, 1)
            g = g_form if g_form < cap else cap  # conservation clamp
            eaten_total = max_eatable * g

        for k in range(n_prey):
            share = prey_eligible[k] / total_available
            eaten_from_prey = eaten_total * share

            if prey_type[k] == 0:  # school
                q_idx = prey_id[k]
                if weight[q_idx] > 0:
                    n_dead_prey = eaten_from_prey / weight[q_idx]
                    # `_kill`: guard, record, then `_consume` (behaviour 2, PREDATION).
                    if n_dead_prey > 0.0:
                        n_dead[q_idx, 0] += n_dead_prey
                        _consume_numba(
                            q_idx,
                            n_dead_prey,
                            inst_abd,
                            preyed_biomass,
                            e_net,
                            is_background,
                            bioen,
                        )

                if tl_tracking:
                    prey_tl = trophic_level[q_idx]
                    if prey_tl <= 0:
                        prey_tl = 1.0
                    tl_weighted_sum[p_idx] += prey_tl * eaten_from_prey

                if diet_enabled:
                    prey_sp = species_id[q_idx]
                    if p_idx < diet_matrix.shape[0] and prey_sp < diet_matrix.shape[1]:
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
                    if p_idx < diet_matrix.shape[0] and rsc_col < diet_matrix.shape[1]:
                        diet_matrix[p_idx, rsc_col] += eaten_from_prey

        # Phase 3: Update predation success rate
        success = min(eaten_total / max_eatable, 1.0)
        pred_success_rate[p_idx] += success / n_subdt
        preyed_biomass[p_idx] += eaten_total
        if bioen:
            # Behaviour 5. Java keeps TWO accumulators: `ingestion` (rescaled by the
            # survivor fraction at every death, feeds the energy budget) and
            # `preyedBiomass` (raw, feeds the trophic-level update at
            # MortalityProcess:396-401). `preyed_biomass` here is the rescaled one, so the
            # TL denominator has to come from this raw accumulator.
            raw_preyed[p_idx] += eaten_total

    @njit(cache=True)
    def _apply_single_cause(
        cause,
        idx,
        inst_abd,
        n_dead,
        eff_starv,
        eff_additional,
        eff_fishing,
        fishing_discard,
        eff_foraging,
        bioen,
        is_background,
        preyed_biomass,
        e_net,
        gonad_weight,
        weight,
        eta_school,
        age_dt,
        first_feeding_age_dt,
        n_subdt,
    ):
        """Apply one non-predation mortality cause to one school.

        Shared by all three interleaved loops, so behaviours 2, 3 and 4 land in all three
        by construction. The trailing parameters are the bioen extension; every use is
        gated on `bioen`, and with it false the arithmetic and the cause set are exactly
        what they were (which is what protects `tests/test_engine_parity.py`).
        """
        if cause == 1:  # STARVATION
            if bioen:
                # Behaviour 3: `_apply_starvation_for_school`'s bioen branch plus
                # `bioen_starvation_substep`, inlined. Java
                # `BioenStarvationMortality.computeStarvation` for ONE sub-step, on the
                # PREVIOUS step's `e_net` (Java step order mortality -> EnergyBudget ->
                # reproduction).
                if is_background[idx]:
                    return
                # Java `Species.isStarvationEnabledBioen`: ageDt > firstFeedingAgeDt,
                # STRICT (the non-bioen sibling uses >=).
                if age_dt[idx] <= first_feeding_age_dt[idx]:
                    return
                # Checked BEFORE any gonad/e_net write, exactly as the reference does:
                # a school whose abundance has already gone non-positive (reachable, since
                # the starvation toll is not clamped by the abundance) gets no writes.
                if inst_abd[idx] <= 0:
                    return
                en = e_net[idx]
                if en >= 0.0:
                    return
                eta = eta_school[idx]
                deficit = abs(en) / n_subdt
                gonad = gonad_weight[idx]
                if gonad >= eta * deficit:
                    # Enough gonadic energy: pay maintenance from the gonad, repay E_net,
                    # kill nobody (so `_kill`'s `n_dead <= 0` guard means no rescale).
                    gonad_weight[idx] = gonad - eta * deficit
                    e_net[idx] = en + deficit
                    return
                w = weight[idx]
                if w <= 0.0:
                    # Java would divide by zero here; flush the gonad as it does and kill
                    # nobody.
                    gonad_weight[idx] = 0.0
                    return
                # Java flushes the gonad BEFORE reading the repayment credit and the toll
                # from it, so the credit is zero and the toll is the whole deficit
                # (`BioenStarvationMortality.java:186-193`). `e_net` is left unchanged by
                # the substep; the survivor rescale below is what moves it.
                gonad_weight[idx] = 0.0
                dead = deficit / w
                if dead > 0.0:
                    n_dead[idx, 1] += dead
                    _consume_numba(idx, dead, inst_abd, preyed_biomass, e_net, is_background, bioen)
                return
            D = eff_starv[idx]
            if D > 0:
                abd = inst_abd[idx]
                if abd > 0:
                    dead = abd * (1.0 - np.exp(-D))
                    n_dead[idx, 1] += dead
                    _consume_numba(idx, dead, inst_abd, preyed_biomass, e_net, is_background, bioen)
        elif cause == 2:  # ADDITIONAL
            D = eff_additional[idx]
            if D > 0:
                abd = inst_abd[idx]
                if abd > 0:
                    dead = abd * (1.0 - np.exp(-D))
                    n_dead[idx, 2] += dead
                    _consume_numba(idx, dead, inst_abd, preyed_biomass, e_net, is_background, bioen)
        elif cause == 3:  # FISHING
            F = eff_fishing[idx]
            if F > 0:
                abd = inst_abd[idx]
                if abd > 0:
                    dead = abd * (1.0 - np.exp(-F))
                    discard_r = fishing_discard[idx]
                    # Java issues TWO incrementNdead calls whose survivor factors telescope
                    # to (I - nDead) / I, so the split is recorded first and the rescale
                    # runs ONCE with the full count.
                    if discard_r > 0:
                        n_dead[idx, 3] += dead * (1.0 - discard_r)
                        n_dead[idx, 6] += dead * discard_r
                    else:
                        n_dead[idx, 3] += dead
                    _consume_numba(idx, dead, inst_abd, preyed_biomass, e_net, is_background, bioen)
        elif cause == 5:  # FORAGING (bioen only)
            D = eff_foraging[idx]
            if D > 0:
                abd = inst_abd[idx]
                if abd > 0:
                    dead = abd * (1.0 - np.exp(-D))
                    n_dead[idx, 5] += dead
                    _consume_numba(idx, dead, inst_abd, preyed_biomass, e_net, is_background, bioen)

    @njit(cache=True)
    def _mortality_in_cell_numba(
        cell_indices,
        seq_pred,
        seq_starv,
        seq_fish,
        seq_nat,
        seq_for,
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
        fr_shape,
        fr_halfsat,
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
        egg_retained,
        bioen,
        cap_fish,
        raw_preyed,
        e_net,
        gonad_weight,
        eta_school,
        is_background,
        eff_foraging,
    ):
        """Numba-compiled full interleaved mortality: 4 causes, or 5 under bioen.

        The cause count is read from `cause_orders.shape[1]`, so this kernel cannot
        disagree with the width of the orders its caller drew. `seq_for` is the fifth
        school sequence (the reference's fifth permutation); it is unused when
        `cause_orders` is 4 wide.
        """
        n_local = len(cell_indices)
        n_causes = cause_orders.shape[1]

        # K1: per-cell scratch (this kernel processes one cell, allocated
        # once and reused across all predators in the per-school loop).
        max_prey_cell = n_local + n_resources
        prey_type_buf = np.empty(max_prey_cell, dtype=np.int32)
        prey_id_buf = np.empty(max_prey_cell, dtype=np.int32)
        prey_eligible_buf = np.empty(max_prey_cell, dtype=np.float64)

        for i in range(n_local):
            for c in range(n_causes):
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
                        fr_shape,
                        fr_halfsat,
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
                        prey_type_buf,
                        prey_id_buf,
                        prey_eligible_buf,
                        egg_retained,
                        bioen,
                        cap_fish,
                        raw_preyed,
                        e_net,
                        is_background,
                    )
                elif cause == 1:
                    idx = cell_indices[seq_starv[i]]
                    _apply_single_cause(
                        cause,
                        idx,
                        inst_abd,
                        n_dead,
                        eff_starv,
                        eff_additional,
                        eff_fishing,
                        fishing_discard,
                        eff_foraging,
                        bioen,
                        is_background,
                        preyed_biomass,
                        e_net,
                        gonad_weight,
                        weight,
                        eta_school,
                        age_dt,
                        first_feeding_age_dt,
                        n_subdt,
                    )
                elif cause == 2:
                    idx = cell_indices[seq_nat[i]]
                    _apply_single_cause(
                        cause,
                        idx,
                        inst_abd,
                        n_dead,
                        eff_starv,
                        eff_additional,
                        eff_fishing,
                        fishing_discard,
                        eff_foraging,
                        bioen,
                        is_background,
                        preyed_biomass,
                        e_net,
                        gonad_weight,
                        weight,
                        eta_school,
                        age_dt,
                        first_feeding_age_dt,
                        n_subdt,
                    )
                elif cause == 3:
                    idx = cell_indices[seq_fish[i]]
                    _apply_single_cause(
                        cause,
                        idx,
                        inst_abd,
                        n_dead,
                        eff_starv,
                        eff_additional,
                        eff_fishing,
                        fishing_discard,
                        eff_foraging,
                        bioen,
                        is_background,
                        preyed_biomass,
                        e_net,
                        gonad_weight,
                        weight,
                        eta_school,
                        age_dt,
                        first_feeding_age_dt,
                        n_subdt,
                    )
                elif cause == 5:  # FORAGING -- bioen only, absent from the 4-wide order
                    idx = cell_indices[seq_for[i]]
                    _apply_single_cause(
                        cause,
                        idx,
                        inst_abd,
                        n_dead,
                        eff_starv,
                        eff_additional,
                        eff_fishing,
                        fishing_discard,
                        eff_foraging,
                        bioen,
                        is_background,
                        preyed_biomass,
                        e_net,
                        gonad_weight,
                        weight,
                        eta_school,
                        age_dt,
                        first_feeding_age_dt,
                        n_subdt,
                    )

    @njit(cache=True)
    def _mortality_all_cells_numba(
        rng_seed,
        sorted_indices,
        boundaries,
        n_cells,
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
        fr_shape,
        fr_halfsat,
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
        tl_weighted_sum,
        tl_tracking,
        diet_matrix,
        diet_enabled,
        egg_retained,
        bioen,
        cap_fish,
        raw_preyed,
        e_net,
        gonad_weight,
        eta_school,
        is_background,
        eff_foraging,
    ):
        """Numba-compiled batch mortality for ALL cells in one call.

        RNG is generated inline using Numba's np.random (seeded from Python).
        This avoids the Python loop overhead of pre-generating RNG data.
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
            # Behaviour 4. The fifth permutation and the fifth cause code exist under
            # bioen ONLY: drawing them unconditionally would shift every bioen-OFF
            # stream and move `tests/test_engine_parity.py`'s committed baselines.
            # `_get_mortality_causes` is the Python-side authority for this list;
            # the literal below must track it (nothing but the equality gates pins the 5).
            if bioen:
                seq_for = np.random.permutation(n_local).astype(np.int32)
                causes = np.array(
                    [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING, _FORAGING], dtype=np.int32
                )
            else:
                seq_for = seq_pred  # never read: the 4-wide order has no cause 5
                causes = np.array([_PREDATION, _STARVATION, _ADDITIONAL, _FISHING], dtype=np.int32)
            n_causes = len(causes)
            cause_orders = np.empty((n_local, n_causes), dtype=np.int32)
            for ii in range(n_local):
                np.random.shuffle(causes)
                for cc in range(n_causes):
                    cause_orders[ii, cc] = causes[cc]

            # K1: per-cell scratch for _apply_predation_numba's prey scan.
            # Reused across every predator in this cell — write-then-read
            # so no zero-init is required.
            max_prey_cell = n_local + n_resources
            prey_type_buf = np.empty(max_prey_cell, dtype=np.int32)
            prey_id_buf = np.empty(max_prey_cell, dtype=np.int32)
            prey_eligible_buf = np.empty(max_prey_cell, dtype=np.float64)

            for i in range(n_local):
                for c in range(n_causes):
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
                            fr_shape,
                            fr_halfsat,
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
                            prey_type_buf,
                            prey_id_buf,
                            prey_eligible_buf,
                            egg_retained,
                            bioen,
                            cap_fish,
                            raw_preyed,
                            e_net,
                            is_background,
                        )
                    elif cause == 1:
                        idx = cell_indices[seq_starv[i]]
                        _apply_single_cause(
                            cause,
                            idx,
                            inst_abd,
                            n_dead,
                            eff_starv,
                            eff_additional,
                            eff_fishing,
                            fishing_discard,
                            eff_foraging,
                            bioen,
                            is_background,
                            preyed_biomass,
                            e_net,
                            gonad_weight,
                            weight,
                            eta_school,
                            age_dt,
                            first_feeding_age_dt,
                            n_subdt,
                        )
                    elif cause == 2:
                        idx = cell_indices[seq_nat[i]]
                        _apply_single_cause(
                            cause,
                            idx,
                            inst_abd,
                            n_dead,
                            eff_starv,
                            eff_additional,
                            eff_fishing,
                            fishing_discard,
                            eff_foraging,
                            bioen,
                            is_background,
                            preyed_biomass,
                            e_net,
                            gonad_weight,
                            weight,
                            eta_school,
                            age_dt,
                            first_feeding_age_dt,
                            n_subdt,
                        )
                    elif cause == 3:
                        idx = cell_indices[seq_fish[i]]
                        _apply_single_cause(
                            cause,
                            idx,
                            inst_abd,
                            n_dead,
                            eff_starv,
                            eff_additional,
                            eff_fishing,
                            fishing_discard,
                            eff_foraging,
                            bioen,
                            is_background,
                            preyed_biomass,
                            e_net,
                            gonad_weight,
                            weight,
                            eta_school,
                            age_dt,
                            first_feeding_age_dt,
                            n_subdt,
                        )
                    elif cause == 5:  # FORAGING -- bioen only, absent from the 4-wide order
                        idx = cell_indices[seq_for[i]]
                        _apply_single_cause(
                            cause,
                            idx,
                            inst_abd,
                            n_dead,
                            eff_starv,
                            eff_additional,
                            eff_fishing,
                            fishing_discard,
                            eff_foraging,
                            bioen,
                            is_background,
                            preyed_biomass,
                            e_net,
                            gonad_weight,
                            weight,
                            eta_school,
                            age_dt,
                            first_feeding_age_dt,
                            n_subdt,
                        )

    @njit(cache=True, parallel=True)
    def _mortality_all_cells_parallel(
        rng_seed,
        sorted_indices,
        boundaries,
        n_cells,
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
        fr_shape,
        fr_halfsat,
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
        tl_weighted_sum,
        tl_tracking,
        diet_matrix,
        diet_enabled,
        egg_retained,
        bioen,
        cap_fish,
        raw_preyed,
        e_net,
        gonad_weight,
        eta_school,
        is_background,
        eff_foraging,
    ):
        """Parallel batch mortality — prange over cells for multi-core execution.

        Each cell gets a deterministic seed derived from rng_seed + cell index.
        RNG is generated inline per cell (same as sequential version) to avoid
        the overhead of a separate pre-generation loop.

        Deterministic because:
        1. np.random.seed() resets the thread-local PRNG per cell iteration.
        2. Each cell's school index range [start, end) is disjoint by
           construction (sorted_indices partitioned by cell boundaries),
           so no cross-cell write conflicts exist even under prange.

        This means results are reproducible for a given seed regardless of
        thread scheduling order, provided the disjoint-index invariant holds.

        The bioen extension adds three per-school WRITE targets — `e_net`,
        `gonad_weight` and `raw_preyed` — and they inherit that same argument without
        weakening it. Every one of them is written at an index that came from
        `cell_indices = sorted_indices[start:end]`:

        * `_consume_numba` writes `preyed_biomass[idx]` / `e_net[idx]` at the index it is
          killing, which is either the school the cause was applied to
          (`cell_indices[seq_*[i]]`) or a prey `q_idx` drawn from `cell_indices`;
        * the starvation branch writes `gonad_weight[idx]` / `e_net[idx]` at
          `cell_indices[seq_starv[i]]`;
        * `raw_preyed[p_idx]` is written at the predator index, `cell_indices[seq_pred[i]]`.

        No index escapes the cell's own slice, so the disjointness that already made
        `n_dead`, `inst_abd`, `pred_success_rate` and `preyed_biomass` race-free covers the
        new writes unchanged. The READ-only bioen inputs (`cap_fish`, `eta_school`,
        `eff_foraging`, `is_background`, `weight`, `age_dt`, `first_feeding_age_dt`) are
        never written by any iteration.
        """
        # Invariant: boundaries partition sorted_indices into disjoint slices
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
            # Behaviour 4. The fifth permutation and the fifth cause code exist under
            # bioen ONLY: drawing them unconditionally would shift every bioen-OFF
            # stream and move `tests/test_engine_parity.py`'s committed baselines.
            # `_get_mortality_causes` is the Python-side authority for this list;
            # the literal below must track it (nothing but the equality gates pins the 5).
            if bioen:
                seq_for = np.random.permutation(n_local).astype(np.int32)
                causes = np.array(
                    [_PREDATION, _STARVATION, _ADDITIONAL, _FISHING, _FORAGING], dtype=np.int32
                )
            else:
                seq_for = seq_pred  # never read: the 4-wide order has no cause 5
                causes = np.array([_PREDATION, _STARVATION, _ADDITIONAL, _FISHING], dtype=np.int32)
            n_causes = len(causes)
            cause_orders = np.empty((n_local, n_causes), dtype=np.int32)
            for ii in range(n_local):
                np.random.shuffle(causes)
                for cc in range(n_causes):
                    cause_orders[ii, cc] = causes[cc]

            # K1: per-cell scratch (each prange iteration owns its own
            # copy — thread-safe).
            max_prey_cell = n_local + n_resources
            prey_type_buf = np.empty(max_prey_cell, dtype=np.int32)
            prey_id_buf = np.empty(max_prey_cell, dtype=np.int32)
            prey_eligible_buf = np.empty(max_prey_cell, dtype=np.float64)

            for i in range(n_local):
                for c in range(n_causes):
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
                            fr_shape,
                            fr_halfsat,
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
                            prey_type_buf,
                            prey_id_buf,
                            prey_eligible_buf,
                            egg_retained,
                            bioen,
                            cap_fish,
                            raw_preyed,
                            e_net,
                            is_background,
                        )
                    elif cause == 1:
                        idx = cell_indices[seq_starv[i]]
                        _apply_single_cause(
                            cause,
                            idx,
                            inst_abd,
                            n_dead,
                            eff_starv,
                            eff_additional,
                            eff_fishing,
                            fishing_discard,
                            eff_foraging,
                            bioen,
                            is_background,
                            preyed_biomass,
                            e_net,
                            gonad_weight,
                            weight,
                            eta_school,
                            age_dt,
                            first_feeding_age_dt,
                            n_subdt,
                        )
                    elif cause == 2:
                        idx = cell_indices[seq_nat[i]]
                        _apply_single_cause(
                            cause,
                            idx,
                            inst_abd,
                            n_dead,
                            eff_starv,
                            eff_additional,
                            eff_fishing,
                            fishing_discard,
                            eff_foraging,
                            bioen,
                            is_background,
                            preyed_biomass,
                            e_net,
                            gonad_weight,
                            weight,
                            eta_school,
                            age_dt,
                            first_feeding_age_dt,
                            n_subdt,
                        )
                    elif cause == 3:
                        idx = cell_indices[seq_fish[i]]
                        _apply_single_cause(
                            cause,
                            idx,
                            inst_abd,
                            n_dead,
                            eff_starv,
                            eff_additional,
                            eff_fishing,
                            fishing_discard,
                            eff_foraging,
                            bioen,
                            is_background,
                            preyed_biomass,
                            e_net,
                            gonad_weight,
                            weight,
                            eta_school,
                            age_dt,
                            first_feeding_age_dt,
                            n_subdt,
                        )
                    elif cause == 5:  # FORAGING -- bioen only, absent from the 4-wide order
                        idx = cell_indices[seq_for[i]]
                        _apply_single_cause(
                            cause,
                            idx,
                            inst_abd,
                            n_dead,
                            eff_starv,
                            eff_additional,
                            eff_fishing,
                            fishing_discard,
                            eff_foraging,
                            bioen,
                            is_background,
                            preyed_biomass,
                            e_net,
                            gonad_weight,
                            weight,
                            eta_school,
                            age_dt,
                            first_feeding_age_dt,
                            n_subdt,
                        )


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
    ctx: SimulationContext | None = None,
    egg_retained: NDArray[np.float64] | None = None,
    cap_fish: NDArray[np.float64] | None = None,
    raw_preyed: NDArray[np.float64] | None = None,
    eta_school: NDArray[np.float64] | None = None,
    eff_foraging: NDArray[np.float64] | None = None,
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
    seq_for = rng.permutation(n_local).astype(np.int32)

    # Full Numba path: all causes compiled (Tier 3), 4 or 5 under bioen (Task 2 of the
    # bioen-Numba-kernel plan taught the kernels FORAGING, the per-fish allometric cap,
    # the survivor rescaling and the gonad-depletion starvation; Task 3 removed the
    # `bioen_enabled` exclusion below that used to force this cell's loop to the Python
    # reference under bioen — this is the SECOND, inner gate; the outer one is in
    # `mortality()`'s own dispatch).
    use_full_numba = (
        _HAS_NUMBA and inst_abd is not None and rsc_size_min is not None and eff_starv is not None
    )
    if use_full_numba:
        rsc_bio = resources.biomass if resources is not None else _DUMMY_RSC_2D
        cell_id = cell_y * grid_nx + cell_x
        _tl_weighted_sum = ctx.tl_weighted_sum if ctx else None
        _diet_tracking_enabled = ctx.diet_tracking_enabled if ctx else False
        _diet_mat = ctx.diet_matrix if ctx else None
        tl_ws = _tl_weighted_sum if _tl_weighted_sum is not None else _DUMMY_RSC_1D
        tl_track = _tl_weighted_sum is not None
        d_mat = _diet_mat if _diet_tracking_enabled and _diet_mat is not None else _DUMMY_DIET
        d_en = _diet_tracking_enabled and _diet_mat is not None

        # Pre-generate cause orders (must use same RNG sequence as Python path). The list
        # comes from `_get_mortality_causes` rather than a literal so the FORAGING code and
        # the bioen/non-bioen width are decided in exactly one place; with bioen off it is
        # the same 4-element list as before, so the RNG consumption is unchanged.
        causes = _get_mortality_causes(config)
        n_causes = len(causes)
        cause_orders = np.zeros((n_local, n_causes), dtype=np.int32)
        for i in range(n_local):
            rng.shuffle(causes)
            cause_orders[i, :] = causes

        # Bioen-only kernel inputs. They are read ONLY under `bioen` (which is
        # `config.bioen_enabled`, the same flag that makes `mortality()` build them), so a
        # zero-length dummy keeps Numba's argument types stable without a per-cell
        # allocation on the bioen-OFF hot path.
        k_cap = cap_fish if cap_fish is not None else _DUMMY_RSC_1D
        k_raw = raw_preyed if raw_preyed is not None else _DUMMY_RSC_1D
        k_eta = eta_school if eta_school is not None else _DUMMY_RSC_1D
        k_for = eff_foraging if eff_foraging is not None else _DUMMY_RSC_1D
        if config.bioen_enabled and (
            cap_fish is None or raw_preyed is None or eta_school is None or eff_foraging is None
        ):
            # Under bioen these ARE read, and the dummies are zero-length: Numba compiles
            # with boundscheck off, so a missing one would be an out-of-bounds read rather
            # than an error. `mortality()` always supplies all four.
            missing = [
                n
                for n, v in (
                    ("cap_fish", cap_fish),
                    ("raw_preyed", raw_preyed),
                    ("eta_school", eta_school),
                    ("eff_foraging", eff_foraging),
                )
                if v is None
            ]
            raise RuntimeError(
                "the bioen Numba path needs cap_fish, raw_preyed, eta_school and "
                f"eff_foraging; got {missing} as None"
            )

        _mortality_in_cell_numba(
            cell_indices,
            seq_pred,
            seq_starv,
            seq_fish,
            seq_nat,
            seq_for,
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
            config.fr_shape,
            config.fr_halfsat,
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
            egg_retained
            if egg_retained is not None
            else np.zeros(len(state.abundance), dtype=np.float64),
            config.bioen_enabled,
            k_cap,
            k_raw,
            state.e_net,
            state.gonad_weight,
            k_eta,
            state.is_background,
            k_for,
        )
        return

    # Python fallback path
    causes = _get_mortality_causes(config)

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
                    ctx=ctx,
                    cap_fish=cap_fish,
                    raw_preyed=raw_preyed,
                )
            elif cause == _STARVATION:
                if inst_abd is None:
                    raise RuntimeError("inst_abd must not be None for STARVATION cause")
                s_local = seq_starv[i]
                s_idx = int(cell_indices[s_local])
                _apply_starvation_for_school(s_idx, state, config, n_subdt, inst_abd)
            elif cause == _ADDITIONAL:
                if inst_abd is None:
                    raise RuntimeError("inst_abd must not be None for ADDITIONAL cause")
                a_local = seq_nat[i]
                a_idx = int(cell_indices[a_local])
                _apply_additional_for_school(a_idx, state, config, n_subdt, inst_abd, step=step)
            elif cause == _FISHING:
                if inst_abd is None:
                    raise RuntimeError("inst_abd must not be None for FISHING cause")
                f_local = seq_fish[i]
                f_idx = int(cell_indices[f_local])
                _apply_fishing_for_school(
                    f_idx,
                    state,
                    config,
                    n_subdt,
                    inst_abd,
                    step=step,
                    fleet_state=(ctx.fleet_state if ctx is not None else None),
                )
            elif cause == _FORAGING:
                if inst_abd is None:
                    raise RuntimeError("inst_abd must not be None for FORAGING cause")
                fo_local = seq_for[i]
                fo_idx = int(cell_indices[fo_local])
                _apply_foraging_for_school(fo_idx, state, config, n_subdt, inst_abd)


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
    ctx: SimulationContext | None = None,
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
    n_subdt = config.mortality_subdt

    # Initialize TL tracking accumulator on context
    if ctx is not None:
        ctx.tl_weighted_sum = np.zeros(len(state), dtype=np.float64)

    # Pre-pass: larva mortality on eggs
    # larva_mortality both reduces abundance AND records in n_dead[:, ADDITIONAL]
    state = larva_mortality(state, config, step=step)

    # Save larva deaths for output. Was: also state.replace(n_dead=zeros)
    # then n_dead = state.n_dead.copy() — three same-shape allocations of
    # the same zero data. H6 (2026-05-08): allocate the kernel scratch
    # directly as zeros and let work_state.n_dead point to it via the
    # replace below. The interleaved loop never reads state.n_dead between
    # here and the post-loop combined_n_dead merge (which reads
    # work_state.n_dead), so we don't need to update state.n_dead.
    larva_deaths = state.n_dead.copy()
    n_dead = np.zeros_like(state.n_dead)

    # Retain eggs: withheld from prey pool
    egg_retained = np.where(state.is_egg, state.abundance, 0.0)
    state = state.replace(egg_retained=egg_retained)

    # Make working copies for in-place modification. gonad_weight and e_net join the
    # set because the bioen starvation cause writes them from inside the interleaved
    # loop (Java BioenStarvationMortality.computeStarvation) and _consume rescales
    # e_net at every death — neither should reach back into the caller's state object.
    pred_success_rate = state.pred_success_rate.copy()
    preyed_biomass = state.preyed_biomass.copy()
    gonad_weight = state.gonad_weight.copy()
    e_net = state.e_net.copy()

    work_state = state.replace(
        n_dead=n_dead,
        pred_success_rate=pred_success_rate,
        preyed_biomass=preyed_biomass,
        gonad_weight=gonad_weight,
        e_net=e_net,
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

    # Apply dynamic accessibility scaling if active
    if ctx is not None and ctx.prey_density_scale is not None and has_access:
        from osmose.engine.processes.dynamic_accessibility import apply_prey_scale_to_matrix

        assert access_matrix is not None, "access_matrix must be populated when has_access is True"

        if use_stage_access:
            sa_obj = config.stage_accessibility
            assert sa_obj is not None, (
                "stage_accessibility must be loaded when use_stage_access is True"
            )
            # Build stage-to-species mapping for prey rows
            stage_to_sp = np.full(access_matrix.shape[0], -1, dtype=np.int32)
            for sp_idx, sp_name in enumerate(config.all_species_names[: config.n_species]):
                for _norm in (sp_name, sp_name.lower(), sp_name.replace(" ", "")):
                    if _norm in sa_obj.prey_lookup:
                        for si in sa_obj.prey_lookup[_norm]:
                            stage_to_sp[si.matrix_index] = sp_idx
                        break
                    mapped = sa_obj._species_name_map.get(_norm)
                    if mapped and mapped in sa_obj.prey_lookup:
                        for si in sa_obj.prey_lookup[mapped]:
                            stage_to_sp[si.matrix_index] = sp_idx
                        break
            access_matrix = apply_prey_scale_to_matrix(
                access_matrix,
                ctx.prey_density_scale,
                config.n_species,
                is_stage_indexed=True,
                stage_to_species=stage_to_sp,
            )
        else:
            access_matrix = apply_prey_scale_to_matrix(
                access_matrix,
                ctx.prey_density_scale,
                config.n_species,
                is_stage_indexed=False,
            )

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
    fleet_state = ctx.fleet_state if ctx is not None else None
    eff_s, eff_a, eff_f, f_disc = _precompute_effective_rates(
        work_state, config, n_subdt, step, fleet_state=fleet_state
    )

    # Bioen only: per-fish allometric ingestion cap (tonnes/fish/sub-step) used at every
    # predator visit, and a raw ingestion accumulator that keeps the trophic-level
    # denominator free of the survivor rescaling (Java keeps `preyedBiomass` raw while
    # `ingestion` is rescaled). Both stay None off the bioen path, which therefore keeps
    # its exact current arithmetic and allocations.
    cap_fish: NDArray[np.float64] | None = None
    raw_preyed: NDArray[np.float64] | None = None
    if config.bioen_enabled:
        from osmose.engine.processes.bioen_predation import per_fish_ingestion_cap

        cap_fish = per_fish_ingestion_cap(
            work_state.weight,
            work_state.species_id,
            work_state.age_dt,
            config.bioen_i_max_all,
            config.bioen_beta,
            config.bioen_larvae_thres_dt,
            config.bioen_theta,
            config.bioen_c_rate,
            config.n_species,
            config.n_dt_per_year,
            n_subdt,
        )
        raw_preyed = np.zeros(len(work_state), dtype=np.float64)

    # Kernel-side bioen inputs, built ONCE here rather than per cell per sub-step. The
    # FORAGING rate joins `eff_starv`/`eff_additional`/`eff_fishing` (the kernel only has
    # to evaluate `1 - exp(-D)`), and `eta` is resolved per school so the kernel never
    # indexes an `n_species`-long array with a background school's id. Both are real
    # arrays whatever `bioen_enabled` says — Numba needs stable argument types — and are
    # read only under the `bioen` flag.
    eff_for = _precompute_foraging_rates(work_state, config, n_subdt)
    eta_school = _precompute_bioen_eta(work_state, config)
    k_cap = cap_fish if cap_fish is not None else _DUMMY_RSC_1D
    k_raw = raw_preyed if raw_preyed is not None else _DUMMY_RSC_1D

    for _sub in range(n_subdt):
        # Release fraction of eggs into prey pool
        release = np.where(
            work_state.is_egg & (work_state.egg_retained > 0),
            work_state.abundance / n_subdt,
            0.0,
        )
        new_retained = np.maximum(0, work_state.egg_retained - release)
        work_state = work_state.replace(egg_retained=new_retained)

        # Per-cell mortality. The batched kernels now carry the per-fish cap, the
        # survivor rescaling and the interleaved bioen starvation (bioen-Numba-kernel
        # plan Task 2), so bioen runs the same batched path as bioen-off runs — this is
        # the FIRST, outer gate (see `_mortality_in_cell`'s `use_full_numba` for the
        # second, inner one). Reverses spec decision 14 (a bioen kernel was a non-goal).
        if _HAS_NUMBA and len(valid_indices) > 0:
            # Generate a seed from Python RNG for Numba's internal PRNG
            rng_seed = int(rng.integers(0, 2**63))

            # Extract tracking arrays from context BEFORE Numba call
            rsc_bio = resources.biomass if resources is not None else _DUMMY_RSC_2D
            _tl_weighted_sum = ctx.tl_weighted_sum if ctx else None
            _diet_tracking_enabled = ctx.diet_tracking_enabled if ctx else False
            _diet_mat = ctx.diet_matrix if ctx else None
            tl_ws = _tl_weighted_sum if _tl_weighted_sum is not None else _DUMMY_RSC_1D
            tl_track = _tl_weighted_sum is not None
            d_mat = _diet_mat if _diet_tracking_enabled and _diet_mat is not None else _DUMMY_DIET
            d_en = _diet_tracking_enabled and _diet_mat is not None

            # Single Numba call for all cells (RNG generated inside)
            _batch_fn = _mortality_all_cells_parallel if parallel else _mortality_all_cells_numba
            _batch_fn(
                rng_seed,
                sorted_indices,
                boundaries,
                n_cells,
                inst_abd,
                work_state.n_dead,
                eff_s,
                eff_a,
                eff_f,
                f_disc,
                work_state.species_id,
                work_state.length,
                work_state.weight,
                work_state.age_dt,
                work_state.first_feeding_age_dt,
                work_state.feeding_stage,
                work_state.pred_success_rate,
                work_state.preyed_biomass,
                work_state.trophic_level,
                config.size_ratio_min,
                config.size_ratio_max,
                config.ingestion_rate,
                config.fr_shape,
                config.fr_halfsat,
                config.n_dt_per_year,
                n_subdt,
                access_matrix,
                has_access,
                use_stage_access,
                prey_access_idx,
                pred_access_idx,
                rsc_bio,
                rsc_sm,
                rsc_sx,
                rsc_tl,
                rsc_ar,
                n_rsc,
                config.n_species,
                tl_ws,
                tl_track,
                d_mat,
                d_en,
                work_state.egg_retained,
                config.bioen_enabled,
                k_cap,
                k_raw,
                work_state.e_net,
                work_state.gonad_weight,
                eta_school,
                work_state.is_background,
                eff_for,
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
                    inst_abd=inst_abd,
                    step=step,
                    rsc_size_min=rsc_sm,
                    rsc_size_max=rsc_sx,
                    rsc_tl=rsc_tl,
                    rsc_access_rows=rsc_ar,
                    n_rsc=n_rsc,
                    eff_starv=eff_s,
                    eff_additional=eff_a,
                    eff_fishing=eff_f,
                    fishing_discard=f_disc,
                    grid_nx=grid.nx,
                    ctx=ctx,
                    egg_retained=work_state.egg_retained,
                    cap_fish=cap_fish,
                    raw_preyed=raw_preyed,
                    eta_school=eta_school,
                    eff_foraging=eff_for,
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
        # Written from inside the interleaved loop under bioen (starvation + survivor
        # rescaling); identical to the inputs otherwise.
        gonad_weight=work_state.gonad_weight,
        e_net=work_state.e_net,
    )

    # Post-loop: out-of-domain mortality
    state = out_mortality(state, config)

    # Compute new starvation rate for NEXT step (lagged)
    state = update_starvation_rate(state, config)

    # Update trophic level from predation: TL = 1 + sum(prey_TL * eaten) / total_preyed.
    # Java divides by the RAW preyedBiomass (MortalityProcess:396-401), which it never
    # rescales; under bioen `state.preyed_biomass` is the survivor-scaled ingestion, so
    # the raw accumulator is the correct denominator there.
    _tl_weighted_sum = ctx.tl_weighted_sum if ctx else None
    tl_denominator = raw_preyed if raw_preyed is not None else state.preyed_biomass
    mask = tl_denominator > 0
    if mask.any() and _tl_weighted_sum is not None:
        new_tl = state.trophic_level.copy()
        # Handle schools that may have been appended after tl_weighted_sum was created
        tl_ws = (
            _tl_weighted_sum[: len(state)]
            if len(_tl_weighted_sum) >= len(state)
            else np.pad(_tl_weighted_sum, (0, len(state) - len(_tl_weighted_sum)))
        )
        valid = mask & (tl_ws > 0)
        if valid.any():
            new_tl[valid] = 1.0 + tl_ws[valid] / tl_denominator[valid]
        state = state.replace(trophic_level=new_tl)

    if ctx is not None:
        ctx.tl_weighted_sum = None
    return state
