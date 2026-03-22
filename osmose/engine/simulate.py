"""Main simulation loop for the Python OSMOSE engine.

Follows Java's SimulationStep.step() ordering:
  incoming_flux -> reset -> resources.update -> movement ->
  mortality (interleaved) -> growth -> aging_mortality ->
  reproduction -> collect_outputs -> compact
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from osmose.engine.background import BackgroundState
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.incoming_flux import IncomingFluxState
from osmose.engine.physical_data import PhysicalData
from osmose.engine.resources import ResourceState
from osmose.engine.state import MortalityCause, SchoolState


@dataclass
class StepOutput:
    """Aggregated output for a single simulation timestep."""

    step: int
    biomass: NDArray[np.float64]
    abundance: NDArray[np.float64]
    mortality_by_cause: NDArray[np.float64]  # (n_species, n_causes)
    yield_by_species: NDArray[np.float64] | None = None  # fishing yield biomass per species
    # Per-species age/size distribution dicts (sp_idx -> 1-D array), or None if disabled
    biomass_by_age: dict[int, NDArray[np.float64]] | None = None
    abundance_by_age: dict[int, NDArray[np.float64]] | None = None
    biomass_by_size: dict[int, NDArray[np.float64]] | None = None
    abundance_by_size: dict[int, NDArray[np.float64]] | None = None

    # Bioenergetics: mean net energy per species, shape (n_species,), or None if bioen disabled
    bioen_e_net_by_species: NDArray[np.float64] | None = None
    bioen_ingestion_by_species: NDArray[np.float64] | None = None
    bioen_maint_by_species: NDArray[np.float64] | None = None
    bioen_rho_by_species: NDArray[np.float64] | None = None
    bioen_size_inf_by_species: NDArray[np.float64] | None = None


# ---------------------------------------------------------------------------
# Stub process functions (replaced in Phase 2-7)
# ---------------------------------------------------------------------------


def _incoming_flux(
    state: SchoolState,
    flux_state: IncomingFluxState | None,
    step: int,
    rng: np.random.Generator,
) -> SchoolState:
    """Inject external biomass from incoming flux CSV time-series."""
    if flux_state is None:
        return state
    new_schools = flux_state.get_incoming_schools(step, rng)
    if new_schools is not None:
        state = state.append(new_schools)
    return state


def _reset_step_variables(state: SchoolState) -> SchoolState:
    """Reset per-step tracking variables at the start of each timestep."""
    return state.replace(
        abundance=state.abundance.copy(),
        n_dead=np.zeros_like(state.n_dead),
        pred_success_rate=np.zeros(len(state), dtype=np.float64),
        preyed_biomass=np.zeros(len(state), dtype=np.float64),
        length_start=state.length.copy(),
        is_out=np.zeros(len(state), dtype=np.bool_),
    )


def _movement(
    state: SchoolState,
    grid: Grid,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
    map_sets: dict | None = None,
    random_patches: dict | None = None,
    species_rngs: list[np.random.Generator] | None = None,
    flat_map_data=None,
) -> SchoolState:
    """Apply spatial movement."""
    from osmose.engine.processes.movement import movement

    return movement(
        state, grid, config, step, rng,
        map_sets=map_sets,
        random_patches=random_patches,
        species_rngs=species_rngs,
        flat_map_data=flat_map_data,
    )


def _mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
    grid: Grid,
    step: int = 0,
    species_rngs: list[np.random.Generator] | None = None,
) -> SchoolState:
    """Apply all mortality sources with interleaved ordering."""
    from osmose.engine.processes.mortality import mortality

    return mortality(state, resources, config, rng, grid, step=step, species_rngs=species_rngs)


def _growth(state: SchoolState, config: EngineConfig, rng: np.random.Generator) -> SchoolState:
    """Apply Von Bertalanffy growth gated by predation success."""
    from osmose.engine.processes.growth import growth

    return growth(state, config, rng)


def _bioen_step(
    state: SchoolState,
    config: EngineConfig,
    temp_data: PhysicalData | None,
    step: int,
    o2_data: PhysicalData | None = None,
) -> SchoolState:
    """Replace _growth() when bioenergetics is enabled.

    Steps (matches Java Bioen ordering):
      1. Cap preyed_biomass at allometric ingestion maximum.
      2. Compute per-school temperature response phi_t (or 1.0 if disabled).
      3. Run energy budget per species (compute_energy_budget).
      4. Apply starvation mortality per species (bioen_starvation).
      5. Update weight, length, gonad weight, abundance, e_net_avg.
    """
    from osmose.engine.processes.bioen_predation import bioen_ingestion_cap
    from osmose.engine.processes.bioen_starvation import bioen_starvation
    from osmose.engine.processes.energy_budget import compute_energy_budget, update_e_net_avg
    from osmose.engine.processes.temp_function import phi_t as phi_t_fn

    if len(state) == 0:
        return state

    assert config.bioen_beta is not None
    assert config.bioen_assimilation is not None
    assert config.bioen_c_m is not None
    assert config.bioen_eta is not None
    assert config.bioen_r is not None
    assert config.bioen_m0 is not None
    assert config.bioen_m1 is not None
    assert config.bioen_e_mobi is not None
    assert config.bioen_e_d is not None
    assert config.bioen_tp is not None
    assert config.bioen_e_maint is not None
    assert config.bioen_i_max is not None
    assert config.bioen_theta is not None
    assert config.bioen_c_rate is not None

    n_subdt = config.mortality_subdt

    # ------------------------------------------------------------------ #
    # Step 1: Cap ingested biomass per school at the allometric maximum.
    # ------------------------------------------------------------------ #
    is_larvae = state.is_egg  # larvae flag uses egg field (age 0 schools)
    capped_ingestion = state.preyed_biomass.copy()
    for sp in range(config.n_species):
        mask = state.species_id == sp
        if not mask.any():
            continue
        cap = bioen_ingestion_cap(
            weight=state.weight[mask],
            i_max=float(config.bioen_i_max[sp]),
            beta=float(config.bioen_beta[sp]),
            n_dt_per_year=config.n_dt_per_year,
            n_subdt=n_subdt,
            is_larvae=is_larvae[mask],
            theta=float(config.bioen_theta[sp]),
            c_rate=float(config.bioen_c_rate[sp]),
        )
        # cap is max per-subdt; total cap for the full timestep = cap * n_subdt
        capped_ingestion[mask] = np.minimum(state.preyed_biomass[mask], cap * n_subdt)

    # ------------------------------------------------------------------ #
    # Step 2: Temperature response phi_t per school.
    # ------------------------------------------------------------------ #
    if config.bioen_phit_enabled and temp_data is not None:
        if temp_data.is_constant:
            temp_scalar = temp_data.get_value(step, 0, 0)
            phi_t_arr = np.empty(len(state), dtype=np.float64)
            for sp in range(config.n_species):
                mask = state.species_id == sp
                if mask.any():
                    phi_t_arr[mask] = phi_t_fn(
                        np.full(mask.sum(), temp_scalar),
                        float(config.bioen_e_mobi[sp]),
                        float(config.bioen_e_d[sp]),
                        float(config.bioen_tp[sp]),
                    )
        else:
            # Spatially explicit: look up each school's cell
            phi_t_arr = np.empty(len(state), dtype=np.float64)
            for sp in range(config.n_species):
                mask = state.species_id == sp
                if not mask.any():
                    continue
                temps = np.array([
                    temp_data.get_value(step, int(state.cell_y[i]), int(state.cell_x[i]))
                    for i in np.where(mask)[0]
                ])
                phi_t_arr[mask] = phi_t_fn(
                    temps,
                    float(config.bioen_e_mobi[sp]),
                    float(config.bioen_e_d[sp]),
                    float(config.bioen_tp[sp]),
                )
    else:
        phi_t_arr = np.ones(len(state), dtype=np.float64)

    # Oxygen limitation
    if config.bioen_fo2_enabled and o2_data is not None:
        from osmose.engine.processes.oxygen_function import f_o2

        f_o2_arr = np.ones(len(state), dtype=np.float64)
        if o2_data.is_constant:
            o2_scalar = o2_data.get_value(step, 0, 0)
            for sp in range(config.n_species):
                mask = state.species_id == sp
                if mask.any():
                    o2_vals = np.full(mask.sum(), o2_scalar)
                    f_o2_arr[mask] = f_o2(
                        o2_vals,
                        float(config.bioen_o2_c1[sp]),
                        float(config.bioen_o2_c2[sp]),
                    )
    else:
        f_o2_arr = np.ones(len(state), dtype=np.float64)

    # Build per-school temperature array for Arrhenius maintenance calculation
    if temp_data is not None and temp_data.is_constant:
        temp_c_arr = np.full(len(state), temp_data.get_value(step, 0, 0), dtype=np.float64)
    elif temp_data is not None:
        temp_c_arr = np.array([
            temp_data.get_value(step, int(state.cell_y[i]), int(state.cell_x[i]))
            for i in range(len(state))
        ])
    else:
        # No temperature data: default to 15°C (neutral for most species)
        temp_c_arr = np.full(len(state), 15.0, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Step 3: Energy budget per species.
    # ------------------------------------------------------------------ #
    dw_tonnes = np.zeros(len(state), dtype=np.float64)
    dg_tonnes = np.zeros(len(state), dtype=np.float64)
    e_net_arr = np.zeros(len(state), dtype=np.float64)
    e_gross_arr = np.zeros(len(state), dtype=np.float64)
    e_maint_arr = np.zeros(len(state), dtype=np.float64)
    rho_arr = np.zeros(len(state), dtype=np.float64)

    for sp in range(config.n_species):
        mask = state.species_id == sp
        if not mask.any():
            continue
        dw_sp, dg_sp, en_sp, eg_sp, em_sp, rho_sp = compute_energy_budget(
            ingestion=capped_ingestion[mask],
            weight=state.weight[mask],
            gonad_weight=state.gonad_weight[mask],
            age_dt=state.age_dt[mask],
            length=state.length[mask],
            temp_c=temp_c_arr[mask],   # raw temperature for Arrhenius maintenance
            assimilation=float(config.bioen_assimilation[sp]),
            c_m=float(config.bioen_c_m[sp]),
            beta=float(config.bioen_beta[sp]),
            eta=float(config.bioen_eta[sp]),
            r=float(config.bioen_r[sp]),
            m0=float(config.bioen_m0[sp]),
            m1=float(config.bioen_m1[sp]),
            e_maint_energy=float(config.bioen_e_maint[sp]),
            phi_t=phi_t_arr[mask],   # Johnson thermal performance (applied to assimilation)
            f_o2=f_o2_arr[mask],
            n_dt_per_year=config.n_dt_per_year,
            e_net_avg=state.e_net_avg[mask],
        )
        dw_tonnes[mask] = dw_sp
        dg_tonnes[mask] = dg_sp
        e_net_arr[mask] = en_sp
        e_gross_arr[mask] = eg_sp
        e_maint_arr[mask] = em_sp
        rho_arr[mask] = rho_sp

    # ------------------------------------------------------------------ #
    # Step 4: Starvation mortality per species.
    # ------------------------------------------------------------------ #
    starvation_dead = np.zeros(len(state), dtype=np.float64)
    new_gonad = state.gonad_weight.copy()

    for sp in range(config.n_species):
        mask = state.species_id == sp
        if not mask.any():
            continue
        n_dead_sp, gonad_sp = bioen_starvation(
            e_net=e_net_arr[mask],
            gonad_weight=state.gonad_weight[mask],
            weight=state.weight[mask],
            eta=float(config.bioen_eta[sp]),
            n_subdt=n_subdt,
        )
        starvation_dead[mask] = n_dead_sp
        new_gonad[mask] = gonad_sp

    # ------------------------------------------------------------------ #
    # Step 5: Apply updates to state arrays.
    # ------------------------------------------------------------------ #
    new_weight = np.maximum(state.weight + dw_tonnes, 0.0)
    new_gonad = np.maximum(new_gonad + dg_tonnes, 0.0)

    # Update length from weight via allometric inverse (W = a * L^b)
    # L = (W / a)^(1/b); use species-level a (condition factor * 1e-6) and b
    new_length = state.length.copy()
    for sp in range(config.n_species):
        mask = state.species_id == sp
        if not mask.any():
            continue
        # W_tonnes = condition_factor * L^b * 1e-6  =>  L = (W_tonnes * 1e6 / condition_factor)^(1/b)
        a = float(config.condition_factor[sp])
        b = float(config.allometric_power[sp])
        safe_a = max(a, 1e-20)
        new_length[mask] = np.power(
            np.maximum(new_weight[mask] * 1e6 / safe_a, 1e-20), 1.0 / b
        )

    # Reduce abundance by starvation deaths (clamp to zero)
    new_abundance = np.maximum(state.abundance - starvation_dead, 0.0)

    # Track starvation in n_dead
    new_n_dead = state.n_dead.copy()
    new_n_dead[:, int(MortalityCause.STARVATION)] += starvation_dead

    # Update running e_net_avg
    new_e_net_avg = np.zeros(len(state), dtype=np.float64)
    for sp in range(config.n_species):
        mask = state.species_id == sp
        if not mask.any():
            continue
        new_e_net_avg[mask] = update_e_net_avg(
            e_net_avg=state.e_net_avg[mask],
            e_net=e_net_arr[mask],
            weight=state.weight[mask],
            age_dt=state.age_dt[mask],
            first_feeding_age_dt=state.first_feeding_age_dt[mask],
            n_dt_per_year=config.n_dt_per_year,
        )

    new_biomass = new_abundance * new_weight

    return state.replace(
        weight=new_weight,
        length=new_length,
        biomass=new_biomass,
        gonad_weight=new_gonad,
        abundance=new_abundance,
        n_dead=new_n_dead,
        e_net_avg=new_e_net_avg,
        e_net=e_net_arr,
        e_gross=e_gross_arr,
        e_maint=e_maint_arr,
        rho=rho_arr,
    )


def _aging_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Kill schools exceeding species lifespan."""
    from osmose.engine.processes.natural import aging_mortality

    return aging_mortality(state, config)


def _reproduction(
    state: SchoolState,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
    grid_ny: int = 10,
    grid_nx: int = 10,
) -> SchoolState:
    """Egg production + age increment."""
    from osmose.engine.processes.reproduction import reproduction

    return reproduction(state, config, step, rng, grid_ny=grid_ny, grid_nx=grid_nx)


def _bioen_reproduction(
    state: SchoolState,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
    grid_ny: int = 10,
    grid_nx: int = 10,
) -> SchoolState:
    """Bioen reproduction: create egg schools from gonad weight (replaces SSB method)."""
    from osmose.engine.processes.bioen_reproduction import bioen_egg_production

    new_egg_schools = []
    gonad = state.gonad_weight.copy()

    for sp in range(config.n_species):
        mask = state.species_id == sp
        if not mask.any():
            continue

        # Get egg weight (handle NaN from missing config)
        ew = np.nan
        if config.egg_weight_override is not None:
            ew = config.egg_weight_override[sp]
        if np.isnan(ew):
            # Fallback: allometric weight at egg size
            ew = config.condition_factor[sp] * config.egg_size[sp] ** config.allometric_power[sp] * 1e-6

        eggs = bioen_egg_production(
            gonad_weight=state.gonad_weight[mask],
            length=state.length[mask],
            age_dt=state.age_dt[mask],
            m0=float(config.bioen_m0[sp]),
            m1=float(config.bioen_m1[sp]),
            egg_weight=float(ew),
            n_dt_per_year=config.n_dt_per_year,
        )

        total_eggs = eggs.sum()
        if total_eggs <= 0:
            continue

        # Reset gonad weight for schools that spawned
        spawned = eggs > 0
        indices = np.where(mask)[0]
        gonad[indices[spawned]] = 0.0

        # Create a single egg school per species (matching Java convention)
        # Place in random cell occupied by parent schools
        parent_cells_y = state.cell_y[mask][spawned]
        parent_cells_x = state.cell_x[mask][spawned]
        if len(parent_cells_y) > 0:
            idx = rng.integers(len(parent_cells_y))
            egg_school = SchoolState.create(n_schools=1, species_id=np.array([sp], dtype=np.int32))
            egg_school = egg_school.replace(
                abundance=np.array([total_eggs]),
                weight=np.array([float(ew)]),
                biomass=np.array([total_eggs * float(ew)]),
                length=np.array([config.egg_size[sp]]),
                cell_x=np.array([parent_cells_x[idx]], dtype=np.int32),
                cell_y=np.array([parent_cells_y[idx]], dtype=np.int32),
                is_egg=np.array([True]),
                first_feeding_age_dt=np.array([1], dtype=np.int32),
            )
            new_egg_schools.append(egg_school)

    state = state.replace(gonad_weight=gonad)

    # Append egg schools
    for egg_school in new_egg_schools:
        state = state.append(egg_school)

    # Age increment for existing schools only (NOT new eggs — they start at age 0)
    # Java's BioenReproductionProcess increments age separately from egg creation
    n_existing = len(state) - sum(len(e) for e in new_egg_schools)
    new_age = state.age_dt.copy()
    new_age[:n_existing] += 1
    state = state.replace(age_dt=new_age)

    return state


# ---------------------------------------------------------------------------
# Background inject/strip helpers
# ---------------------------------------------------------------------------


def _collect_background_outputs(
    state: SchoolState, config: EngineConfig, n_focal: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate background species biomass/abundance before stripping."""
    n_total = config.n_species + config.n_background
    bkg_biomass = np.zeros(n_total, dtype=np.float64)
    bkg_abundance = np.zeros(n_total, dtype=np.float64)
    if len(state) > n_focal:
        bkg_ids = state.species_id[n_focal:]
        np.add.at(bkg_biomass, bkg_ids, state.biomass[n_focal:])
        np.add.at(bkg_abundance, bkg_ids, state.abundance[n_focal:])
    return bkg_biomass, bkg_abundance


def _strip_background(state: SchoolState, n_focal: int) -> SchoolState:
    """Remove background schools by slicing to first n_focal entries."""
    from dataclasses import fields

    sliced = {}
    for f in fields(state):
        arr = getattr(state, f.name)
        sliced[f.name] = arr[:n_focal] if arr.ndim == 1 else arr[:n_focal, :]
    return SchoolState(**sliced)


# ---------------------------------------------------------------------------
# Output collection
# ---------------------------------------------------------------------------


def _collect_outputs(
    state: SchoolState,
    config: EngineConfig,
    step: int,
    bkg_output: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> StepOutput:
    """Aggregate per-species biomass and abundance from current state."""
    n_total = config.n_species + config.n_background
    biomass = np.zeros(n_total, dtype=np.float64)
    abundance = np.zeros(n_total, dtype=np.float64)
    if len(state) > 0:
        # Apply output cutoff age filter (Java convention: exclude young-of-year)
        if config.output_cutoff_age is not None:
            age_years = state.age_dt.astype(np.float64) / config.n_dt_per_year
            cutoff = config.output_cutoff_age[state.species_id]
            include = age_years >= cutoff
            np.add.at(biomass, state.species_id[include], state.biomass[include])
            np.add.at(abundance, state.species_id[include], state.abundance[include])
        else:
            np.add.at(biomass, state.species_id, state.biomass)
            np.add.at(abundance, state.species_id, state.abundance)
    if bkg_output is not None:
        biomass += bkg_output[0]
        abundance += bkg_output[1]

    # Aggregate mortality by cause per species (focal only)
    n_causes = len(MortalityCause)
    mortality_by_cause = np.zeros((config.n_species, n_causes), dtype=np.float64)
    if len(state) > 0:
        focal_mask = state.species_id < config.n_species
        for cause in range(n_causes):
            np.add.at(
                mortality_by_cause[:, cause],
                state.species_id[focal_mask],
                state.n_dead[focal_mask, cause],
            )

    # Yield: fishing deaths * weight per species
    yield_by_species = np.zeros(config.n_species, dtype=np.float64)
    if len(state) > 0:
        fishing_dead = state.n_dead[:, int(MortalityCause.FISHING)]
        fishing_yield = fishing_dead * state.weight  # weight already in tonnes
        focal_mask = state.species_id < config.n_species
        np.add.at(yield_by_species, state.species_id[focal_mask], fishing_yield[focal_mask])

    # Age binning
    biomass_by_age: dict[int, NDArray[np.float64]] | None = None
    abundance_by_age: dict[int, NDArray[np.float64]] | None = None
    if config.output_biomass_byage or config.output_abundance_byage:
        bba: dict[int, NDArray[np.float64]] = {}
        aba: dict[int, NDArray[np.float64]] = {}
        for sp in range(config.n_species):
            max_age_yr = int(np.ceil(config.lifespan_dt[sp] / config.n_dt_per_year))
            n_bins = max_age_yr + 1
            bba[sp] = np.zeros(n_bins, dtype=np.float64)
            aba[sp] = np.zeros(n_bins, dtype=np.float64)
            if len(state) > 0:
                sp_mask = state.species_id == sp
                if sp_mask.any():
                    age_yr = state.age_dt[sp_mask] // config.n_dt_per_year
                    age_yr = np.clip(age_yr, 0, max_age_yr)
                    if config.output_biomass_byage:
                        np.add.at(bba[sp], age_yr, state.biomass[sp_mask])
                    if config.output_abundance_byage:
                        np.add.at(aba[sp], age_yr, state.abundance[sp_mask])
        if config.output_biomass_byage:
            biomass_by_age = bba
        if config.output_abundance_byage:
            abundance_by_age = aba

    # Size binning
    biomass_by_size: dict[int, NDArray[np.float64]] | None = None
    abundance_by_size: dict[int, NDArray[np.float64]] | None = None
    if config.output_biomass_bysize or config.output_abundance_bysize:
        edges = np.arange(
            config.output_size_min,
            config.output_size_max + config.output_size_incr,
            config.output_size_incr,
        )
        n_bins = len(edges)  # number of bins = number of left edges (searchsorted gives 0..n_bins)
        bbs: dict[int, NDArray[np.float64]] = {}
        abs_: dict[int, NDArray[np.float64]] = {}
        for sp in range(config.n_species):
            bbs[sp] = np.zeros(n_bins, dtype=np.float64)
            abs_[sp] = np.zeros(n_bins, dtype=np.float64)
            if len(state) > 0:
                sp_mask = state.species_id == sp
                if sp_mask.any():
                    bin_idx = np.searchsorted(edges, state.length[sp_mask], side="right") - 1
                    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                    if config.output_biomass_bysize:
                        np.add.at(bbs[sp], bin_idx, state.biomass[sp_mask])
                    if config.output_abundance_bysize:
                        np.add.at(abs_[sp], bin_idx, state.abundance[sp_mask])
        if config.output_biomass_bysize:
            biomass_by_size = bbs
        if config.output_abundance_bysize:
            abundance_by_size = abs_

    # Bioen: mean e_net per focal species (zeros when state is empty or no focal schools)
    bioen_e_net_by_species: NDArray[np.float64] | None = None
    bioen_ingestion_by_species: NDArray[np.float64] | None = None
    bioen_maint_by_species: NDArray[np.float64] | None = None
    bioen_rho_by_species: NDArray[np.float64] | None = None
    bioen_size_inf_by_species: NDArray[np.float64] | None = None
    if config.bioen_enabled:
        bioen_e_net_by_species = np.zeros(config.n_species, dtype=np.float64)
        if len(state) > 0:
            counts = np.zeros(config.n_species, dtype=np.float64)
            focal = state.species_id < config.n_species
            np.add.at(bioen_e_net_by_species, state.species_id[focal], state.e_net[focal])
            np.add.at(counts, state.species_id[focal], 1)
            safe_counts = np.where(counts > 0, counts, 1)
            bioen_e_net_by_species /= safe_counts

        bioen_ingestion = np.zeros(config.n_species, dtype=np.float64)
        bioen_maint = np.zeros(config.n_species, dtype=np.float64)
        bioen_rho = np.zeros(config.n_species, dtype=np.float64)
        bioen_sizeinf = np.zeros(config.n_species, dtype=np.float64)
        if len(state) > 0:
            counts2 = np.zeros(config.n_species, dtype=np.float64)
            focal = state.species_id < config.n_species
            # e_gross = ingestion * assimilation * phi_T * f_O2 (Java's "ingestion" output)
            np.add.at(bioen_ingestion, state.species_id[focal], state.e_gross[focal])
            np.add.at(bioen_maint, state.species_id[focal], state.e_maint[focal])
            np.add.at(bioen_rho, state.species_id[focal], state.rho[focal])
            np.add.at(counts2, state.species_id[focal], 1)
            safe2 = np.where(counts2 > 0, counts2, 1)
            bioen_ingestion /= safe2
            bioen_maint /= safe2
            bioen_rho /= safe2
            # sizeInf: max observed length per species
            for sp in range(config.n_species):
                sp_mask = (state.species_id == sp) & focal
                if sp_mask.any():
                    bioen_sizeinf[sp] = state.length[sp_mask].max()
        bioen_ingestion_by_species = bioen_ingestion
        bioen_maint_by_species = bioen_maint
        bioen_rho_by_species = bioen_rho
        bioen_size_inf_by_species = bioen_sizeinf

    return StepOutput(
        step=step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=mortality_by_cause,
        yield_by_species=yield_by_species,
        biomass_by_age=biomass_by_age,
        abundance_by_age=abundance_by_age,
        biomass_by_size=biomass_by_size,
        abundance_by_size=abundance_by_size,
        bioen_e_net_by_species=bioen_e_net_by_species,
        bioen_ingestion_by_species=bioen_ingestion_by_species,
        bioen_maint_by_species=bioen_maint_by_species,
        bioen_rho_by_species=bioen_rho_by_species,
        bioen_size_inf_by_species=bioen_size_inf_by_species,
    )


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def initialize(config: EngineConfig, grid: Grid, rng: np.random.Generator) -> SchoolState:
    """Create an empty initial population (Java convention).

    Java's PopulatingProcess creates zero initial schools. All schools
    are created by reproduction's seeding mechanism (SSB=0 triggers
    seeding_biomass injection as SSB for egg production).
    """
    return SchoolState.create(n_schools=0)


def _average_step_outputs(
    accumulated: list[StepOutput], freq: int, record_step: int
) -> StepOutput:
    """Average accumulated StepOutputs over recording frequency."""
    def _avg_bioen(attr: str) -> NDArray[np.float64] | None:
        arrays = [getattr(o, attr) for o in accumulated if getattr(o, attr) is not None]
        return np.mean(arrays, axis=0) if arrays else None

    bioen_e_net_avg = _avg_bioen("bioen_e_net_by_species")
    bioen_ingestion_avg = _avg_bioen("bioen_ingestion_by_species")
    bioen_maint_avg = _avg_bioen("bioen_maint_by_species")
    bioen_rho_avg = _avg_bioen("bioen_rho_by_species")
    bioen_size_inf_avg = _avg_bioen("bioen_size_inf_by_species")

    if len(accumulated) == 1:
        return StepOutput(
            step=record_step,
            biomass=accumulated[0].biomass,
            abundance=accumulated[0].abundance,
            mortality_by_cause=accumulated[0].mortality_by_cause,
            yield_by_species=accumulated[0].yield_by_species,
            bioen_e_net_by_species=bioen_e_net_avg,
            bioen_ingestion_by_species=bioen_ingestion_avg,
            bioen_maint_by_species=bioen_maint_avg,
            bioen_rho_by_species=bioen_rho_avg,
            bioen_size_inf_by_species=bioen_size_inf_avg,
        )
    biomass = np.mean([o.biomass for o in accumulated], axis=0)
    abundance = np.mean([o.abundance for o in accumulated], axis=0)
    mortality = np.sum([o.mortality_by_cause for o in accumulated], axis=0)
    yield_sum = np.sum(
        [o.yield_by_species for o in accumulated if o.yield_by_species is not None], axis=0
    )
    return StepOutput(
        step=record_step,
        biomass=biomass,
        abundance=abundance,
        mortality_by_cause=mortality,
        yield_by_species=yield_sum,
        bioen_e_net_by_species=bioen_e_net_avg,
        bioen_ingestion_by_species=bioen_ingestion_avg,
        bioen_maint_by_species=bioen_maint_avg,
        bioen_rho_by_species=bioen_rho_avg,
        bioen_size_inf_by_species=bioen_size_inf_avg,
    )


def simulate(
    config: EngineConfig,
    grid: Grid,
    rng: np.random.Generator,
    movement_rngs: list[np.random.Generator] | None = None,
    mortality_rngs: list[np.random.Generator] | None = None,
) -> list[StepOutput]:
    """Run the OSMOSE simulation loop.

    Process ordering matches Java's SimulationStep.step().
    """
    if movement_rngs is None:
        movement_rngs = [rng] * config.n_species
    if mortality_rngs is None:
        mortality_rngs = [rng] * config.n_species
    state = initialize(config, grid, rng)
    resources = ResourceState(config=config.raw_config, grid=grid)
    background = BackgroundState(config=config.raw_config, grid=grid, engine_config=config)
    flux_state = IncomingFluxState(config=config.raw_config, engine_config=config, grid=grid)
    outputs: list[StepOutput] = []

    # Bioenergetics: load temperature forcing when enabled
    temp_data = None
    if config.bioen_enabled:
        temp_val = config.raw_config.get("temperature.value", "")
        if temp_val:
            temp_data = PhysicalData.from_constant(float(temp_val))

    # Bioenergetics: load O2 forcing when enabled
    o2_data = None
    if config.bioen_enabled:
        o2_val = config.raw_config.get("oxygen.value", "")
        if o2_val:
            o2_data = PhysicalData.from_constant(float(o2_val))

    from osmose.engine.movement_maps import MovementMapSet

    map_sets: dict[int, MovementMapSet] = {}
    for sp in range(config.n_species):
        if config.movement_method[sp] == "maps":
            sp_name = config.species_names[sp]
            map_sets[sp] = MovementMapSet(
                config=config.raw_config,
                species_name=sp_name,
                n_dt_per_year=config.n_dt_per_year,
                n_years=config.n_year,
                lifespan_dt=int(config.lifespan_dt[sp]),
                ny=grid.ny,
                nx=grid.nx,
                config_dir=config.raw_config.get("_osmose.config.dir", ""),
            )

    # Pre-flatten map data for Numba movement path
    from osmose.engine.processes.movement import _flatten_all_map_sets
    flat_map_data = _flatten_all_map_sets(map_sets, config.n_species, grid.ny, grid.nx) if map_sets else None

    # Phase 4: Build random distribution patches
    from osmose.engine.processes.movement import build_random_patches

    random_patches = build_random_patches(config, grid, rng)

    # Phase 5: Output recording frequency
    record_freq = config.output_record_frequency
    accumulated: list[StepOutput] = []

    # Phase 5: Initial state output (step -1)
    if config.output_step0_include:
        outputs.append(_collect_outputs(state, config, step=-1))

    for step in range(config.n_steps):
        state = _incoming_flux(state, flux_state, step, rng)
        state = _reset_step_variables(state)
        resources.update(step)
        state = _movement(
            state, grid, config, step, rng,
            map_sets=map_sets,
            random_patches=random_patches,
            species_rngs=movement_rngs,
            flat_map_data=flat_map_data,
        )

        # Inject background schools before mortality
        bkg_schools = background.get_schools(step)
        n_focal = len(state)
        if len(bkg_schools) > 0:
            state = state.append(bkg_schools)

        state = _mortality(state, resources, config, rng, grid, step=step, species_rngs=mortality_rngs)

        # Collect background output BEFORE stripping
        bkg_output = _collect_background_outputs(state, config, n_focal)

        # Strip background schools
        state = _strip_background(state, n_focal)

        if config.bioen_enabled:
            state = _bioen_step(state, config, temp_data, step, o2_data=o2_data)
        else:
            state = _growth(state, config, rng)
        state = _aging_mortality(state, config)
        if config.bioen_enabled:
            state = _bioen_reproduction(state, config, step, rng, grid_ny=grid.ny, grid_nx=grid.nx)
        else:
            state = _reproduction(state, config, step, rng, grid_ny=grid.ny, grid_nx=grid.nx)

        # Collect focal outputs after reproduction
        step_out = _collect_outputs(state, config, step, bkg_output)
        accumulated.append(step_out)

        # Write averaged output at recording frequency
        if (step + 1) % record_freq == 0:
            outputs.append(_average_step_outputs(accumulated, record_freq, step))
            accumulated = []

        state = state.compact()

    # Flush any remaining accumulated steps (if n_steps not divisible by freq)
    if accumulated:
        outputs.append(_average_step_outputs(accumulated, len(accumulated), config.n_steps - 1))

    return outputs
