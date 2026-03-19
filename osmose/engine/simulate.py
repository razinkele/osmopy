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
from osmose.engine.resources import ResourceState
from osmose.engine.state import MortalityCause, SchoolState


@dataclass
class StepOutput:
    """Aggregated output for a single simulation timestep."""

    step: int
    biomass: NDArray[np.float64]
    abundance: NDArray[np.float64]
    mortality_by_cause: NDArray[np.float64]  # (n_species, n_causes)


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
) -> SchoolState:
    """Apply spatial movement."""
    from osmose.engine.processes.movement import movement

    return movement(state, grid, config, step, rng, map_sets=map_sets)


def _mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
    grid: Grid,
    step: int = 0,
) -> SchoolState:
    """Apply all mortality sources with interleaved ordering."""
    from osmose.engine.processes.mortality import mortality

    return mortality(state, resources, config, rng, grid, step=step)


def _growth(state: SchoolState, config: EngineConfig, rng: np.random.Generator) -> SchoolState:
    """Apply Von Bertalanffy growth gated by predation success."""
    from osmose.engine.processes.growth import growth

    return growth(state, config, rng)


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

    return StepOutput(
        step=step, biomass=biomass, abundance=abundance, mortality_by_cause=mortality_by_cause
    )


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def initialize(config: EngineConfig, grid: Grid, rng: np.random.Generator) -> SchoolState:
    """Create an empty initial population (Java convention).

    Java's PopulatingProcess creates zero initial schools by default.
    All schools are created by reproduction's seeding mechanism in the
    first timestep (SSB=0 triggers seeding_biomass injection).
    """
    return SchoolState.create(n_schools=0)


def simulate(
    config: EngineConfig,
    grid: Grid,
    rng: np.random.Generator,
) -> list[StepOutput]:
    """Run the OSMOSE simulation loop.

    Process ordering matches Java's SimulationStep.step().
    """
    state = initialize(config, grid, rng)
    resources = ResourceState(config=config.raw_config, grid=grid)
    background = BackgroundState(config=config.raw_config, grid=grid, engine_config=config)
    flux_state = IncomingFluxState(config=config.raw_config, engine_config=config, grid=grid)
    outputs: list[StepOutput] = []

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
            )

    for step in range(config.n_steps):
        state = _incoming_flux(state, flux_state, step, rng)
        state = _reset_step_variables(state)
        resources.update(step)
        state = _movement(state, grid, config, step, rng, map_sets=map_sets)

        # Inject background schools before mortality
        bkg_schools = background.get_schools(step)
        n_focal = len(state)
        if len(bkg_schools) > 0:
            state = state.append(bkg_schools)

        state = _mortality(state, resources, config, rng, grid, step=step)

        # Collect background output BEFORE stripping
        bkg_output = _collect_background_outputs(state, config, n_focal)

        # Strip background schools
        state = _strip_background(state, n_focal)

        state = _growth(state, config, rng)
        state = _aging_mortality(state, config)
        state = _reproduction(state, config, step, rng, grid_ny=grid.ny, grid_nx=grid.nx)

        # Collect focal outputs after reproduction
        outputs.append(_collect_outputs(state, config, step, bkg_output))
        state = state.compact()

    return outputs
