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

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.resources import ResourceState
from osmose.engine.state import SchoolState


@dataclass
class StepOutput:
    """Aggregated output for a single simulation timestep."""

    step: int
    biomass: NDArray[np.float64]
    abundance: NDArray[np.float64]


# ---------------------------------------------------------------------------
# Stub process functions (replaced in Phase 2-7)
# ---------------------------------------------------------------------------


def _incoming_flux(
    state: SchoolState, config: EngineConfig, step: int, rng: np.random.Generator
) -> SchoolState:
    """Phase 1 stub: incoming flux (migration injection)."""
    return state


def _reset_step_variables(state: SchoolState) -> SchoolState:
    """Reset per-step tracking variables at the start of each timestep."""
    return state.replace(
        abundance=state.abundance.copy(),
        n_dead=np.zeros_like(state.n_dead),
        pred_success_rate=np.zeros(len(state), dtype=np.float64),
        preyed_biomass=np.zeros(len(state), dtype=np.float64),
        length_start=state.length.copy(),
    )


def _movement(
    state: SchoolState,
    grid: Grid,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
) -> SchoolState:
    """Apply spatial movement."""
    from osmose.engine.processes.movement import movement

    return movement(state, grid, config, step, rng)


def _mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
    grid: Grid,
) -> SchoolState:
    """Apply all mortality sources. Phase 6: predation + starvation + fishing + additional."""
    from osmose.engine.processes.fishing import fishing_mortality
    from osmose.engine.processes.natural import additional_mortality, larva_mortality, out_mortality
    from osmose.engine.processes.predation import predation
    from osmose.engine.processes.starvation import starvation_mortality, update_starvation_rate

    n_subdt = config.mortality_subdt

    # Pre-pass: larva mortality on eggs (before main loop)
    state = larva_mortality(state, config)

    # Main mortality sub-timestep loop
    for _sub in range(n_subdt):
        state = predation(state, config, rng, n_subdt, grid.ny, grid.nx, resources=resources)
        state = starvation_mortality(state, config, n_subdt)
        state = fishing_mortality(state, config, n_subdt)
        state = additional_mortality(state, config, n_subdt)

    # Out-of-domain mortality
    state = out_mortality(state, config)
    # Compute new starvation rate for NEXT step (lagged)
    state = update_starvation_rate(state, config)
    return state


def _growth(state: SchoolState, config: EngineConfig, rng: np.random.Generator) -> SchoolState:
    """Apply Von Bertalanffy growth gated by predation success."""
    from osmose.engine.processes.growth import growth

    return growth(state, config, rng)


def _aging_mortality(state: SchoolState, config: EngineConfig) -> SchoolState:
    """Kill schools exceeding species lifespan."""
    from osmose.engine.processes.natural import aging_mortality

    return aging_mortality(state, config)


def _reproduction(
    state: SchoolState, config: EngineConfig, step: int, rng: np.random.Generator
) -> SchoolState:
    """Egg production + age increment."""
    from osmose.engine.processes.reproduction import reproduction

    return reproduction(state, config, step, rng)


# ---------------------------------------------------------------------------
# Output collection
# ---------------------------------------------------------------------------


def _collect_outputs(state: SchoolState, config: EngineConfig, step: int) -> StepOutput:
    """Aggregate per-species biomass and abundance from current state."""
    biomass = np.zeros(config.n_species, dtype=np.float64)
    abundance = np.zeros(config.n_species, dtype=np.float64)
    if len(state) > 0:
        np.add.at(biomass, state.species_id, state.biomass)
        np.add.at(abundance, state.species_id, state.abundance)
    return StepOutput(step=step, biomass=biomass, abundance=abundance)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def initialize(config: EngineConfig, grid: Grid, rng: np.random.Generator) -> SchoolState:
    """Create the initial population of schools with seeding biomass.

    Each species gets n_schools schools with biomass split equally from
    seeding_biomass. Schools start at age 0 with egg size.
    """
    total_schools = int(config.n_schools.sum())
    species_ids = np.repeat(np.arange(config.n_species, dtype=np.int32), config.n_schools)
    state = SchoolState.create(n_schools=total_schools, species_id=species_ids)

    # Initial length = egg size, weight from allometry
    lengths = config.egg_size[species_ids]
    weights = config.condition_factor[species_ids] * lengths ** config.allometric_power[species_ids]

    # Split seeding biomass equally among schools of each species
    biomass_per_school = np.zeros(total_schools, dtype=np.float64)
    for sp in range(config.n_species):
        mask = species_ids == sp
        n = mask.sum()
        if n > 0 and config.seeding_biomass[sp] > 0:
            biomass_per_school[mask] = config.seeding_biomass[sp] / n

    abundance = np.where(weights > 0, biomass_per_school / weights, 0.0)

    # Random placement on grid
    cell_x = rng.integers(0, max(1, grid.nx), size=total_schools).astype(np.int32)
    cell_y = rng.integers(0, max(1, grid.ny), size=total_schools).astype(np.int32)

    state = state.replace(
        length=lengths,
        weight=weights,
        abundance=abundance,
        biomass=biomass_per_school,
        cell_x=cell_x,
        cell_y=cell_y,
        is_egg=np.ones(total_schools, dtype=np.bool_),
    )
    return state


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
    outputs: list[StepOutput] = []

    for step in range(config.n_steps):
        state = _incoming_flux(state, config, step, rng)
        state = _reset_step_variables(state)
        resources.update(step)
        state = _movement(state, grid, config, step, rng)
        state = _mortality(state, resources, config, rng, grid)
        state = _growth(state, config, rng)
        state = _aging_mortality(state, config)
        state = _reproduction(state, config, step, rng)
        outputs.append(_collect_outputs(state, config, step))
        state = state.compact()

    return outputs
