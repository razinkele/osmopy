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
    """Phase 1 stub: spatial movement."""
    return state


def _mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
) -> SchoolState:
    """Apply mortality sources. Phase 2: only additional mortality."""
    from osmose.engine.processes.natural import additional_mortality

    for _sub in range(config.mortality_subdt):
        state = additional_mortality(state, config, config.mortality_subdt)
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
    """Phase 1 stub: egg production + age increment."""
    return state


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
    """Create the initial population of schools.

    Phase 1: creates empty state. Phase 3 will add seeding/population init.
    """
    total_schools = int(config.n_schools.sum())
    species_ids = np.repeat(np.arange(config.n_species, dtype=np.int32), config.n_schools)
    return SchoolState.create(n_schools=total_schools, species_id=species_ids)


def simulate(
    config: EngineConfig,
    grid: Grid,
    rng: np.random.Generator,
) -> list[StepOutput]:
    """Run the OSMOSE simulation loop.

    Process ordering matches Java's SimulationStep.step().
    """
    state = initialize(config, grid, rng)
    resources = ResourceState(config={}, grid=grid)
    outputs: list[StepOutput] = []

    for step in range(config.n_steps):
        state = _incoming_flux(state, config, step, rng)
        state = _reset_step_variables(state)
        resources.update(step)
        state = _movement(state, grid, config, step, rng)
        state = _mortality(state, resources, config, rng)
        state = _growth(state, config, rng)
        state = _aging_mortality(state, config)
        state = _reproduction(state, config, step, rng)
        outputs.append(_collect_outputs(state, config, step))
        state = state.compact()

    return outputs
