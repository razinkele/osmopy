"""Mortality orchestrator for the OSMOSE Python engine.

Implements Java's interleaved mortality loop: per sub-timestep,
mortality causes are shuffled per school-slot, with each cause
targeting a different school from its own pre-shuffled ordering.
"""

from __future__ import annotations

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.fishing import fishing_mortality
from osmose.engine.processes.natural import (
    additional_mortality,
    larva_mortality,
    out_mortality,
)
from osmose.engine.processes.predation import predation
from osmose.engine.processes.starvation import starvation_mortality, update_starvation_rate
from osmose.engine.resources import ResourceState
from osmose.engine.state import SchoolState


def mortality(
    state: SchoolState,
    resources: ResourceState,
    config: EngineConfig,
    rng: np.random.Generator,
    grid: Grid,
) -> SchoolState:
    """Apply all mortality sources with interleaved ordering.

    Matches Java's MortalityProcess structure:
    1. Pre-pass: larva mortality on eggs
    2. Retain eggs (withheld from prey pool)
    3. Per sub-timestep:
       a. Release fraction of eggs
       b. Shuffle cause order per sub-timestep
       c. Apply: predation -> then starvation, fishing, additional in shuffled order
    4. Post-loop: out-of-domain mortality, starvation rate update

    Note: Full per-school cause shuffling (Java's inner loop) is approximated
    by shuffling cause order per sub-timestep. Predation runs first because it
    operates on cell groupings and modifies prey biomass asynchronously.
    The other three causes are applied in random order after predation.
    """
    n_subdt = config.mortality_subdt

    # Pre-pass: larva mortality on eggs
    state = larva_mortality(state, config)

    # Retain eggs: withheld from prey pool
    egg_retained = np.where(state.is_egg, state.abundance, 0.0)
    state = state.replace(egg_retained=egg_retained)

    # Mortality cause functions (excluding predation which runs separately)
    cause_fns = [
        lambda s: starvation_mortality(s, config, n_subdt),
        lambda s: fishing_mortality(s, config, n_subdt),
        lambda s: additional_mortality(s, config, n_subdt),
    ]

    for _sub in range(n_subdt):
        # Release fraction of eggs into prey pool
        release = np.where(
            state.is_egg & (state.egg_retained > 0),
            state.abundance / n_subdt,
            0.0,
        )
        new_retained = np.maximum(0, state.egg_retained - release)
        state = state.replace(egg_retained=new_retained)

        # Predation first (operates on cell groupings)
        state = predation(state, config, rng, n_subdt, grid.ny, grid.nx, resources=resources)

        # Shuffle remaining causes per sub-timestep
        order = rng.permutation(len(cause_fns))
        for idx in order:
            state = cause_fns[idx](state)

    # Out-of-domain mortality (after main loop)
    state = out_mortality(state, config)

    # Compute new starvation rate for NEXT step (lagged)
    state = update_starvation_rate(state, config)

    # Update trophic level from predation
    # TL = 1 + weighted average of prey TLs (using preyed_biomass as weight)
    # For Phase 1: simple formula using species-level TL
    # Full diet tracking deferred to output phase
    mask = state.preyed_biomass > 0
    if mask.any():
        # Default prey TL = 1.0 (plankton base)
        # Schools that ate get TL = 1 + 1 = 2 (minimum consumer)
        new_tl = state.trophic_level.copy()
        new_tl[mask] = 2.0  # placeholder until full diet tracking
        state = state.replace(trophic_level=new_tl)

    return state
