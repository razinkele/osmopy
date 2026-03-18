"""Movement process functions for the OSMOSE Python engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.state import SchoolState


def random_walk(
    state: SchoolState,
    grid: Grid,
    walk_range: NDArray[np.int32],
    rng: np.random.Generator,
) -> SchoolState:
    """Move schools by random displacement within walk_range cells.

    Displacement clamped to grid bounds; land cells avoided.
    """
    if len(state) == 0:
        return state

    dx = np.array([rng.integers(-int(r), int(r) + 1) for r in walk_range], dtype=np.int32)
    dy = np.array([rng.integers(-int(r), int(r) + 1) for r in walk_range], dtype=np.int32)

    new_x = np.clip(state.cell_x + dx, 0, grid.nx - 1)
    new_y = np.clip(state.cell_y + dy, 0, grid.ny - 1)

    # Avoid land cells: stay at old position if new is land
    on_land = ~grid.ocean_mask[new_y, new_x]
    new_x = np.where(on_land, state.cell_x, new_x)
    new_y = np.where(on_land, state.cell_y, new_y)

    return state.replace(cell_x=new_x, cell_y=new_y)


def movement(
    state: SchoolState,
    grid: Grid,
    config: EngineConfig,
    step: int,
    rng: np.random.Generator,
) -> SchoolState:
    """Move all schools according to their species' movement method."""
    if len(state) == 0:
        return state

    sp = state.species_id
    walk_range = config.random_walk_range[sp]
    state = random_walk(state, grid, walk_range, rng)

    return state
