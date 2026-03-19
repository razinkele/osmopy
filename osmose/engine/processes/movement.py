"""Movement process functions for the OSMOSE Python engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.movement_maps import MovementMapSet
from osmose.engine.state import SchoolState


def _map_move_school(
    age_dt: int,
    cx: int,
    cy: int,
    grid_ny: int,
    grid_nx: int,
    ocean_mask: NDArray[np.bool_],
    map_set: MovementMapSet,
    walk_range: int,
    step: int,
    rng: np.random.Generator,
) -> tuple[int, int, bool]:
    """Move a single school using map-based distribution.

    Returns (new_x, new_y, is_out).

    Parameters
    ----------
    age_dt : int
        School age in time-step units.
    cx, cy : int
        Current cell coordinates (-1 if unlocated).
    grid_ny, grid_nx : int
        Grid dimensions.
    ocean_mask : NDArray[np.bool_]
        Boolean mask of ocean cells (True = ocean).
    map_set : MovementMapSet
        Pre-loaded movement maps for this species.
    walk_range : int
        Maximum random-walk displacement (cells).
    step : int
        Global simulation step.
    rng : np.random.Generator
        Random number generator.
    """
    # Step 1 — Out-of-domain check
    current_map = map_set.get_map(age_dt, step)
    if current_map is None:
        return -1, -1, True

    # Step 2 — Same-map detection
    index_map = map_set.get_index(age_dt, step)
    same_map = False
    if age_dt > 0 and step > 0:
        prev_index = map_set.get_index(age_dt - 1, step - 1)
        same_map = (index_map == prev_index)

    # Step 3a — New placement (rejection sampling)
    if not same_map or cx < 0:
        n_cells = grid_nx * grid_ny
        max_p = map_set.max_proba[index_map]
        for _ in range(10_000):
            flat_idx = int(round((n_cells - 1) * rng.random()))
            j = flat_idx // grid_nx
            i = flat_idx % grid_nx
            proba = current_map[j, i]
            if proba > 0 and not np.isnan(proba):
                if max_p == 0.0 or proba >= rng.random() * max_p:
                    return i, j, False
        raise RuntimeError("Map placement failed after 10000 attempts")

    # Step 3b — Random walk (same map, school is located)
    accessible: list[tuple[int, int]] = []
    for yi in range(max(0, cy - walk_range), min(grid_ny, cy + walk_range + 1)):
        for xi in range(max(0, cx - walk_range), min(grid_nx, cx + walk_range + 1)):
            if ocean_mask[yi, xi] and current_map[yi, xi] > 0 and not np.isnan(current_map[yi, xi]):
                accessible.append((xi, yi))
    if len(accessible) == 0:
        return cx, cy, False  # stranded — stay in place
    idx = int(round((len(accessible) - 1) * rng.random()))
    return accessible[idx][0], accessible[idx][1], False


def build_random_patches(
    config: EngineConfig,
    grid: Grid,
    rng: np.random.Generator,
) -> dict[int, set[tuple[int, int]]]:
    """Build BFS-connected patches for species with random distribution + ncell constraint.

    Java's RandomDistribution builds a connected patch of ncell ocean cells
    starting from a random ocean cell. The patch constrains INITIAL placement only.

    Returns a dict mapping species index -> set of (x, y) patch cells.
    """
    patches: dict[int, set[tuple[int, int]]] = {}
    if config.random_distribution_ncell is None:
        return patches

    # Build list of ocean cells
    ocean_cells: list[tuple[int, int]] = []
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.ocean_mask[y, x]:
                ocean_cells.append((x, y))

    n_ocean = len(ocean_cells)
    if n_ocean == 0:
        return patches

    for sp in range(config.n_species):
        if config.movement_method[sp] != "random":
            continue
        ncell = int(config.random_distribution_ncell[sp])
        if ncell <= 0 or ncell >= n_ocean:
            continue

        # Pick random starting ocean cell
        start_idx = rng.integers(0, n_ocean)
        start_x, start_y = ocean_cells[start_idx]

        # BFS to collect ncell connected ocean cells
        patch: set[tuple[int, int]] = set()
        queue: list[tuple[int, int]] = [(start_x, start_y)]
        patch.add((start_x, start_y))

        while len(patch) < ncell and queue:
            cx, cy = queue.pop(0)
            for ny, nx in grid.neighbors(cy, cx):
                if len(patch) >= ncell:
                    break
                if grid.ocean_mask[ny, nx] and (nx, ny) not in patch:
                    patch.add((nx, ny))
                    queue.append((nx, ny))

        patches[sp] = patch

    return patches


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

    # Batch RNG: generate max-range displacements, then clip to per-school range
    max_r = int(walk_range.max()) if len(walk_range) > 0 else 0
    if max_r == 0:
        return state
    dx = rng.integers(-max_r, max_r + 1, size=len(state)).astype(np.int32)
    dy = rng.integers(-max_r, max_r + 1, size=len(state)).astype(np.int32)
    dx = np.clip(dx, -walk_range, walk_range)
    dy = np.clip(dy, -walk_range, walk_range)

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
    map_sets: dict[int, MovementMapSet] | None = None,
    random_patches: dict[int, set[tuple[int, int]]] | None = None,
) -> SchoolState:
    """Move all schools according to their species' movement method."""
    if len(state) == 0:
        return state

    sp = state.species_id

    # Determine which schools use which method
    uses_random = np.array([config.movement_method[s] == "random" for s in sp])
    uses_maps = np.array([config.movement_method[s] == "maps" for s in sp])

    # Phase 4: Place unlocated schools on patch cells (before random walk)
    if random_patches and uses_random.any():
        new_cx = state.cell_x.copy()
        new_cy = state.cell_y.copy()
        changed = False
        for i in np.where(uses_random)[0]:
            sp_id = int(sp[i])
            if sp_id in random_patches and new_cx[i] < 0:
                # Unlocated school: place on random patch cell
                patch_cells = list(random_patches[sp_id])
                idx = rng.integers(0, len(patch_cells))
                new_cx[i], new_cy[i] = patch_cells[idx]
                changed = True
        if changed:
            state = state.replace(cell_x=new_cx, cell_y=new_cy)

    # Random walk for "random" species (batch vectorized)
    if uses_random.any():
        walk_range = config.random_walk_range[sp]
        walk_range_masked = np.where(uses_random, walk_range, 0)
        state = random_walk(state, grid, walk_range_masked, rng)

    # Map-based movement for "maps" species (per-school scalar loop)
    if uses_maps.any() and map_sets is not None:
        new_cx = state.cell_x.copy()
        new_cy = state.cell_y.copy()
        new_out = state.is_out.copy()
        for i in np.where(uses_maps)[0]:
            sp_id = int(sp[i])
            if sp_id in map_sets:
                x, y, out = _map_move_school(
                    int(state.age_dt[i]),
                    int(new_cx[i]),
                    int(new_cy[i]),
                    grid.ny,
                    grid.nx,
                    grid.ocean_mask,
                    map_sets[sp_id],
                    int(config.random_walk_range[sp_id]),
                    step,
                    rng,
                )
                new_cx[i], new_cy[i], new_out[i] = x, y, out
        state = state.replace(cell_x=new_cx, cell_y=new_cy, is_out=new_out)

    return state
