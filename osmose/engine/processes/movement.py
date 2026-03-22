"""Movement process functions for the OSMOSE Python engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.movement_maps import MovementMapSet
from osmose.engine.state import SchoolState

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


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
    species_rngs: list[np.random.Generator] | None = None,
    flat_map_data=None,
) -> SchoolState:
    """Move all schools according to their species' movement method.

    Parameters
    ----------
    species_rngs : list of Generator, optional
        Per-species random number generators. When provided and
        ``config.movement_seed_fixed`` is True, each species uses its own
        independent RNG for patch placement and map-based movement.
        Falls back to the global ``rng`` when None or when
        ``movement_seed_fixed`` is False.
    """
    if len(state) == 0:
        return state

    sp = state.species_id

    # Determine which schools use which method
    uses_random = np.array([config.movement_method[s] == "random" for s in sp])
    uses_maps = np.array([config.movement_method[s] == "maps" for s in sp])

    def _rng_for(sp_id: int) -> np.random.Generator:
        """Return per-species RNG if available and fixed seeds enabled, else global."""
        if species_rngs is not None and config.movement_seed_fixed and sp_id < len(species_rngs):
            return species_rngs[sp_id]
        return rng

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
                _sp_rng = _rng_for(sp_id)
                idx = _sp_rng.integers(0, len(patch_cells))
                new_cx[i], new_cy[i] = patch_cells[idx]
                changed = True
        if changed:
            state = state.replace(cell_x=new_cx, cell_y=new_cy)

    # Random walk for "random" species (batch vectorized)
    if uses_random.any():
        walk_range = config.random_walk_range[sp]
        walk_range_masked = np.where(uses_random, walk_range, 0)
        state = random_walk(state, grid, walk_range_masked, rng)

    # Map-based movement for "maps" species
    if uses_maps.any() and map_sets is not None:
        if _HAS_NUMBA and flat_map_data is not None:
            flat_maps, flat_max_proba, flat_is_null, sp_offsets = flat_map_data
            map_school_indices = np.where(uses_maps)[0].astype(np.int32)
            current_idx, same_map_flags = _precompute_map_indices(
                state.species_id, state.age_dt, uses_maps, map_sets, step
            )
            rng_seed = int(rng.integers(0, 2**63))
            new_cx = state.cell_x.copy()
            new_cy = state.cell_y.copy()
            new_out = state.is_out.copy()
            _map_move_batch_numba(
                rng_seed,
                map_school_indices, current_idx, same_map_flags,
                new_cx, new_cy, state.species_id,
                flat_maps, flat_max_proba, flat_is_null, sp_offsets,
                grid.ocean_mask, config.random_walk_range.astype(np.int32),
                grid.ny, grid.nx,
                new_cx, new_cy, new_out,
            )
            state = state.replace(cell_x=new_cx, cell_y=new_cy, is_out=new_out)
        else:
            # Python fallback (per-school scalar loop)
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
                        _rng_for(sp_id),
                    )
                    new_cx[i], new_cy[i], new_out[i] = x, y, out
            state = state.replace(cell_x=new_cx, cell_y=new_cy, is_out=new_out)

    return state


if _HAS_NUMBA:

    @njit(cache=True)
    def _map_move_batch_numba(
        rng_seed,
        school_indices, current_map_idx, same_map,
        cell_x, cell_y, sp_ids,
        all_maps, all_max_proba, all_is_null, sp_map_offset,
        ocean_mask, walk_range,
        ny, nx,
        out_cx, out_cy, out_is_out,
    ):
        """Numba-compiled batch map-based movement for all schools."""
        np.random.seed(rng_seed)
        n_cells = ny * nx
        for k in range(len(school_indices)):
            idx = school_indices[k]
            sp = sp_ids[idx]
            map_idx = current_map_idx[k]

            global_map_idx = sp_map_offset[sp] + map_idx

            if map_idx < 0 or all_is_null[global_map_idx]:
                out_cx[idx] = -1
                out_cy[idx] = -1
                out_is_out[idx] = True
                continue

            current_map = all_maps[global_map_idx]
            max_p = all_max_proba[global_map_idx]

            if not same_map[k] or cell_x[idx] < 0:
                placed = False
                for _ in range(10_000):
                    flat_idx = int(round((n_cells - 1) * np.random.random()))
                    j = flat_idx // nx
                    i = flat_idx % nx
                    proba = current_map[j, i]
                    if proba > 0 and not np.isnan(proba):
                        if max_p == 0.0 or proba >= np.random.random() * max_p:
                            out_cx[idx] = i
                            out_cy[idx] = j
                            out_is_out[idx] = False
                            placed = True
                            break
                if not placed:
                    out_cx[idx] = -1
                    out_cy[idx] = -1
                    out_is_out[idx] = True
                continue

            cx_k = cell_x[idx]
            cy_k = cell_y[idx]
            wr = walk_range[sp]
            n_accessible = 0
            y_lo = max(0, cy_k - wr)
            y_hi = min(ny, cy_k + wr + 1)
            x_lo = max(0, cx_k - wr)
            x_hi = min(nx, cx_k + wr + 1)
            for yi in range(y_lo, y_hi):
                for xi in range(x_lo, x_hi):
                    if ocean_mask[yi, xi] and current_map[yi, xi] > 0:
                        if not np.isnan(current_map[yi, xi]):
                            n_accessible += 1

            if n_accessible == 0:
                out_cx[idx] = cx_k
                out_cy[idx] = cy_k
                out_is_out[idx] = False
                continue

            target = int(round((n_accessible - 1) * np.random.random()))
            count = 0
            for yi in range(y_lo, y_hi):
                for xi in range(x_lo, x_hi):
                    if ocean_mask[yi, xi] and current_map[yi, xi] > 0:
                        if not np.isnan(current_map[yi, xi]):
                            if count == target:
                                out_cx[idx] = xi
                                out_cy[idx] = yi
                                out_is_out[idx] = False
                            count += 1
                            if count > target:
                                break
                if count > target:
                    break


def _flatten_all_map_sets(
    map_sets: dict[int, MovementMapSet],
    n_species: int,
    ny: int,
    nx: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Stack all species' movement maps into contiguous arrays for Numba.

    Returns:
        all_maps: float64[total_maps, ny, nx] — stacked maps (None → zeros)
        all_max_proba: float64[total_maps] — max probability per map
        all_is_null: bool[total_maps] — True for None (out-of-domain) maps
        sp_map_offset: int32[n_species] — offset into all_maps per species (-1 if no maps)
    """
    total_maps = sum(len(ms.maps) for ms in map_sets.values())
    all_maps = np.zeros((total_maps, ny, nx), dtype=np.float64)
    all_max_proba = np.zeros(total_maps, dtype=np.float64)
    all_is_null = np.zeros(total_maps, dtype=np.bool_)
    sp_map_offset = np.full(n_species, -1, dtype=np.int32)

    pos = 0
    for sp, ms in map_sets.items():
        sp_map_offset[sp] = pos
        for k, m in enumerate(ms.maps):
            if m is not None:
                all_maps[pos + k] = m
            else:
                all_is_null[pos + k] = True
            all_max_proba[pos + k] = ms.max_proba[k]
        pos += len(ms.maps)

    return all_maps, all_max_proba, all_is_null, sp_map_offset


def _precompute_map_indices(
    species_id: NDArray[np.int32],
    age_dt: NDArray[np.int32],
    uses_maps: NDArray[np.bool_],
    map_sets: dict[int, MovementMapSet],
    step: int,
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    """Pre-compute per-school map indices and same-map flags.

    Returns:
        current_idx: int32[n_map_schools] — current map index (-1 if out of range)
        same_map: bool[n_map_schools] — True if same map as previous step
    """
    map_school_mask = np.where(uses_maps)[0]
    n = len(map_school_mask)
    current_idx = np.full(n, -1, dtype=np.int32)
    prev_idx = np.full(n, -1, dtype=np.int32)

    for k, i in enumerate(map_school_mask):
        sp = int(species_id[i])
        age = int(age_dt[i])
        if sp not in map_sets:
            continue
        ms = map_sets[sp]
        if 0 <= age < ms.index_maps.shape[0] and 0 <= step < ms.index_maps.shape[1]:
            current_idx[k] = ms.index_maps[age, step]
        prev_age = age - 1
        prev_step = step - 1
        if 0 <= prev_age < ms.index_maps.shape[0] and 0 <= prev_step < ms.index_maps.shape[1]:
            prev_idx[k] = ms.index_maps[prev_age, prev_step]

    same_map = (current_idx == prev_idx) & (age_dt[map_school_mask] > 0) & (step > 0)
    return current_idx, same_map
