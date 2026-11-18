"""Run-setup helpers that turn a config into flat engine inputs, kept free of Numba.

The Python engine uses them at the start of a run (``simulate``). The C# engine's bundle builder
(``osmose.engine_cs.build``) calls the same functions, so both engines start from identical
inputs. They live here, not in ``processes.mortality`` / ``processes.movement`` (which re-export
them), because those modules import Numba at import time: a C# process that imported them for
these plain helpers paid 64 MB of Numba and LLVM it never used (measured 2026-11-18, task 1.11).
``tests/test_csharp_build_no_numba.py`` keeps it that way.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from osmose.engine.config import EngineConfig
    from osmose.engine.grid import Grid
    from osmose.engine.movement_maps import MovementMapSet

# Empty resource arrays handed to the kernels when a run has no resources.
_DUMMY_RSC_1D = np.zeros(0, dtype=np.float64)
_DUMMY_RSC_I1D = np.zeros(0, dtype=np.int32)


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


def build_random_patches(
    config: EngineConfig,
    grid: Grid,
    rng: np.random.Generator,
) -> dict[int, NDArray[np.int32]]:
    """Build the placement domain of every species that uses the "random" distribution.

    Mirrors Java's ``RandomDistribution.createRandomMap`` (4.3.3): when
    ``movement.distribution.ncell`` is unset, or covers at least every ocean cell, the
    domain is the WHOLE ocean and no random number is drawn; otherwise it is a
    BFS-connected patch of ``ncell`` ocean cells grown from a random ocean cell.

    The domain is where UNLOCATED schools (new eggs) are placed. Located schools
    random-walk beyond it, as in Java, whose ``getAccessibleCells`` ignores the domain.

    Returns a dict mapping species index -> ``(n_cells, 2)`` int32 array of ``(x, y)``
    cells. A species is absent only when the grid has no ocean cell at all.
    """
    domains: dict[int, NDArray[np.int32]] = {}
    ocean_ys, ocean_xs = np.nonzero(grid.ocean_mask)  # row-major: y outer, x inner
    n_ocean = len(ocean_xs)
    if n_ocean == 0:
        return domains
    whole_ocean = np.column_stack((ocean_xs, ocean_ys)).astype(np.int32)
    whole_ocean.setflags(write=False)  # one array shared by every whole-domain species

    for sp in range(config.n_species):
        if config.movement_method[sp] != "random":
            continue
        # 0 is the parser's placeholder for an unset key (Java: unset -> whole domain).
        ncell = (
            0
            if config.random_distribution_ncell is None
            else int(config.random_distribution_ncell[sp])
        )
        if ncell <= 0 or ncell >= n_ocean:
            domains[sp] = whole_ocean
            continue

        # Pick random starting ocean cell
        start_idx = int(rng.integers(0, n_ocean))
        start = (int(ocean_xs[start_idx]), int(ocean_ys[start_idx]))

        # BFS to collect ncell connected ocean cells (the list keeps insertion order)
        patch: list[tuple[int, int]] = [start]
        seen: set[tuple[int, int]] = {start}
        queue: deque[tuple[int, int]] = deque([start])

        while len(patch) < ncell and queue:
            cx, cy = queue.popleft()
            for ny, nx in grid.neighbors(cy, cx):
                if len(patch) >= ncell:
                    break
                if grid.ocean_mask[ny, nx] and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    patch.append((nx, ny))
                    queue.append((nx, ny))

        domains[sp] = np.array(patch, dtype=np.int32)

    return domains


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
        if sp >= n_species:
            raise ValueError(f"Movement map species ID {sp} exceeds n_species={n_species}")
        sp_map_offset[sp] = pos
        for k, m in enumerate(ms.maps):
            if m is not None:
                all_maps[pos + k] = m
            else:
                all_is_null[pos + k] = True
            all_max_proba[pos + k] = ms.max_proba[k]
        pos += len(ms.maps)

    return all_maps, all_max_proba, all_is_null, sp_map_offset


def _movement_salinity_weight(config, grid, step) -> NDArray[np.float64] | None:
    """Per-step salinity occupancy-weight grid for the gate, or None when off.

    Reads config.salinity_gate_enabled / .salinity_field / .salinity_gate_s_low /
    .salinity_gate_s_high and grid.ny/nx. Constant fields broadcast to the grid.
    """
    if not (config.salinity_gate_enabled and config.salinity_field is not None):
        return None
    from osmose.engine.processes.salinity_gate import salinity_weight

    if config.salinity_field.is_constant:
        S = np.full((grid.ny, grid.nx), config.salinity_field.get_scalar())
    else:
        S = config.salinity_field.get_grid(step)
    # S is always an ndarray here, so salinity_weight returns an ndarray.
    return cast(
        "NDArray[np.float64]",
        salinity_weight(S, config.salinity_gate_s_low, config.salinity_gate_s_high),
    )
