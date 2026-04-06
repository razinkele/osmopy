# osmose/engine/economics/costs.py
"""Cost and revenue calculations for DSVM fleet dynamics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_travel_costs(
    current_y: int,
    current_x: int,
    ny: int,
    nx: int,
    fuel_cost_per_cell: float,
) -> NDArray[np.float64]:
    """Compute travel cost from current position to every cell (Manhattan distance).

    Returns flat array of shape (ny * nx,).
    """
    ys = np.arange(ny, dtype=np.float64)
    xs = np.arange(nx, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    dist = np.abs(grid_y - current_y) + np.abs(grid_x - current_x)
    return (dist * fuel_cost_per_cell).ravel()


def compute_expected_revenue(
    biomass_by_cell: NDArray[np.float64],
    price: NDArray[np.float64],
    elasticity: NDArray[np.float64],
    target_species: list[int],
    ref_biomass: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute expected revenue per cell with stock-dependent catchability.

    Args:
        biomass_by_cell: Shape (n_species, ny, nx).
        price: Per-species price, shape (n_species,).
        elasticity: Stock elasticity, shape (n_species,).
        target_species: Species indices this fleet targets.
        ref_biomass: Reference biomass for catchability scaling, shape (n_species,).

    Returns:
        Revenue per cell, flat array shape (ny * nx,).
    """
    n_species, ny, nx = biomass_by_cell.shape
    n_cells = ny * nx
    revenue = np.zeros(n_cells, dtype=np.float64)

    for sp in target_species:
        if sp >= n_species:
            continue
        bio_flat = biomass_by_cell[sp].ravel()
        ref = max(ref_biomass[sp], 1e-20)
        catchability = np.power(np.maximum(bio_flat / ref, 0.0), elasticity[sp])
        revenue += catchability * bio_flat * price[sp]

    return revenue
