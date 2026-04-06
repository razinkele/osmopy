# osmose/engine/economics/choice.py
"""DSVM discrete choice model: multinomial logit for vessel location decisions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from osmose.engine.economics.fleet import FleetState


def logit_probabilities(
    values: NDArray[np.float64],
    beta: float,
) -> NDArray[np.float64]:
    """Multinomial logit probabilities.

    P(i) = exp(β × V(i)) / Σ exp(β × V(j))

    Uses log-sum-exp trick for numerical stability.
    When β=0, returns uniform probabilities.
    """
    if beta == 0.0:
        n = len(values)
        return np.full(n, 1.0 / n)

    scaled = beta * values
    max_v = scaled.max()
    exp_v = np.exp(scaled - max_v)
    return exp_v / exp_v.sum()


def aggregate_effort(
    vessel_fleet: NDArray[np.int32],
    vessel_cell_y: NDArray[np.int32],
    vessel_cell_x: NDArray[np.int32],
    n_fleets: int,
    ny: int,
    nx: int,
) -> NDArray[np.float64]:
    """Count vessels per cell per fleet → effort map (n_fleets, ny, nx)."""
    effort = np.zeros((n_fleets, ny, nx), dtype=np.float64)
    for i in range(len(vessel_fleet)):
        fi = vessel_fleet[i]
        cy = vessel_cell_y[i]
        cx = vessel_cell_x[i]
        if 0 <= cy < ny and 0 <= cx < nx:
            effort[fi, cy, cx] += 1.0
    return effort


def fleet_decision(
    fleet_state: FleetState,
    biomass_by_cell_species: NDArray[np.float64],
    rng: np.random.Generator,
) -> FleetState:
    """Execute DSVM decision for all vessels: compute expected revenue per cell, apply logit.

    MVP: Revenue-only (no travel costs, no memory blending).
    """
    n_species, ny, nx = biomass_by_cell_species.shape
    n_cells = ny * nx

    for fi, fleet in enumerate(fleet_state.fleets):
        revenue_map = np.zeros(n_cells, dtype=np.float64)
        for sp in fleet.target_species:
            if sp < n_species:
                revenue_map += biomass_by_cell_species[sp].ravel() * fleet.price_per_tonne[sp]

        values = np.append(revenue_map, 0.0)  # last element = port
        probs = logit_probabilities(values, fleet_state.rationality)

        vessel_mask = fleet_state.vessel_fleet == fi
        vessel_indices = np.where(vessel_mask)[0]

        for vi in vessel_indices:
            choice = rng.choice(len(values), p=probs)
            if choice == n_cells:
                fleet_state.vessel_cell_y[vi] = fleet.home_port_y
                fleet_state.vessel_cell_x[vi] = fleet.home_port_x
            else:
                fleet_state.vessel_cell_y[vi] = choice // nx
                fleet_state.vessel_cell_x[vi] = choice % nx

    fleet_state.effort_map = aggregate_effort(
        fleet_state.vessel_fleet,
        fleet_state.vessel_cell_y,
        fleet_state.vessel_cell_x,
        n_fleets=len(fleet_state.fleets),
        ny=ny,
        nx=nx,
    )

    return fleet_state
