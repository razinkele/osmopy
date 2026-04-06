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


def update_catch_memory(
    memory: NDArray[np.float64],
    realized_catch: NDArray[np.float64],
    decay: float,
) -> NDArray[np.float64]:
    """Exponential moving average update for catch memory.

    memory = decay × memory + (1 - decay) × realized_catch
    """
    return decay * memory + (1.0 - decay) * realized_catch


def fleet_decision(
    fleet_state: FleetState,
    biomass_by_cell_species: NDArray[np.float64],
    rng: np.random.Generator,
) -> FleetState:
    """Execute DSVM decision with full cost model and memory.

    V(c) = expected_revenue(c) - travel_cost(c) - operating_cost
    Biomass estimate = (1-decay) × observed + decay × memory
    """
    from osmose.engine.economics.costs import compute_expected_revenue, compute_travel_costs

    n_species, ny, nx = biomass_by_cell_species.shape
    n_cells = ny * nx

    for fi, fleet in enumerate(fleet_state.fleets):
        # Blend total biomass with catch memory for this fleet
        biomass_total = biomass_by_cell_species.sum(axis=0)  # (ny, nx)
        memory_layer = fleet_state.catch_memory[fi]
        blended_total = (
            1.0 - fleet_state.memory_decay
        ) * biomass_total + fleet_state.memory_decay * memory_layer
        # Scale per-species biomass proportionally by blend factor
        safe_total = np.where(biomass_total > 0, biomass_total, 1.0)
        scale = np.where(biomass_total > 0, blended_total / safe_total, 1.0)
        blended_species = biomass_by_cell_species * scale[np.newaxis, :, :]

        # Ref biomass: total observed biomass per species (across all cells)
        ref_biomass = np.maximum(biomass_by_cell_species.sum(axis=(1, 2)), 1.0)

        # Expected revenue with stock-dependent catchability
        revenue = compute_expected_revenue(
            blended_species,
            fleet.price_per_tonne,
            fleet.stock_elasticity,
            fleet.target_species,
            ref_biomass,
        )

        vessel_mask = fleet_state.vessel_fleet == fi
        vessel_indices = np.where(vessel_mask)[0]

        for vi in vessel_indices:
            # Skip vessels that exceeded days-at-sea — force to port
            if fleet_state.vessel_days_used[vi] >= fleet.max_days_at_sea:
                fleet_state.vessel_cell_y[vi] = fleet.home_port_y
                fleet_state.vessel_cell_x[vi] = fleet.home_port_x
                continue

            # Costs from current position
            travel = compute_travel_costs(
                int(fleet_state.vessel_cell_y[vi]),
                int(fleet_state.vessel_cell_x[vi]),
                ny,
                nx,
                fleet.fuel_cost_per_cell,
            )
            total_cost = travel + fleet.base_operating_cost

            # V(c) = revenue - cost; V(port) = 0
            profit = revenue - total_cost
            values = np.append(profit, 0.0)
            probs = logit_probabilities(values, fleet_state.rationality)

            choice = rng.choice(len(values), p=probs)
            if choice == n_cells:
                fleet_state.vessel_cell_y[vi] = fleet.home_port_y
                fleet_state.vessel_cell_x[vi] = fleet.home_port_x
            else:
                cy = choice // nx
                cx = choice % nx
                fleet_state.vessel_cell_y[vi] = cy
                fleet_state.vessel_cell_x[vi] = cx
                fleet_state.vessel_days_used[vi] += 1
                fleet_state.vessel_costs[vi] += travel[choice] + fleet.base_operating_cost

    fleet_state.effort_map = aggregate_effort(
        fleet_state.vessel_fleet,
        fleet_state.vessel_cell_y,
        fleet_state.vessel_cell_x,
        n_fleets=len(fleet_state.fleets),
        ny=ny,
        nx=nx,
    )

    return fleet_state
