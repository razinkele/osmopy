# osmose/engine/economics/fleet.py
"""Fleet configuration, state, and config parsing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FleetConfig:
    """Immutable per-fleet configuration."""

    name: str
    n_vessels: int
    home_port_y: int
    home_port_x: int
    gear_type: str
    max_days_at_sea: int
    fuel_cost_per_cell: float
    base_operating_cost: float
    stock_elasticity: NDArray[np.float64]  # shape (n_species,)
    target_species: list[int]
    price_per_tonne: NDArray[np.float64]  # shape (n_species,)


@dataclass
class FleetState:
    """Mutable per-simulation fleet state."""

    fleets: list[FleetConfig]
    vessel_fleet: NDArray[np.int32]
    vessel_cell_y: NDArray[np.int32]
    vessel_cell_x: NDArray[np.int32]
    vessel_days_used: NDArray[np.int32]
    vessel_revenue: NDArray[np.float64]
    vessel_costs: NDArray[np.float64]
    effort_map: NDArray[np.float64]
    catch_memory: NDArray[np.float64]
    memory_decay: float
    rationality: float


def parse_fleets(cfg: dict[str, str], n_species: int) -> list[FleetConfig]:
    """Parse fleet definitions from OSMOSE config."""
    n_fleets = int(cfg.get("economic.fleet.number", "0"))
    if n_fleets == 0:
        return []

    fleets: list[FleetConfig] = []
    for fi in range(n_fleets):
        fid = f"fsh{fi}"
        prefix = "economic.fleet"

        name = cfg.get(f"{prefix}.name.{fid}", f"Fleet{fi}")
        n_vessels = int(cfg.get(f"{prefix}.nvessels.{fid}", "1"))
        home_y = int(cfg.get(f"{prefix}.homeport.y.{fid}", "0"))
        home_x = int(cfg.get(f"{prefix}.homeport.x.{fid}", "0"))
        gear = cfg.get(f"{prefix}.gear.{fid}", "generic")
        max_days = int(cfg.get(f"{prefix}.max.days.{fid}", "200"))
        fuel_cost = float(cfg.get(f"{prefix}.fuel.cost.{fid}", "0.0"))
        op_cost = float(cfg.get(f"{prefix}.operating.cost.{fid}", "0.0"))

        target_str = cfg.get(f"{prefix}.target.species.{fid}", "")
        target_species = [int(s.strip()) for s in target_str.split(",") if s.strip()]

        price = np.array(
            [float(cfg.get(f"{prefix}.price.sp{sp}.{fid}", "0.0")) for sp in range(n_species)]
        )
        elasticity = np.array(
            [
                float(cfg.get(f"{prefix}.stock.elasticity.sp{sp}.{fid}", "0.0"))
                for sp in range(n_species)
            ]
        )

        fleets.append(
            FleetConfig(
                name=name,
                n_vessels=n_vessels,
                home_port_y=home_y,
                home_port_x=home_x,
                gear_type=gear,
                max_days_at_sea=max_days,
                fuel_cost_per_cell=fuel_cost,
                base_operating_cost=op_cost,
                stock_elasticity=elasticity,
                target_species=target_species,
                price_per_tonne=price,
            )
        )

    return fleets


def create_fleet_state(
    fleets: list[FleetConfig],
    grid_ny: int,
    grid_nx: int,
    rationality: float = 1.0,
    memory_decay: float = 0.7,
) -> FleetState:
    """Initialize fleet state with all vessels at their home ports."""
    total_vessels = sum(f.n_vessels for f in fleets)
    n_fleets = len(fleets)

    vessel_fleet = np.empty(total_vessels, dtype=np.int32)
    vessel_cell_y = np.empty(total_vessels, dtype=np.int32)
    vessel_cell_x = np.empty(total_vessels, dtype=np.int32)

    offset = 0
    for fi, fleet in enumerate(fleets):
        end = offset + fleet.n_vessels
        vessel_fleet[offset:end] = fi
        vessel_cell_y[offset:end] = fleet.home_port_y
        vessel_cell_x[offset:end] = fleet.home_port_x
        offset = end

    return FleetState(
        fleets=fleets,
        vessel_fleet=vessel_fleet,
        vessel_cell_y=vessel_cell_y,
        vessel_cell_x=vessel_cell_x,
        vessel_days_used=np.zeros(total_vessels, dtype=np.int32),
        vessel_revenue=np.zeros(total_vessels, dtype=np.float64),
        vessel_costs=np.zeros(total_vessels, dtype=np.float64),
        effort_map=np.zeros((n_fleets, grid_ny, grid_nx), dtype=np.float64),
        catch_memory=np.zeros((n_fleets, grid_ny, grid_nx), dtype=np.float64),
        memory_decay=memory_decay,
        rationality=rationality,
    )
