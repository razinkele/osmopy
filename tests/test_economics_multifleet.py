# tests/test_economics_multifleet.py
"""Tests for multi-fleet operation and non-interference."""

import numpy as np
import pytest

from osmose.engine.economics.choice import fleet_decision
from osmose.engine.economics.fleet import FleetConfig, create_fleet_state


class TestMultiFleet:
    def test_two_fleets_different_targets(self):
        """Two fleets targeting different species should distribute independently."""
        fleet_a = FleetConfig(
            name="Trawlers",
            n_vessels=20,
            home_port_y=0,
            home_port_x=0,
            gear_type="bottom_trawl",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0, 0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0, 0.0]),
        )
        fleet_b = FleetConfig(
            name="Longliners",
            n_vessels=20,
            home_port_y=2,
            home_port_x=2,
            gear_type="longline",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0, 0.0]),
            target_species=[1],
            price_per_tonne=np.array([0.0, 2000.0]),
        )
        fs = create_fleet_state([fleet_a, fleet_b], grid_ny=3, grid_nx=3, rationality=5.0)

        # Species 0 at (0,0), species 1 at (2,2)
        biomass = np.zeros((2, 3, 3), dtype=np.float64)
        biomass[0, 0, 0] = 5000.0
        biomass[1, 2, 2] = 5000.0

        rng = np.random.default_rng(42)
        fs = fleet_decision(fs, biomass, rng)

        # Fleet A (trawlers) should concentrate near (0,0)
        trawler_mask = fs.vessel_fleet == 0
        trawler_at_00 = np.sum(
            (fs.vessel_cell_y[trawler_mask] == 0) & (fs.vessel_cell_x[trawler_mask] == 0)
        )
        assert trawler_at_00 > 10  # most of 20

        # Fleet B (longliners) should concentrate near (2,2)
        liner_mask = fs.vessel_fleet == 1
        liner_at_22 = np.sum(
            (fs.vessel_cell_y[liner_mask] == 2) & (fs.vessel_cell_x[liner_mask] == 2)
        )
        assert liner_at_22 > 10  # most of 20

    def test_effort_map_per_fleet(self):
        """Effort map should have separate layers per fleet."""
        fleet_a = FleetConfig(
            name="A",
            n_vessels=5,
            home_port_y=0,
            home_port_x=0,
            gear_type="a",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0]),
        )
        fleet_b = FleetConfig(
            name="B",
            n_vessels=3,
            home_port_y=1,
            home_port_x=1,
            gear_type="b",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0]),
        )
        fs = create_fleet_state([fleet_a, fleet_b], grid_ny=2, grid_nx=2, rationality=0.0)

        biomass = np.zeros((1, 2, 2))
        rng = np.random.default_rng(42)
        fs = fleet_decision(fs, biomass, rng)

        assert fs.effort_map.shape == (2, 2, 2)  # 2 fleets, 2x2 grid
        assert fs.effort_map[0].sum() == pytest.approx(5.0)  # fleet A: 5 vessels
        assert fs.effort_map[1].sum() == pytest.approx(3.0)  # fleet B: 3 vessels
