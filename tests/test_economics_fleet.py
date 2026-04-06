# tests/test_economics_fleet.py
"""Tests for FleetConfig, FleetState, and config parsing."""

import numpy as np
import pytest

from osmose.engine.economics.fleet import FleetConfig, create_fleet_state, parse_fleets


class TestParseFleets:
    def test_single_fleet(self):
        cfg = {
            "simulation.economic.enabled": "true",
            "economic.fleet.number": "1",
            "economic.fleet.name.fsh0": "Trawlers",
            "economic.fleet.nvessels.fsh0": "10",
            "economic.fleet.homeport.y.fsh0": "2",
            "economic.fleet.homeport.x.fsh0": "3",
            "economic.fleet.gear.fsh0": "bottom_trawl",
            "economic.fleet.max.days.fsh0": "200",
            "economic.fleet.fuel.cost.fsh0": "500.0",
            "economic.fleet.operating.cost.fsh0": "1000.0",
            "economic.fleet.target.species.fsh0": "0,1",
            "economic.fleet.price.sp0.fsh0": "2500.0",
            "economic.fleet.price.sp1.fsh0": "1800.0",
            "economic.fleet.stock.elasticity.sp0.fsh0": "0.5",
            "economic.fleet.stock.elasticity.sp1.fsh0": "0.3",
        }
        fleets = parse_fleets(cfg, n_species=2)
        assert len(fleets) == 1
        f = fleets[0]
        assert f.name == "Trawlers"
        assert f.n_vessels == 10
        assert f.home_port_y == 2
        assert f.home_port_x == 3
        assert f.target_species == [0, 1]
        assert f.price_per_tonne[0] == pytest.approx(2500.0)
        assert f.stock_elasticity[1] == pytest.approx(0.3)

    def test_empty_when_disabled(self):
        fleets = parse_fleets({}, n_species=2)
        assert len(fleets) == 0


class TestCreateFleetState:
    def test_vessels_start_at_home_port(self):
        fleet = FleetConfig(
            name="Trawlers",
            n_vessels=5,
            home_port_y=2,
            home_port_x=3,
            gear_type="bottom_trawl",
            max_days_at_sea=200,
            fuel_cost_per_cell=500.0,
            base_operating_cost=1000.0,
            stock_elasticity=np.array([0.5]),
            target_species=[0],
            price_per_tonne=np.array([2500.0]),
        )
        state = create_fleet_state(
            fleets=[fleet],
            grid_ny=5,
            grid_nx=5,
            rationality=1.0,
            memory_decay=0.7,
        )
        assert len(state.vessel_fleet) == 5
        assert np.all(state.vessel_cell_y == 2)
        assert np.all(state.vessel_cell_x == 3)
        assert np.all(state.vessel_days_used == 0)
        assert state.effort_map.shape == (1, 5, 5)
        assert state.rationality == pytest.approx(1.0)
