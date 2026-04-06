# tests/test_economics_days.py
"""Tests for days-at-sea tracking and forced port return."""

import numpy as np

from osmose.engine.economics.choice import fleet_decision
from osmose.engine.economics.fleet import FleetConfig, create_fleet_state


class TestDaysAtSea:
    def _make_fleet(self, max_days: int = 5) -> FleetConfig:
        return FleetConfig(
            name="Trawlers",
            n_vessels=1,
            home_port_y=0,
            home_port_x=0,
            gear_type="bottom_trawl",
            max_days_at_sea=max_days,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0]),
        )

    def test_days_increment_when_fishing(self):
        """Days used should increment when vessel goes to a fishing cell."""
        fleet = self._make_fleet(max_days=100)
        fs = create_fleet_state([fleet], grid_ny=3, grid_nx=3, rationality=1.0)
        biomass = np.zeros((1, 3, 3), dtype=np.float64)
        biomass[0, 1, 1] = 10000.0  # fish at (1,1)

        rng = np.random.default_rng(42)
        fs = fleet_decision(fs, biomass, rng)
        # Vessel should have gone fishing → days_used incremented
        if fs.vessel_cell_y[0] != 0 or fs.vessel_cell_x[0] != 0:
            assert fs.vessel_days_used[0] == 1

    def test_forced_port_at_limit(self):
        """Vessel at days-at-sea limit should be forced to port."""
        fleet = self._make_fleet(max_days=0)  # already at limit
        fs = create_fleet_state([fleet], grid_ny=3, grid_nx=3, rationality=1.0)
        fs.vessel_days_used[0] = 0  # At limit (max_days=0)

        biomass = np.zeros((1, 3, 3), dtype=np.float64)
        biomass[0, 1, 1] = 10000.0

        rng = np.random.default_rng(42)
        fs = fleet_decision(fs, biomass, rng)
        # Should be at home port
        assert fs.vessel_cell_y[0] == fleet.home_port_y
        assert fs.vessel_cell_x[0] == fleet.home_port_x
