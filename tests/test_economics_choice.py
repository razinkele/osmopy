# tests/test_economics_choice.py
"""Tests for DSVM discrete choice and effort aggregation."""

import numpy as np
import pytest

from osmose.engine.economics.choice import aggregate_effort, fleet_decision, logit_probabilities
from osmose.engine.economics.fleet import FleetConfig, FleetState, create_fleet_state


class TestLogitProbabilities:
    def test_uniform_when_beta_zero(self):
        """β=0 → uniform probability across all cells."""
        values = np.array([10.0, 20.0, 5.0, 0.0])
        probs = logit_probabilities(values, beta=0.0)
        assert probs.shape == (4,)
        assert np.allclose(probs, 0.25)

    def test_deterministic_when_beta_large(self):
        """Large β → probability concentrated on highest-value cell."""
        values = np.array([10.0, 100.0, 5.0, 0.0])
        probs = logit_probabilities(values, beta=50.0)
        assert probs[1] > 0.99

    def test_probabilities_sum_to_one(self):
        values = np.array([1.0, 2.0, 3.0, 0.0])
        probs = logit_probabilities(values, beta=1.0)
        assert np.sum(probs) == pytest.approx(1.0)


class TestAggregateEffort:
    def test_counts_vessels_per_cell(self):
        vessel_fleet = np.array([0, 0, 0], dtype=np.int32)
        vessel_cell_y = np.array([0, 0, 1], dtype=np.int32)
        vessel_cell_x = np.array([1, 1, 0], dtype=np.int32)
        effort = aggregate_effort(vessel_fleet, vessel_cell_y, vessel_cell_x, n_fleets=1, ny=2, nx=2)
        assert effort.shape == (1, 2, 2)
        assert effort[0, 0, 1] == 2.0
        assert effort[0, 1, 0] == 1.0
        assert effort[0, 0, 0] == 0.0
        assert effort[0, 1, 1] == 0.0


class TestFleetDecision:
    def _make_fleet_and_state(self, n_vessels: int = 20) -> tuple[FleetConfig, FleetState]:
        fleet = FleetConfig(
            name="Trawlers",
            n_vessels=n_vessels,
            home_port_y=0,
            home_port_x=0,
            gear_type="bottom_trawl",
            max_days_at_sea=200,
            fuel_cost_per_cell=0.0,
            base_operating_cost=0.0,
            stock_elasticity=np.array([0.0]),
            target_species=[0],
            price_per_tonne=np.array([1000.0]),
        )
        state = create_fleet_state([fleet], grid_ny=3, grid_nx=3, rationality=1.0)
        return fleet, state

    def test_vessels_move_to_fish(self):
        """With high rationality, vessels should concentrate where biomass is highest."""
        fleet, fs = self._make_fleet_and_state(n_vessels=100)

        biomass_by_cell = np.zeros((3, 3), dtype=np.float64)
        biomass_by_cell[1, 1] = 1000.0

        rng = np.random.default_rng(42)
        fs = fleet_decision(
            fleet_state=fs,
            biomass_by_cell_species=biomass_by_cell.reshape(1, 3, 3),
            rng=rng,
        )
        at_target = np.sum((fs.vessel_cell_y == 1) & (fs.vessel_cell_x == 1))
        assert at_target > 50

    def test_port_option_chosen_when_no_fish(self):
        """When no biomass anywhere, vessels should stay at port (home)."""
        fleet, fs = self._make_fleet_and_state(n_vessels=50)
        biomass_by_cell = np.zeros((1, 3, 3), dtype=np.float64)
        rng = np.random.default_rng(42)
        fs = fleet_decision(fleet_state=fs, biomass_by_cell_species=biomass_by_cell, rng=rng)
        assert fs.effort_map.sum() == pytest.approx(50.0)
