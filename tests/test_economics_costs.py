# tests/test_economics_costs.py
"""Tests for economic cost calculations."""

import numpy as np
import pytest

from osmose.engine.economics.costs import (
    compute_expected_revenue,
    compute_travel_costs,
)


class TestTravelCosts:
    def test_manhattan_distance(self):
        """Travel cost = Manhattan distance × fuel_cost_per_cell."""
        current_y, current_x = 2, 3
        ny, nx = 5, 5
        fuel_cost = 100.0
        costs = compute_travel_costs(current_y, current_x, ny, nx, fuel_cost)
        assert costs.shape == (ny * nx,)
        # Same cell → 0 cost
        assert costs[current_y * nx + current_x] == 0.0
        # Adjacent cell (2,4) → distance 1, cost 100
        assert costs[current_y * nx + 4] == pytest.approx(100.0)
        # Cell (0,0) → distance |2-0| + |3-0| = 5, cost 500
        assert costs[0] == pytest.approx(500.0)

    def test_zero_fuel_cost(self):
        costs = compute_travel_costs(0, 0, 3, 3, 0.0)
        assert np.all(costs == 0.0)


class TestExpectedRevenue:
    def test_revenue_with_catchability(self):
        """Revenue = Σ_sp catchability × biomass × price."""
        biomass_by_cell = np.array([[[100.0, 0.0], [50.0, 200.0]]])  # (1 species, 2, 2)
        price = np.array([10.0])
        elasticity = np.array([0.5])
        target_species = [0]
        ref_biomass = np.array([100.0])

        revenue = compute_expected_revenue(
            biomass_by_cell, price, elasticity, target_species, ref_biomass
        )
        assert revenue.shape == (4,)  # ny*nx = 2*2
        # Cell (0,0): biomass=100, catchability = (100/100)^0.5 = 1.0, revenue = 1*100*10 = 1000
        assert revenue[0] == pytest.approx(1000.0)
        # Cell (0,1): biomass=0, revenue = 0
        assert revenue[1] == pytest.approx(0.0)

    def test_no_target_species(self):
        biomass_by_cell = np.array([[[100.0]]])
        revenue = compute_expected_revenue(
            biomass_by_cell, np.array([10.0]), np.array([0.5]), [], np.array([100.0])
        )
        assert revenue[0] == pytest.approx(0.0)
