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


from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState


class TestEffortFishingIntegration:
    def test_effort_scales_fishing_mortality(self):
        """Fishing mortality should be higher where fleet effort is concentrated."""
        cfg_dict = {
            "simulation.time.ndtperyear": "12",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "1",
            "simulation.nschool.sp0": "2",
            "species.name.sp0": "TestFish",
            "species.linf.sp0": "20.0",
            "species.k.sp0": "0.3",
            "species.t0.sp0": "-0.1",
            "species.egg.size.sp0": "0.1",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.lifespan.sp0": "3",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.efficiency.critical.sp0": "0.57",
            "mortality.fishing.rate.sp0": "0.5",
        }
        config = EngineConfig.from_dict(cfg_dict)

        state = SchoolState.create(n_schools=2, species_id=np.array([0, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            biomass=np.array([100.0, 100.0]),
            length=np.array([15.0, 15.0]),
            weight=np.array([0.1, 0.1]),
            age_dt=np.array([24, 24], dtype=np.int32),
            cell_y=np.array([0, 1], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
        )

        fleet = FleetConfig(
            name="Trawlers",
            n_vessels=10,
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
        fs = create_fleet_state([fleet], grid_ny=2, grid_nx=1, rationality=1.0)
        fs.effort_map[0, 0, 0] = 10.0
        fs.effort_map[0, 1, 0] = 0.0

        from osmose.engine.processes.mortality import _precompute_effective_rates

        # Without fleet state — both schools get same fishing rate
        _, _, eff_fishing_base, _ = _precompute_effective_rates(state, config, 10, 0)
        assert eff_fishing_base[0] == eff_fishing_base[1]
        assert eff_fishing_base[0] > 0

        # With fleet state — school at (0,0) should have fishing, school at (1,0) should have zero
        _, _, eff_fishing_effort, _ = _precompute_effective_rates(
            state, config, 10, 0, fleet_state=fs
        )

        assert eff_fishing_effort[0] > 0
        assert eff_fishing_effort[1] == 0.0
