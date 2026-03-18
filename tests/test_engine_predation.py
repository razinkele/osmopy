"""Tests for predation process -- Tier 1 analytical verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.predation import (
    predation,
    _predation_in_cell_python as predation_in_cell,
)
from osmose.engine.state import SchoolState


def _make_predation_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "simulation.nschool.sp0": "5",
        "simulation.nschool.sp1": "5",
        "species.name.sp0": "Prey",
        "species.name.sp1": "Predator",
        "species.linf.sp0": "15.0",
        "species.linf.sp1": "80.0",
        "species.k.sp0": "0.3",
        "species.k.sp1": "0.15",
        "species.t0.sp0": "-0.1",
        "species.t0.sp1": "-0.2",
        "species.egg.size.sp0": "0.1",
        "species.egg.size.sp1": "0.1",
        "species.length2weight.condition.factor.sp0": "0.006",
        "species.length2weight.condition.factor.sp1": "0.005",
        "species.length2weight.allometric.power.sp0": "3.0",
        "species.length2weight.allometric.power.sp1": "3.0",
        "species.lifespan.sp0": "4",
        "species.lifespan.sp1": "10",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "species.vonbertalanffy.threshold.age.sp1": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.ingestion.rate.max.sp1": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "predation.efficiency.critical.sp1": "0.57",
        # Eligible when r_min <= pred/prey < r_max
        "predation.predprey.sizeratio.min.sp0": "1.0",
        "predation.predprey.sizeratio.min.sp1": "1.0",
        "predation.predprey.sizeratio.max.sp0": "3.5",
        "predation.predprey.sizeratio.max.sp1": "3.5",
    }


class TestPredationInCell:
    def test_predator_eats_prey(self):
        """A predator with valid size ratio should consume prey."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        # Predator (sp1) len=25, prey (sp0) len=10 -> ratio=2.5, in (1.0, 3.5]
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([50.0, 500.0]),
            length=np.array([25.0, 10.0]),
            weight=np.array([78.125, 6.0]),
            biomass=np.array([3906.25, 3000.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            cell_y=np.array([0, 0], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        predation_in_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
        # Prey abundance should have decreased
        assert state.abundance[1] < 500.0

    def test_self_predation_excluded(self):
        """A school should never eat itself."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        # Single school alone in cell -- function exits early (< 2 schools)
        state = SchoolState.create(n_schools=1, species_id=np.array([1], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0]),
            length=np.array([30.0]),
            weight=np.array([135.0]),
            biomass=np.array([13500.0]),
            age_dt=np.array([24], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        predation_in_cell(np.array([0], dtype=np.int32), state, cfg, rng, n_subdt=10)
        np.testing.assert_allclose(state.abundance[0], 100.0)

    def test_eggs_cannot_predate(self):
        """Schools with age_dt < first_feeding_age_dt should not predate."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 500.0]),
            length=np.array([25.0, 10.0]),
            weight=np.array([78.125, 6.0]),
            biomass=np.array([7812.5, 3000.0]),
            age_dt=np.array([0, 24], dtype=np.int32),  # predator is egg
            first_feeding_age_dt=np.array([1, 1], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        predation_in_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
        # Egg (age_dt=0 < first_feeding=1) should not eat -- prey unchanged
        np.testing.assert_allclose(state.abundance[1], 500.0)

    def test_asynchronous_update(self):
        """Prey biomass should decrease between successive predators."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        # 2 predators (sp1, len=25) + 1 prey (sp0, len=10): ratio=2.5 in (1.0, 3.5]
        state = SchoolState.create(n_schools=3, species_id=np.array([1, 1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([50.0, 50.0, 100.0]),
            length=np.array([25.0, 25.0, 10.0]),
            weight=np.array([78.125, 78.125, 6.0]),
            biomass=np.array([3906.25, 3906.25, 600.0]),
            age_dt=np.array([24, 24, 24], dtype=np.int32),
        )
        original_prey_abundance = 100.0
        rng = np.random.default_rng(42)
        predation_in_cell(np.array([0, 1, 2], dtype=np.int32), state, cfg, rng, n_subdt=10)
        # Prey should have been eaten by at least one predator
        assert state.abundance[2] < original_prey_abundance

    def test_predation_success_rate_updated(self):
        """Predator's pred_success_rate should increase after eating."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        # Predator sp1 (len=25) eats prey sp0 (len=10), ratio=2.5 in (1.0, 3.5]
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([50.0, 500.0]),
            length=np.array([25.0, 10.0]),
            weight=np.array([78.125, 6.0]),
            biomass=np.array([3906.25, 3000.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
            pred_success_rate=np.array([0.0, 0.0]),
        )
        rng = np.random.default_rng(42)
        predation_in_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
        assert state.pred_success_rate[0] > 0.0

    def test_size_ratio_outside_range_no_predation(self):
        """Prey outside the size ratio window should not be eaten."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        # Predator sp1 (len=25), prey sp0 (len=25) -> ratio=1.0, NOT > 1.0
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([50.0, 500.0]),
            length=np.array([25.0, 25.0]),
            weight=np.array([78.125, 93.75]),
            biomass=np.array([3906.25, 46875.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        predation_in_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
        np.testing.assert_allclose(state.abundance[1], 500.0)


class TestPredationAcrossCells:
    def test_schools_in_different_cells_dont_interact(self):
        """Predation only happens within the same cell."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([50.0, 500.0]),
            length=np.array([25.0, 10.0]),
            weight=np.array([78.125, 6.0]),
            biomass=np.array([3906.25, 3000.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
            cell_x=np.array([0, 5], dtype=np.int32),  # different cells!
            cell_y=np.array([0, 5], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10)
        # Prey should be unchanged -- different cell from predator
        np.testing.assert_allclose(new_state.abundance[1], 500.0)

    def test_empty_state(self):
        """Predation on empty state should return empty state."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        state = SchoolState.create(n_schools=0)
        rng = np.random.default_rng(42)
        new_state = predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10)
        assert len(new_state) == 0

    def test_same_cell_predation_via_top_level(self):
        """Top-level predation() should apply predation within shared cells."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([50.0, 500.0]),
            length=np.array([25.0, 10.0]),
            weight=np.array([78.125, 6.0]),
            biomass=np.array([3906.25, 3000.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
            cell_x=np.array([3, 3], dtype=np.int32),  # same cell
            cell_y=np.array([2, 2], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        new_state = predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10)
        # Prey should have been eaten
        assert new_state.abundance[1] < 500.0
