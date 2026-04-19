"""Tests for predation process -- Tier 1 analytical verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.predation import (
    _predation_on_resources,
    predation_for_cell,
)
from osmose.engine.resources import ResourceState
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
        predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
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
        predation_for_cell(np.array([0], dtype=np.int32), state, cfg, rng, n_subdt=10)
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
        predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
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
        predation_for_cell(np.array([0, 1, 2], dtype=np.int32), state, cfg, rng, n_subdt=10)
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
        predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
        assert state.pred_success_rate[0] > 0.0

    def test_size_ratio_outside_range_no_predation(self):
        """Prey outside the size ratio window should not be eaten."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        # Predator sp1 (len=25), prey sp0 (len=50) -> ratio=0.5, below r_min=1.0
        # Also sp0 (len=50) vs sp1 (len=25) -> ratio=2.0 in range, but sp0 is
        # assigned as "prey" here so we only check sp0 abundance stays unchanged.
        # Use sp0 as predator (index 0) and sp1 as prey (index 1) to avoid
        # mutual predation: sp1 (len=6) cannot eat sp0 (len=25) at ratio 6/25=0.24.
        state = SchoolState.create(n_schools=2, species_id=np.array([0, 1], dtype=np.int32))
        state = state.replace(
            abundance=np.array([50.0, 500.0]),
            length=np.array([6.0, 25.0]),
            weight=np.array([1.296, 78.125]),
            biomass=np.array([64.8, 39062.5]),
            age_dt=np.array([24, 24], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
        # sp0 (len=6) trying to eat sp1 (len=25): ratio=6/25=0.24, below r_min=1.0 -> no predation
        # sp1 (len=25) trying to eat sp0 (len=6): ratio=25/6=4.17, above r_max=3.5 -> no predation
        np.testing.assert_allclose(state.abundance[0], 50.0)
        np.testing.assert_allclose(state.abundance[1], 500.0)


class TestPredationAcrossCells:
    def test_school_outside_cell_indices_is_untouched(self):
        """predation_for_cell must not mutate schools outside cell_indices.

        Note: predation_for_cell's dispatch is governed by cell_indices, NOT
        by state.cell_x/cell_y. We leave cell_x/cell_y at their default zeros
        to emphasize this -- the "bystander" tag comes from not being in
        cell_indices, not from any spatial-coordinate difference.
        """
        cfg = EngineConfig.from_dict(_make_predation_config())
        state = SchoolState.create(
            n_schools=3, species_id=np.array([1, 0, 0], dtype=np.int32)
        )
        state = state.replace(
            abundance=np.array([50.0, 500.0, 123.0]),
            length=np.array([25.0, 10.0, 10.0]),
            weight=np.array([78.125, 6.0, 6.0]),
            biomass=np.array([3906.25, 3000.0, 738.0]),
            age_dt=np.array([24, 24, 24], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        # Only schools 0 and 1 are in cell_indices; school 2 is the bystander.
        predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
        assert state.abundance[1] < 500.0
        np.testing.assert_allclose(state.abundance[2], 123.0)

    def test_empty_state(self):
        """Predation on empty state should return empty state."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        state = SchoolState.create(n_schools=0)
        rng = np.random.default_rng(42)
        predation_for_cell(np.array([], dtype=np.int32), state, cfg, rng, n_subdt=10)
        assert len(state) == 0

    def test_same_cell_predation(self):
        """Top-level predation should apply predation within shared cells."""
        cfg = EngineConfig.from_dict(_make_predation_config())
        state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([50.0, 500.0]),
            length=np.array([25.0, 10.0]),
            weight=np.array([78.125, 6.0]),
            biomass=np.array([3906.25, 3000.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
            cell_x=np.array([3, 3], dtype=np.int32),
            cell_y=np.array([2, 2], dtype=np.int32),
        )
        rng = np.random.default_rng(42)
        predation_for_cell(np.array([0, 1], dtype=np.int32), state, cfg, rng, n_subdt=10)
        assert state.abundance[1] < 500.0


def _make_ltl_config() -> dict[str, str]:
    """Minimal config for one focal species + one LTL resource (ltl.* keys)."""
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nresource": "1",
        "simulation.nschool.sp0": "5",
        "species.name.sp0": "Anchovy",
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
        # size ratios: predator (len=15) / prey range [0.001, 0.5] -> ratio in [30, 15000]
        "predation.predprey.sizeratio.min.sp0": "1.0",
        "predation.predprey.sizeratio.max.sp0": "10000.0",
        # LTL resource
        "ltl.name.rsc0": "Plankton",
        "ltl.size.min.rsc0": "0.001",
        "ltl.size.max.rsc0": "0.5",
        "ltl.tl.rsc0": "1.0",
        "ltl.accessibility2fish.rsc0": "0.5",
        "ltl.biomass.total.rsc0": "1000.0",
    }


def test_predation_on_resources_removes_biomass() -> None:
    """_predation_on_resources should reduce resource biomass in a cell.

    A mature focal school (sp0, len=15) with size-ratio window [1, 10000] is
    placed in cell (0, 0).  The LTL resource has size range [0.001, 0.5] cm,
    which falls entirely within the predator's prey window (15/10000=0.0015 to
    15/1.0=15).  After one call the resource biomass in cell (0, 0) must be
    strictly less than the initial value.
    """
    cfg_dict = _make_ltl_config()
    cfg = EngineConfig.from_dict(cfg_dict)
    grid = Grid.from_dimensions(ny=5, nx=5)

    # Build ResourceState with uniform biomass
    resources = ResourceState(config=cfg_dict, grid=grid)
    resources.update(step=0)

    # Record initial biomass in cell (0, 0)
    cell_y, cell_x = 0, 0
    biomass_before = resources.get_cell_biomass(0, cell_y, cell_x)
    assert biomass_before > 0.0, "ResourceState.update() should set non-zero biomass"

    # Single feeding-age school in cell (0, 0)
    state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
    state = state.replace(
        abundance=np.array([1000.0]),
        length=np.array([15.0]),  # mature anchovy
        weight=np.array([8.1]),   # ~0.006 * 15^3
        biomass=np.array([8100.0]),
        age_dt=np.array([24], dtype=np.int32),   # past first feeding age
        first_feeding_age_dt=np.array([1], dtype=np.int32),
        pred_success_rate=np.array([0.0]),        # no prior satiation
        preyed_biomass=np.array([0.0]),
        cell_x=np.array([cell_x], dtype=np.int32),
        cell_y=np.array([cell_y], dtype=np.int32),
    )

    rng = np.random.default_rng(42)
    cell_indices = np.array([0], dtype=np.int32)

    _predation_on_resources(
        cell_indices,
        state,
        cfg,
        resources,
        cell_y,
        cell_x,
        rng,
        n_subdt=10,
    )

    biomass_after = resources.get_cell_biomass(0, cell_y, cell_x)
    assert biomass_after < biomass_before, (
        f"Resource biomass should decrease after predation: "
        f"before={biomass_before:.4f}, after={biomass_after:.4f}"
    )
