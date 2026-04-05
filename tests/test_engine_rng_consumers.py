"""Tests for per-species RNG wiring into movement and predation consumers (Gap 3)."""

from __future__ import annotations

import inspect

import numpy as np

from osmose.engine.processes.movement import movement
from osmose.engine.processes.mortality import mortality
from osmose.engine.processes.predation import predation


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------


class TestSignatures:
    def test_movement_accepts_species_rngs(self):
        """movement() should accept species_rngs as optional keyword argument."""
        sig = inspect.signature(movement)
        assert "species_rngs" in sig.parameters, "movement() must accept species_rngs parameter"
        param = sig.parameters["species_rngs"]
        assert param.default is None, "species_rngs default should be None"

    def test_mortality_accepts_species_rngs(self):
        """mortality() should accept species_rngs as optional keyword argument."""
        sig = inspect.signature(mortality)
        assert "species_rngs" in sig.parameters, "mortality() must accept species_rngs parameter"
        param = sig.parameters["species_rngs"]
        assert param.default is None, "species_rngs default should be None"

    def test_predation_accepts_species_rngs(self):
        """predation() should accept species_rngs as optional keyword argument."""
        sig = inspect.signature(predation)
        assert "species_rngs" in sig.parameters, "predation() must accept species_rngs parameter"
        param = sig.parameters["species_rngs"]
        assert param.default is None, "species_rngs default should be None"


# ---------------------------------------------------------------------------
# Backward compatibility tests (species_rngs=None must work like before)
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def _make_base_config(self) -> dict:
        return {
            "simulation.time.ndtperyear": "24",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "2",
            "simulation.nschool.sp0": "5",
            "simulation.nschool.sp1": "5",
            "species.name.sp0": "Anchovy",
            "species.name.sp1": "Sardine",
            "species.linf.sp0": "15.0",
            "species.linf.sp1": "25.0",
            "species.k.sp0": "0.4",
            "species.k.sp1": "0.3",
            "species.t0.sp0": "-0.1",
            "species.t0.sp1": "-0.2",
            "species.egg.size.sp0": "0.1",
            "species.egg.size.sp1": "0.15",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.condition.factor.sp1": "0.008",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.length2weight.allometric.power.sp1": "3.1",
            "species.lifespan.sp0": "3",
            "species.lifespan.sp1": "5",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "species.vonbertalanffy.threshold.age.sp1": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.ingestion.rate.max.sp1": "3.0",
            "predation.efficiency.critical.sp0": "0.57",
            "predation.efficiency.critical.sp1": "0.57",
        }

    def test_movement_none_species_rngs_works(self):
        """movement() with species_rngs=None should run without error."""
        from osmose.engine.config import EngineConfig
        from osmose.engine.grid import Grid
        from osmose.engine.state import SchoolState

        config = EngineConfig.from_dict(self._make_base_config())
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)

        state = SchoolState.create(n_schools=4, species_id=np.zeros(4, dtype=np.int32))
        state = state.replace(
            abundance=np.full(4, 100.0),
            length=np.full(4, 5.0),
            weight=np.full(4, 0.001),
            cell_x=np.array([1, 2, 1, 2], dtype=np.int32),
            cell_y=np.array([1, 1, 2, 2], dtype=np.int32),
            age_dt=np.full(4, 10, dtype=np.int32),
        )

        # Should not raise
        result = movement(state, grid, config, step=0, rng=rng, species_rngs=None)
        assert result is not None
        assert len(result) == 4

    def test_mortality_none_species_rngs_works(self):
        """mortality() with species_rngs=None should run without error."""
        from osmose.engine.config import EngineConfig
        from osmose.engine.grid import Grid
        from osmose.engine.resources import ResourceState
        from osmose.engine.state import SchoolState

        config = EngineConfig.from_dict(self._make_base_config())
        grid = Grid.from_dimensions(ny=5, nx=5)
        resources = ResourceState(config=config.raw_config, grid=grid)
        rng = np.random.default_rng(42)

        state = SchoolState.create(n_schools=4, species_id=np.zeros(4, dtype=np.int32))
        state = state.replace(
            abundance=np.full(4, 100.0),
            length=np.full(4, 5.0),
            weight=np.full(4, 0.001),
            cell_x=np.array([1, 2, 1, 2], dtype=np.int32),
            cell_y=np.array([1, 1, 2, 2], dtype=np.int32),
            age_dt=np.full(4, 10, dtype=np.int32),
            first_feeding_age_dt=np.ones(4, dtype=np.int32),
            pred_success_rate=np.zeros(4),
            preyed_biomass=np.zeros(4),
            starvation_rate=np.zeros(4),
        )

        # Should not raise
        result = mortality(state, resources, config, rng, grid, step=0, species_rngs=None)
        assert result is not None
        assert len(result) == 4

    def test_predation_none_species_rngs_works(self):
        """predation() with species_rngs=None should run without error."""
        from osmose.engine.config import EngineConfig
        from osmose.engine.grid import Grid
        from osmose.engine.resources import ResourceState
        from osmose.engine.state import SchoolState

        config = EngineConfig.from_dict(self._make_base_config())
        grid = Grid.from_dimensions(ny=5, nx=5)
        resources = ResourceState(config=config.raw_config, grid=grid)
        rng = np.random.default_rng(42)

        state = SchoolState.create(n_schools=4, species_id=np.zeros(4, dtype=np.int32))
        state = state.replace(
            abundance=np.full(4, 100.0),
            length=np.full(4, 5.0),
            weight=np.full(4, 0.001),
            cell_x=np.array([1, 2, 1, 2], dtype=np.int32),
            cell_y=np.array([1, 1, 2, 2], dtype=np.int32),
            age_dt=np.full(4, 10, dtype=np.int32),
            first_feeding_age_dt=np.ones(4, dtype=np.int32),
            pred_success_rate=np.zeros(4),
            preyed_biomass=np.zeros(4),
        )

        # Should not raise
        result = predation(
            state,
            config,
            rng,
            n_subdt=1,
            grid_ny=grid.ny,
            grid_nx=grid.nx,
            resources=resources,
            species_rngs=None,
        )
        assert result is not None
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Functional tests: movement uses per-species rng when seed_fixed=True
# ---------------------------------------------------------------------------


class TestMovementPerSpeciesRNG:
    def _make_config(self, seed_fixed: bool) -> dict:
        cfg = {
            "simulation.time.ndtperyear": "24",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "2",
            "simulation.nschool.sp0": "5",
            "simulation.nschool.sp1": "5",
            "species.name.sp0": "Anchovy",
            "species.name.sp1": "Sardine",
            "species.linf.sp0": "15.0",
            "species.linf.sp1": "25.0",
            "species.k.sp0": "0.4",
            "species.k.sp1": "0.3",
            "species.t0.sp0": "-0.1",
            "species.t0.sp1": "-0.2",
            "species.egg.size.sp0": "0.1",
            "species.egg.size.sp1": "0.15",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.condition.factor.sp1": "0.008",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.length2weight.allometric.power.sp1": "3.1",
            "species.lifespan.sp0": "3",
            "species.lifespan.sp1": "5",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "species.vonbertalanffy.threshold.age.sp1": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.ingestion.rate.max.sp1": "3.0",
            "predation.efficiency.critical.sp0": "0.57",
            "predation.efficiency.critical.sp1": "0.57",
            "movement.randomseed.fixed": "true" if seed_fixed else "false",
        }
        return cfg

    def test_movement_with_species_rngs_accepted(self):
        """movement() accepts and uses species_rngs without error."""
        from osmose.engine.config import EngineConfig
        from osmose.engine.grid import Grid
        from osmose.engine.state import SchoolState

        config = EngineConfig.from_dict(self._make_config(seed_fixed=True))
        grid = Grid.from_dimensions(ny=5, nx=5)
        rng = np.random.default_rng(42)
        sp_rngs = [np.random.default_rng(i) for i in range(config.n_species)]

        state = SchoolState.create(
            n_schools=4,
            species_id=np.array([0, 0, 1, 1], dtype=np.int32),
        )
        state = state.replace(
            abundance=np.full(4, 100.0),
            length=np.full(4, 5.0),
            weight=np.full(4, 0.001),
            cell_x=np.array([1, 2, 1, 2], dtype=np.int32),
            cell_y=np.array([1, 1, 2, 2], dtype=np.int32),
            age_dt=np.full(4, 10, dtype=np.int32),
        )

        result = movement(state, grid, config, step=0, rng=rng, species_rngs=sp_rngs)
        assert len(result) == 4

    def test_movement_seed_fixed_false_ignores_species_rngs(self):
        """When movement_seed_fixed=False, species_rngs are ignored (uses global rng)."""
        from osmose.engine.config import EngineConfig
        from osmose.engine.grid import Grid
        from osmose.engine.state import SchoolState

        config = EngineConfig.from_dict(self._make_config(seed_fixed=False))
        grid = Grid.from_dimensions(ny=5, nx=5)

        state = SchoolState.create(
            n_schools=2,
            species_id=np.array([0, 1], dtype=np.int32),
        )
        state = state.replace(
            abundance=np.full(2, 100.0),
            length=np.full(2, 5.0),
            weight=np.full(2, 0.001),
            cell_x=np.array([1, 2], dtype=np.int32),
            cell_y=np.array([1, 1], dtype=np.int32),
            age_dt=np.full(2, 10, dtype=np.int32),
        )

        # Same global rng seed, different species_rngs → identical results
        rng1 = np.random.default_rng(99)
        sp_rngs_a = [np.random.default_rng(1), np.random.default_rng(2)]
        result_a = movement(state, grid, config, step=0, rng=rng1, species_rngs=sp_rngs_a)

        rng2 = np.random.default_rng(99)
        sp_rngs_b = [np.random.default_rng(100), np.random.default_rng(200)]
        result_b = movement(state, grid, config, step=0, rng=rng2, species_rngs=sp_rngs_b)

        # Should be identical because movement_seed_fixed=False → same global rng used
        assert np.array_equal(result_a.cell_x, result_b.cell_x)
        assert np.array_equal(result_a.cell_y, result_b.cell_y)
