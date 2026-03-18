"""Tests for the simulation loop skeleton."""

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import StepOutput, simulate


@pytest.fixture
def minimal_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "10",
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
    }


class TestSimulate:
    def test_simulate_returns_outputs(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12

    def test_step_output_has_biomass(self, minimal_config):
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert isinstance(outputs[0], StepOutput)
        assert outputs[0].step == 0
        assert outputs[0].biomass.shape == (1,)

    def test_simulate_correct_step_count(self, minimal_config):
        minimal_config["simulation.time.nyear"] = "2"
        cfg = EngineConfig.from_dict(minimal_config)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 24
        assert outputs[-1].step == 23
