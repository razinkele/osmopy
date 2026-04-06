# tests/test_economics_integration.py
"""Integration test: full simulation with DSVM fleet economics enabled."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate


def _economics_config() -> dict[str, str]:
    """Minimal config with economics enabled and 1 fleet."""
    return {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "5",
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
        # Economics
        "simulation.economic.enabled": "true",
        "simulation.economic.rationality": "1.0",
        "simulation.economic.memory.decay": "0.7",
        "economic.fleet.number": "1",
        "economic.fleet.name.fsh0": "Trawlers",
        "economic.fleet.nvessels.fsh0": "10",
        "economic.fleet.homeport.y.fsh0": "1",
        "economic.fleet.homeport.x.fsh0": "1",
        "economic.fleet.gear.fsh0": "bottom_trawl",
        "economic.fleet.max.days.fsh0": "200",
        "economic.fleet.fuel.cost.fsh0": "0.0",
        "economic.fleet.operating.cost.fsh0": "0.0",
        "economic.fleet.target.species.fsh0": "0",
        "economic.fleet.price.sp0.fsh0": "1000.0",
        "economic.fleet.stock.elasticity.sp0.fsh0": "0.0",
    }


class TestEconomicsIntegration:
    def test_simulation_completes_with_economics(self):
        """Full sim with economics should complete without errors."""
        cfg = EngineConfig.from_dict(_economics_config())
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12

    def test_both_modules_enabled(self):
        """Simulation with BOTH genetics and economics enabled should complete."""
        cfg_dict = _economics_config()
        cfg_dict["simulation.genetic.enabled"] = "true"
        cfg_dict["evolution.trait.imax.mean.sp0"] = "3.5"
        cfg_dict["evolution.trait.imax.var.sp0"] = "0.1"
        cfg_dict["evolution.trait.imax.envvar.sp0"] = "0.05"
        cfg_dict["evolution.trait.imax.nlocus.sp0"] = "5"
        cfg_dict["evolution.trait.imax.nval.sp0"] = "20"
        cfg_dict["evolution.trait.imax.target"] = "ingestion_rate"

        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12
