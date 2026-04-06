# tests/test_genetics_integration.py
"""Integration test: full simulation with Ev-OSMOSE genetics enabled."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate


def _genetics_config() -> dict[str, str]:
    """Minimal config with genetics enabled and 1 evolving trait (imax)."""
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
        # Genetics
        "simulation.genetic.enabled": "true",
        "evolution.trait.imax.mean.sp0": "3.5",
        "evolution.trait.imax.var.sp0": "0.1",
        "evolution.trait.imax.envvar.sp0": "0.05",
        "evolution.trait.imax.nlocus.sp0": "5",
        "evolution.trait.imax.nval.sp0": "20",
        "evolution.trait.imax.target": "ingestion_rate",
    }


class TestGeneticsIntegration:
    def test_simulation_completes_with_genetics(self):
        """Full sim with genetics should complete without errors."""
        cfg = EngineConfig.from_dict(_genetics_config())
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12

    def test_genetics_disabled_matches_baseline(self):
        """Disabling genetics should produce identical results to no-genetics config."""
        base_cfg = _genetics_config()
        base_cfg["simulation.genetic.enabled"] = "false"
        cfg_off = EngineConfig.from_dict(base_cfg)
        grid = Grid.from_dimensions(ny=3, nx=3)
        outputs_off = simulate(cfg_off, grid, np.random.default_rng(42))

        cfg_plain = _genetics_config()
        del cfg_plain["simulation.genetic.enabled"]
        del cfg_plain["evolution.trait.imax.mean.sp0"]
        del cfg_plain["evolution.trait.imax.var.sp0"]
        del cfg_plain["evolution.trait.imax.envvar.sp0"]
        del cfg_plain["evolution.trait.imax.nlocus.sp0"]
        del cfg_plain["evolution.trait.imax.nval.sp0"]
        del cfg_plain["evolution.trait.imax.target"]
        cfg_none = EngineConfig.from_dict(cfg_plain)
        outputs_none = simulate(cfg_none, grid, np.random.default_rng(42))

        for a, b in zip(outputs_off, outputs_none):
            np.testing.assert_array_almost_equal(a.biomass, b.biomass)
