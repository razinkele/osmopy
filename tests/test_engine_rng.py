"""Tests for the per-species deterministic RNG factory."""

from __future__ import annotations

import numpy as np

from osmose.engine.rng import build_rng


class TestBuildRng:
    def test_shared_rng_when_not_fixed(self):
        """All species get same Generator when fixed=False."""
        rngs = build_rng(seed=42, n_species=3, fixed=False)
        assert rngs[0] is rngs[1] is rngs[2]

    def test_independent_rng_when_fixed(self):
        """Each species gets distinct Generator when fixed=True."""
        rngs = build_rng(seed=42, n_species=3, fixed=True)
        assert rngs[0] is not rngs[1]
        assert rngs[0].random() != rngs[1].random()

    def test_fixed_rng_reproducible(self):
        """Same seed produces same sequences."""
        r1 = build_rng(42, 3, True)
        r2 = build_rng(42, 3, True)
        for i in range(3):
            assert r1[i].random() == r2[i].random()

    def test_adding_species_doesnt_change_existing(self):
        """SeedSequence guarantees independence."""
        r3 = build_rng(42, 3, True)
        r4 = build_rng(42, 4, True)
        for i in range(3):
            assert r3[i].random() == r4[i].random()


class TestSimulateAcceptsRngs:
    def test_simulate_accepts_rng_lists(self):
        """simulate() should accept movement_rngs and mortality_rngs."""
        from osmose.engine.config import EngineConfig
        from osmose.engine.grid import Grid
        from osmose.engine.simulate import simulate

        cfg_dict = {
            "simulation.time.ndtperyear": "12",
            "simulation.time.nyear": "1",
            "simulation.nspecies": "2",
            "simulation.nschool.sp0": "5",
            "simulation.nschool.sp1": "5",
            "species.name.sp0": "FishA",
            "species.name.sp1": "FishB",
            "species.linf.sp0": "20.0",
            "species.linf.sp1": "30.0",
            "species.k.sp0": "0.3",
            "species.k.sp1": "0.2",
            "species.t0.sp0": "-0.1",
            "species.t0.sp1": "-0.2",
            "species.egg.size.sp0": "0.1",
            "species.egg.size.sp1": "0.1",
            "species.length2weight.condition.factor.sp0": "0.006",
            "species.length2weight.condition.factor.sp1": "0.006",
            "species.length2weight.allometric.power.sp0": "3.0",
            "species.length2weight.allometric.power.sp1": "3.0",
            "species.lifespan.sp0": "3",
            "species.lifespan.sp1": "4",
            "species.vonbertalanffy.threshold.age.sp0": "1.0",
            "species.vonbertalanffy.threshold.age.sp1": "1.0",
            "mortality.subdt": "10",
            "predation.ingestion.rate.max.sp0": "3.5",
            "predation.ingestion.rate.max.sp1": "3.5",
            "predation.efficiency.critical.sp0": "0.57",
            "predation.efficiency.critical.sp1": "0.57",
        }
        cfg = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        movement_rngs = build_rng(42, cfg.n_species, fixed=True)
        mortality_rngs = build_rng(43, cfg.n_species, fixed=True)

        outputs = simulate(
            cfg, grid, rng, movement_rngs=movement_rngs, mortality_rngs=mortality_rngs
        )
        assert len(outputs) == 12
