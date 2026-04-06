# tests/test_genetics_bioen_integration.py
"""Tests for genetics trait override integration with bioenergetics."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.simulate import simulate


def _bioen_genetics_config() -> dict[str, str]:
    """Config with bioenergetics AND genetics enabled, 4 evolving traits."""
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
        # Bioenergetics
        "simulation.bioen.enabled": "true",
        "species.bioen.beta.sp0": "0.8",
        "species.bioen.assimilation.sp0": "0.6",
        "species.bioen.c.m.sp0": "5.258",
        "species.bioen.eta.sp0": "1.0",
        "species.bioen.r.sp0": "0.5",
        "species.bioen.m0.sp0": "10.0",
        "species.bioen.m1.sp0": "0.5",
        "species.bioen.e.mobi.sp0": "0.45",
        "species.bioen.e.d.sp0": "2.46",
        "species.bioen.tp.sp0": "14.0",
        "species.bioen.e.maint.sp0": "0.45",
        "species.bioen.i.max.sp0": "3.5",
        "species.bioen.theta.sp0": "1.0",
        "species.bioen.c.rate.sp0": "0.0",
        "species.bioen.k.for.sp0": "0.0",
        # Genetics — all 4 traits
        "simulation.genetic.enabled": "true",
        "evolution.trait.imax.mean.sp0": "3.5",
        "evolution.trait.imax.var.sp0": "0.1",
        "evolution.trait.imax.envvar.sp0": "0.0",
        "evolution.trait.imax.nlocus.sp0": "5",
        "evolution.trait.imax.nval.sp0": "20",
        "evolution.trait.imax.target": "bioen_i_max",
        "evolution.trait.gsi.mean.sp0": "0.5",
        "evolution.trait.gsi.var.sp0": "0.01",
        "evolution.trait.gsi.envvar.sp0": "0.0",
        "evolution.trait.gsi.nlocus.sp0": "5",
        "evolution.trait.gsi.nval.sp0": "20",
        "evolution.trait.gsi.target": "bioen_r",
        "evolution.trait.m0.mean.sp0": "10.0",
        "evolution.trait.m0.var.sp0": "1.0",
        "evolution.trait.m0.envvar.sp0": "0.0",
        "evolution.trait.m0.nlocus.sp0": "5",
        "evolution.trait.m0.nval.sp0": "20",
        "evolution.trait.m0.target": "bioen_m0",
        "evolution.trait.m1.mean.sp0": "0.5",
        "evolution.trait.m1.var.sp0": "0.01",
        "evolution.trait.m1.envvar.sp0": "0.0",
        "evolution.trait.m1.nlocus.sp0": "5",
        "evolution.trait.m1.nval.sp0": "20",
        "evolution.trait.m1.target": "bioen_m1",
    }


class TestBioenGeneticsIntegration:
    def test_simulation_completes_with_4_traits_and_bioen(self):
        """Full simulation with bioenergetics + all 4 evolving traits should complete."""
        cfg = EngineConfig.from_dict(_bioen_genetics_config())
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(42)
        outputs = simulate(cfg, grid, rng)
        assert len(outputs) == 12

    def test_trait_overrides_affect_growth(self):
        """_bioen_step must apply per-school trait overrides when genetics is active."""
        from osmose.engine.simulate import _bioen_step
        from osmose.engine.state import SchoolState

        cfg = EngineConfig.from_dict(_bioen_genetics_config())

        # Create 2 schools with large preyed_biomass so ingestion cap matters
        state = SchoolState.create(n_schools=2, species_id=np.array([0, 0], dtype=np.int32))
        state = state.replace(
            weight=np.array([10.0, 10.0]),
            biomass=np.array([1e6, 1e6]),
            length=np.array([15.0, 15.0]),
            abundance=np.array([1e6, 1e6]),
            preyed_biomass=np.array([500.0, 500.0]),  # very large → cap will differ
            e_net_avg=np.array([0.0, 0.0]),
            cell_y=np.array([0, 0], dtype=np.int32),
            cell_x=np.array([0, 0], dtype=np.int32),
            age_dt=np.array([36, 36], dtype=np.int32),
            first_feeding_age_dt=np.array([1, 1], dtype=np.int32),
        )

        # School 0: high imax (cap = 100×), school 1: very low imax (cap ≈ 0)
        out_hi = _bioen_step(
            state, cfg, None, step=0, trait_overrides={"bioen_i_max": np.array([10.0, 0.0001])}
        )
        out_lo = _bioen_step(
            state, cfg, None, step=0, trait_overrides={"bioen_i_max": np.array([0.0001, 10.0])}
        )

        # School 0 should do better under hi, school 1 should do better under lo
        assert out_hi.weight[0] > out_lo.weight[0]
        assert out_hi.weight[1] < out_lo.weight[1]


class TestBioenReproductionOverride:
    def test_different_m0_affects_maturity(self):
        """Per-school m0 from genetics should change which schools mature."""
        from osmose.engine.processes.bioen_reproduction import bioen_egg_production

        length = np.array([12.0, 12.0, 12.0])
        age_dt = np.array([24, 24, 24], dtype=np.int32)
        gonad = np.array([0.01, 0.01, 0.01])

        # Scalar m0=10 → all mature (12 > 10 + 0.5*1 = 10.5)
        eggs_scalar = bioen_egg_production(
            gonad, length, age_dt, m0=10.0, m1=0.5, egg_weight=1e-6, n_dt_per_year=24
        )
        assert (eggs_scalar > 0).all()

        # Array m0 → school 2 has high m0, won't mature
        m0_arr = np.array([10.0, 10.0, 20.0])
        eggs_arr = bioen_egg_production(
            gonad, length, age_dt, m0=m0_arr, m1=0.5, egg_weight=1e-6, n_dt_per_year=24
        )
        assert eggs_arr[0] > 0
        assert eggs_arr[2] == 0.0  # 12 < 20 + 0.5*1 = 20.5, not mature
