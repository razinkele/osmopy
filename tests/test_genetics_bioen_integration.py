# tests/test_genetics_bioen_integration.py
"""Tests for genetics trait override integration with bioenergetics."""

import numpy as np
import pytest

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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN GAP, owner: C3 Stage-1 Task 5 (Ev-OSMOSE). The allometric ingestion cap "
            "moved out of _bioen_step and into the mortality loop (spec decision 14 / Java "
            "BioenPredationMortality), so _bioen_step no longer reads "
            "trait_overrides['bioen_i_max'] at all. mortality() cannot read it either, because "
            "simulate.py expresses traits AFTER mortality (express_traits at :1812 vs "
            "_mortality at :1743). Java DOES honour the per-school trait "
            "(getMaxPredationRate: existsTrait('imax')), so this is a real parity gap; closing "
            "it needs the phenotypes available before mortality, which is a step-order change "
            "in Ev-OSMOSE's territory. The other three evolving traits (r, m0, m1) still reach "
            "compute_energy_budget and are covered by the tests below. strict=True so this "
            "flips loudly the moment the plumbing is restored."
        ),
    )
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
        """Per-school m0 from genetics should change which schools mature.

        Maturity moved out of `bioen_egg_production` (removed in the Java-parity task 5)
        into `_bioen_reproduction`, which is where Java decides it — `isMature` is a
        School flag set by `EnergyBudget.getMaturation`, not by the reproduction process.
        The assertion is therefore made through `_bioen_reproduction`; the standalone
        LMRN cases live in
        `test_engine_bioen_reproduction_wiring.py::TestBioenReproductionMaturity`.
        """
        from osmose.engine.simulate import _bioen_reproduction
        from osmose.engine.state import SchoolState

        cfg_dict = _bioen_genetics_config()
        cfg_dict["simulation.time.ndtperyear"] = "24"
        config = EngineConfig.from_dict(cfg_dict)
        # NOTE: this fixture's bioen keys predate the 4.4.0 renames (`species.bioen.m0.sp0`
        # is not a key the reader knows — hence the "unknown keys" log), so m0/m1 come
        # from config as 0.0. Both arms therefore pass m0/m1 explicitly through
        # trait_overrides, which is exactly the genetics path under test.
        m1 = np.full(3, 0.5)

        state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
        state = state.replace(
            abundance=np.full(3, 1e4),
            length=np.full(3, 12.0),
            weight=np.full(3, 1e-6),
            age_dt=np.full(3, 24, dtype=np.int32),
            gonad_weight=np.full(3, 0.01),
        )
        n_before = len(state)

        # Uniform m0 = 10 → all three mature (12 >= 10 + 0.5*1 = 10.5).
        out_all = _bioen_reproduction(
            state,
            config,
            step=0,
            rng=np.random.default_rng(0),
            trait_overrides={"bioen_m0": np.full(3, 10.0), "bioen_m1": m1},
        )
        eggs_all = out_all.abundance[n_before:].sum()
        assert eggs_all > 0
        assert np.all(out_all.gonad_weight[:n_before] < 0.01)  # all three released

        # Per-school m0 → school 2 needs 20 cm and stays immature.
        out_two = _bioen_reproduction(
            state,
            config,
            step=0,
            rng=np.random.default_rng(0),
            trait_overrides={"bioen_m0": np.array([10.0, 10.0, 20.0]), "bioen_m1": m1},
        )
        eggs_two = out_two.abundance[n_before:].sum()
        assert eggs_two == pytest.approx(eggs_all * 2.0 / 3.0, rel=1e-9)
        assert out_two.gonad_weight[2] == pytest.approx(0.01)  # kept its gonad
