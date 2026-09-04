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
            "simulate.py expresses traits AFTER mortality (express_traits at :1818 vs "
            "_mortality at :1749). Java DOES honour the per-school trait "
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


class TestBioenReproductionSeededOutFlag:
    """Regression for Task 6: `_bioen_reproduction`'s population-seeding window
    (`population.seeding.year.max`, default the species lifespan) is independent of and
    can outlast genetics' `evolution.seeding.year`. Discovered live on baltic_ev sp6
    (smelt, step 343, 15y genetics-on run,
    `test_ev_osmose_activation.py::test_baltic_ev_runs_15_years_with_genetics_on`):
    once cod's parity fix (Tasks 1-5) made the preflight pass for the first time, smelt's
    SSB was structurally 0 (unrelated calibration gap, plan Task 7.4) past year 10, so
    every step kept bootstrapping eggs from `population.seeding.biomass` with zero real
    parents, while genetics' own seeding window had already closed and demanded a real
    parent pool -- `create_offspring_genotypes` raised. The fix threads a `seeded_out`
    dict through `_bioen_reproduction` so the caller in `simulate.py`'s main loop can OR
    the bioen-level seeding flag into its own per-species `seeding` decision.
    """

    def test_seeded_out_flags_species_bootstrapped_this_step(self):
        """SSB == 0 with the population-seeding window still open must set the flag."""
        from osmose.engine.simulate import _bioen_reproduction
        from osmose.engine.state import SchoolState

        cfg_dict = _bioen_genetics_config()
        cfg_dict["population.seeding.biomass.sp0"] = "1000.0"
        # Long past a plausible genetics.seeding.year (10) -- the whole point of the gap.
        cfg_dict["population.seeding.year.max"] = "100"
        config = EngineConfig.from_dict(cfg_dict)

        # Zero schools of sp0 alive: SSB is structurally 0, exactly like collapsed smelt.
        state = SchoolState.create(n_schools=0, species_id=np.zeros(0, dtype=np.int32))

        seeded_out: dict = {}
        out = _bioen_reproduction(
            state, config, step=50, rng=np.random.default_rng(0), seeded_out=seeded_out
        )
        assert "seeded_this_step" in seeded_out
        assert bool(seeded_out["seeded_this_step"][0]) is True
        # The bootstrap must actually have produced eggs, or the flag would be moot.
        assert len(out) > len(state)

    def test_seeded_out_omitted_is_backward_compatible(self):
        """The default (`seeded_out=None`) must not change behaviour or crash -- every
        call in test_engine_bioen_reproduction_wiring.py relies on this."""
        from osmose.engine.simulate import _bioen_reproduction
        from osmose.engine.state import SchoolState

        cfg_dict = _bioen_genetics_config()
        cfg_dict["population.seeding.biomass.sp0"] = "1000.0"
        cfg_dict["population.seeding.year.max"] = "100"
        config = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(n_schools=0, species_id=np.zeros(0, dtype=np.int32))

        out = _bioen_reproduction(state, config, step=50, rng=np.random.default_rng(0))
        assert len(out) > len(state)

    def test_genetics_inheritance_survives_a_bootstrapped_species_past_its_transmission_year(
        self,
    ):
        """End-to-end through `simulate()`: a species with zero SSB whose population-seeding
        window outlives `evolution.seeding.year` must not raise. Direct regression for the
        crash this class documents -- reverting the `seeded_out` wiring in
        `osmose/engine/simulate.py` (the `sp_seeding` OR) reproduces the ValueError from
        `osmose.engine.genetics.inheritance.create_offspring_genotypes`.
        """
        cfg_dict = _bioen_genetics_config()
        # Genetics seeding window closes almost immediately...
        cfg_dict["evolution.seeding.year"] = "0"
        # ...while the population-seeding (bioen) window stays open the whole run, and
        # sp0 never has a mature school (m0 far above any reachable length) so SSB == 0
        # every step -- forcing the bootstrap branch on every reproduction call.
        cfg_dict["population.seeding.biomass.sp0"] = "1000.0"
        cfg_dict["population.seeding.year.max"] = "100"
        cfg_dict["evolution.trait.m0.mean.sp0"] = "1000.0"
        cfg_dict["simulation.time.nyear"] = "1"
        config = EngineConfig.from_dict(cfg_dict)
        grid = Grid.from_dimensions(ny=3, nx=3)
        rng = np.random.default_rng(7)

        outputs = simulate(config, grid, rng)  # must not raise
        assert len(outputs) == 12
