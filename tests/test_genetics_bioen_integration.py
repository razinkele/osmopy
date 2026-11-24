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
        # Task 7 requires a configured temperature source whenever bioen is enabled
        # (simulate() now raises otherwise). 15.0 reproduces the pre-Task-7 fallback
        # _bioen_step used when no source was configured, so this is a no-op for every
        # test built on this fixture that predates Task 7.
        "temperature.value": "15.0",
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

    # The evolving imax acts in MORTALITY (bioen ingestion cap + genetic foraging), as Java's
    # getMaxPredationRate / ForagingMortality do -- not in _bioen_step's energy budget, which
    # Java never feeds imax either. A former strict xfail asserted it here (the wrong layer);
    # the behaviour is now tested where it lives: tests/test_genetics_imax_wiring.py.


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
    (`population.seeding.year.max`, default the longest lifespan of any species) is independent of and
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


class TestSeedingConspecificFilter:
    """Regression for ruling R25 (Task 6 fix round 1): the `seeding=True` branch of
    `create_offspring_genotypes` used to draw its donor school from the ENTIRE
    multi-species `parent_gs.alleles[name]` pool, unfiltered by species. `sp_seeding`
    (see `TestBioenReproductionSeededOutFlag` above) makes that branch reachable
    mid-run for a single collapsed species while its neighbours carry real, non-zero,
    evolved allele deviations -- so a `var=0.0` species could silently inherit another
    species' genetics. `TestBioenReproductionSeededOutFlag`'s three tests are
    single-species (`simulation.nspecies: "1"`) and structurally cannot see this: with
    only one species in the population, the unfiltered pool and the conspecific pool
    are the same set. This class uses two species specifically so the two pools
    differ.
    """

    def _two_species_genetic_state(self):
        """sp0: var > 0 with real, non-zero, "evolved" allele deviations (stands in for
        cod). sp1: var = 0.0, all-zero alleles, exactly like the rest of baltic_ev's
        species before any cross-species leak. 3 living schools of each."""
        from osmose.engine.genetics.genotype import GeneticState
        from osmose.engine.genetics.trait import Trait, TraitRegistry

        n_loci = np.array([3, 3], dtype=np.int32)
        trait = Trait(
            name="imax",
            species_mean=np.array([3.5, 2.0]),
            species_var=np.array([0.5, 0.0]),
            env_var=np.array([0.0, 0.0]),
            n_loci=n_loci,
            n_alleles=np.array([5, 5], dtype=np.int32),
            target_param="bioen_i_max",
            allele_pool=[
                [np.full(5, 9.0) for _ in range(3)],  # sp0 fallback pool -- also non-zero
                [],  # sp1: var=0 -> no pool built (mirrors trait.py:84's guard)
            ],
        )
        registry = TraitRegistry()
        registry.register(trait)

        species_id = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        alleles = np.zeros((6, 3, 2), dtype=np.float64)
        alleles[0:3, :, :] = 9.0  # sp0's "evolved" deviation: unmistakably non-zero
        # sp1 rows (3:6) stay all-zero, matching a var=0 species whose own
        # reproduction has only ever gone through the (correctly-filtered)
        # non-seeding branch or the zero-filled fallback.
        gs = GeneticState(
            alleles={"imax": alleles},
            env_noise={"imax": np.zeros(6)},
            registry=registry,
        )
        gonad_weight = np.zeros(6)  # irrelevant to the seeding branch by design (R25)
        return gs, species_id, gonad_weight

    def test_var0_species_seeding_bootstrap_does_not_inherit_other_species_alleles(self):
        """The whole point of the fix: sp1 (var=0.0) offspring bootstrapped via
        seeding=True must come out all-zero, never carrying sp0's 9.0 deviation."""
        from osmose.engine.genetics.inheritance import create_offspring_genotypes

        gs, species_id, gonad_weight = self._two_species_genetic_state()
        rng = np.random.default_rng(12345)

        offspring = create_offspring_genotypes(
            parent_gs=gs,
            gonad_weight=gonad_weight,
            species_id=species_id,
            offspring_species=1,
            n_offspring=20,
            rng=rng,
            seeding=True,
        )

        assert np.all(offspring.alleles["imax"] == 0.0), (
            "sp1 (var=0.0) offspring carry non-zero alleles after a seeding bootstrap -- "
            "cross-species contamination from sp0's donor pool (ruling R25)."
        )

    def test_var_gt0_species_seeding_bootstrap_still_draws_from_its_own_pool(self):
        """Sanity check on the other side of the mask: sp0's own seeding bootstrap must
        still draw from (only) sp0's real pool -- i.e. the fix must not zero everyone
        out, just stop the cross-species leak."""
        from osmose.engine.genetics.inheritance import create_offspring_genotypes

        gs, species_id, gonad_weight = self._two_species_genetic_state()
        rng = np.random.default_rng(999)

        offspring = create_offspring_genotypes(
            parent_gs=gs,
            gonad_weight=gonad_weight,
            species_id=species_id,
            offspring_species=0,
            n_offspring=20,
            rng=rng,
            seeding=True,
        )

        assert np.all(offspring.alleles["imax"] == 9.0)

    def test_var0_species_with_zero_living_conspecifics_falls_back_not_to_cross_species(
        self,
    ):
        """The exact smelt scenario: the collapsed species (sp1) has NO living schools
        at all (SSB structurally 0, like baltic_ev smelt past step ~342), while sp0
        is alive and carries real, non-zero alleles. Pre-fix, `parent_gs.alleles[name]`
        was the WHOLE population -- with only sp0 alive, every donor draw was
        guaranteed (not just likely) to land on sp0, so this is the single strongest
        red case for the mask. Post-fix, `conspecific_indices` for sp1 is empty, so
        the seeding branch must fall through to `trait.allele_pool[1]` (also empty for
        a var=0 species), producing all-zero offspring -- never sp0's 9.0."""
        from osmose.engine.genetics.genotype import GeneticState
        from osmose.engine.genetics.inheritance import create_offspring_genotypes
        from osmose.engine.genetics.trait import Trait, TraitRegistry

        n_loci = np.array([3, 3], dtype=np.int32)
        trait = Trait(
            name="imax",
            species_mean=np.array([3.5, 2.0]),
            species_var=np.array([0.5, 0.0]),
            env_var=np.array([0.0, 0.0]),
            n_loci=n_loci,
            n_alleles=np.array([5, 5], dtype=np.int32),
            target_param="bioen_i_max",
            allele_pool=[
                [np.full(5, 9.0) for _ in range(3)],  # sp0 pool
                [],  # sp1: var=0 -> no pool, exactly as production
            ],
        )
        registry = TraitRegistry()
        registry.register(trait)

        # Only sp0 is alive. sp1 has zero living schools -- SSB == 0, no conspecifics
        # of any kind (not even non-mature ones) to draw from.
        species_id = np.array([0, 0, 0], dtype=np.int32)
        alleles = np.full((3, 3, 2), 9.0)
        gs = GeneticState(
            alleles={"imax": alleles}, env_noise={"imax": np.zeros(3)}, registry=registry
        )
        gonad_weight = np.zeros(3)

        offspring = create_offspring_genotypes(
            parent_gs=gs,
            gonad_weight=gonad_weight,
            species_id=species_id,
            offspring_species=1,
            n_offspring=20,
            rng=np.random.default_rng(42),
            seeding=True,
        )

        assert np.all(offspring.alleles["imax"] == 0.0), (
            "sp1 has zero living conspecifics but sp0 is alive with real alleles -- "
            "offspring must fall back to the empty trait.allele_pool (all zeros), "
            "never draw a sp0 donor from the whole-population pool (ruling R25)."
        )


class TestBioenParentWeight:
    """Review of #343. Java weighs a parent by the eggs it releases
    (`BioenReproductionProcess.run`: `weight_rand.add(nEgg, school)`). The engine used the gonad
    left AFTER release, which ranks parents by what they kept, and is 0 for every school when
    the season factor is 1: then no parent had a weight and inheritance stopped the run."""

    def _mature_schools(self, config):
        from osmose.engine.state import SchoolState

        state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
        return state.replace(
            abundance=np.array([100.0, 400.0, 50.0]),
            length=np.full(3, 100.0),  # far above the maturity line
            weight=np.full(3, 0.01),
            age_dt=np.full(3, 30, dtype=np.int32),
            gonad_weight=np.array([2e-6, 1e-6, 0.0]),  # the third has nothing to release
        )

    def test_the_parent_weight_is_the_eggs_released(self):
        from dataclasses import replace

        from osmose.engine.simulate import _bioen_reproduction

        config = EngineConfig.from_dict(_bioen_genetics_config())
        n_sp, n_dt = config.n_species, config.n_dt_per_year
        for season in (0.25, 1.0):
            cfg = replace(config, spawning_season=np.full((n_sp, n_dt), season))
            state = self._mature_schools(cfg)
            seeded_out: dict = {}
            out = _bioen_reproduction(
                state, cfg, step=20, rng=np.random.default_rng(0), seeded_out=seeded_out
            )
            w = seeded_out["parent_weight"]
            assert len(w) == len(state)
            # Proportional to released gonad x abundance: 2e-6 x 100 against 1e-6 x 400.
            assert w[2] == 0.0 and w[0] > 0 and w[1] > 0
            assert w[1] / w[0] == pytest.approx(2.0)
            if season == 1.0:
                # Everything was released: no gonad left, and still two parents.
                assert (out.gonad_weight[:3] == 0.0).all()

    def test_a_full_release_does_not_stop_inheritance(self, monkeypatch):
        """End to end: genetics past its seeding window, the season factor 1 every step."""
        import osmose.engine.config as config_mod
        from osmose.engine import genetics

        cfg_dict = _bioen_genetics_config()
        cfg_dict["evolution.seeding.year"] = "0"
        cfg_dict["simulation.time.nyear"] = "2"
        cfg_dict["population.seeding.biomass.sp0"] = "1000.0"  # the fixture has no fish without it
        # A maturity line every school crosses early (the fixture's 10 cm + 0.5 cm/yr is not
        # reached in two years), so that real parents exist and inheritance is exercised.
        for key in ("species.bioen.m0.sp0", "evolution.trait.m0.mean.sp0"):
            cfg_dict[key] = "0.11"  # just above the 0.1 cm egg: the first cohorts mature
        for key in ("species.bioen.m1.sp0", "evolution.trait.m1.mean.sp0"):
            cfg_dict[key] = "0.0"
        # No genetic variance, so the trajectory does not depend on the allele pools (seeded
        # from hash(name), which Python randomises per process).
        for key in [k for k in cfg_dict if k.startswith("evolution.trait.") and ".var." in k]:
            cfg_dict[key] = "0.0"
        real_seasons = config_mod._load_spawning_seasons

        def every_step_full(cfg, n_sp, n_dt):
            real_seasons(cfg, n_sp, n_dt)
            return np.ones((n_sp, n_dt))

        monkeypatch.setattr(config_mod, "_load_spawning_seasons", every_step_full)
        inherited: list = []
        real = genetics.create_offspring_genotypes

        def spy(**kw):
            if not kw["seeding"]:
                inherited.append(
                    kw["gonad_weight"][kw["species_id"] == kw["offspring_species"]].copy()
                )
            return real(**kw)

        monkeypatch.setattr(genetics, "create_offspring_genotypes", spy)
        config = EngineConfig.from_dict(cfg_dict)
        assert (config.spawning_season == 1.0).all()
        outputs = simulate(config, Grid.from_dimensions(ny=3, nx=3), np.random.default_rng(7))
        assert len(outputs) == 24
        assert inherited and all((w > 0).any() for w in inherited)
