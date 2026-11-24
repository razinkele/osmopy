"""Genetics on the standard (non-bioenergetic) reproduction path (#338).

Java 4.4.1 does it in ``ReproductionProcess.run_genet``: the two parents of an offspring school
are drawn with the weight of each mature school's egg production (which, within a species, is
proportional to its spawning biomass), and a species bootstrapped from its seeding biomass gets
fresh genotypes. Before this, the Python engine chose parents by gonad weight on both paths,
and only the bioenergetic module sets gonad weight: a run with genetics and no bioenergetics
stopped at its first reproduction outside the genetics seeding window.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine import genetics
from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.simulate import simulate
from osmose.engine.state import SchoolState


def _two_species(**extra: str) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "mortality.subdt": "10",
    }
    for i, name in enumerate(("Spawner", "Bootstrapped")):
        cfg |= {
            f"simulation.nschool.sp{i}": "5",
            f"species.name.sp{i}": name,
            f"species.linf.sp{i}": "30.0",
            f"species.k.sp{i}": "0.3",
            f"species.t0.sp{i}": "-0.1",
            f"species.egg.size.sp{i}": "0.1",
            f"species.length2weight.condition.factor.sp{i}": "0.006",
            f"species.length2weight.allometric.power.sp{i}": "3.0",
            f"species.lifespan.sp{i}": "5",
            f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
            f"predation.ingestion.rate.max.sp{i}": "3.5",
            f"predation.efficiency.critical.sp{i}": "0.57",
            f"species.sexratio.sp{i}": "0.5",
            f"species.relativefecundity.sp{i}": "800",
            f"species.maturity.size.sp{i}": "12.0",
            f"population.seeding.biomass.sp{i}": "50000",
        }
    return cfg | extra


def _schools() -> SchoolState:
    """Five schools. Species 0: two mature ones, one too short, one mature in size but empty.
    Species 1: one school, too short, so its spawning stock is zero."""
    state = SchoolState.create(n_schools=5, species_id=np.array([0, 0, 0, 0, 1], dtype=np.int32))
    return state.replace(
        abundance=np.array([1000.0, 250.0, 4000.0, 0.0, 600.0]),
        length=np.array([20.0, 28.0, 5.0, 25.0, 6.0]),
        weight=np.array([0.05, 0.13, 0.001, 0.09, 0.002]),
        age_dt=np.array([24, 40, 3, 30, 4], dtype=np.int32),
    )


class TestWhatReproductionReports:
    def test_each_mature_school_weighs_its_spawning_biomass(self):
        cfg = EngineConfig.from_dict(_two_species())
        state = _schools()
        spawn: dict = {}
        after = reproduction(state, cfg, 0, np.random.default_rng(1), spawn_out=spawn)
        # One weight per school handed in, in their order: abundance x weight where the school
        # is mature, nothing where it is too short or holds no fish.
        assert spawn["parent_weight"].tolist() == [1000.0 * 0.05, 250.0 * 0.13, 0.0, 0.0, 0.0]
        # Species 1 has no spawning stock and is inside its seeding period: bootstrapped.
        assert spawn["seeded_this_step"].tolist() == [False, True]
        # Those schools are the first rows of the state that comes back, so the weights line
        # up with the parents the genetics step looks at; both species got offspring.
        assert np.array_equal(after.species_id[:5], state.species_id)
        assert set(after.species_id[5:].tolist()) == {0, 1}

    def test_after_the_seeding_period_a_species_without_spawners_gets_no_offspring(self):
        cfg = EngineConfig.from_dict(_two_species())
        late = int(cfg.seeding_max_step.max())  # the first step after every seeding period
        spawn: dict = {}
        after = reproduction(_schools(), cfg, late, np.random.default_rng(1), spawn_out=spawn)
        assert spawn["seeded_this_step"].tolist() == [False, False]
        assert set(after.species_id[5:].tolist()) == {0}  # so no offspring is ever parentless

    def test_asking_for_the_report_changes_nothing_else(self):
        cfg = EngineConfig.from_dict(_two_species())
        plain = reproduction(_schools(), cfg, 0, np.random.default_rng(1))
        asked = reproduction(_schools(), cfg, 0, np.random.default_rng(1), spawn_out={})
        for name in ("species_id", "abundance", "length", "weight", "age_dt", "cell_x", "cell_y"):
            assert np.array_equal(getattr(plain, name), getattr(asked, name)), name


def _evolving(**extra: str) -> dict[str, str]:
    cfg = _two_species(**{"simulation.nspecies": "1"})
    cfg = {k: v for k, v in cfg.items() if not k.endswith(".sp1")}
    cfg |= {
        "species.lifespan.sp0": "3",
        "species.linf.sp0": "20.0",
        "species.maturity.size.sp0": "4.0",  # reached within the first year
        "population.seeding.biomass.sp0": "1000",
        "simulation.genetic.enabled": "true",
        "evolution.trait.imax.mean.sp0": "3.5",
        "evolution.trait.imax.var.sp0": "0.1",
        "evolution.trait.imax.envvar.sp0": "0.05",
        "evolution.trait.imax.nlocus.sp0": "5",
        "evolution.trait.imax.nval.sp0": "20",
        "evolution.trait.imax.target": "bioen_i_max",
    }
    return cfg | extra


def _run_spied(raw: dict[str, str], monkeypatch) -> tuple[list, list[dict]]:
    calls: list[dict] = []
    real = genetics.create_offspring_genotypes

    def spy(**kw):
        calls.append(
            {k: kw[k] for k in ("gonad_weight", "species_id", "offspring_species", "seeding")}
        )
        return real(**kw)

    monkeypatch.setattr(genetics, "create_offspring_genotypes", spy)
    cfg = EngineConfig.from_dict(raw)
    out = simulate(cfg, Grid.from_dimensions(ny=3, nx=3), np.random.default_rng(42))
    return out, calls


class TestAWholeRunWithoutBioenergetics:
    def test_the_run_completes_and_offspring_inherit_from_spawners(self, monkeypatch):
        out, calls = _run_spied(_evolving(), monkeypatch)
        assert len(out) == 12
        bootstrapped = [c for c in calls if c["seeding"]]
        inherited = [c for c in calls if not c["seeding"]]
        # The first cohorts come from the seeding biomass (no spawner yet): fresh genotypes.
        # Once schools mature, offspring have parents, and every such call is handed a
        # positive weight for some school of the offspring's species.
        assert bootstrapped and inherited
        for c in inherited:
            of_species = c["species_id"] == c["offspring_species"]
            assert len(c["gonad_weight"]) == len(c["species_id"])
            assert (c["gonad_weight"][of_species] > 0).any()
            assert (c["gonad_weight"] >= 0).all()

    def test_inside_the_genetics_seeding_window_every_cohort_is_fresh(self, monkeypatch):
        out, calls = _run_spied(_evolving(**{"evolution.seeding.year": "2"}), monkeypatch)
        assert len(out) == 12
        assert calls and all(c["seeding"] for c in calls)

    def test_genetics_off_takes_no_extra_draw(self):
        # The same config with genetics off, twice, with and without the keys that genetics
        # reads: identical biomass. (The fixed-seed baselines of tests/test_engine_parity.py
        # are the wider guard on the standard path's random stream.)
        raw = _evolving(**{"simulation.genetic.enabled": "false"})
        grid = Grid.from_dimensions(ny=3, nx=3)
        a = simulate(EngineConfig.from_dict(raw), grid, np.random.default_rng(42))
        plain = {
            k: v for k, v in raw.items() if not k.startswith(("evolution.", "simulation.genetic"))
        }
        b = simulate(EngineConfig.from_dict(plain), grid, np.random.default_rng(42))
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x.biomass, y.biomass)


class TestTheTransmissionKey:
    def test_javas_key_is_read(self):
        """Java 4.4.1 reads `population.genotype.transmission.year.start`; the Python engine
        read only its own `evolution.seeding.year`."""
        java = EngineConfig.from_dict(
            _evolving(**{"population.genotype.transmission.year.start": "3"})
        )
        assert java.genetics_transmission_year == 3
        own = EngineConfig.from_dict(_evolving(**{"evolution.seeding.year": "4"}))
        assert own.genetics_transmission_year == 4
        both = EngineConfig.from_dict(
            _evolving(
                **{
                    "population.genotype.transmission.year.start": "3",
                    "evolution.seeding.year": "4",
                }
            )
        )
        assert both.genetics_transmission_year == 3  # Java's key wins when both are given
        assert EngineConfig.from_dict(_evolving()).genetics_transmission_year == 0


def test_no_eligible_parent_is_still_an_error_at_the_function():
    """The routine itself still refuses to invent genotypes: asked for inherited offspring with
    no school of the species carrying a parent weight, it raises. The engine no longer gets
    there on the standard path; a caller that does has a bug."""
    registry = genetics.TraitRegistry()
    registry.register(
        genetics.Trait(
            name="imax", species_mean=np.array([3.5]), species_var=np.array([0.1]),
            env_var=np.array([0.05]), n_loci=np.array([3], dtype=np.int32),
            n_alleles=np.array([20], dtype=np.int32), target_param="ingestion_rate",
        )
    )  # fmt: skip
    parents = genetics.GeneticState(
        alleles={"imax": np.ones((2, 3, 2))}, env_noise={"imax": np.zeros(2)}, registry=registry
    )
    with pytest.raises(ValueError, match="no eligible parents"):
        genetics.create_offspring_genotypes(
            parent_gs=parents, gonad_weight=np.zeros(2), species_id=np.zeros(2, dtype=np.int32),
            offspring_species=0, n_offspring=5, rng=np.random.default_rng(1),
        )  # fmt: skip
