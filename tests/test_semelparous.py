"""OSMOSE 4.4.x post-reproduction mortality, ``species.reproduction.strategy = semelparous`` (#137).

Java (v4.4.1): reproduction marks a school that lays eggs (``school_nEgg > 0``) with
``setHasSpawned(true, step)``, keeping the step of its FIRST spawning. Every step, after growth and
aging and before reproduction, ``updateSpawningMortality`` kills, cause SPAWNING, a school that has
spawned once ``step - timeAtSpawning >= survival time``. The survival time is
``round(value * ndtperyear)`` from the key Java actually reads, the un-indexed
``species.reproduction.postspawning.survivaltime.sp`` (its format string has no %d), 0 when unset;
iteroparous species never die of it. Java's bioenergetic reproduction never sets the flag.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.config import POSTSPAWNING_SURVIVAL_KEY, EngineConfig
from osmose.engine.processes.natural import spawning_mortality
from osmose.engine.processes.reproduction import record_spawning
from osmose.engine.state import MortalityCause, SchoolState

NDT = 4


def _config(**extra: str) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": str(NDT),
        "simulation.time.nyear": "3",
        "simulation.nspecies": "2",
        "mortality.subdt": "10",
    }
    for i in range(2):
        cfg |= {
            f"simulation.nschool.sp{i}": "5",
            f"species.name.sp{i}": f"Fish{i}",
            f"species.linf.sp{i}": "30.0",
            f"species.k.sp{i}": "0.5",
            f"species.t0.sp{i}": "-0.1",
            f"species.egg.size.sp{i}": "0.1",
            f"species.length2weight.condition.factor.sp{i}": "0.006",
            f"species.length2weight.allometric.power.sp{i}": "3.0",
            f"species.lifespan.sp{i}": "5",
            f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
            f"predation.ingestion.rate.max.sp{i}": "3.5",
            f"predation.efficiency.critical.sp{i}": "0.57",
            f"species.maturity.size.sp{i}": "8.0",
            f"species.sexratio.sp{i}": "0.5",
            f"species.relativefecundity.sp{i}": "500",
            f"population.seeding.biomass.sp{i}": "1000",
        }
    return cfg | {"species.reproduction.strategy.sp0": "semelparous"} | extra


class TestTheKeys:
    def test_absent_means_iteroparous(self):
        c = EngineConfig.from_dict(_config(**{"species.reproduction.strategy.sp0": "iteroparous"}))
        assert c.reproduction_semelparous is None

    def test_semelparous_in_any_case(self):
        c = EngineConfig.from_dict(_config(**{"species.reproduction.strategy.sp0": "SemelParous"}))
        assert c.reproduction_semelparous is not None
        assert list(c.reproduction_semelparous) == [True, False]
        assert c.postspawning_survival_dt == 0

    def test_another_strategy_is_refused(self):
        with pytest.raises(ValueError, match="iteroparous or semelparous"):
            EngineConfig.from_dict(_config(**{"species.reproduction.strategy.sp0": "annual"}))

    def test_the_survival_time_comes_from_javas_unindexed_key(self):
        c = EngineConfig.from_dict(_config(**{POSTSPAWNING_SURVIVAL_KEY: "0.5"}))
        assert c.postspawning_survival_dt == 2  # 0.5 yr x 4 steps

    def test_it_rounds_halves_up_as_javas_math_round(self):
        c = EngineConfig.from_dict(_config(**{POSTSPAWNING_SURVIVAL_KEY: "0.625"}))  # 2.5 steps
        assert c.postspawning_survival_dt == 3

    def test_the_indexed_key_java_never_reads_is_refused(self):
        with pytest.raises(ValueError, match="never read by Java"):
            EngineConfig.from_dict(_config(**{f"{POSTSPAWNING_SURVIVAL_KEY}0": "0.5"}))


def _state(spawned=None, at=None) -> SchoolState:
    """Species 0 (semelparous) and 1 (iteroparous): a mature school each, and an immature one
    of species 0."""
    sid = np.array([0, 1, 0], dtype=np.int32)
    s = SchoolState.create(n_schools=3, species_id=sid).replace(
        abundance=np.array([100.0, 100.0, 100.0]),
        weight=np.array([0.01, 0.01, 0.001]),
        length=np.array([20.0, 20.0, 4.0]),
        age_dt=np.array([8, 8, 2], dtype=np.int32),
    )
    if spawned is not None:
        s = s.replace(has_spawned=np.array(spawned), spawn_step=np.array(at, dtype=np.int32))
    return s


class TestTheFlag:
    def test_it_is_set_for_a_school_that_lays_eggs(self):
        c = EngineConfig.from_dict(_config())
        mature = np.array([True, True, False])
        out = record_spawning(_state(), c, mature, np.array([0.25, 0.25]), 5)
        assert list(out.has_spawned) == [True, False, False]  # species 1 is iteroparous
        assert list(out.spawn_step) == [5, -1, -1]

    def test_it_latches_the_first_spawning(self):
        c = EngineConfig.from_dict(_config())
        s = _state([True, False, False], [3, -1, -1])
        out = record_spawning(s, c, np.array([True, True, False]), np.array([0.25, 0.25]), 7)
        assert list(out.spawn_step) == [3, -1, -1]

    def test_no_eggs_no_flag(self):
        c = EngineConfig.from_dict(_config())
        mature = np.array([True, True, False])
        out = record_spawning(_state(), c, mature, np.array([0.0, 0.25]), 5)  # out of season
        assert not out.has_spawned.any()
        c0 = EngineConfig.from_dict(_config(**{"species.relativefecundity.sp0": "0"}))
        assert not record_spawning(
            _state(), c0, mature, np.array([0.25, 0.25]), 5
        ).has_spawned.any()


class TestTheDeath:
    def test_it_kills_at_the_survival_time_with_cause_spawning(self):
        c = EngineConfig.from_dict(_config(**{POSTSPAWNING_SURVIVAL_KEY: "0.5"}))  # 2 steps
        s = _state([True, True, False], [4, 4, -1])
        assert (spawning_mortality(s, c, 5).abundance == 100.0).all()  # 1 step after spawning
        out = spawning_mortality(s, c, 6)
        assert list(out.abundance) == [0.0, 100.0, 100.0]  # species 1 is iteroparous
        assert out.n_dead[0, MortalityCause.SPAWNING] == 100.0
        assert out.n_dead[:, MortalityCause.SPAWNING].sum() == 100.0

    def test_with_the_default_it_kills_the_step_after_spawning(self):
        c = EngineConfig.from_dict(_config())
        s = _state([True, False, False], [4, -1, -1])
        assert spawning_mortality(s, c, 5).abundance[0] == 0.0


def _run(cfg: dict[str, str]):
    import osmose.engine.simulate as sim
    from osmose.engine import PythonEngine

    config, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg | {"_osmose.config.dir": "."}, 4)
    return config, sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)


class TestAWholeRun:
    def test_spawning_deaths_appear_only_for_the_semelparous_species(self):
        _, out = _run(_config())
        spawning = np.array([o.mortality_by_cause[:, MortalityCause.SPAWNING] for o in out])
        assert spawning[:, 0].sum() > 0 and spawning[:, 1].sum() == 0

    def test_without_the_strategy_no_spawning_death(self):
        _, out = _run(_config(**{"species.reproduction.strategy.sp0": "iteroparous"}))
        assert all((o.mortality_by_cause[:, MortalityCause.SPAWNING] == 0).all() for o in out)

    def test_the_death_runs_after_growth_and_before_the_next_reproduction(self, monkeypatch):
        import osmose.engine.simulate as sim
        from osmose.engine.processes import natural

        order: list[str] = []
        real_growth, real_repro, real_spawn = (
            sim._growth,
            sim._reproduction,
            natural.spawning_mortality,
        )

        def growth(*a, **k):
            order.append("growth")
            return real_growth(*a, **k)

        def repro(*a, **k):
            order.append("reproduction")
            return real_repro(*a, **k)

        def spawn(*a, **k):
            order.append("spawning")
            return real_spawn(*a, **k)

        monkeypatch.setattr(sim, "_growth", growth)
        monkeypatch.setattr(sim, "_reproduction", repro)
        monkeypatch.setattr(natural, "spawning_mortality", spawn)
        _run(_config(**{"simulation.time.nyear": "1"}))
        assert order[:6] == ["growth", "spawning", "reproduction"] * 2


_BIOEN = {
    "module.bioenergetics.enabled": "true",
    "temperature.value": "14.0",
    "simulation.bioen.fo2.enabled": "false",
    **{
        f"{k}.sp{i}": v
        for i in range(2)
        for k, v in {
            "species.beta": "0.75",
            "species.bioen.assimilation": "0.68",
            "species.bioen.maint.energy.c_m": "0.00123",
            "species.maturity.m0": "0.5",
            "species.maturity.m1": "0.1",
            "species.maturity.eta": "1.4",
            "species.maturity.r": "0.45",
            "predation.larval.ingestion.rate.increase.ratio": "1.5",
        }.items()
    },
}


def test_a_bioenergetic_run_has_no_spawning_death(monkeypatch):
    """Java's BioenReproductionProcess has no setHasSpawned: under bioenergetics a semelparous
    species never dies of spawning. Control arm: the same species without bioenergetics does."""
    from osmose.engine.processes import reproduction

    calls: list[int] = []
    real = reproduction.record_spawning

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(reproduction, "record_spawning", spy)
    _, out = _run(_config(**_BIOEN, **{"simulation.time.nyear": "2"}))
    assert calls == []
    assert all((o.mortality_by_cause[:, MortalityCause.SPAWNING] == 0).all() for o in out)

    _, out = _run(_config(**{"simulation.time.nyear": "2"}))
    assert calls  # the standard path records spawning ...
    assert sum(o.mortality_by_cause[0, MortalityCause.SPAWNING] for o in out) > 0  # ... and kills


def test_the_csharp_engine_refuses_it_and_only_it():
    from osmose.engine_cs.capabilities import csharp_engine_block_reason

    assert "semelparous" in (csharp_engine_block_reason(_config()) or "")
    iter_cfg = _config(**{"species.reproduction.strategy.sp0": "iteroparous"})
    assert "semelparous" not in (csharp_engine_block_reason(iter_cfg) or "")
