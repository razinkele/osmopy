"""OSMOSE 4.4.0's stochastic maturity ogive (#137): ``species.maturity.mode.sp# = stochastic``.

Java (``Species.isSexuallyMature``, v4.4.1): a school not yet mature becomes mature this step with
probability (F(L) - F(L0)) / (1 - F(L0)), F the normal CDF with mean l50 and sd (l75 - l50) /
0.6745, L its length now and L0 its length at the start of the step; once mature, it stays so.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from osmose.engine.config import EngineConfig
from osmose.engine.grid import Grid
from osmose.engine.processes.maturity import (
    Z75,
    maturation_probability,
    mature_mask,
    mature_stochastically,
)
from osmose.engine.processes.reproduction import reproduction
from osmose.engine.simulate import simulate
from osmose.engine.state import SchoolState


def _config(**extra: str) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "3",
        "simulation.nspecies": "2",
        "mortality.subdt": "10",
    }
    for i, name in enumerate(("Ogive", "Threshold")):
        cfg |= {
            f"simulation.nschool.sp{i}": "5",
            f"species.name.sp{i}": name,
            f"species.linf.sp{i}": "30.0",
            f"species.k.sp{i}": "0.5",
            f"species.t0.sp{i}": "-0.1",
            f"species.egg.size.sp{i}": "0.1",
            f"species.length2weight.condition.factor.sp{i}": "0.006",
            f"species.length2weight.allometric.power.sp{i}": "3.0",
            f"species.lifespan.sp{i}": "4",
            f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
            f"predation.ingestion.rate.max.sp{i}": "3.5",
            f"predation.efficiency.critical.sp{i}": "0.57",
            f"species.sexratio.sp{i}": "0.5",
            f"species.relativefecundity.sp{i}": "800",
            f"species.maturity.size.sp{i}": "12.0",
            f"population.seeding.biomass.sp{i}": "50000",
        }
    return cfg | extra


STOCHASTIC = {
    "species.maturity.mode.sp0": "stochastic",
    "species.maturity.l50.sp0": "15.0",
    "species.maturity.l75.sp0": "18.0",
}


class TestTheKeys:
    def test_no_mode_means_no_stochastic_species(self):
        c = EngineConfig.from_dict(_config())
        assert c.maturity_stochastic is None
        assert (
            EngineConfig.from_dict(
                _config(**{"species.maturity.mode.sp0": "legacy"})
            ).maturity_stochastic
            is None
        )

    def test_a_stochastic_species_carries_its_ogive(self):
        c = EngineConfig.from_dict(_config(**STOCHASTIC))
        assert c.maturity_stochastic.tolist() == [True, False]
        assert c.maturity_l50[0] == 15.0
        assert c.maturity_sd[0] == (18.0 - 15.0) / Z75
        assert Z75 == pytest.approx(norm.ppf(0.75), abs=1e-15)

    @pytest.mark.parametrize(
        "bad",
        [
            {"species.maturity.l75.sp0": "15.0"},  # l75 == l50: no spread
            {"species.maturity.l75.sp0": "12.0"},  # l75 < l50
            {"species.maturity.l50.sp0": "nan"},
        ],
    )
    def test_an_ogive_without_a_spread_is_refused(self, bad):
        with pytest.raises(ValueError, match="species.maturity.l75.sp0"):
            EngineConfig.from_dict(_config(**(STOCHASTIC | bad)))

    def test_bioenergetics_ignores_the_mode_as_java_does(self):
        from osmose.engine.config import _parse_stochastic_maturity

        cfg = _config(**STOCHASTIC)
        assert _parse_stochastic_maturity(cfg, 2, 1, bioen=True) == (None, None, None)
        stochastic, l50, sd = _parse_stochastic_maturity(cfg, 2, 1, bioen=False)
        assert stochastic.tolist() == [True, False, False]  # padded for one background species
        assert np.isnan(l50[2]) and np.isnan(sd[1])

    def test_missing_l50_is_refused(self):
        cfg = _config(**STOCHASTIC)
        del cfg["species.maturity.l50.sp0"]
        with pytest.raises(ValueError, match="species.maturity.l50.sp0"):
            EngineConfig.from_dict(cfg)


class TestTheProbability:
    def test_it_is_the_conditional_chance_of_maturing_this_step(self):
        length = np.array([14.0, 15.0, 20.0, 15.0, 30.0])
        start = np.array([13.0, 14.0, 10.0, 15.0, 29.0])
        l50 = np.full(5, 15.0)
        sd = np.full(5, 3.0 / Z75)
        got = maturation_probability(length, start, l50, sd)
        f = norm(15.0, 3.0 / Z75).cdf
        want = (f(length) - f(start)) / (1 - f(start))
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert got[3] == 0.0  # no growth, no chance

    def test_a_school_far_beyond_the_ogive_gets_no_probability_not_an_error(self):
        # F(L0) == 1 in double precision: Java computes 0/0, a NaN, and the draw is never below
        # it. Same here, without a warning.
        with np.errstate(all="raise"):
            p = maturation_probability(
                np.array([200.0]), np.array([199.0]), np.array([15.0]), np.array([1.0])
            )
        assert np.isnan(p[0])


def _schools(cfg: EngineConfig, lengths, starts, species=None, mature=None) -> SchoolState:
    n = len(lengths)
    sid = np.zeros(n, dtype=np.int32) if species is None else np.asarray(species, dtype=np.int32)
    s = SchoolState.create(n_schools=n, species_id=sid)
    return s.replace(
        abundance=np.full(n, 1000.0),
        length=np.asarray(lengths, dtype=float),
        length_start=np.asarray(starts, dtype=float),
        weight=np.full(n, 0.05),
        age_dt=np.full(n, 30, dtype=np.int32),
        **({} if mature is None else {"is_mature": np.asarray(mature, dtype=bool)}),
    )


class TestTheDraw:
    def test_certain_and_impossible_cases(self):
        cfg = EngineConfig.from_dict(_config(**STOCHASTIC))
        # 0: grew from far below to far above l50 (probability 1); 1: did not grow (0);
        # 2: did not grow but is already mature, so stays mature; 3: the threshold species.
        s = _schools(
            cfg,
            [40.0, 16.0, 16.0, 13.0],
            [1.0, 16.0, 16.0, 13.0],
            species=[0, 0, 0, 1],
            mature=[False, False, True, False],
        )
        out = mature_stochastically(s, cfg, np.random.default_rng(0))
        assert out.is_mature[:3].tolist() == [True, False, True]
        assert mature_mask(out, cfg).tolist() == [
            True,
            False,
            True,
            True,
        ]  # sp1 by its 12 cm threshold

    def test_the_frequency_matches_the_probability(self):
        cfg = EngineConfig.from_dict(_config(**STOCHASTIC))
        n = 20000
        s = _schools(cfg, np.full(n, 16.0), np.full(n, 14.0))
        out = mature_stochastically(s, cfg, np.random.default_rng(7))
        p = float(
            maturation_probability(
                np.array([16.0]), np.array([14.0]), np.array([15.0]), np.array([3.0 / Z75])
            )[0]
        )
        assert abs(out.is_mature.mean() - p) < 4 * np.sqrt(p * (1 - p) / n)

    def test_nothing_is_drawn_for_a_config_without_the_ogive(self):
        cfg = EngineConfig.from_dict(_config())
        s = _schools(cfg, [16.0, 20.0], [14.0, 18.0])
        rng = np.random.default_rng(3)
        assert mature_stochastically(s, cfg, rng) is s
        assert rng.random() == np.random.default_rng(3).random()  # the stream is untouched

    def test_a_species_that_does_not_reproduce_is_never_drawn(self):
        cfg = EngineConfig.from_dict(
            _config(**(STOCHASTIC | {"species.relativefecundity.sp0": "0"}))
        )
        s = _schools(cfg, [40.0], [1.0])
        assert mature_stochastically(s, cfg, np.random.default_rng(0)).is_mature.tolist() == [False]


class TestReproductionAndOutputs:
    def test_reproduction_spawns_from_drawn_maturity_and_eggs_are_not_mature(self):
        cfg = EngineConfig.from_dict(_config(**STOCHASTIC))
        late = int(cfg.seeding_max_step.max())  # past seeding: eggs only from real spawners
        # Both schools are above the 12 cm threshold; only school 0 is mature by its draw.
        s = _schools(cfg, [20.0, 20.0], [20.0, 20.0], mature=[True, False])
        out = reproduction(s, cfg, late, np.random.default_rng(1))
        assert out.is_mature[:2].tolist() == [True, False]
        eggs = out.is_mature[2:]
        assert len(eggs) > 0 and not eggs.any()
        # The eggs came from one school's biomass, not two: half of what the threshold gives.
        thr = EngineConfig.from_dict(_config())
        out_thr = reproduction(
            _schools(thr, [20.0, 20.0], [20.0, 20.0]), thr, late, np.random.default_rng(1)
        )
        assert out.abundance[2:].sum() == pytest.approx(out_thr.abundance[2:].sum() / 2)

    def test_ssb_and_life_stage_outputs_use_the_drawn_flag(self):
        from osmose.engine.simulate import _collect_by_life_stage, _collect_ssb

        cfg = EngineConfig.from_dict(_config(**STOCHASTIC))
        s = _schools(cfg, [20.0, 20.0], [20.0, 20.0], mature=[True, False])
        assert _collect_ssb(s, cfg)[0] == pytest.approx(1000.0 * 0.05)
        abundance, _ = _collect_by_life_stage(s, cfg)
        assert abundance[0].tolist() == [0.0, 1000.0, 1000.0]  # one juvenile, one adult

    def test_a_subset_reads_the_same_flags(self):
        # The live-movement view tests only the schools it draws.
        cfg = EngineConfig.from_dict(_config(**STOCHASTIC))
        s = _schools(
            cfg,
            [20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            species=[0, 0, 1],
            mature=[True, False, False],
        )
        where = np.array([False, True, True])
        assert (
            mature_mask(s, cfg, where=where).tolist()
            == mature_mask(s, cfg)[where].tolist()
            == [False, True]
        )

    def test_a_merged_state_without_the_flag_reads_not_mature(self):
        cfg = EngineConfig.from_dict(_config(**STOCHASTIC))
        with_flag = _schools(cfg, [20.0], [20.0], mature=[True])
        without = _schools(cfg, [20.0, 20.0], [20.0, 20.0])
        assert with_flag.append(without).is_mature.tolist() == [True, False, False]
        assert without.append(with_flag).is_mature.tolist() == [False, False, True]


def test_a_whole_run_differs_only_when_asked_for():
    grid = Grid.from_dimensions(ny=3, nx=3)
    plain = simulate(EngineConfig.from_dict(_config()), grid, np.random.default_rng(5))
    legacy = simulate(
        EngineConfig.from_dict(_config(**{"species.maturity.mode.sp0": "legacy"})),
        grid,
        np.random.default_rng(5),
    )
    ogive = simulate(EngineConfig.from_dict(_config(**STOCHASTIC)), grid, np.random.default_rng(5))
    for a, b in zip(plain, legacy, strict=True):
        assert np.array_equal(a.biomass, b.biomass)
    assert any(not np.array_equal(a.biomass, b.biomass) for a, b in zip(plain, ogive, strict=True))


def test_the_csharp_engine_refuses_it():
    from osmose.engine_cs.capabilities import csharp_engine_block_reason

    assert "maturity_stochastic" not in (csharp_engine_block_reason(_config()) or "")
    assert "maturity_stochastic" in (csharp_engine_block_reason(_config(**STOCHASTIC)) or "")
