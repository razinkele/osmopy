"""OSMOSE 4.4.x multiplier and seasonality of the constant mortality rates (#137).

Java (``AnnualAdditionalMortality``, ``AnnualLarvaMortality``, ``SeasonTimeSeries``, v4.4.1): the
rate of step t is ``multiplier * rate * season[t] / nStepYear``. The multiplier defaults to 1 and
the season to 1 at every step; the season is read from ``<prefix>.seasonality.file.sp#`` or from
``<prefix>.seasonality.sp#`` (one value per step of a year, repeated, or one per step of the run),
and is not normalised. Both are read only in the constant scenarios: a by-step rate replaces them.
The larval prefix is ``mortality.additional.larva.rate``, or ``mortality.additional.neonate.rate``
for a species on Java's VCONSTANT ("neonate") scenario, which is the same process under its own
keys.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import additional_mortality, larva_mortality
from osmose.engine.state import SchoolState

ADD = "mortality.additional.rate"
LARVA = "mortality.additional.larva.rate"
NEO = "mortality.additional.neonate.rate"
NDT = 4
NYEAR = 2
NSTEP = NDT * NYEAR


def _config(**extra: str) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": str(NDT),
        "simulation.time.nyear": str(NYEAR),
        "simulation.nspecies": "2",
        "mortality.subdt": "10",
    }
    for i in range(2):
        cfg |= {
            f"simulation.nschool.sp{i}": "5",
            f"species.name.sp{i}": f"Fish{i}",
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
            f"species.maturity.size.sp{i}": "12.0",
            f"{ADD}.sp{i}": "0.5",
            f"{LARVA}.sp{i}": "1.0",
            f"population.seeding.biomass.sp{i}": "1000",
        }
    return cfg | extra


def _load(**extra: str) -> EngineConfig:
    return EngineConfig.from_dict(_config(**extra))


class TestTheKeys:
    def test_without_them_nothing_is_set(self):
        c = _load()
        assert c.additional_mortality_factor is None and c.larva_mortality_factor is None

    def test_a_multiplier_scales_one_species(self):
        f = _load(**{f"{ADD}.multiplier.sp1": "2.5"}).additional_mortality_factor
        assert f is not None and f.shape == (2, NSTEP)
        assert (f[0] == 1.0).all() and (f[1] == 2.5).all()

    def test_a_season_of_one_year_repeats(self):
        f = _load(**{f"{ADD}.seasonality.sp0": "0.5;1.0;1.5;2.0"}).additional_mortality_factor
        assert f is not None
        assert list(f[0]) == [0.5, 1.0, 1.5, 2.0] * NYEAR
        assert (f[1] == 1.0).all()

    def test_a_season_of_the_whole_run_is_taken_as_is(self):
        vals = [0.25 * (k + 1) for k in range(NSTEP)]
        f = _load(
            **{f"{ADD}.seasonality.sp0": ";".join(map(str, vals))}
        ).additional_mortality_factor
        assert f is not None and list(f[0]) == vals

    def test_a_season_of_another_length_is_refused(self):
        with pytest.raises(ValueError, match="seasonality.sp0 has 3 values"):
            _load(**{f"{ADD}.seasonality.sp0": "1;2;3"})

    def test_a_season_from_a_file(self, tmp_path):
        path = tmp_path / "season.csv"
        path.write_text("step;value\n0;0.5\n1;1.0\n2;1.5\n3;2.0\n")
        cfg = _config(**{f"{ADD}.seasonality.file.sp1": str(path)})
        f = EngineConfig.from_dict(cfg).additional_mortality_factor
        assert f is not None and list(f[1]) == [0.5, 1.0, 1.5, 2.0] * NYEAR

    def test_multiplier_and_season_multiply_and_are_not_normalised(self):
        f = _load(
            **{f"{ADD}.multiplier.sp0": "2.0", f"{ADD}.seasonality.sp0": "1;1;1;5"}
        ).additional_mortality_factor
        assert f is not None and list(f[0]) == [2.0, 2.0, 2.0, 10.0] * NYEAR

    def test_null_means_unset_as_in_java(self):
        assert _load(**{f"{ADD}.multiplier.sp0": "null"}).additional_mortality_factor is None

    def test_the_larval_keys_are_separate(self):
        c = _load(**{f"{LARVA}.multiplier.sp0": "3.0"})
        assert c.additional_mortality_factor is None
        assert c.larva_mortality_factor is not None and (c.larva_mortality_factor[0] == 3.0).all()


class TestTheNeonateScenario:
    def _neo(self, **extra: str) -> dict[str, str]:
        cfg = _config(**extra)
        del cfg[f"{LARVA}.sp1"]
        return cfg | {f"{NEO}.sp1": "0.75"}

    def test_its_rate_is_the_larval_rate(self):
        c = EngineConfig.from_dict(self._neo())
        assert c.larva_mortality_rate[1] == 0.75 and c.larva_mortality_rate[0] == 1.0

    def test_its_own_multiplier_and_season_apply(self):
        c = EngineConfig.from_dict(
            self._neo(**{f"{NEO}.multiplier.sp1": "2.0", f"{LARVA}.multiplier.sp1": "9.0"})
        )
        assert c.larva_mortality_factor is not None and (c.larva_mortality_factor[1] == 2.0).all()

    def test_with_the_larval_rate_too_it_is_refused(self):
        with pytest.raises(ValueError, match="choose one larval scenario"):
            _load(**{f"{NEO}.sp0": "0.75"})

    def test_it_ignores_the_reference_density_as_java_does(self):
        cfg = self._neo(**{"mortality.additional.larva.referencedensity.sp1": "100"})
        assert EngineConfig.from_dict(cfg).larva_reference_density is None

    def test_the_reader_converts_it_from_per_year_like_the_larval_rate(self):
        from osmose.config.aliases import _migrate_larva_rate

        out = _migrate_larva_rate(
            {
                "simulation.time.ndtperyear": "4",
                f"{NEO}.sp0": "8",
                f"{LARVA}.sp1": "8",
                f"{NEO}.multiplier.sp0": "8",
                f"{LARVA}.seasonality.sp1": "8;8;8;8",
            },
            0.25,
            warn_bydt=False,
        )
        assert out[f"{NEO}.sp0"] == "2" and out[f"{LARVA}.sp1"] == "2"
        assert out[f"{NEO}.multiplier.sp0"] == "8" and out[f"{LARVA}.seasonality.sp1"] == "8;8;8;8"


def _state() -> SchoolState:
    """Species 0 and 1: one egg school and one older school each."""
    sid = np.array([0, 0, 1, 1], dtype=np.int32)
    s = SchoolState.create(n_schools=4, species_id=sid)
    return s.replace(
        abundance=np.full(4, 1e6),
        weight=np.array([1e-9, 0.5, 1e-9, 0.5]),
        length=np.array([0.1, 20.0, 0.1, 20.0]),
        age_dt=np.array([0, 12, 0, 12], dtype=np.int32),
        is_egg=np.array([True, False, True, False]),
    )


def _deaths(fn, config: EngineConfig, step: int, **kw) -> np.ndarray:
    return fn(_state(), config, step=step, **kw).n_dead[:, 2]  # ADDITIONAL


class TestTheRates:
    def test_the_additional_rate_of_a_step_is_scaled(self):
        season = "0.5;1.0;1.5;2.0"
        c = _load(**{f"{ADD}.multiplier.sp0": "2.0", f"{ADD}.seasonality.sp0": season})
        base = _load()
        for step in (0, 3, 6):
            got = _deaths(additional_mortality, c, step, n_subdt=10)
            want = _deaths(additional_mortality, base, step, n_subdt=10)
            factor = 2.0 * [0.5, 1.0, 1.5, 2.0][step % NDT]
            d = 0.5 * factor / NDT / 10
            assert got[1] == pytest.approx(1e6 * (1 - math.exp(-d)), rel=1e-12)
            assert got[3] == want[3]  # species 1 unchanged

    def test_a_by_step_additional_rate_replaces_the_factor(self, tmp_path):
        path = tmp_path / "add.csv"
        path.write_text("\n".join(["0.25"] * NDT) + "\n")
        c = _load(**{f"{ADD}.multiplier.sp0": "8.0", f"{ADD}.bytdt.file.sp0": str(path)})
        got = _deaths(additional_mortality, c, 2, n_subdt=10)
        d = 0.25 / NDT / 10
        assert got[1] == pytest.approx(1e6 * (1 - math.exp(-d)), rel=1e-12)

    def test_the_larval_rate_of_a_step_is_scaled(self):
        c = _load(**{f"{LARVA}.seasonality.sp1": "0.5;1.0;1.5;2.0"})
        got = _deaths(larva_mortality, c, 3)
        assert got[2] == pytest.approx(1e6 * (1 - math.exp(-1.0 * 2.0)), rel=1e-12)
        assert got[0] == pytest.approx(1e6 * (1 - math.exp(-1.0)), rel=1e-12)

    def test_a_by_step_larval_rate_replaces_the_factor(self, tmp_path):
        path = tmp_path / "larva.csv"
        path.write_text("step;value\n" + "".join(f"{k};0.25\n" for k in range(NDT)))
        c = _load(**{f"{LARVA}.multiplier.sp0": "8.0", f"{LARVA}.bytdt.file.sp0": str(path)})
        got = _deaths(larva_mortality, c, 1)
        assert got[0] == pytest.approx(1e6 * (1 - math.exp(-0.25)), rel=1e-12)

    def test_the_batch_precompute_is_scaled(self):
        from osmose.engine.processes import mortality as m

        c = _load(**{f"{ADD}.multiplier.sp0": "2.0", f"{ADD}.seasonality.sp0": "1;1;3;1"})
        state = _state()
        eff_add = m._precompute_effective_rates(state, c, 10, 2)[1]
        base = m._precompute_effective_rates(state, _load(), 10, 2)[1]
        assert eff_add[1] == pytest.approx(base[1] * 6.0, rel=1e-12)
        assert eff_add[3] == base[3]

    def test_the_per_cell_path_is_scaled(self):
        from osmose.engine.processes import mortality as m

        c = _load(**{f"{ADD}.multiplier.sp0": "2.0", f"{ADD}.seasonality.sp0": "1;1;3;1"})
        out = {}
        for name, config in (("factor", c), ("base", _load())):
            state = _state().replace(first_feeding_age_dt=np.zeros(4, dtype=np.int32))
            m._apply_additional_for_school(1, state, config, 10, state.abundance.copy(), step=2)
            out[name] = state.n_dead[1, 2]
        d_base = 0.5 / NDT / 10
        assert out["base"] == pytest.approx(1e6 * (1 - math.exp(-d_base)), rel=1e-12)
        assert out["factor"] == pytest.approx(1e6 * (1 - math.exp(-6.0 * d_base)), rel=1e-12)


def _run(cfg: dict[str, str]):
    import osmose.engine.simulate as sim
    from osmose.engine import PythonEngine

    config, grid, rng, mv, mo = PythonEngine()._prepare_run(cfg | {"_osmose.config.dir": "."}, 7)
    return sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)


def _same(a, b) -> bool:
    return all(
        np.array_equal(x.biomass, y.biomass) and np.array_equal(x.abundance, y.abundance)
        for x, y in zip(a, b, strict=True)
    )


class TestAWholeRun:
    """Exact binary values, so the factor and an equivalent written-out rate agree bit for bit."""

    def test_a_multiplier_equals_the_multiplied_rate(self):
        a = _run(_config(**{f"{ADD}.sp0": "0.25", f"{ADD}.multiplier.sp0": "2.0"}))
        b = _run(_config(**{f"{ADD}.sp0": "0.5"}))
        c = _run(_config(**{f"{ADD}.sp0": "0.25"}))
        assert _same(a, b) and not _same(a, c)

    def test_a_season_equals_the_same_by_step_file(self, tmp_path):
        season = [0.5, 1.0, 1.5, 2.0]
        path = tmp_path / "add.csv"
        path.write_text("\n".join(str(0.5 * s) for s in season) + "\n")
        a = _run(_config(**{f"{ADD}.seasonality.sp1": ";".join(map(str, season))}))
        b = _run(_config(**{f"{ADD}.bytdt.file.sp1": str(path)}))
        assert _same(a, b) and not _same(a, _run(_config()))

    def test_a_larval_multiplier_equals_the_multiplied_rate(self):
        a = _run(_config(**{f"{LARVA}.sp1": "0.5", f"{LARVA}.multiplier.sp1": "2.0"}))
        b = _run(_config(**{f"{LARVA}.sp1": "1.0"}))
        assert _same(a, b) and not _same(a, _run(_config(**{f"{LARVA}.sp1": "0.5"})))


def test_the_csharp_engine_refuses_the_keys_and_only_them():
    from osmose.engine_cs.capabilities import csharp_engine_block_reason

    assert "mortality_rate_factor" not in (csharp_engine_block_reason(_config()) or "")
    for key in (f"{ADD}.multiplier.sp0", f"{LARVA}.seasonality.sp1"):
        value = "2" if "multiplier" in key else "1;1;1;2"
        assert "mortality_rate_factor" in (
            csharp_engine_block_reason(_config(**{key: value})) or ""
        )


def test_the_sp1b_guard_refuses_a_larval_multiplier_on_the_stock():
    from osmose.calibration.larva_recal import resolved_d0

    cfg = _config()
    assert resolved_d0(cfg, "Fish0") == 1.0
    with pytest.raises(ValueError, match="larva.rate.multiplier.sp0 is set"):
        resolved_d0(cfg | {f"{LARVA}.multiplier.sp0": "2"}, "Fish0")
    assert resolved_d0(cfg | {f"{LARVA}.multiplier.sp1": "2"}, "Fish0") == 1.0
