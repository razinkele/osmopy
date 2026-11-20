"""OSMOSE 4.4.x's simplified ("lite") bioenergetic model (#137).

Java (``EnergyBudget``, v4.4.1): ``species.bioenergetics.model.sp#`` unset or exactly ``full``
keeps the full model; any other value selects ``computeEnetLite``:

    E_net = assimilation * (ingestion * f_T * f_O2
                            - Imax / nStepYear * theta * xi_crit * w[g]^beta * N * 1e-6)

with ``f_T = (T - T1)(T - T2) / ((T - T1)(T - T2) - (T - Topt)^2)``, the roots T1, T2 derived from
``species.temperature.tmin/tmax/topt`` and ``predation.efficiency.critical``, and ``theta`` the
larval ingestion ratio below the larval threshold age. Java never sets E_gross or E_maint for a
lite school.
"""

from __future__ import annotations

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.energy_budget import (
    energy_terms_lite,
    lite_temperature_bounds,
    lite_temperature_factor,
)

LITE = {
    "species.bioenergetics.model.sp0": "lite",
    "species.temperature.tmin.sp0": "2.0",
    "species.temperature.tmax.sp0": "26.0",
    "species.temperature.topt.sp0": "16.0",
}


def _bioen(**extra: str) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "2",
        "mortality.subdt": "10",
        "module.bioenergetics.enabled": "true",
        "temperature.value": "14.0",
        "simulation.bioen.fo2.enabled": "false",
    }
    for i in range(2):
        cfg |= {
            f"simulation.nschool.sp{i}": "8",
            f"species.name.sp{i}": f"Fish{i}",
            f"species.linf.sp{i}": "20.0",
            f"species.k.sp{i}": "0.3",
            f"species.t0.sp{i}": "-0.1",
            f"species.egg.size.sp{i}": "0.1",
            f"species.length2weight.condition.factor.sp{i}": "0.006",
            f"species.length2weight.allometric.power.sp{i}": "3.0",
            f"species.lifespan.sp{i}": "3",
            f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
            f"predation.ingestion.rate.max.sp{i}": "3.5",
            f"predation.efficiency.critical.sp{i}": "0.57",
            f"species.beta.sp{i}": "0.75",
            f"species.bioen.assimilation.sp{i}": "0.68",
            f"species.bioen.maint.energy.c_m.sp{i}": "0.00123",
            f"species.maturity.m0.sp{i}": "4.5",
            f"species.maturity.m1.sp{i}": "1.8",
            f"species.maturity.eta.sp{i}": "1.4",
            f"species.maturity.r.sp{i}": "0.45",
            f"predation.larval.ingestion.rate.increase.ratio.sp{i}": "1.5",
            f"species.larvae.growth.threshold.age.sp{i}": "0.25",
            f"population.seeding.biomass.sp{i}": "5000",
        }
    return cfg | extra


class TestTheKey:
    def test_absent_means_the_full_model(self):
        assert EngineConfig.from_dict(_bioen()).bioen_lite is None

    def test_full_means_the_full_model(self):
        c = EngineConfig.from_dict(_bioen(**{"species.bioenergetics.model.sp0": "full"}))
        assert c.bioen_lite is None

    @pytest.mark.parametrize("value", ["lite", "simple", "Full", "FULL"])
    def test_any_other_value_is_lite_as_javas_equals(self, value):
        c = EngineConfig.from_dict(_bioen(**(LITE | {"species.bioenergetics.model.sp0": value})))
        assert c.bioen_lite is not None and list(c.bioen_lite) == [True, False]

    def test_a_lite_species_needs_its_temperatures(self):
        cfg = _bioen(**LITE)
        del cfg["species.temperature.topt.sp0"]
        with pytest.raises(ValueError, match="species.temperature.topt.sp0"):
            EngineConfig.from_dict(cfg)

    def test_tmax_not_above_tmin_is_refused(self):
        with pytest.raises(ValueError, match="tmax.sp0 must be above"):
            EngineConfig.from_dict(_bioen(**(LITE | {"species.temperature.tmax.sp0": "2.0"})))

    @pytest.mark.parametrize("xi", ["1.0", "0.0"])
    def test_a_degenerate_critical_efficiency_is_refused(self, xi):
        with pytest.raises(ValueError, match="efficiency.critical.sp0 must be in"):
            EngineConfig.from_dict(_bioen(**(LITE | {"predation.efficiency.critical.sp0": xi})))

    @pytest.mark.parametrize("key", ["tmin", "tmax", "topt"])
    def test_an_infinite_temperature_is_refused(self, key):
        with pytest.raises(ValueError, match="finite temperatures"):
            EngineConfig.from_dict(_bioen(**(LITE | {f"species.temperature.{key}.sp0": "inf"})))

    @pytest.mark.parametrize("limit", ["2.0", "26.0"])
    def test_an_optimum_on_a_limit_is_refused(self, limit):
        with pytest.raises(ValueError, match="equals a thermal limit"):
            EngineConfig.from_dict(_bioen(**(LITE | {"species.temperature.topt.sp0": limit})))

    def test_the_roots_are_carried(self):
        c = EngineConfig.from_dict(_bioen(**LITE))
        t1, t2 = lite_temperature_bounds(2.0, 26.0, 16.0, 0.57)
        assert c.bioen_lite_t1[0] == t1 and c.bioen_lite_t2[0] == t2  # type: ignore[index]
        assert c.bioen_lite_topt[0] == 16.0  # type: ignore[index]


class TestTheTemperatureResponse:
    @pytest.mark.parametrize("tmin,tmax,topt,xi", [(2, 26, 16, 0.57), (0, 10, 3, 0.3)])
    def test_one_at_the_optimum_critical_at_the_limits(self, tmin, tmax, topt, xi):
        t1, t2 = lite_temperature_bounds(tmin, tmax, topt, xi)
        assert t1 < tmin and t2 > tmax
        f = lite_temperature_factor(np.array([tmin, topt, tmax], dtype=float), t1, t2, topt)
        assert f == pytest.approx([xi, 1.0, xi], rel=1e-12)


class TestTheNetEnergy:
    def test_it_is_javas_formula(self):
        ing = np.array([2e-3, 5e-4])
        w = np.array([2e-5, 1e-6])
        n = np.array([1000.0, 3000.0])
        age = np.array([12, 1], dtype=np.int32)  # the second is a larva (threshold 3 steps)
        t1, t2 = lite_temperature_bounds(2.0, 26.0, 16.0, 0.57)
        eg, em, en = energy_terms_lite(
            ing, w, n, age, 14.0, 0.68, 0.75, 3.5, 1.5, 3, 0.57, t1, t2, 16.0, 0.9, 12
        )
        f_t = float(lite_temperature_factor(14.0, t1, t2, 16.0))
        theta = np.array([1.0, 1.5])
        want = 0.68 * (ing * f_t * 0.9 - 3.5 / 12 * theta * 0.57 * (w * 1e6) ** 0.75 * n * 1e-6)
        assert en == pytest.approx(want, rel=1e-12)
        assert (eg == 0).all() and (em == 0).all()

    def test_zero_at_the_thermal_limit_when_eating_at_the_maximum(self):
        w, n = np.array([3e-5]), np.array([500.0])
        t1, t2 = lite_temperature_bounds(2.0, 26.0, 16.0, 0.57)
        i_max_step = 3.5 / 12 * (w * 1e6) ** 0.75 * n * 1e-6
        for temp in (2.0, 26.0):
            _, _, en = energy_terms_lite(
                i_max_step, w, n, np.array([24]), temp, 0.68, 0.75, 3.5, 1.5, 3, 0.57,
                t1, t2, 16.0, 1.0, 12,
            )  # fmt: skip
            assert abs(en[0]) < 1e-15


def _step(cfg: dict[str, str]):
    from osmose.engine.physical_data import PhysicalData
    from osmose.engine.simulate import _bioen_step
    from osmose.engine.state import SchoolState

    config = EngineConfig.from_dict(cfg)
    sid = np.array([0, 0, 1], dtype=np.int32)
    state = SchoolState.create(n_schools=3, species_id=sid).replace(
        abundance=np.array([1000.0, 4000.0, 1000.0]),
        weight=np.array([2e-5, 1e-6, 2e-5]),
        length=np.array([15.0, 4.0, 15.0]),
        age_dt=np.array([12, 1, 12], dtype=np.int32),
        preyed_biomass=np.array([2e-3, 3e-4, 2e-3]),
        first_feeding_age_dt=np.ones(3, dtype=np.int32),
    )
    temp = PhysicalData.from_constant(14.0)
    return config, _bioen_step(state, config, temp, 0)


def test_the_step_uses_the_lite_budget_for_lite_species_only():
    _, full = _step(_bioen())
    _, lite = _step(_bioen(**LITE))
    t1, t2 = lite_temperature_bounds(2.0, 26.0, 16.0, 0.57)
    _, _, want = energy_terms_lite(
        np.array([2e-3, 3e-4]), np.array([2e-5, 1e-6]), np.array([1000.0, 4000.0]),
        np.array([12, 1], dtype=np.int32), 14.0, 0.68, 0.75, 3.5, 1.5, 3, 0.57, t1, t2, 16.0,
        1.0, 12,
    )  # fmt: skip
    assert lite.e_net[:2] == pytest.approx(want, rel=1e-12)
    assert (lite.e_gross[:2] == 0).all() and (lite.e_maint[:2] == 0).all()
    assert lite.e_net[2] == full.e_net[2] and lite.weight[2] == full.weight[2]  # full species
    assert lite.e_net[0] != full.e_net[0]


def test_a_whole_run_with_a_lite_species_runs_and_differs():
    import osmose.engine.simulate as sim
    from osmose.engine import PythonEngine

    def run(cfg: dict[str, str]):
        config, grid, rng, mv, mo = PythonEngine()._prepare_run(
            cfg | {"_osmose.config.dir": "."}, 3
        )
        return sim.simulate(config, grid, rng, movement_rngs=mv, mortality_rngs=mo)

    full, lite = run(_bioen()), run(_bioen(**LITE))
    assert all(np.isfinite(r.biomass).all() for r in lite)
    assert any(not np.array_equal(a.biomass, b.biomass) for a, b in zip(full, lite, strict=True))
