"""OSMOSE 4.4.x's density-dependent larval mortality (#137).

Java (``AnnualLarvaMortality`` / ``ByDtLarvaMortality``, v4.4.1): with
``mortality.additional.larva.referencedensity.sp#`` set, the larval rate of a step becomes

    M * (1 + a * SSB / reference),   a = ln 2 / M,

so at a spawning stock equal to the reference, larval survival halves. Java applies larval
mortality to eggs in a pre-pass before the sub-step loop (``MortalityProcess``), so the SSB it
reads is the one totalled at the start of the step, after movement: the instantaneous biomass of
every mature focal school, out-of-domain schools included. Without the key ``a`` is 0 and
nothing changes. Where the base rate of a step is 0, Java's formula is 0 x infinity; the engine
uses its limit, ln 2 * SSB / reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import larva_mortality
from osmose.engine.state import SchoolState

KEY = "mortality.additional.larva.referencedensity.sp0"


def _config(**extra: str) -> dict[str, str]:
    cfg = {
        "simulation.time.ndtperyear": "12",
        "simulation.time.nyear": "1",
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
            # Per step: a dict handed to EngineConfig skips the reader, which is where the
            # file's per-year rate is divided by the steps per year.
            f"mortality.additional.larva.rate.sp{i}": "2.0",
            f"population.seeding.biomass.sp{i}": "1000",
        }
    return cfg | extra


def _state() -> SchoolState:
    """Species 0: an egg school, two mature schools (one out of the domain), one immature.
    Species 1: an egg school and a mature school, whose biomass must not count for species 0."""
    sid = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    s = SchoolState.create(n_schools=6, species_id=sid)
    return s.replace(
        abundance=np.array([1e6, 100.0, 300.0, 5000.0, 1e6, 700.0]),
        weight=np.array([1e-9, 0.5, 0.25, 0.01, 1e-9, 0.4]),
        length=np.array([0.1, 20.0, 15.0, 5.0, 0.1, 18.0]),
        age_dt=np.array([0, 30, 24, 6, 0, 30], dtype=np.int32),
        is_egg=np.array([True, False, False, False, True, False]),
        is_out=np.array([False, False, True, False, False, False]),
    )


SSB0 = 100.0 * 0.5 + 300.0 * 0.25  # the two mature schools of species 0: 125 t


class TestTheKey:
    def test_absent_means_off(self):
        assert EngineConfig.from_dict(_config()).larva_reference_density is None

    def test_a_negative_reference_means_off_as_in_java(self):
        assert EngineConfig.from_dict(_config(**{KEY: "-999"})).larva_reference_density is None

    def test_a_reference_is_carried_per_species(self):
        ref = EngineConfig.from_dict(_config(**{KEY: "250"})).larva_reference_density
        assert ref is not None and ref[0] == 250.0 and ref[1] < 0

    def test_zero_is_refused(self):
        with pytest.raises(ValueError, match="referencedensity.sp0"):
            EngineConfig.from_dict(_config(**{KEY: "0"}))


def _egg_deaths(cfg: dict[str, str], state: SchoolState | None = None) -> np.ndarray:
    config = EngineConfig.from_dict(cfg)
    out = larva_mortality(state or _state(), config, step=3)
    return out.n_dead[:, 2]  # ADDITIONAL


class TestTheRate:
    def test_at_ssb_equal_to_the_reference_survival_halves(self):
        base = _egg_deaths(_config())
        dd = _egg_deaths(_config(**{KEY: str(SSB0)}))
        n = 1e6
        survival_base = (n - base[0]) / n
        survival_dd = (n - dd[0]) / n
        assert survival_dd == pytest.approx(survival_base / 2, rel=1e-12)
        assert dd[4] == base[4]  # species 1 has no reference: unchanged

    def test_it_is_javas_formula(self):
        ref = 400.0
        m = 2.0  # 24 per year over 12 steps
        want_rate = m * (1 + (math.log(2) / m) * SSB0 / ref)
        dd = _egg_deaths(_config(**{KEY: str(ref)}))
        assert dd[0] == pytest.approx(1e6 * (1 - math.exp(-want_rate)), rel=1e-12)

    def test_only_mature_focal_schools_of_the_species_count(self):
        # Make the immature school and species 1's mature school enormous: no change.
        heavy = _state().replace(
            abundance=np.array([1e6, 100.0, 300.0, 5e9, 1e6, 7e9]),
        )
        assert (
            _egg_deaths(_config(**{KEY: "400"}), heavy)[0]
            == _egg_deaths(_config(**{KEY: "400"}))[0]
        )
        # Drop the out-of-domain mature school: the stock, and so the deaths, fall.
        no_out = _state().replace(abundance=np.array([1e6, 100.0, 0.0, 5000.0, 1e6, 700.0]))
        assert (
            _egg_deaths(_config(**{KEY: "400"}), no_out)[0]
            < _egg_deaths(_config(**{KEY: "400"}))[0]
        )

    def test_a_stochastic_species_counts_only_its_drawn_mature_schools(self):
        # Species 0 on the stochastic ogive: its two big schools are past the 12 cm threshold,
        # but only the first has been drawn mature, so only its biomass counts.
        cfg = _config(**{
            KEY: str(100.0 * 0.5),
            "species.maturity.mode.sp0": "stochastic",
            "species.maturity.l50.sp0": "14.0",
            "species.maturity.l75.sp0": "16.0",
        })  # fmt: skip
        drawn = _state().replace(is_mature=np.array([False, True, False, False, False, False]))
        base = _egg_deaths(
            _config(
                **{
                    "species.maturity.mode.sp0": "stochastic",
                    "species.maturity.l50.sp0": "14.0",
                    "species.maturity.l75.sp0": "16.0",
                }
            ),
            drawn,
        )
        dd = _egg_deaths(cfg, drawn)
        n = 1e6
        assert (n - dd[0]) / n == pytest.approx(
            (n - base[0]) / n / 2, rel=1e-12
        )  # SSB = 50 = reference

    def test_a_zero_base_rate_uses_the_limit_not_a_nan(self):
        cfg = _config(**{KEY: "400", "mortality.additional.larva.rate.sp0": "0"})
        dd = _egg_deaths(cfg)
        assert np.isfinite(dd).all()
        assert dd[0] == pytest.approx(1e6 * (1 - math.exp(-math.log(2) * SSB0 / 400.0)), rel=1e-12)


def test_a_whole_run_changes_only_with_a_positive_reference():
    from osmose.engine.grid import Grid
    from osmose.engine.simulate import simulate

    grid = Grid.from_dimensions(ny=3, nx=3)
    runs = {
        label: simulate(
            EngineConfig.from_dict(
                _config(
                    **extra,
                    # mature within the run, so that there is a spawning stock to scale by
                    **{"simulation.time.nyear": "2", "species.maturity.size.sp0": "4.0"},
                )
            ),
            grid,
            np.random.default_rng(6),
        )
        for label, extra in (("absent", {}), ("off", {KEY: "-1"}), ("on", {KEY: "50"}))
    }
    for a, b in zip(runs["absent"], runs["off"], strict=True):
        assert np.array_equal(a.biomass, b.biomass)
    assert any(
        not np.array_equal(a.biomass, b.biomass)
        for a, b in zip(runs["absent"], runs["on"], strict=True)
    )


def test_the_csharp_engine_refuses_it():
    from osmose.engine_cs.capabilities import csharp_engine_block_reason

    assert "larva_density_dependence" not in (csharp_engine_block_reason(_config()) or "")
    assert "larva_density_dependence" in (csharp_engine_block_reason(_config(**{KEY: "400"})) or "")
