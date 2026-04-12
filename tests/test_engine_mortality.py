"""Tests for natural and aging mortality — Tier 1 analytical verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import (
    additional_mortality,
    aging_mortality,
    larva_mortality,
    out_mortality,
)
from osmose.engine.state import MortalityCause, SchoolState


def _make_mortality_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "24",
        "simulation.time.nyear": "1",
        "simulation.nspecies": "1",
        "simulation.nschool.sp0": "1",
        "species.name.sp0": "TestFish",
        "species.linf.sp0": "30.0",
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
        "mortality.additional.rate.sp0": "0.2",
    }


class TestAdditionalMortality:
    def test_mortality_per_substep(self):
        """Per sub-step: D = M_annual / (n_dt_per_year * n_subdt)."""
        cfg = EngineConfig.from_dict(_make_mortality_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            biomass=np.array([6000.0]),
            weight=np.array([6.0]),
            age_dt=np.array([24], dtype=np.int32),  # non-zero: eggs are skipped
        )
        n_subdt = 10
        new_state = additional_mortality(state, cfg, n_subdt)
        # D = 0.2 / (24 * 10) = 0.000833...
        d = 0.2 / (24 * 10)
        expected_dead = 1000.0 * (1 - np.exp(-d))
        actual_dead = new_state.n_dead[0, MortalityCause.ADDITIONAL]
        np.testing.assert_allclose(actual_dead, expected_dead, rtol=1e-10)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0 - expected_dead, rtol=1e-10)

    def test_annual_decay_after_full_year(self):
        """After n_dt * n_subdt applications, total ≈ 1 - exp(-M_annual)."""
        cfg = EngineConfig.from_dict(_make_mortality_config())
        n_subdt = 10
        n_dt = 24  # timesteps per year

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([10000.0]),
            weight=np.array([6.0]),
            biomass=np.array([60000.0]),
            age_dt=np.array([24], dtype=np.int32),  # non-zero: eggs are skipped
        )

        # Apply for a full year: n_dt timesteps, each with n_subdt sub-steps
        for _step in range(n_dt):
            for _sub in range(n_subdt):
                state = additional_mortality(state, cfg, n_subdt)

        # Should approximate exp(-M_annual) = exp(-0.2)
        expected = 10000.0 * np.exp(-0.2)
        np.testing.assert_allclose(state.abundance[0], expected, rtol=1e-4)

    def test_zero_rate_no_mortality(self):
        # Two species: sp0 rate=0, sp1 rate=0.2 — verifies zero-rate school is
        # specifically unaffected while a non-zero control school in the same
        # state DOES lose abundance (makes the test non-tautological).
        cfg_dict = _make_mortality_config()
        cfg_dict["simulation.nspecies"] = "2"
        cfg_dict["simulation.nschool.sp1"] = "1"
        cfg_dict["species.name.sp1"] = "ControlFish"
        cfg_dict["species.linf.sp1"] = "30.0"
        cfg_dict["species.k.sp1"] = "0.3"
        cfg_dict["species.t0.sp1"] = "-0.1"
        cfg_dict["species.egg.size.sp1"] = "0.1"
        cfg_dict["species.length2weight.condition.factor.sp1"] = "0.006"
        cfg_dict["species.length2weight.allometric.power.sp1"] = "3.0"
        cfg_dict["species.lifespan.sp1"] = "3"
        cfg_dict["species.vonbertalanffy.threshold.age.sp1"] = "1.0"
        cfg_dict["predation.ingestion.rate.max.sp1"] = "3.5"
        cfg_dict["predation.efficiency.critical.sp1"] = "0.57"
        cfg_dict["mortality.additional.rate.sp0"] = "0.0"
        cfg_dict["mortality.additional.rate.sp1"] = "0.2"
        cfg = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(
            n_schools=2,
            species_id=np.array([0, 1], dtype=np.int32),
        )
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
        )
        new_state = additional_mortality(state, cfg, n_subdt=10)
        # sp0 (rate=0): unchanged abundance and zero n_dead
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)
        assert new_state.n_dead[0, MortalityCause.ADDITIONAL] == 0.0
        # sp1 (rate=0.2): abundance must have decreased
        assert new_state.abundance[1] < 1000.0

    def test_skips_age_zero_schools(self):
        """Java skips age_dt==0 schools — eggs handled by larva_mortality."""
        cfg = EngineConfig.from_dict(_make_mortality_config())
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([0.001, 6.0]),
            age_dt=np.array([0, 24], dtype=np.int32),  # egg vs adult
        )
        new_state = additional_mortality(state, cfg, n_subdt=10)
        # Egg (age_dt=0) should be untouched
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)
        # Adult should have mortality applied
        assert new_state.abundance[1] < 1000.0


class TestOutMortality:
    def test_out_mortality_formula(self):
        """n_dead = N * (1 - exp(-rate / n_dt_per_year)) for is_out schools."""
        cfg_dict = _make_mortality_config()
        cfg_dict["mortality.out.rate.sp0"] = "0.3"
        cfg = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([6.0, 6.0]),
            biomass=np.array([6000.0, 6000.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
            is_out=np.array([True, False]),
        )
        new_state = out_mortality(state, cfg)
        rate = 0.3
        n_dt = 24
        expected_dead = 1000.0 * (1 - np.exp(-rate / n_dt))
        np.testing.assert_allclose(
            new_state.n_dead[0, MortalityCause.OUT], expected_dead, rtol=1e-10
        )
        np.testing.assert_allclose(new_state.abundance[0], 1000.0 - expected_dead, rtol=1e-10)
        # School not marked is_out is unaffected
        np.testing.assert_allclose(new_state.abundance[1], 1000.0)
        np.testing.assert_allclose(new_state.n_dead[1, MortalityCause.OUT], 0.0)


class TestAgingMortality:
    def test_kills_old_schools(self):
        cfg = EngineConfig.from_dict(_make_mortality_config())
        state = SchoolState.create(n_schools=3, species_id=np.zeros(3, dtype=np.int32))
        state = state.replace(
            abundance=np.array([100.0, 100.0, 100.0]),
            age_dt=np.array([70, 71, 72], dtype=np.int32),
        )
        new_state = aging_mortality(state, cfg)
        np.testing.assert_allclose(new_state.abundance, [100.0, 0.0, 0.0])
        np.testing.assert_allclose(new_state.n_dead[1, MortalityCause.AGING], 100.0)
        np.testing.assert_allclose(new_state.n_dead[2, MortalityCause.AGING], 100.0)

    def test_young_schools_survive(self):
        cfg = EngineConfig.from_dict(_make_mortality_config())
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            abundance=np.array([500.0]),
            age_dt=np.array([24], dtype=np.int32),
        )
        new_state = aging_mortality(state, cfg)
        np.testing.assert_allclose(new_state.abundance, [500.0])


class TestLarvaMortality:
    def test_kills_only_eggs(self):
        """Larva mortality should only affect schools where is_egg=True."""
        cfg_dict = _make_mortality_config()
        cfg_dict["mortality.additional.larva.rate.sp0"] = "1.0"
        cfg = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(n_schools=2, species_id=np.zeros(2, dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([0.001, 6.0]),
            is_egg=np.array([True, False]),
        )
        new_state = larva_mortality(state, cfg)
        assert new_state.abundance[0] < 1000.0
        np.testing.assert_allclose(new_state.abundance[1], 1000.0)

    def test_zero_rate_no_mortality(self):
        # Two species: sp0 larva rate=0 (default), sp1 larva rate=1.0 — verifies
        # the zero-rate egg is specifically unaffected while a non-zero control
        # egg in the same state DOES lose abundance (makes the test non-tautological).
        cfg_dict = _make_mortality_config()
        cfg_dict["simulation.nspecies"] = "2"
        cfg_dict["simulation.nschool.sp1"] = "1"
        cfg_dict["species.name.sp1"] = "ControlFish"
        cfg_dict["species.linf.sp1"] = "30.0"
        cfg_dict["species.k.sp1"] = "0.3"
        cfg_dict["species.t0.sp1"] = "-0.1"
        cfg_dict["species.egg.size.sp1"] = "0.1"
        cfg_dict["species.length2weight.condition.factor.sp1"] = "0.006"
        cfg_dict["species.length2weight.allometric.power.sp1"] = "3.0"
        cfg_dict["species.lifespan.sp1"] = "3"
        cfg_dict["species.vonbertalanffy.threshold.age.sp1"] = "1.0"
        cfg_dict["predation.ingestion.rate.max.sp1"] = "3.5"
        cfg_dict["predation.efficiency.critical.sp1"] = "0.57"
        cfg_dict["mortality.additional.rate.sp1"] = "0.0"
        cfg_dict["mortality.additional.larva.rate.sp0"] = "0.0"
        cfg_dict["mortality.additional.larva.rate.sp1"] = "1.0"
        cfg = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(
            n_schools=2,
            species_id=np.array([0, 1], dtype=np.int32),
        )
        state = state.replace(
            abundance=np.array([1000.0, 1000.0]),
            is_egg=np.array([True, True]),
        )
        new_state = larva_mortality(state, cfg)
        # sp0 (larva rate=0): unchanged abundance and zero n_dead
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)
        assert new_state.n_dead[0, MortalityCause.ADDITIONAL] == 0.0
        # sp1 (larva rate=1.0): abundance must have decreased
        assert new_state.abundance[1] < 1000.0

    def test_larva_mortality_formula(self):
        """D = M_larva (full rate), applied once per egg cohort."""
        cfg_dict = _make_mortality_config()
        cfg_dict["mortality.additional.larva.rate.sp0"] = "2.0"
        cfg = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(n_schools=1, species_id=np.zeros(1, dtype=np.int32))
        state = state.replace(
            abundance=np.array([5000.0]),
            weight=np.array([0.001]),
            is_egg=np.array([True]),
        )
        new_state = larva_mortality(state, cfg)
        # D = 2.0 (full rate, applied once per cohort — Java convention)
        d = 2.0
        expected_dead = 5000.0 * (1 - np.exp(-d))
        np.testing.assert_allclose(
            new_state.n_dead[0, MortalityCause.ADDITIONAL], expected_dead, rtol=1e-10
        )
