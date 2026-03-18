"""Tests for natural and aging mortality — Tier 1 analytical verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.natural import additional_mortality, aging_mortality
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
    def test_mortality_reduces_abundance(self):
        cfg = EngineConfig.from_dict(_make_mortality_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            biomass=np.array([6000.0]),
            weight=np.array([6.0]),
        )
        n_subdt = 10
        new_state = additional_mortality(state, cfg, n_subdt)
        m = 0.2
        expected_dead = 1000.0 * (1 - np.exp(-m / n_subdt))
        actual_dead = new_state.n_dead[0, MortalityCause.ADDITIONAL]
        np.testing.assert_allclose(actual_dead, expected_dead, rtol=1e-10)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0 - expected_dead, rtol=1e-10)

    def test_zero_rate_no_mortality(self):
        cfg_dict = _make_mortality_config()
        cfg_dict["mortality.additional.rate.sp0"] = "0.0"
        cfg = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(abundance=np.array([1000.0]))
        new_state = additional_mortality(state, cfg, n_subdt=10)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)


class TestAgingMortality:
    def test_kills_old_schools(self):
        cfg = EngineConfig.from_dict(_make_mortality_config())
        # lifespan = 3 years * 24 dt = 72 dt. Aging kills at age_dt >= 71.
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
