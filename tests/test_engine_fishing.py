"""Tests for fishing mortality — Tier 1 verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.fishing import fishing_mortality
from osmose.engine.state import MortalityCause, SchoolState


def _make_fishing_config() -> dict[str, str]:
    return {
        "simulation.time.ndtperyear": "24",
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
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
        "simulation.fishing.mortality.enabled": "true",
        "fishing.rate.sp0": "0.5",
    }


class TestFishingMortality:
    def test_fishing_reduces_abundance(self):
        cfg = EngineConfig.from_dict(_make_fishing_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            weight=np.array([6.0]),
            age_dt=np.array([24], dtype=np.int32),
        )
        new_state = fishing_mortality(state, cfg, n_subdt=10)
        d = 0.5 / (24 * 10)
        expected_dead = 1000.0 * (1 - np.exp(-d))
        np.testing.assert_allclose(
            new_state.n_dead[0, MortalityCause.FISHING], expected_dead, rtol=1e-10
        )

    def test_disabled_fishing(self):
        cfg_dict = _make_fishing_config()
        cfg_dict["simulation.fishing.mortality.enabled"] = "false"
        cfg = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(abundance=np.array([1000.0]), age_dt=np.array([24], dtype=np.int32))
        new_state = fishing_mortality(state, cfg, n_subdt=10)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)

    def test_skips_eggs(self):
        cfg = EngineConfig.from_dict(_make_fishing_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(abundance=np.array([1000.0]), age_dt=np.array([0], dtype=np.int32))
        new_state = fishing_mortality(state, cfg, n_subdt=10)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)

    def test_annual_fishing_decay(self):
        """Full year of fishing should give approximately exp(-F)."""
        cfg = EngineConfig.from_dict(_make_fishing_config())
        n_subdt = 10
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([10000.0]),
            weight=np.array([6.0]),
            age_dt=np.array([24], dtype=np.int32),
        )
        for _step in range(24):
            for _sub in range(n_subdt):
                state = fishing_mortality(state, cfg, n_subdt)
        expected = 10000.0 * np.exp(-0.5)
        np.testing.assert_allclose(state.abundance[0], expected, rtol=1e-4)
