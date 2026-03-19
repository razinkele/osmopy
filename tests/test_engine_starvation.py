"""Tests for starvation mortality — Tier 1 verification."""

import numpy as np

from osmose.engine.config import EngineConfig
from osmose.engine.processes.starvation import starvation_mortality, update_starvation_rate
from osmose.engine.state import MortalityCause, SchoolState


def _make_starv_config() -> dict[str, str]:
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
        "mortality.starvation.rate.max.sp0": "3.0",
    }


class TestStarvationMortality:
    def test_applies_lagged_rate(self):
        """Starvation should use the stored starvation_rate, not compute new one."""
        cfg = EngineConfig.from_dict(_make_starv_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            weight=np.array([6.0]),
            age_dt=np.array([24], dtype=np.int32),
            starvation_rate=np.array([1.5]),  # pre-set lagged rate
        )
        new_state = starvation_mortality(state, cfg, n_subdt=10)
        # D = 1.5 / (n_dt_per_year * n_subdt) = 1.5 / (24 * 10) = 0.00625
        expected_dead = 1000.0 * (1 - np.exp(-1.5 / (24 * 10)))
        np.testing.assert_allclose(
            new_state.n_dead[0, MortalityCause.STARVATION], expected_dead, rtol=1e-10
        )

    def test_zero_rate_no_mortality(self):
        cfg = EngineConfig.from_dict(_make_starv_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            age_dt=np.array([24], dtype=np.int32),
            starvation_rate=np.array([0.0]),
        )
        new_state = starvation_mortality(state, cfg, n_subdt=10)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)

    def test_skips_eggs(self):
        cfg = EngineConfig.from_dict(_make_starv_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            age_dt=np.array([0], dtype=np.int32),
            starvation_rate=np.array([2.0]),
        )
        new_state = starvation_mortality(state, cfg, n_subdt=10)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)


class TestUpdateStarvationRate:
    def test_full_success_gives_zero_rate(self):
        """pred_success_rate >= critical -> starvation_rate = 0."""
        cfg = EngineConfig.from_dict(_make_starv_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(pred_success_rate=np.array([1.0]))
        new_state = update_starvation_rate(state, cfg)
        np.testing.assert_allclose(new_state.starvation_rate[0], 0.0)

    def test_zero_success_gives_max_rate(self):
        """pred_success_rate = 0 -> starvation_rate = M_max."""
        cfg = EngineConfig.from_dict(_make_starv_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(pred_success_rate=np.array([0.0]))
        new_state = update_starvation_rate(state, cfg)
        np.testing.assert_allclose(new_state.starvation_rate[0], 3.0)

    def test_partial_success(self):
        """Intermediate success gives proportional rate."""
        cfg = EngineConfig.from_dict(_make_starv_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        # S_R = 0.285 = C_SR/2 -> rate = M_max * (1 - 0.285/0.57) = M_max * 0.5 = 1.5
        state = state.replace(pred_success_rate=np.array([0.285]))
        new_state = update_starvation_rate(state, cfg)
        np.testing.assert_allclose(new_state.starvation_rate[0], 1.5, rtol=1e-6)
