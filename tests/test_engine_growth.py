"""Tests for growth process functions — Tier 1 analytical verification."""

import numpy as np

from osmose.engine.processes.growth import expected_length_gompertz, expected_length_vb, growth
from osmose.engine.config import EngineConfig
from osmose.engine.state import SchoolState


class TestExpectedLengthVB:
    """Verify Von Bertalanffy expected length against known formula."""

    def test_age_zero_returns_egg_size(self):
        result = expected_length_vb(
            age_dt=np.array([0]),
            linf=np.array([30.0]),
            k=np.array([0.3]),
            t0=np.array([-0.1]),
            egg_size=np.array([0.1]),
            vb_threshold_age=np.array([1.0]),
            n_dt_per_year=24,
        )
        np.testing.assert_allclose(result, [0.1], atol=1e-10)

    def test_young_of_year_linear(self):
        linf, k, t0, a_thres = 30.0, 0.3, -0.1, 1.0
        l_thres = linf * (1 - np.exp(-k * (a_thres - t0)))
        l_egg = 0.1
        n_dt = 24
        half_age_dt = int(a_thres * n_dt / 2)  # 12 dt = 0.5 years
        result = expected_length_vb(
            age_dt=np.array([half_age_dt]),
            linf=np.array([linf]),
            k=np.array([k]),
            t0=np.array([t0]),
            egg_size=np.array([l_egg]),
            vb_threshold_age=np.array([a_thres]),
            n_dt_per_year=n_dt,
        )
        expected = l_egg + (l_thres - l_egg) * 0.5
        np.testing.assert_allclose(result, [expected], atol=1e-10)

    def test_above_threshold_vb_formula(self):
        linf, k, t0 = 30.0, 0.3, -0.1
        n_dt = 24
        age_years = 3.0
        age_dt = int(age_years * n_dt)
        result = expected_length_vb(
            age_dt=np.array([age_dt]),
            linf=np.array([linf]),
            k=np.array([k]),
            t0=np.array([t0]),
            egg_size=np.array([0.1]),
            vb_threshold_age=np.array([1.0]),
            n_dt_per_year=n_dt,
        )
        expected = linf * (1 - np.exp(-k * (age_years - t0)))
        np.testing.assert_allclose(result, [expected], atol=1e-10)

    def test_vectorized_multiple_species(self):
        n_dt = 24
        result = expected_length_vb(
            age_dt=np.array([48, 48]),
            linf=np.array([30.0, 50.0]),
            k=np.array([0.3, 0.2]),
            t0=np.array([-0.1, -0.2]),
            egg_size=np.array([0.1, 0.2]),
            vb_threshold_age=np.array([1.0, 1.0]),
            n_dt_per_year=n_dt,
        )
        age = 2.0
        e0 = 30.0 * (1 - np.exp(-0.3 * (age - (-0.1))))
        e1 = 50.0 * (1 - np.exp(-0.2 * (age - (-0.2))))
        np.testing.assert_allclose(result, [e0, e1], atol=1e-10)


def _make_growth_config() -> dict[str, str]:
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
        "species.lifespan.sp0": "5",
        "species.vonbertalanffy.threshold.age.sp0": "1.0",
        "mortality.subdt": "10",
        "predation.ingestion.rate.max.sp0": "3.5",
        "predation.efficiency.critical.sp0": "0.57",
    }


class TestGrowthGating:
    def test_no_growth_below_critical(self):
        cfg = EngineConfig.from_dict(_make_growth_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([10.0]),
            weight=np.array([6.0]),
            age_dt=np.array([48], dtype=np.int32),
            abundance=np.array([100.0]),
            biomass=np.array([600.0]),
            pred_success_rate=np.array([0.3]),  # below 0.57 critical
        )
        new_state = growth(state, cfg, np.random.default_rng(42))
        np.testing.assert_allclose(new_state.length, [10.0], atol=1e-10)

    def test_max_growth_at_full_success(self):
        cfg = EngineConfig.from_dict(_make_growth_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([10.0]),
            weight=np.array([6.0]),
            age_dt=np.array([48], dtype=np.int32),
            abundance=np.array([100.0]),
            biomass=np.array([600.0]),
            pred_success_rate=np.array([1.0]),
        )
        new_state = growth(state, cfg, np.random.default_rng(42))
        # At sr=1.0, growth_factor = delta_lmax_factor * delta_L
        l_cur = expected_length_vb(
            np.array([48], dtype=np.int32),
            np.array([30.0]),
            np.array([0.3]),
            np.array([-0.1]),
            np.array([0.1]),
            np.array([1.0]),
            24,
        )
        l_nxt = expected_length_vb(
            np.array([49], dtype=np.int32),
            np.array([30.0]),
            np.array([0.3]),
            np.array([-0.1]),
            np.array([0.1]),
            np.array([1.0]),
            24,
        )
        max_delta = 2.0 * (l_nxt[0] - l_cur[0])
        expected = min(10.0 + max_delta, 30.0)
        np.testing.assert_allclose(new_state.length[0], expected, atol=1e-10)

    def test_egg_always_grows(self):
        cfg = EngineConfig.from_dict(_make_growth_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([0.1]),
            weight=np.array([0.000006]),
            age_dt=np.array([0], dtype=np.int32),
            abundance=np.array([1000.0]),
            biomass=np.array([0.006]),
            pred_success_rate=np.array([0.0]),
        )
        new_state = growth(state, cfg, np.random.default_rng(42))
        # Egg gets exactly delta_L = L_expected(1) - L_expected(0)
        l_next = expected_length_vb(
            np.array([1], dtype=np.int32),
            np.array([30.0]),
            np.array([0.3]),
            np.array([-0.1]),
            np.array([0.1]),
            np.array([1.0]),
            24,
        )
        expected = 0.1 + (l_next[0] - 0.1)
        np.testing.assert_allclose(new_state.length[0], expected, atol=1e-10)

    def test_weight_updated_after_growth(self):
        cfg = EngineConfig.from_dict(_make_growth_config())
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([10.0]),
            weight=np.array([6.0]),
            age_dt=np.array([48], dtype=np.int32),
            abundance=np.array([100.0]),
            biomass=np.array([600.0]),
            pred_success_rate=np.array([1.0]),
        )
        new_state = growth(state, cfg, np.random.default_rng(42))
        expected_weight = 0.006 * new_state.length[0] ** 3.0
        np.testing.assert_allclose(new_state.weight[0], expected_weight, rtol=1e-10)

    def test_csr_equals_one_full_success_gets_max_delta(self):
        """When critical success rate = 1.0 and sr = 1.0, growth = max_delta."""
        cfg_dict = _make_growth_config()
        cfg_dict["predation.efficiency.critical.sp0"] = "1.0"
        cfg = EngineConfig.from_dict(cfg_dict)
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            length=np.array([10.0]),
            weight=np.array([6.0]),
            age_dt=np.array([48], dtype=np.int32),
            abundance=np.array([100.0]),
            biomass=np.array([600.0]),
            pred_success_rate=np.array([1.0]),
        )
        new_state = growth(state, cfg, np.random.default_rng(42))
        # Should get max_delta = delta_lmax_factor * delta_L
        l_cur = expected_length_vb(
            np.array([48], dtype=np.int32),
            np.array([30.0]),
            np.array([0.3]),
            np.array([-0.1]),
            np.array([0.1]),
            np.array([1.0]),
            24,
        )
        l_nxt = expected_length_vb(
            np.array([49], dtype=np.int32),
            np.array([30.0]),
            np.array([0.3]),
            np.array([-0.1]),
            np.array([0.1]),
            np.array([1.0]),
            24,
        )
        max_delta = 2.0 * (l_nxt[0] - l_cur[0])
        expected = min(10.0 + max_delta, 30.0)
        np.testing.assert_allclose(new_state.length[0], expected, atol=1e-10)


class TestExpectedLengthGompertz:
    def test_age_zero_returns_egg_size(self):
        result = expected_length_gompertz(
            age_dt=np.array([0]),
            linf=np.array([30.0]),
            k_gom=np.array([0.5]),
            t_gom=np.array([1.5]),
            k_exp=np.array([2.0]),
            a_exp_dt=np.array([6]),
            a_gom_dt=np.array([12]),
            egg_size=np.array([0.1]),
            n_dt_per_year=24,
        )
        np.testing.assert_allclose(result, [0.1], atol=1e-10)

    def test_gompertz_phase(self):
        """Above a_gom, should follow Gompertz curve."""
        linf, k_g, t_g = 30.0, 0.5, 1.5
        n_dt = 24
        age_years = 3.0
        age_dt = int(age_years * n_dt)
        result = expected_length_gompertz(
            age_dt=np.array([age_dt]),
            linf=np.array([linf]),
            k_gom=np.array([k_g]),
            t_gom=np.array([t_g]),
            k_exp=np.array([2.0]),
            a_exp_dt=np.array([6]),
            a_gom_dt=np.array([12]),
            egg_size=np.array([0.1]),
            n_dt_per_year=n_dt,
        )
        expected = linf * np.exp(-np.exp(-k_g * (age_years - t_g)))
        np.testing.assert_allclose(result, [expected], atol=1e-10)
