"""Tests for fishing system variants and selectivity types."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.fishing import fishing_mortality
from osmose.engine.processes.selectivity import (
    gaussian,
    log_normal,
    sigmoid,
)
from osmose.engine.state import MortalityCause, SchoolState


class TestSigmoidSelectivity:
    """Type 1: Logistic sigmoid parameterized by L50/L75 (Java formula)."""

    def test_half_at_l50(self) -> None:
        """Selectivity = 0.5 at L50."""
        sel = sigmoid(np.array([20.0]), l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(0.5, rel=1e-6)

    def test_75pct_at_l75(self) -> None:
        """Selectivity = 0.75 at L75."""
        sel = sigmoid(np.array([25.0]), l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(0.75, rel=1e-6)


class TestGaussianSelectivity:
    """Type 2: Normal distribution, normalized by peak."""

    def test_peak_at_l50(self) -> None:
        """Maximum selectivity (1.0) at L50."""
        length = np.array([20.0])
        sel = gaussian(length, l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(1.0)

    def test_symmetric_bell(self) -> None:
        """Symmetric around L50."""
        l50, l75 = 20.0, 25.0
        lengths = np.array([15.0, 25.0])  # equidistant from l50
        sel = gaussian(lengths, l50=l50, l75=l75)
        assert sel[0] == pytest.approx(sel[1], rel=1e-6)

    def test_decreases_from_peak(self) -> None:
        """Selectivity decreases away from L50."""
        lengths = np.array([10.0, 20.0, 30.0])
        sel = gaussian(lengths, l50=20.0, l75=25.0)
        assert sel[1] > sel[0]
        assert sel[1] > sel[2]

    def test_at_l75_approx_75pct(self) -> None:
        """At L75, selectivity should be approximately 0.7978 of peak.

        Java: sd = (L75 - L50) / qnorm(0.75), so at L75 the normal PDF
        ratio = density(L75)/density(L50) = exp(-0.5 * 0.6745**2) ~ 0.7978.
        """
        sel = gaussian(np.array([25.0]), l50=20.0, l75=25.0)
        assert sel[0] == pytest.approx(0.7978, rel=0.01)


class TestLogNormalSelectivity:
    """Type 3: Log-normal distribution, normalized by mode."""

    def test_peak_at_mode(self) -> None:
        """Maximum selectivity at the mode = exp(mean - sd**2)."""
        l50, l75 = 20.0, 30.0
        mean = np.log(l50)
        q75 = 0.674489750196082
        sd = np.log(l75 / l50) / q75
        mode = np.exp(mean - sd**2)
        sel = log_normal(np.array([mode]), l50=l50, l75=l75)
        assert sel[0] == pytest.approx(1.0, rel=1e-4)

    def test_asymmetric(self) -> None:
        """Log-normal is right-skewed -- more selectivity above mode than below."""
        l50, l75 = 20.0, 30.0
        lengths = np.array([5.0, 10.0, 20.0, 30.0, 50.0])
        sel = log_normal(lengths, l50=l50, l75=l75)
        # Should be highest near mode, asymmetric
        peak_idx = np.argmax(sel)
        assert sel[peak_idx] > sel[0]
        assert sel[peak_idx] > sel[-1]


# ---------------------------------------------------------------------------
# Helpers for fishing_mortality integration tests
# ---------------------------------------------------------------------------


def _mock_config(**overrides):
    """Create a minimal MagicMock EngineConfig for fishing tests."""
    config = MagicMock(spec=EngineConfig)
    defaults = {
        "fishing_enabled": True,
        "n_species": 1,
        "n_dt_per_year": 24,
        "n_year": 1,
        "fishing_rate": np.array([0.0]),
        "fishing_rate_by_year": None,
        "fishing_seasonality": None,
        "fishing_rate_by_dt_by_class": None,
        "fishing_catches": None,
        "fishing_catches_by_year": None,
        "fishing_catches_season": None,
        "fishing_selectivity_type": np.array([-1], dtype=np.int32),
        "fishing_selectivity_a50": np.array([np.nan]),
        "fishing_selectivity_l50": np.array([0.0]),
        "fishing_selectivity_slope": np.array([0.0]),
        "fishing_selectivity_l75": np.array([0.0]),
        "fishing_spatial_maps": [None],
        "mpa_zones": None,
        "fishing_discard_rate": None,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(config, k, v)
    return config


def _make_state(n_schools=1, **overrides):
    """Create a minimal SchoolState for fishing tests."""
    defaults = {
        "abundance": np.array([1000.0] * n_schools),
        "weight": np.array([0.01] * n_schools),
        "length": np.array([10.0] * n_schools),
        "age_dt": np.array([24] * n_schools, dtype=np.int32),
        "cell_x": np.array([0] * n_schools, dtype=np.int32),
        "cell_y": np.array([0] * n_schools, dtype=np.int32),
    }
    defaults.update(overrides)
    sp_id = defaults.pop("species_id", np.zeros(n_schools, dtype=np.int32))
    state = SchoolState.create(n_schools=n_schools, species_id=sp_id)
    return state.replace(**defaults)


class TestRateByDtByClass:
    """RateByDtByClassFishingMortality -- rate varies per (dt, age/size class)."""

    def test_rate_from_age_class(self) -> None:
        """Rate looked up by simulation step and school age class."""
        from osmose.engine.timeseries import ByClassTimeSeries

        # Create a ByClassTimeSeries with 2 age classes
        # Class thresholds: 0 and 48 (age in dt)
        classes = np.array([0.0, 48.0])
        # Step 0: class 0 = 0.1, class 1 = 0.5
        values = np.array([[0.1, 0.5]])  # 1 step, 2 classes
        ts = ByClassTimeSeries(classes, values)

        config = _mock_config(
            fishing_rate_by_dt_by_class=[ts],
            n_year=1,
        )

        # Young school (age_dt=10 < 48) -> class 0 -> rate 0.1
        state = _make_state(age_dt=np.array([10], dtype=np.int32))
        result = fishing_mortality(state, config, n_subdt=1, step=0)
        dead = result.n_dead[0, MortalityCause.FISHING]

        # With F=0.1, d = 0.1 / (24 * 1), mortality = 1 - exp(-d) ~ d
        d = 0.1 / 24
        expected = 1000.0 * (1 - np.exp(-d))
        assert dead == pytest.approx(expected, rel=1e-4)

    def test_old_school_gets_higher_rate(self) -> None:
        """Older schools should get a higher rate from the class lookup."""
        from osmose.engine.timeseries import ByClassTimeSeries

        classes = np.array([0.0, 48.0])
        values = np.array([[0.1, 0.5]])
        ts = ByClassTimeSeries(classes, values)

        config = _mock_config(
            fishing_rate_by_dt_by_class=[ts],
            n_year=1,
        )

        # Old school (age_dt=100 >= 48) -> class 1 -> rate 0.5
        state = _make_state(age_dt=np.array([100], dtype=np.int32))
        result = fishing_mortality(state, config, n_subdt=1, step=0)
        dead_old = result.n_dead[0, MortalityCause.FISHING]

        # Young school -> class 0 -> rate 0.1
        state_young = _make_state(age_dt=np.array([10], dtype=np.int32))
        result_young = fishing_mortality(state_young, config, n_subdt=1, step=0)
        dead_young = result_young.n_dead[0, MortalityCause.FISHING]

        assert dead_old > dead_young


class TestCatchBasedFishing:
    """Catch-based proportional allocation."""

    def test_proportional_allocation(self) -> None:
        """Catch distributed proportionally to school biomass."""
        config = _mock_config(
            fishing_catches=np.array([100.0]),
            n_dt_per_year=12,
            fishing_selectivity_type=np.array([-1], dtype=np.int32),
        )

        # Two schools of same species, different biomass
        state = _make_state(
            n_schools=2,
            species_id=np.array([0, 0], dtype=np.int32),
            abundance=np.array([1000.0, 3000.0]),
            weight=np.array([0.001, 0.001]),  # biomass: 1.0, 3.0 tonnes
            length=np.array([10.0, 10.0]),
            age_dt=np.array([24, 24], dtype=np.int32),
        )

        result = fishing_mortality(state, config, n_subdt=1, step=0)

        dead_0 = result.n_dead[0, MortalityCause.FISHING]
        dead_1 = result.n_dead[1, MortalityCause.FISHING]
        # 3:1 biomass ratio -> 3:1 catch ratio
        assert dead_1 / dead_0 == pytest.approx(3.0, rel=0.1)

    def test_zero_fishable_biomass(self) -> None:
        """Zero fishable biomass (via selectivity) -> no catch."""
        config = _mock_config(
            fishing_catches=np.array([100.0]),
            n_dt_per_year=12,
            fishing_selectivity_type=np.array([0], dtype=np.int32),
            fishing_selectivity_a50=np.array([100.0]),  # no school old enough
        )

        state = _make_state(
            abundance=np.array([1000.0]),
            weight=np.array([0.001]),
            length=np.array([5.0]),
            age_dt=np.array([1], dtype=np.int32),
        )

        result = fishing_mortality(state, config, n_subdt=1, step=0)
        assert result.n_dead[0, MortalityCause.FISHING] == 0.0

    def test_catch_not_exceed_biomass(self) -> None:
        """Catch should not exceed available biomass."""
        config = _mock_config(
            fishing_catches=np.array([1e10]),  # very high catch target
            n_dt_per_year=1,
        )

        state = _make_state(
            abundance=np.array([100.0]),
            weight=np.array([0.001]),
        )

        result = fishing_mortality(state, config, n_subdt=1, step=0)
        dead = result.n_dead[0, MortalityCause.FISHING]
        assert dead <= 100.0


class TestSelectivityIntegration:
    """Selectivity types 2 and 3 wired into fishing_mortality."""

    def test_gaussian_selectivity_in_fishing(self) -> None:
        """Type 2 (Gaussian) reduces fishing on small/large fish."""
        config = _mock_config(
            fishing_rate=np.array([1.0]),
            fishing_selectivity_type=np.array([2], dtype=np.int32),
            fishing_selectivity_l50=np.array([20.0]),
            fishing_selectivity_l75=np.array([25.0]),
        )

        # Two schools: one at L50 (peak), one far from it
        state = _make_state(
            n_schools=2,
            species_id=np.array([0, 0], dtype=np.int32),
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([0.01, 0.01]),
            length=np.array([20.0, 5.0]),
            age_dt=np.array([48, 48], dtype=np.int32),
        )

        result = fishing_mortality(state, config, n_subdt=1, step=0)
        dead_at_peak = result.n_dead[0, MortalityCause.FISHING]
        dead_far = result.n_dead[1, MortalityCause.FISHING]
        assert dead_at_peak > dead_far

    def test_lognormal_selectivity_in_fishing(self) -> None:
        """Type 3 (log-normal) applies asymmetric selectivity."""
        config = _mock_config(
            fishing_rate=np.array([1.0]),
            fishing_selectivity_type=np.array([3], dtype=np.int32),
            fishing_selectivity_l50=np.array([20.0]),
            fishing_selectivity_l75=np.array([30.0]),
        )

        # School near mode vs very small school
        state = _make_state(
            n_schools=2,
            species_id=np.array([0, 0], dtype=np.int32),
            abundance=np.array([1000.0, 1000.0]),
            weight=np.array([0.01, 0.01]),
            length=np.array([15.0, 1.0]),
            age_dt=np.array([48, 48], dtype=np.int32),
        )

        result = fishing_mortality(state, config, n_subdt=1, step=0)
        dead_near_mode = result.n_dead[0, MortalityCause.FISHING]
        dead_tiny = result.n_dead[1, MortalityCause.FISHING]
        assert dead_near_mode > dead_tiny

    def test_sigmoid_l75_formula_in_fishing(self) -> None:
        """Type 1 with L75 uses Java L50/L75 formula."""
        config = _mock_config(
            fishing_rate=np.array([1.0]),
            fishing_selectivity_type=np.array([1], dtype=np.int32),
            fishing_selectivity_l50=np.array([20.0]),
            fishing_selectivity_l75=np.array([25.0]),
        )

        # At L50, selectivity = 0.5; at L75, selectivity = 0.75
        state_l50 = _make_state(length=np.array([20.0]))
        state_l75 = _make_state(length=np.array([25.0]))
        state_big = _make_state(length=np.array([40.0]))

        result_l50 = fishing_mortality(state_l50, config, n_subdt=1, step=0)
        result_l75 = fishing_mortality(state_l75, config, n_subdt=1, step=0)
        result_big = fishing_mortality(state_big, config, n_subdt=1, step=0)

        dead_l50 = result_l50.n_dead[0, MortalityCause.FISHING]
        dead_l75 = result_l75.n_dead[0, MortalityCause.FISHING]
        dead_big = result_big.n_dead[0, MortalityCause.FISHING]

        # Selectivity increases: L50 < L75 < big
        assert dead_l50 < dead_l75
        assert dead_l75 < dead_big


class TestFishingScenarioDispatch:
    """Config auto-detects fishing scenario from keys."""

    def test_rate_annual_detected(self) -> None:
        """mortality.fishing.rate.sp0 -> rate-based."""
        from osmose.engine.config import detect_fishing_scenario

        config = {"mortality.fishing.rate.sp0": "0.3"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "rate_annual"

    def test_rate_by_year_detected(self) -> None:
        from osmose.engine.config import detect_fishing_scenario

        config = {"mortality.fishing.rate.byYear.file.sp0": "/path/to/file.csv"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "rate_by_year"

    def test_catches_annual_detected(self) -> None:
        from osmose.engine.config import detect_fishing_scenario

        config = {"mortality.fishing.catches.sp0": "1000"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "catches_annual"

    def test_rate_by_dt_by_age_detected(self) -> None:
        from osmose.engine.config import detect_fishing_scenario

        config = {"mortality.fishing.rate.byDt.byAge.file.sp0": "/path.csv"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "rate_by_dt_by_class"

    def test_no_fishing_returns_none(self) -> None:
        from osmose.engine.config import detect_fishing_scenario

        config = {"species.name.sp0": "TestFish"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario is None

    def test_catches_by_year_detected(self) -> None:
        from osmose.engine.config import detect_fishing_scenario

        config = {"mortality.fishing.catches.byYear.file.sp0": "/path.csv"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "catches_by_year"

    def test_catches_by_dt_by_class_detected(self) -> None:
        from osmose.engine.config import detect_fishing_scenario

        config = {"mortality.fishing.catches.byDt.byAge.file.sp0": "/path.csv"}
        scenario = detect_fishing_scenario(config, 0)
        assert scenario == "catches_by_dt_by_class"
