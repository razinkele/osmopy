"""Tests for composable banded loss objectives."""

from __future__ import annotations

import math

import pytest

from osmose.calibration.losses import (
    banded_log_ratio_loss,
    make_banded_objective,
    stability_penalty,
    worst_species_penalty,
)
from osmose.calibration.targets import BiomassTarget


class TestBandedLogRatioLoss:
    def test_within_range_returns_zero(self) -> None:
        assert banded_log_ratio_loss(150000, 60000, 250000) == 0.0

    def test_at_lower_bound_returns_zero(self) -> None:
        assert banded_log_ratio_loss(60000, 60000, 250000) == 0.0

    def test_at_upper_bound_returns_zero(self) -> None:
        assert banded_log_ratio_loss(250000, 60000, 250000) == 0.0

    def test_below_range(self) -> None:
        result = banded_log_ratio_loss(6000, 60000, 250000)
        expected = math.log10(60000 / 6000) ** 2
        assert result == pytest.approx(expected)

    def test_above_range(self) -> None:
        result = banded_log_ratio_loss(500000, 60000, 250000)
        expected = math.log10(500000 / 250000) ** 2
        assert result == pytest.approx(expected)

    def test_zero_biomass(self) -> None:
        assert banded_log_ratio_loss(0, 60000, 250000) == 100.0

    def test_negative_biomass(self) -> None:
        assert banded_log_ratio_loss(-100, 60000, 250000) == 100.0


class TestStabilityPenalty:
    def test_both_below_threshold(self) -> None:
        assert stability_penalty(0.1, 0.02) == 0.0

    def test_cv_above_threshold(self) -> None:
        result = stability_penalty(0.4, 0.02)
        assert result == pytest.approx((0.4 - 0.2) ** 2)

    def test_trend_above_threshold(self) -> None:
        result = stability_penalty(0.1, 0.1)
        assert result == pytest.approx((0.1 - 0.05) ** 2)

    def test_both_above_threshold(self) -> None:
        result = stability_penalty(0.5, 0.15)
        expected = (0.5 - 0.2) ** 2 + (0.15 - 0.05) ** 2
        assert result == pytest.approx(expected)

    def test_custom_thresholds(self) -> None:
        result = stability_penalty(0.6, 0.3, cv_threshold=0.5, trend_threshold=0.2)
        expected = (0.6 - 0.5) ** 2 + (0.3 - 0.2) ** 2
        assert result == pytest.approx(expected)


class TestWorstSpeciesPenalty:
    def test_returns_max(self) -> None:
        assert worst_species_penalty([0.1, 0.5, 0.3]) == 0.5

    def test_single_species(self) -> None:
        assert worst_species_penalty([0.7]) == 0.7

    def test_all_zero(self) -> None:
        assert worst_species_penalty([0.0, 0.0]) == 0.0

    def test_empty_list(self) -> None:
        assert worst_species_penalty([]) == 0.0


class TestMakeBandedObjective:
    @pytest.fixture()
    def targets(self) -> list[BiomassTarget]:
        return [
            BiomassTarget(species="cod", target=120000, lower=60000, upper=250000, weight=1.0),
            BiomassTarget(
                species="herring", target=1500000, lower=800000, upper=3000000, weight=0.5
            ),
        ]

    def test_all_within_range(self, targets: list[BiomassTarget]) -> None:
        obj, _ = make_banded_objective(targets, ["cod", "herring"])
        stats = {
            "cod_mean": 150000,
            "cod_cv": 0.1,
            "cod_trend": 0.01,
            "herring_mean": 2000000,
            "herring_cv": 0.05,
            "herring_trend": 0.02,
        }
        assert obj(stats) == 0.0

    def test_one_below_range(self, targets: list[BiomassTarget]) -> None:
        obj, _ = make_banded_objective(targets, ["cod", "herring"])
        stats = {
            "cod_mean": 10000,
            "cod_cv": 0.1,
            "cod_trend": 0.01,
            "herring_mean": 2000000,
            "herring_cv": 0.05,
            "herring_trend": 0.02,
        }
        result = obj(stats)
        assert result > 0.0

    def test_missing_species_key_penalty(self, targets: list[BiomassTarget]) -> None:
        obj, _ = make_banded_objective(targets, ["cod", "herring"])
        stats = {"cod_mean": 150000, "cod_cv": 0.1, "cod_trend": 0.01}
        # herring keys missing -> 100.0 penalty * 0.5 weight = 50.0 + worst term
        result = obj(stats)
        assert result >= 50.0

    def test_stability_penalty_applied(self, targets: list[BiomassTarget]) -> None:
        obj, _ = make_banded_objective(targets, ["cod", "herring"], w_stability=5.0)
        stats = {
            "cod_mean": 150000,
            "cod_cv": 0.5,
            "cod_trend": 0.01,
            "herring_mean": 2000000,
            "herring_cv": 0.05,
            "herring_trend": 0.02,
        }
        result = obj(stats)
        assert result > 0.0

    def test_worst_species_term(self, targets: list[BiomassTarget]) -> None:
        obj_with, _ = make_banded_objective(targets, ["cod", "herring"], w_worst=1.0)
        obj_without, _ = make_banded_objective(targets, ["cod", "herring"], w_worst=0.0)
        stats = {
            "cod_mean": 10000,
            "cod_cv": 0.1,
            "cod_trend": 0.01,
            "herring_mean": 2000000,
            "herring_cv": 0.05,
            "herring_trend": 0.02,
        }
        assert obj_with(stats) > obj_without(stats)


def test_make_banded_objective_returns_tuple(synthetic_two_species_targets):
    targets, species_names = synthetic_two_species_targets
    result = make_banded_objective(targets, species_names)
    assert isinstance(result, tuple)
    assert len(result) == 2
    obj, accessor = result
    assert callable(obj)
    assert callable(accessor)


def test_residuals_accessor_returns_none_before_first_call(synthetic_two_species_targets):
    targets, species_names = synthetic_two_species_targets
    _, accessor = make_banded_objective(targets, species_names)
    assert accessor() is None


def test_residuals_accessor_returns_most_recent(
    synthetic_two_species_targets, synthetic_stats_in_band,
    synthetic_stats_sp_b_out_of_band,
):
    targets, species_names = synthetic_two_species_targets
    obj, accessor = make_banded_objective(targets, species_names)

    obj(synthetic_stats_in_band)
    labels1, residuals1, sim_biomass1 = accessor()
    assert tuple(labels1) == ("sp_a", "sp_b")
    assert tuple(residuals1) == (0.0, 0.0)

    obj(synthetic_stats_sp_b_out_of_band)
    labels2, residuals2, sim_biomass2 = accessor()
    assert residuals2[1] > 0.0
    assert sim_biomass2[1] == synthetic_stats_sp_b_out_of_band["sp_b_mean"]


def test_make_banded_objective_callable_returns_zero_when_in_band(
    synthetic_two_species_targets, synthetic_stats_in_band,
):
    """Backward-compat: callable behaviour unchanged for in-band stats."""
    targets, species_names = synthetic_two_species_targets
    obj, _ = make_banded_objective(targets, species_names)
    value = obj(synthetic_stats_in_band)
    assert value == 0.0


def test_residuals_accessor_reset_to_none_on_call_failure(synthetic_two_species_targets):
    """Spec §6.5.2 failure-mode parity."""
    targets, species_names = synthetic_two_species_targets
    obj, accessor = make_banded_objective(targets, species_names)

    obj({"sp_a_mean": 1.0, "sp_b_mean": 2.0, "sp_a_cv": 0.0, "sp_b_cv": 0.0,
         "sp_a_trend": 0.0, "sp_b_trend": 0.0})
    assert accessor() is not None

    with pytest.raises((TypeError, AttributeError, KeyError)):
        obj(None)  # type: ignore[arg-type]
    assert accessor() is None
