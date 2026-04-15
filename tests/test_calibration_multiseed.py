"""Tests for multi-seed validation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from osmose.calibration.multiseed import rank_candidates_multiseed, validate_multiseed


def _make_mock_factory(base_value: float = 1.0, seed_noise: float = 0.1):
    """Returns a factory(seed) -> objective(x) -> float."""

    def factory(seed: int):
        rng = np.random.default_rng(seed)

        def objective(x: np.ndarray) -> float:
            return base_value + rng.normal(0, seed_noise) + float(np.sum(x))

        return objective

    return factory


class TestValidateMultiseed:
    def test_returns_expected_keys(self) -> None:
        factory = _make_mock_factory()
        result = validate_multiseed(factory, np.array([0.0]), seeds=[1, 2, 3])
        assert "per_seed" in result
        assert "mean" in result
        assert "std" in result
        assert "cv" in result
        assert "worst_seed" in result
        assert "worst_value" in result

    def test_per_seed_length_matches_seeds(self) -> None:
        factory = _make_mock_factory()
        result = validate_multiseed(factory, np.array([0.0]), seeds=[10, 20, 30, 40])
        assert len(result["per_seed"]) == 4

    def test_mean_is_average_of_per_seed(self) -> None:
        factory = _make_mock_factory()
        result = validate_multiseed(factory, np.array([0.0]), seeds=[1, 2])
        assert result["mean"] == pytest.approx(np.mean(result["per_seed"]))

    def test_worst_seed_is_max(self) -> None:
        factory = _make_mock_factory()
        result = validate_multiseed(factory, np.array([0.0]), seeds=[1, 2, 3])
        worst_idx = np.argmax(result["per_seed"])
        assert result["worst_seed"] == [1, 2, 3][worst_idx]
        assert result["worst_value"] == max(result["per_seed"])

    def test_deterministic(self) -> None:
        factory = _make_mock_factory()
        x = np.array([0.5])
        r1 = validate_multiseed(factory, x, seeds=[42, 123])
        r2 = validate_multiseed(factory, x, seeds=[42, 123])
        assert r1["per_seed"] == r2["per_seed"]


class TestRankCandidatesMultiseed:
    def test_returns_expected_keys(self) -> None:
        factory = _make_mock_factory()
        candidates = np.array([[0.0], [1.0]])
        result = rank_candidates_multiseed(factory, candidates, seeds=[1, 2])
        assert "rankings" in result
        assert "scores" in result

    def test_rankings_length(self) -> None:
        factory = _make_mock_factory()
        candidates = np.array([[0.0], [1.0], [2.0]])
        result = rank_candidates_multiseed(factory, candidates, seeds=[1, 2])
        assert len(result["rankings"]) == 3
        assert len(result["scores"]) == 3

    def test_lower_sum_ranked_first(self) -> None:
        """Candidate with x=[0] should rank above x=[10] (lower objective)."""
        factory = _make_mock_factory(base_value=0.0, seed_noise=0.001)
        candidates = np.array([[0.0], [10.0]])
        result = rank_candidates_multiseed(factory, candidates, seeds=[1, 2, 3])
        assert result["rankings"][0] == 0  # x=[0.0] is better

    def test_scores_have_multiseed_fields(self) -> None:
        factory = _make_mock_factory()
        candidates = np.array([[0.0], [1.0]])
        result = rank_candidates_multiseed(factory, candidates, seeds=[1, 2])
        score = result["scores"][0]
        assert "per_seed" in score
        assert "mean" in score
        assert "cv" in score
