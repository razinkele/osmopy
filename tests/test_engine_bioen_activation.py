"""Tests for bioen process activation (SP-5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes.foraging_mortality import foraging_rate
from osmose.engine.state import MortalityCause, SchoolState


class TestForagingMortality:
    """ForagingMortality.getRate() -- two modes."""

    def test_constant_mode(self) -> None:
        """Without genetics: F = k_for / ndt_per_year."""
        k_for = np.array([0.24])
        ndt_per_year = 24
        rate = foraging_rate(k_for=k_for, ndt_per_year=ndt_per_year)
        assert rate[0] == pytest.approx(0.01)  # 0.24 / 24

    def test_constant_mode_clamped(self) -> None:
        """Negative result clamped to 0."""
        k_for = np.array([-0.1])
        rate = foraging_rate(k_for=k_for, ndt_per_year=24)
        assert rate[0] == 0.0

    def test_genetic_mode(self) -> None:
        """With genetics: F = k1 * exp(k2 * (imax_trait - I_max)) / ndt."""
        k1 = np.array([0.1])
        k2 = np.array([2.0])
        imax_trait = np.array([5.0])
        I_max = np.array([5.0])  # trait == baseline -> exp(0) = 1
        ndt_per_year = 24
        rate = foraging_rate(
            k_for=None,
            ndt_per_year=ndt_per_year,
            k1_for=k1,
            k2_for=k2,
            imax_trait=imax_trait,
            I_max=I_max,
        )
        assert rate[0] == pytest.approx(0.1 / 24)  # k1 * exp(0) / 24

    def test_genetic_mode_penalty(self) -> None:
        """Trait below baseline -> exponential penalty increases rate."""
        k1 = np.array([0.1])
        k2 = np.array([1.0])
        imax_trait = np.array([3.0])  # below baseline
        I_max = np.array([5.0])
        rate = foraging_rate(
            k_for=None,
            ndt_per_year=24,
            k1_for=k1,
            k2_for=k2,
            imax_trait=imax_trait,
            I_max=I_max,
        )
        # exp(1.0 * (3 - 5)) = exp(-2) ~ 0.135
        expected = 0.1 * np.exp(-2.0) / 24
        assert rate[0] == pytest.approx(expected, rel=1e-6)


class TestBioenStarvationSwitch:
    """When bioen_enabled=True, starvation uses gonad-depletion formula."""

    def test_bioen_starvation_uses_gonad(self) -> None:
        """With bioen enabled, starvation depletes gonad weight before killing."""
        from osmose.engine.processes.bioen_starvation import bioen_starvation

        e_net = np.array([-1.0])  # negative -> starvation
        gonad_weight = np.array([0.5])
        weight = np.array([0.01])
        eta = 0.8

        n_dead, new_gonad = bioen_starvation(e_net, gonad_weight, weight, eta, n_subdt=1)
        # Gonad should absorb some deficit
        assert new_gonad[0] < gonad_weight[0]

    def test_standard_starvation_when_bioen_disabled(self) -> None:
        """With bioen disabled, standard starvation rate applies."""
        from osmose.engine.processes.mortality import _apply_starvation_for_school

        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            weight=np.array([0.01]),
            starvation_rate=np.array([0.5]),
            age_dt=np.array([48], dtype=np.int32),
        )

        config = MagicMock(spec=EngineConfig)
        config.bioen_enabled = False
        config.n_dt_per_year = 24

        inst_abd = state.abundance.copy()
        _apply_starvation_for_school(0, state, config, n_subdt=1, inst_abd=inst_abd)
        assert state.n_dead[0, MortalityCause.STARVATION] > 0
