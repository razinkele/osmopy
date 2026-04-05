"""Tests for bioen_reproduction.bioen_egg_production."""

import numpy as np
import pytest
from osmose.engine.processes.bioen_reproduction import bioen_egg_production


class TestBioenReproduction:
    def test_immature_no_eggs(self):
        """Fish below L_mature produce no eggs."""
        eggs = bioen_egg_production(
            gonad_weight=np.array([0.01]),
            length=np.array([5.0]),
            age_dt=np.array([1], dtype=np.int32),
            m0=10.0,
            m1=2.0,
            egg_weight=0.001,
            n_dt_per_year=24,
        )
        # L_mature = 10 + 2*(1/24) ≈ 10.08, length 5 < 10.08
        assert eggs[0] == 0.0

    def test_mature_with_gonad_produces_eggs(self):
        """Mature fish with gonad weight produce eggs."""
        eggs = bioen_egg_production(
            gonad_weight=np.array([0.5]),
            length=np.array([20.0]),
            age_dt=np.array([48], dtype=np.int32),
            m0=5.0,
            m1=2.0,
            egg_weight=0.001,
            n_dt_per_year=24,
        )
        # L_mature = 5 + 2*2 = 9, length 20 >= 9 -> mature
        assert eggs[0] == pytest.approx(0.5 / 0.001)  # 500 eggs

    def test_mature_zero_gonad_no_eggs(self):
        """Mature but empty gonad -> no eggs."""
        eggs = bioen_egg_production(
            gonad_weight=np.array([0.0]),
            length=np.array([20.0]),
            age_dt=np.array([48], dtype=np.int32),
            m0=5.0,
            m1=2.0,
            egg_weight=0.001,
            n_dt_per_year=24,
        )
        assert eggs[0] == 0.0

    def test_vectorized(self):
        """Mix of mature/immature fish."""
        eggs = bioen_egg_production(
            gonad_weight=np.array([0.1, 0.2, 0.0]),
            length=np.array([20.0, 3.0, 20.0]),
            age_dt=np.array([48, 48, 48], dtype=np.int32),
            m0=5.0,
            m1=2.0,
            egg_weight=0.01,
            n_dt_per_year=24,
        )
        assert eggs[0] > 0  # mature + gonad
        assert eggs[1] == 0  # immature (length 3 < 9)
        assert eggs[2] == 0  # mature but no gonad
