"""Tests for `bioen_reproduction.bioen_egg_release` (Java `BioenReproductionProcess.run()`).

Rewritten for the Java-parity contract (task 5): eggs scale with school ABUNDANCE, the
season is the released gonad FRACTION (not a flush), and maturity is decided by the
caller rather than inside the release function.

The maturity cases below previously lived in `bioen_egg_production`, which took
`length`/`age_dt`/`m0`/`m1` and applied the LMRN itself. Maturity now belongs to
`_bioen_reproduction` (Java decides it in `EnergyBudget.getMaturation`, not in the
reproduction process), so the LMRN coverage moved to
`test_engine_bioen_reproduction_wiring.py::TestBioenReproductionMaturity`.
"""

import numpy as np
import pytest

from osmose.engine.processes.bioen_reproduction import bioen_egg_release


class TestBioenEggRelease:
    def test_immature_no_eggs(self):
        """Schools flagged immature by the caller release nothing and keep their gonad."""
        n_eggs, w_egg = bioen_egg_release(
            gonad_weight=np.array([0.01]),
            abundance=np.array([1e5]),
            is_mature=np.array([False]),
            season=0.25,
            sex_ratio=0.5,
            egg_weight_t=1e-9,
        )
        assert n_eggs[0] == 0.0
        assert w_egg[0] == 0.0

    def test_mature_with_gonad_produces_eggs(self):
        """nEgg = gonad*season*sexRatio/eggWeight*N."""
        n_eggs, w_egg = bioen_egg_release(
            gonad_weight=np.array([0.5]),
            abundance=np.array([1e4]),
            is_mature=np.array([True]),
            season=0.25,
            sex_ratio=0.5,
            egg_weight_t=1e-3,
        )
        assert w_egg[0] == pytest.approx(0.5 * 0.25)
        assert n_eggs[0] == pytest.approx(0.5 * 0.25 * 0.5 / 1e-3 * 1e4)

    def test_mature_zero_gonad_no_eggs(self):
        """Mature but empty gonad -> no eggs (Java `wEgg <= 0 -> continue`)."""
        n_eggs, w_egg = bioen_egg_release(
            gonad_weight=np.array([0.0]),
            abundance=np.array([1e5]),
            is_mature=np.array([True]),
            season=0.25,
            sex_ratio=0.5,
            egg_weight_t=1e-3,
        )
        assert n_eggs[0] == 0.0
        assert w_egg[0] == 0.0

    def test_vectorized(self):
        """Mix of mature/immature/spent schools."""
        n_eggs, w_egg = bioen_egg_release(
            gonad_weight=np.array([0.1, 0.2, 0.0]),
            abundance=np.array([1e4, 1e4, 1e4]),
            is_mature=np.array([True, False, True]),
            season=0.25,
            sex_ratio=0.5,
            egg_weight_t=1e-2,
        )
        assert n_eggs[0] > 0  # mature + gonad
        assert n_eggs[1] == 0  # immature
        assert n_eggs[2] == 0  # mature but no gonad
        np.testing.assert_array_equal(w_egg, [0.1 * 0.25, 0.0, 0.0])

    def test_eggs_scale_with_abundance(self):
        """The v1 contract summed `gonad/egg_weight` per SCHOOL with no `* N`, so two
        schools differing only in abundance produced the same egg count. Java multiplies
        by `school.getInstantaneousAbundance()`."""
        n_eggs, _ = bioen_egg_release(
            gonad_weight=np.array([0.1, 0.1]),
            abundance=np.array([1e6, 1e3]),
            is_mature=np.array([True, True]),
            season=0.25,
            sex_ratio=0.5,
            egg_weight_t=1e-2,
        )
        assert n_eggs[0] / n_eggs[1] == pytest.approx(1e3)

    def test_gonad_decrement_is_partial(self):
        """v1 zeroed the gonad of every spawner. Java removes only `gonad*season`."""
        gonad = np.array([0.4])
        _, w_egg = bioen_egg_release(gonad, np.array([1e4]), np.array([True]), 0.25, 0.5, 1e-3)
        remaining = gonad - w_egg
        assert remaining[0] == pytest.approx(0.4 * 0.75)
        assert remaining[0] > 0.0

    def test_sex_ratio_scales_output(self):
        n_hi, _ = bioen_egg_release(
            np.array([0.1]), np.array([1e4]), np.array([True]), 0.25, 0.6, 1e-3
        )
        n_lo, _ = bioen_egg_release(
            np.array([0.1]), np.array([1e4]), np.array([True]), 0.25, 0.3, 1e-3
        )
        assert n_hi[0] / n_lo[0] == pytest.approx(2.0)

    def test_zero_egg_weight_does_not_divide_by_zero(self):
        n_eggs, _ = bioen_egg_release(
            np.array([0.1]), np.array([1e4]), np.array([True]), 0.25, 0.5, 0.0
        )
        assert np.isfinite(n_eggs[0])
