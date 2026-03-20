"""Tests for predation helper functions and diet tracking in both Numba/Python paths."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from osmose.engine.config import EngineConfig
from osmose.engine.processes import predation as predation_module
from osmose.engine.processes.predation import (
    compute_appetite,
    compute_size_overlap,
    disable_diet_tracking,
    enable_diet_tracking,
    get_diet_matrix,
    predation,
)
from osmose.engine.state import SchoolState
from tests.test_engine_predation import _make_predation_config


# ---------------------------------------------------------------------------
# Helper to build a minimal two-school state (one predator, one prey)
# ---------------------------------------------------------------------------


def _make_pred_prey_state() -> tuple[SchoolState, EngineConfig]:
    """Return a (state, config) pair with sp1 predator eating sp0 prey."""
    cfg = EngineConfig.from_dict(_make_predation_config())
    # sp1 (predator, len=25) vs sp0 (prey, len=10): ratio=2.5, in [1.0, 3.5)
    state = SchoolState.create(n_schools=2, species_id=np.array([1, 0], dtype=np.int32))
    state = state.replace(
        abundance=np.array([50.0, 500.0]),
        length=np.array([25.0, 10.0]),
        weight=np.array([78.125, 6.0]),
        biomass=np.array([3906.25, 3000.0]),
        age_dt=np.array([24, 24], dtype=np.int32),
        cell_x=np.array([0, 0], dtype=np.int32),
        cell_y=np.array([0, 0], dtype=np.int32),
    )
    return state, cfg


# ---------------------------------------------------------------------------
# compute_size_overlap
# ---------------------------------------------------------------------------


class TestComputeSizeOverlap:
    def test_ratio_in_range(self):
        # pred=25, prey=10 -> ratio=2.5, r_min=1.0, r_max=3.5 -> True
        assert compute_size_overlap(25.0, 10.0, 1.0, 3.5) is True

    def test_ratio_below_min(self):
        # pred=6, prey=25 -> ratio=0.24 < 1.0 -> False
        assert compute_size_overlap(6.0, 25.0, 1.0, 3.5) is False

    def test_ratio_above_max(self):
        # pred=25, prey=6 -> ratio=4.17 >= 3.5 -> False
        assert compute_size_overlap(25.0, 6.0, 1.0, 3.5) is False

    def test_ratio_at_min_boundary(self):
        # ratio exactly equal to r_min is included
        assert compute_size_overlap(10.0, 10.0, 1.0, 3.5) is True

    def test_ratio_at_max_boundary(self):
        # ratio exactly equal to r_max is excluded (half-open interval)
        assert compute_size_overlap(35.0, 10.0, 1.0, 3.5) is False

    def test_zero_prey_length(self):
        assert compute_size_overlap(25.0, 0.0, 1.0, 3.5) is False

    def test_negative_prey_length(self):
        assert compute_size_overlap(25.0, -5.0, 1.0, 3.5) is False


# ---------------------------------------------------------------------------
# compute_appetite
# ---------------------------------------------------------------------------


class TestComputeAppetite:
    def test_basic_calculation(self):
        # biomass=3906.25, rate=3.5, n_dt=24, n_subdt=10
        expected = 3906.25 * 3.5 / (24 * 10)
        result = compute_appetite(3906.25, 3.5, 24, 10)
        assert abs(result - expected) < 1e-10

    def test_zero_biomass(self):
        assert compute_appetite(0.0, 3.5, 24, 10) == 0.0

    def test_scales_linearly_with_biomass(self):
        a1 = compute_appetite(100.0, 2.0, 24, 10)
        a2 = compute_appetite(200.0, 2.0, 24, 10)
        assert abs(a2 - 2 * a1) < 1e-10

    def test_scales_inversely_with_n_subdt(self):
        a1 = compute_appetite(1000.0, 2.0, 24, 5)
        a2 = compute_appetite(1000.0, 2.0, 24, 10)
        assert abs(a1 - 2 * a2) < 1e-10


# ---------------------------------------------------------------------------
# Diet tracking — Python path
# ---------------------------------------------------------------------------


class TestDietTrackingPythonPath:
    def setup_method(self):
        disable_diet_tracking()

    def teardown_method(self):
        disable_diet_tracking()

    def test_diet_recorded_when_enabled(self):
        """Python path records diet matrix when tracking is enabled."""
        state, cfg = _make_pred_prey_state()
        # 2 schools (school 0 = sp1 predator, school 1 = sp0 prey)
        enable_diet_tracking(n_schools=2, n_species=2)
        rng = np.random.default_rng(0)
        predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10)

        diet = get_diet_matrix()
        assert diet is not None
        # Predator (school 0, sp1) ate prey (school 1, sp0): diet[0, 0] > 0
        assert diet[0, 0] > 0.0, "Predator school should have recorded eating from sp0"
        # Prey did NOT eat predator: diet[1, 1] should be 0
        assert diet[1, 1] == 0.0, "Prey school should not have eaten predator sp1"

    def test_diet_zero_when_disabled(self):
        """No diet accumulation when tracking is disabled (default)."""
        state, cfg = _make_pred_prey_state()
        rng = np.random.default_rng(0)
        predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10)
        assert get_diet_matrix() is None


# ---------------------------------------------------------------------------
# Diet tracking — Numba path (when Numba is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not predation_module._HAS_NUMBA,
    reason="Numba not installed; Numba-path diet tracking tests skipped",
)
class TestDietTrackingNumbaPath:
    def setup_method(self):
        disable_diet_tracking()

    def teardown_method(self):
        disable_diet_tracking()

    def test_numba_records_diet_when_enabled(self):
        """Numba path records diet matrix when tracking is enabled."""
        state, cfg = _make_pred_prey_state()
        enable_diet_tracking(n_schools=2, n_species=2)
        rng = np.random.default_rng(0)
        predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10)

        diet = get_diet_matrix()
        assert diet is not None
        # school 0 is sp1 (predator) that ate sp0 (col 0)
        assert diet[0, 0] > 0.0, "Numba: predator school should have diet[pred, prey_sp] > 0"

    def test_numba_prey_does_not_eat_predator(self):
        """Numba path: prey school should not record eating predator."""
        state, cfg = _make_pred_prey_state()
        enable_diet_tracking(n_schools=2, n_species=2)
        rng = np.random.default_rng(0)
        predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10)

        diet = get_diet_matrix()
        assert diet is not None
        # school 1 is sp0 (prey); sp1 is column 1 — prey cannot eat predator here
        assert diet[1, 1] == 0.0, "Numba: prey school diet[prey, pred_sp] should be 0"


# ---------------------------------------------------------------------------
# Path parity: Numba and Python produce identical results
# ---------------------------------------------------------------------------


class TestNumbaVsPythonParity:
    """Verify that Numba and Python predation paths produce identical outputs."""

    def setup_method(self):
        disable_diet_tracking()

    def teardown_method(self):
        disable_diet_tracking()

    def _run_predation(self, use_numba: bool, seed: int = 42) -> tuple:
        """Run predation with Numba toggled on/off; return (new_state, diet_matrix)."""
        state, cfg = _make_pred_prey_state()
        enable_diet_tracking(n_schools=2, n_species=2)
        rng = np.random.default_rng(seed)

        with mock.patch.object(predation_module, "_HAS_NUMBA", use_numba):
            new_state = predation(state, cfg, rng, n_subdt=10, grid_ny=10, grid_nx=10)

        diet = get_diet_matrix()
        disable_diet_tracking()
        return new_state, diet

    def test_abundance_identical(self):
        """Both paths should yield identical post-predation abundance."""
        if not predation_module._HAS_NUMBA:
            pytest.skip("Numba not installed; parity test requires both paths")

        state_numba, diet_numba = self._run_predation(use_numba=True, seed=42)
        state_python, diet_python = self._run_predation(use_numba=False, seed=42)

        np.testing.assert_allclose(
            state_numba.abundance,
            state_python.abundance,
            rtol=1e-10,
            err_msg="Abundance mismatch between Numba and Python paths",
        )

    def test_pred_success_rate_identical(self):
        """Both paths should yield identical predator success rates."""
        if not predation_module._HAS_NUMBA:
            pytest.skip("Numba not installed; parity test requires both paths")

        state_numba, _ = self._run_predation(use_numba=True, seed=42)
        state_python, _ = self._run_predation(use_numba=False, seed=42)

        np.testing.assert_allclose(
            state_numba.pred_success_rate,
            state_python.pred_success_rate,
            rtol=1e-10,
            err_msg="pred_success_rate mismatch between Numba and Python paths",
        )

    def test_diet_matrix_identical(self):
        """Both paths should produce identical diet matrices when tracking is enabled."""
        if not predation_module._HAS_NUMBA:
            pytest.skip("Numba not installed; parity test requires both paths")

        _, diet_numba = self._run_predation(use_numba=True, seed=42)
        _, diet_python = self._run_predation(use_numba=False, seed=42)

        assert diet_numba is not None and diet_python is not None
        np.testing.assert_allclose(
            diet_numba,
            diet_python,
            rtol=1e-10,
            err_msg="Diet matrix mismatch between Numba and Python paths",
        )
