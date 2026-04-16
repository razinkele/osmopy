"""Tests for density-dependent dynamic predation accessibility."""

import numpy as np

from osmose.engine.processes.dynamic_accessibility import (
    apply_prey_scale_to_matrix,
    compute_prey_density_scale,
)


class TestComputePreyDensityScale:
    def test_at_reference_returns_ones(self):
        ref = np.array([1000.0, 2000.0, 500.0])
        scale = compute_prey_density_scale(ref, ref)
        np.testing.assert_allclose(scale, [1.0, 1.0, 1.0])

    def test_above_reference_capped_at_one(self):
        bio = np.array([2000.0, 4000.0])
        ref = np.array([1000.0, 2000.0])
        scale = compute_prey_density_scale(bio, ref)
        np.testing.assert_allclose(scale, [1.0, 1.0])

    def test_below_reference_scales_down(self):
        bio = np.array([500.0, 100.0])
        ref = np.array([1000.0, 1000.0])
        scale = compute_prey_density_scale(bio, ref, exponent=1.0, floor=0.0)
        np.testing.assert_allclose(scale, [0.5, 0.1])

    def test_floor_prevents_zero(self):
        bio = np.array([0.0, 1.0])
        ref = np.array([1000.0, 1000.0])
        scale = compute_prey_density_scale(bio, ref, floor=0.05)
        assert scale[0] == 0.05
        assert scale[1] >= 0.05

    def test_exponent_controls_sensitivity(self):
        bio = np.array([500.0])
        ref = np.array([1000.0])
        # Linear: 0.5
        s1 = compute_prey_density_scale(bio, ref, exponent=1.0, floor=0.0)
        # Square: 0.25
        s2 = compute_prey_density_scale(bio, ref, exponent=2.0, floor=0.0)
        # Sqrt: ~0.707
        s3 = compute_prey_density_scale(bio, ref, exponent=0.5, floor=0.0)
        assert s2[0] < s1[0] < s3[0]
        np.testing.assert_allclose(s1, [0.5])
        np.testing.assert_allclose(s2, [0.25])

    def test_zero_reference_safe(self):
        bio = np.array([100.0])
        ref = np.array([0.0])
        scale = compute_prey_density_scale(bio, ref)
        assert np.isfinite(scale[0])
        assert scale[0] == 1.0  # bio >> ref → capped at 1


class TestApplyPreyScaleToMatrix:
    def test_species_matrix_scales_columns(self):
        # Non-stage: matrix[predator, prey]
        matrix = np.ones((4, 4))
        scale = np.array([0.5, 1.0])  # 2 focal species
        result = apply_prey_scale_to_matrix(matrix, scale, n_species=2, is_stage_indexed=False)
        # Column 0 (prey sp0) scaled by 0.5
        np.testing.assert_allclose(result[:, 0], 0.5)
        # Column 1 (prey sp1) unchanged
        np.testing.assert_allclose(result[:, 1], 1.0)
        # Columns 2-3 (resources) unchanged
        np.testing.assert_allclose(result[:, 2:], 1.0)

    def test_stage_matrix_scales_rows(self):
        # Stage-indexed: matrix[prey_stage, pred_stage]
        matrix = np.ones((4, 3))
        scale = np.array([0.5, 0.8])
        stage_to_species = np.array([0, 0, 1, -1], dtype=np.int32)
        result = apply_prey_scale_to_matrix(
            matrix, scale, n_species=2,
            is_stage_indexed=True, stage_to_species=stage_to_species,
        )
        np.testing.assert_allclose(result[0, :], 0.5)  # sp0 stage 0
        np.testing.assert_allclose(result[1, :], 0.5)  # sp0 stage 1
        np.testing.assert_allclose(result[2, :], 0.8)  # sp1
        np.testing.assert_allclose(result[3, :], 1.0)  # unmapped (-1)

    def test_does_not_mutate_input(self):
        matrix = np.ones((3, 3))
        scale = np.array([0.5, 0.5, 0.5])
        result = apply_prey_scale_to_matrix(matrix, scale, n_species=3, is_stage_indexed=False)
        np.testing.assert_allclose(matrix, 1.0)  # original unchanged
        np.testing.assert_allclose(result[:, :3], 0.5)

    def test_scale_ones_is_identity(self):
        matrix = np.random.rand(5, 5)
        scale = np.ones(3)
        result = apply_prey_scale_to_matrix(matrix, scale, n_species=3, is_stage_indexed=False)
        np.testing.assert_allclose(result, matrix)
