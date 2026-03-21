"""Tests for osmose.engine.accessibility."""

import numpy as np
import pytest
from pathlib import Path

from osmose.engine.accessibility import (
    AccessibilityMatrix,
    StageInfo,
    _parse_label,
    _parse_labels,
)


class TestParseLabel:
    def test_simple_name(self):
        name, threshold = _parse_label("cod")
        assert name == "cod"
        assert threshold == float("inf")

    def test_name_with_threshold(self):
        name, threshold = _parse_label("cod < 0.45")
        assert name == "cod"
        assert threshold == 0.45

    def test_whitespace_handling(self):
        name, threshold = _parse_label("  hake  < 1.5  ")
        assert name == "hake"
        assert threshold == 1.5


class TestParseLabels:
    def test_single_stage_species(self):
        result = _parse_labels(["cod", "hake"])
        assert len(result["cod"]) == 1
        assert result["cod"][0].threshold == float("inf")
        assert result["cod"][0].matrix_index == 0

    def test_two_stage_species(self):
        result = _parse_labels(["cod < 0.45", "cod"])
        stages = result["cod"]
        assert len(stages) == 2
        assert stages[0].threshold == 0.45
        assert stages[0].matrix_index == 0
        assert stages[1].threshold == float("inf")
        assert stages[1].matrix_index == 1


class TestFromCsv:
    def test_2x2_matrix_shape(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod;hake\ncod;0.8;0.5\nhake;0.3;0.9\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod", "hake"])
        assert matrix.raw_matrix.shape == (2, 2)
        assert "cod" in matrix.prey_lookup
        assert "hake" in matrix.prey_lookup

    def test_matrix_values(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod;hake\ncod;0.8;0.5\nhake;0.3;0.9\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod", "hake"])
        assert matrix.raw_matrix[0, 0] == 0.8
        assert matrix.raw_matrix[1, 0] == 0.3

    def test_staged_labels(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod < 0.45;cod\ncod < 0.45;0.1;0.2\ncod;0.3;0.4\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        assert len(matrix.prey_lookup["cod"]) == 2
        assert matrix.prey_lookup["cod"][0].threshold == 0.45


class TestGetIndex:
    def test_single_stage_any_age(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod\ncod;1.0\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        idx = matrix.get_index("cod", age_years=5.0, role="prey")
        assert idx == 0

    def test_two_stage_young_vs_old(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod < 0.45;cod\ncod < 0.45;0.1;0.2\ncod;0.3;0.4\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        young_idx = matrix.get_index("cod", age_years=0.2, role="prey")
        old_idx = matrix.get_index("cod", age_years=1.0, role="prey")
        assert young_idx != old_idx
        assert young_idx == 0
        assert old_idx == 1

    def test_unknown_species_returns_minus_one(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod\ncod;1.0\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        assert matrix.get_index("unknown_fish", age_years=1.0) == -1


class TestResolveName:
    def test_case_insensitive(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";Cod\nCod;1.0\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        assert matrix.resolve_name("cod") == "Cod"

    def test_missing_species(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod\ncod;1.0\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod"])
        assert matrix.resolve_name("hake") is None


class TestComputeSchoolIndices:
    def test_basic_vectorized(self, tmp_path):
        csv = tmp_path / "access.csv"
        csv.write_text(";cod;hake\ncod;0.8;0.5\nhake;0.3;0.9\n")
        matrix = AccessibilityMatrix.from_csv(csv, species_names=["cod", "hake"])
        indices = matrix.compute_school_indices(
            species_id=np.array([0, 1, 0], dtype=np.int32),
            age_dt=np.array([12, 24, 6], dtype=np.int32),
            n_dt_per_year=12,
            all_species_names=["cod", "hake"],
            role="prey",
        )
        assert indices.shape == (3,)
        assert (indices >= 0).all()
