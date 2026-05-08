"""Tests for osmose.engine.accessibility."""

import numpy as np

from osmose.engine.accessibility import (
    AccessibilityMatrix,
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


class TestComputeSchoolIndicesVectorisedMatchesLoop:
    """A1 acceptance: the vectorised path is element-wise equal to the loop."""

    def _make_matrix(self, tmp_path) -> AccessibilityMatrix:
        # Multi-stage prey + pred layout exercising:
        #   - 2 finite + 1 inf threshold (cod, hake)
        #   - 1 finite + 1 inf threshold (anchovy)
        #   - 1 stage / no threshold (sprat — pred only)
        csv = tmp_path / "access.csv"
        csv.write_text(
            ";cod < 0.4;cod;hake < 0.5;hake;anchovy < 0.2;anchovy;sprat\n"
            "cod < 0.4;0.10;0.20;0.30;0.40;0.50;0.60;0.70\n"
            "cod;0.11;0.21;0.31;0.41;0.51;0.61;0.71\n"
            "hake < 0.5;0.12;0.22;0.32;0.42;0.52;0.62;0.72\n"
            "hake;0.13;0.23;0.33;0.43;0.53;0.63;0.73\n"
            "anchovy < 0.2;0.14;0.24;0.34;0.44;0.54;0.64;0.74\n"
            "anchovy;0.15;0.25;0.35;0.45;0.55;0.65;0.75\n"
        )
        return AccessibilityMatrix.from_csv(
            csv, species_names=["cod", "hake", "anchovy", "sprat"]
        )

    def test_randomised_inputs_match_loop_prey(self, tmp_path):
        m = self._make_matrix(tmp_path)
        rng = np.random.default_rng(0)
        n = 500
        species_id = rng.integers(0, 4, size=n, dtype=np.int32)
        age_dt = rng.integers(0, 60, size=n, dtype=np.int32)
        names = ["cod", "hake", "anchovy", "sprat"]
        vec = m.compute_school_indices(species_id, age_dt, 12, names, role="prey")
        loop = m._compute_school_indices_loop(species_id, age_dt, 12, names, role="prey")
        np.testing.assert_array_equal(vec, loop)

    def test_randomised_inputs_match_loop_pred(self, tmp_path):
        m = self._make_matrix(tmp_path)
        rng = np.random.default_rng(0)
        n = 500
        species_id = rng.integers(0, 4, size=n, dtype=np.int32)
        age_dt = rng.integers(0, 60, size=n, dtype=np.int32)
        names = ["cod", "hake", "anchovy", "sprat"]
        vec = m.compute_school_indices(species_id, age_dt, 12, names, role="pred")
        loop = m._compute_school_indices_loop(species_id, age_dt, 12, names, role="pred")
        np.testing.assert_array_equal(vec, loop)

    def test_age_at_threshold_returns_next_stage(self, tmp_path):
        # cod thresholds = [0.4, +inf], matrix indices = [0, 1].
        # age = 0.4 (== threshold): legacy `if age < threshold` is False at the
        # 0.4 stage, falls through to the +inf stage → matrix index 1.
        # searchsorted-right at age=0.4 in [0.4, +inf] returns 1 → index 1.
        m = self._make_matrix(tmp_path)
        # n_dt_per_year=10 so age_dt=4 → exactly 0.4 years.
        indices = m.compute_school_indices(
            species_id=np.array([0], dtype=np.int32),
            age_dt=np.array([4], dtype=np.int32),
            n_dt_per_year=10,
            all_species_names=["cod", "hake", "anchovy", "sprat"],
            role="prey",
        )
        loop = m._compute_school_indices_loop(
            species_id=np.array([0], dtype=np.int32),
            age_dt=np.array([4], dtype=np.int32),
            n_dt_per_year=10,
            all_species_names=["cod", "hake", "anchovy", "sprat"],
            role="prey",
        )
        np.testing.assert_array_equal(indices, loop)

    def test_age_below_min_threshold_returns_first_stage(self, tmp_path):
        m = self._make_matrix(tmp_path)
        indices = m.compute_school_indices(
            species_id=np.array([0], dtype=np.int32),
            age_dt=np.array([1], dtype=np.int32),  # 0.083 yr < 0.4 threshold
            n_dt_per_year=12,
            all_species_names=["cod", "hake", "anchovy", "sprat"],
            role="prey",
        )
        # cod stage 0 is the row "cod < 0.4" = matrix row 0
        assert indices[0] == 0

    def test_all_finite_thresholds_species_returns_last_stage_for_old_ages(self, tmp_path):
        # CSV without an open-ended adult label for whitefish:
        # only "whitefish < 0.4" and "whitefish < 1.2", no plain "whitefish".
        # Legacy loop falls through both `if age < threshold` checks and
        # returns stages[-1] (the threshold-1.2 row). Vectorised path:
        # searchsorted-right(>=1.2) returns 2, clamp to 1 → matrix index 1.
        csv = tmp_path / "access.csv"
        csv.write_text(
            ";whitefish < 0.4;whitefish < 1.2\n"
            "whitefish < 0.4;0.1;0.2\n"
            "whitefish < 1.2;0.3;0.4\n"
        )
        m = AccessibilityMatrix.from_csv(csv, species_names=["whitefish"])
        names = ["whitefish"]
        species_id = np.array([0, 0, 0], dtype=np.int32)
        age_dt = np.array([2, 8, 24], dtype=np.int32)  # 0.17, 0.67, 2.0 years
        vec = m.compute_school_indices(species_id, age_dt, 12, names, role="prey")
        loop = m._compute_school_indices_loop(species_id, age_dt, 12, names, role="prey")
        np.testing.assert_array_equal(vec, loop)
        # 2.0 yr > all finite thresholds → falls back to stages[-1] = idx 1.
        assert vec[2] == 1

    def test_unresolved_species_keeps_minus_one(self, tmp_path):
        # all_species_names contains "shark" which has no entry in the CSV.
        # Legacy loop: resolved[2] = None, school 2 stays -1.
        # Vectorised path: sp_idx=2 absent from _stages_by_role[role],
        # mask never fires, school 2 stays -1.
        csv = tmp_path / "access.csv"
        csv.write_text(";cod;hake\ncod;0.8;0.5\nhake;0.3;0.9\n")
        m = AccessibilityMatrix.from_csv(
            csv, species_names=["cod", "hake", "shark"]
        )
        species_id = np.array([0, 1, 2, 0], dtype=np.int32)
        age_dt = np.array([12, 12, 12, 12], dtype=np.int32)
        vec = m.compute_school_indices(
            species_id, age_dt, 12, ["cod", "hake", "shark"], role="prey"
        )
        loop = m._compute_school_indices_loop(
            species_id, age_dt, 12, ["cod", "hake", "shark"], role="prey"
        )
        np.testing.assert_array_equal(vec, loop)
        assert vec[2] == -1

    def test_role_only_in_pred(self, tmp_path):
        # sprat appears only as a predator (column), not as a prey (row).
        # Legacy loop role="prey": resolve_name returns "sprat" but
        # prey_lookup.get("sprat") is None → schools stay -1.
        # Vectorised path role="prey": _stages_by_role["prey"] has no sprat
        # entry, mask never fires.
        m = self._make_matrix(tmp_path)
        species_id = np.array([3, 3], dtype=np.int32)  # sprat
        age_dt = np.array([1, 60], dtype=np.int32)
        vec = m.compute_school_indices(
            species_id, age_dt, 12, ["cod", "hake", "anchovy", "sprat"], role="prey"
        )
        loop = m._compute_school_indices_loop(
            species_id, age_dt, 12, ["cod", "hake", "anchovy", "sprat"], role="prey"
        )
        np.testing.assert_array_equal(vec, loop)
        assert (vec == -1).all()

    def test_empty_input_returns_empty(self, tmp_path):
        m = self._make_matrix(tmp_path)
        empty_id = np.array([], dtype=np.int32)
        empty_age = np.array([], dtype=np.int32)
        vec = m.compute_school_indices(
            empty_id, empty_age, 12, ["cod", "hake", "anchovy", "sprat"], role="prey"
        )
        assert vec.shape == (0,)
        assert vec.dtype == np.int32
