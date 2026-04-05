"""Tests for fisheries-based fishing mortality and stage-indexed accessibility."""

import textwrap

import numpy as np
import pytest

from osmose.engine.accessibility import AccessibilityMatrix, _parse_label, _parse_labels
from osmose.engine.config import EngineConfig
from osmose.engine.processes.fishing import fishing_mortality
from osmose.engine.state import MortalityCause, SchoolState


# ---------------------------------------------------------------------------
# Helper: minimal config with fisheries
# ---------------------------------------------------------------------------


def _make_fisheries_config(n_sp: int = 2, n_dt: int = 24, tmp_path=None) -> dict[str, str]:
    """Build a minimal config with fisheries enabled."""
    cfg: dict[str, str] = {
        "simulation.time.ndtperyear": str(n_dt),
        "simulation.time.nyear": "1",
        "simulation.nspecies": str(n_sp),
        "mortality.subdt": "10",
        "fisheries.enabled": "true",
        "simulation.nfisheries": str(n_sp),
    }
    names = ["SpeciesA", "SpeciesB", "SpeciesC", "SpeciesD"]
    for i in range(n_sp):
        cfg.update(
            {
                f"simulation.nschool.sp{i}": "10",
                f"species.name.sp{i}": names[i],
                f"species.linf.sp{i}": "20.0",
                f"species.k.sp{i}": "0.3",
                f"species.t0.sp{i}": "-0.1",
                f"species.egg.size.sp{i}": "0.1",
                f"species.length2weight.condition.factor.sp{i}": "0.006",
                f"species.length2weight.allometric.power.sp{i}": "3.0",
                f"species.lifespan.sp{i}": "3",
                f"species.vonbertalanffy.threshold.age.sp{i}": "1.0",
                f"predation.ingestion.rate.max.sp{i}": "3.5",
                f"predation.efficiency.critical.sp{i}": "0.57",
                # Fishery config
                f"fisheries.name.fsh{i}": f"fishery{i}",
                f"fisheries.rate.base.fsh{i}": str(0.3 + i * 0.1),
                f"fisheries.selectivity.type.fsh{i}": "0",
                f"fisheries.selectivity.a50.fsh{i}": str(1.0 + i * 0.5),
            }
        )

    # Write catchability CSV
    if tmp_path is not None:
        catch_path = tmp_path / "fishery-catchability.csv"
        header = "," + ",".join(f"fishery{i}" for i in range(n_sp))
        rows = [header]
        for i in range(n_sp):
            vals = ["0"] * n_sp
            vals[i] = "1"
            rows.append(f"{names[i]}," + ",".join(vals))
        catch_path.write_text("\n".join(rows))
        cfg["fisheries.catchability.file"] = str(catch_path)
        cfg["_osmose.config.dir"] = str(tmp_path)

    return cfg


# ---------------------------------------------------------------------------
# Test: Fisheries config parsing
# ---------------------------------------------------------------------------


class TestFisheriesConfigParsing:
    def test_fisheries_rates_parsed(self, tmp_path):
        """Fisheries base rates are correctly parsed into per-species fishing_rate."""
        cfg = _make_fisheries_config(n_sp=2, tmp_path=tmp_path)
        ec = EngineConfig.from_dict(cfg)
        # sp0 -> fsh0 rate=0.3, sp1 -> fsh1 rate=0.4
        assert ec.fishing_rate[0] == pytest.approx(0.3)
        assert ec.fishing_rate[1] == pytest.approx(0.4)

    def test_fisheries_a50_parsed(self, tmp_path):
        """Fisheries selectivity a50 correctly parsed."""
        cfg = _make_fisheries_config(n_sp=2, tmp_path=tmp_path)
        ec = EngineConfig.from_dict(cfg)
        assert ec.fishing_selectivity_a50[0] == pytest.approx(1.0)
        assert ec.fishing_selectivity_a50[1] == pytest.approx(1.5)

    def test_fisheries_selectivity_type(self, tmp_path):
        """Fisheries selectivity type is 0 (age-based) for all."""
        cfg = _make_fisheries_config(n_sp=2, tmp_path=tmp_path)
        ec = EngineConfig.from_dict(cfg)
        assert ec.fishing_selectivity_type[0] == 0
        assert ec.fishing_selectivity_type[1] == 0

    def test_fisheries_enabled_sets_fishing_enabled(self, tmp_path):
        """fisheries.enabled=true enables fishing even without legacy key."""
        cfg = _make_fisheries_config(n_sp=1, tmp_path=tmp_path)
        # Remove legacy fishing toggle
        cfg.pop("simulation.fishing.mortality.enabled", None)
        ec = EngineConfig.from_dict(cfg)
        assert ec.fishing_enabled is True

    def test_fisheries_disabled_zero_rates(self):
        """When fisheries.enabled is false, fishing rates stay zero."""
        cfg = _make_fisheries_config(n_sp=1)
        cfg["fisheries.enabled"] = "false"
        ec = EngineConfig.from_dict(cfg)
        assert ec.fishing_rate[0] == 0.0


# ---------------------------------------------------------------------------
# Test: Age-based fishing selectivity
# ---------------------------------------------------------------------------


class TestAgeBasedFishing:
    def test_below_a50_no_fishing(self, tmp_path):
        """Schools below a50 age are not fished."""
        cfg = _make_fisheries_config(n_sp=1, tmp_path=tmp_path)
        ec = EngineConfig.from_dict(cfg)
        # a50 = 1.0 year = 24 dt. Create school with age 12 dt (0.5 years)
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            weight=np.array([5.0]),
            age_dt=np.array([12], dtype=np.int32),
        )
        new_state = fishing_mortality(state, ec, n_subdt=10)
        np.testing.assert_allclose(new_state.abundance[0], 1000.0)

    def test_above_a50_fished(self, tmp_path):
        """Schools at or above a50 age are fished."""
        cfg = _make_fisheries_config(n_sp=1, tmp_path=tmp_path)
        ec = EngineConfig.from_dict(cfg)
        # a50 = 1.0 year = 24 dt. Create school with age 24 dt
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([1000.0]),
            weight=np.array([5.0]),
            age_dt=np.array([24], dtype=np.int32),
        )
        new_state = fishing_mortality(state, ec, n_subdt=10)
        d = 0.3 / (24 * 10)
        expected_dead = 1000.0 * (1 - np.exp(-d))
        np.testing.assert_allclose(
            new_state.n_dead[0, MortalityCause.FISHING], expected_dead, rtol=1e-10
        )

    def test_annual_fishing_decay_age_based(self, tmp_path):
        """Full year of age-based fishing gives exp(-F) decay."""
        cfg = _make_fisheries_config(n_sp=1, tmp_path=tmp_path)
        ec = EngineConfig.from_dict(cfg)
        n_subdt = 10
        state = SchoolState.create(n_schools=1, species_id=np.array([0], dtype=np.int32))
        state = state.replace(
            abundance=np.array([10000.0]),
            weight=np.array([5.0]),
            age_dt=np.array([48], dtype=np.int32),  # well above a50
        )
        for _step in range(24):
            for _sub in range(n_subdt):
                state = fishing_mortality(state, ec, n_subdt)
        expected = 10000.0 * np.exp(-0.3)
        np.testing.assert_allclose(state.abundance[0], expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test: Accessibility label parsing
# ---------------------------------------------------------------------------


class TestAccessibilityParsing:
    def test_parse_label_with_threshold(self):
        """Label 'cod < 0.4' parses to ('cod', 0.4)."""
        name, threshold = _parse_label("cod < 0.4")
        assert name == "cod"
        assert threshold == pytest.approx(0.4)

    def test_parse_label_no_threshold(self):
        """Label 'cod' parses to ('cod', inf)."""
        name, threshold = _parse_label("cod")
        assert name == "cod"
        assert threshold == float("inf")

    def test_parse_label_with_spaces(self):
        """Label with trailing spaces parses correctly."""
        name, threshold = _parse_label("pouting ")
        assert name == "pouting"
        assert threshold == float("inf")

    def test_parse_labels_multi_stage(self):
        """Two-stage species has sorted stages."""
        labels = ["cod < 0.4", "cod", "sole"]
        result = _parse_labels(labels)
        assert "cod" in result
        assert len(result["cod"]) == 2
        # First stage: threshold 0.4 (young), second: inf (adult)
        assert result["cod"][0].threshold == pytest.approx(0.4)
        assert result["cod"][0].matrix_index == 0
        assert result["cod"][1].threshold == float("inf")
        assert result["cod"][1].matrix_index == 1
        # Sole: single stage
        assert len(result["sole"]) == 1


class TestAccessibilityMatrix:
    def _write_simple_csv(self, tmp_path):
        """Write a small accessibility CSV for testing."""
        csv_content = textwrap.dedent("""\
            v Prey / Predator >;cod < 0.4;cod;sole
            cod < 0.4;0.1;0.2;0.3
            cod;0.4;0.5;0.6
            sole;0.7;0.8;0.9
        """)
        csv_path = tmp_path / "access.csv"
        csv_path.write_text(csv_content)
        return csv_path

    def test_matrix_shape(self, tmp_path):
        """Raw matrix has correct shape."""
        path = self._write_simple_csv(tmp_path)
        am = AccessibilityMatrix.from_csv(path, ["cod", "sole"])
        assert am.raw_matrix.shape == (3, 3)

    def test_single_stage_index(self, tmp_path):
        """Single-stage species (sole) gets its index."""
        path = self._write_simple_csv(tmp_path)
        am = AccessibilityMatrix.from_csv(path, ["cod", "sole"])
        # sole has one prey row at index 2
        idx = am.get_index("sole", 0.0, role="prey")
        assert idx == 2
        idx = am.get_index("sole", 5.0, role="prey")
        assert idx == 2

    def test_two_stage_young(self, tmp_path):
        """Young cod (age < 0.4) maps to 'cod < 0.4' row."""
        path = self._write_simple_csv(tmp_path)
        am = AccessibilityMatrix.from_csv(path, ["cod", "sole"])
        idx = am.get_index("cod", 0.2, role="prey")
        assert idx == 0  # "cod < 0.4" is row 0

    def test_two_stage_adult(self, tmp_path):
        """Adult cod (age >= 0.4) maps to 'cod' row."""
        path = self._write_simple_csv(tmp_path)
        am = AccessibilityMatrix.from_csv(path, ["cod", "sole"])
        idx = am.get_index("cod", 0.5, role="prey")
        assert idx == 1  # "cod" is row 1

    def test_resolve_name_case_insensitive(self, tmp_path):
        """Species name resolution is case-insensitive."""
        path = self._write_simple_csv(tmp_path)
        am = AccessibilityMatrix.from_csv(path, ["Cod", "Sole"])
        assert am.resolve_name("Cod") is not None
        assert am.resolve_name("cod") is not None
        assert am.resolve_name("COD") is not None

    def test_compute_school_indices(self, tmp_path):
        """School index arrays computed correctly."""
        path = self._write_simple_csv(tmp_path)
        am = AccessibilityMatrix.from_csv(path, ["cod", "sole"])
        species_id = np.array([0, 0, 1], dtype=np.int32)
        age_dt = np.array([2, 20, 5], dtype=np.int32)  # n_dt=24 → ages: 0.08, 0.83, 0.21 years
        indices = am.compute_school_indices(
            species_id, age_dt, n_dt_per_year=24, all_species_names=["cod", "sole"], role="prey"
        )
        # cod at 0.08 years (< 0.4) → row 0
        assert indices[0] == 0
        # cod at 0.83 years (>= 0.4) → row 1
        assert indices[1] == 1
        # sole at any age → row 2
        assert indices[2] == 2

    def test_species_not_in_matrix(self, tmp_path):
        """Unknown species returns -1."""
        path = self._write_simple_csv(tmp_path)
        am = AccessibilityMatrix.from_csv(path, ["cod", "sole"])
        idx = am.get_index("unknown_fish", 1.0, role="prey")
        assert idx == -1


# ---------------------------------------------------------------------------
# Test: EEC accessibility matrix loads
# ---------------------------------------------------------------------------


class TestEECAccessibility:
    def test_eec_accessibility_shape(self):
        """EEC accessibility CSV loads as 35x25 stage-indexed matrix."""
        from pathlib import Path

        csv_path = Path("data/eec_full/predation-accessibility.csv")
        if not csv_path.exists():
            pytest.skip("EEC data not available")
        am = AccessibilityMatrix.from_csv(csv_path, [])
        assert am.raw_matrix.shape == (35, 25)

    def test_eec_has_stage_labels(self):
        """EEC accessibility has multi-stage species labels."""
        from pathlib import Path

        csv_path = Path("data/eec_full/predation-accessibility.csv")
        if not csv_path.exists():
            pytest.skip("EEC data not available")
        am = AccessibilityMatrix.from_csv(csv_path, [])
        # lesserSpottedDogfish should have 2 prey stages
        assert "lesserSpottedDogfish" in am.prey_lookup
        assert len(am.prey_lookup["lesserSpottedDogfish"]) == 2
