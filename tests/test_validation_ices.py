"""Tests for osmose.validation.ices."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from osmose.validation.ices import (
    SpeciesBiomassComparison,
    compare_outputs_to_ices,
    format_markdown_report,
    load_snapshot,
    model_biomass_window_mean,
)


def _make_snapshot(tmp_path: Path) -> Path:
    """Build a minimal ICES snapshot bundle for tests."""
    d = tmp_path / "ices_snapshots"
    d.mkdir()
    manifest = {
        "advice_year": 2024,
        "model_species_to_ices_stocks": {
            "cod": ["cod.27.22-24"],
            "herring": ["her.27.20-24", "her.27.idx-only"],
            "flounder": [],
            "perch": ["per.27.idx-only"],
        },
        "units_by_stock": {
            "cod.27.22-24": "tonnes",
            "her.27.20-24": "tonnes",
            "her.27.idx-only": "index",
            "per.27.idx-only": "index",
        },
    }
    (d / "index.json").write_text(json.dumps(manifest))

    # Cod tonnes-unit assessment: SSB 100, 110, 120, 130, 140 over 2018-2022
    (d / "cod.27.22-24.assessment.json").write_text(json.dumps([
        {"year": 2018, "ssb": 100.0, "f": 0.30},
        {"year": 2019, "ssb": 110.0, "f": 0.32},
        {"year": 2020, "ssb": 120.0, "f": 0.31},
        {"year": 2021, "ssb": 130.0, "f": 0.29},
        {"year": 2022, "ssb": 140.0, "f": 0.28},
    ]))
    (d / "cod.27.22-24.reference_points.json").write_text(json.dumps({
        "blim": 60.0, "bpa": 90.0, "fmsy": 0.30,
    }))

    # Herring tonnes-unit assessment: SSB 1000-1500
    (d / "her.27.20-24.assessment.json").write_text(json.dumps([
        {"year": 2018, "ssb": 1000.0, "f": 0.20},
        {"year": 2019, "ssb": 1100.0, "f": 0.22},
        {"year": 2020, "ssb": 1200.0, "f": 0.21},
        {"year": 2021, "ssb": 1400.0, "f": 0.19},
        {"year": 2022, "ssb": 1500.0, "f": 0.18},
    ]))
    (d / "her.27.20-24.reference_points.json").write_text(json.dumps({}))

    # Index-unit assessments — exist but should be excluded from envelope
    (d / "her.27.idx-only.assessment.json").write_text(json.dumps([
        {"year": 2018, "ssb": 0.85, "f": 0.20},
        {"year": 2022, "ssb": 0.95, "f": 0.18},
    ]))
    (d / "her.27.idx-only.reference_points.json").write_text(json.dumps({}))
    (d / "per.27.idx-only.assessment.json").write_text(json.dumps([
        {"year": 2020, "ssb": 1.2, "f": 0.10},
    ]))
    (d / "per.27.idx-only.reference_points.json").write_text(json.dumps({}))

    return d


def _fake_results(biomass_by_species: dict[str, list[float]]) -> MagicMock:
    """Build a mock OsmoseResults whose .biomass(species=...) returns a long-form df."""

    def _biomass(species: str | None = None) -> pd.DataFrame | None:
        if species is None or species not in biomass_by_species:
            return None
        values = biomass_by_species[species]
        return pd.DataFrame({
            "species": [species] * len(values),
            "time": list(range(len(values))),
            "value": values,
        })

    mock = MagicMock()
    mock.biomass = _biomass
    mock.output_dir = Path("/tmp/fake-results")
    return mock


class TestLoadSnapshot:
    def test_loads_manifest_and_assessments(self, tmp_path):
        d = _make_snapshot(tmp_path)
        snap = load_snapshot(d)
        assert snap.manifest["advice_year"] == 2024
        assert "cod.27.22-24" in snap.assessments
        assert "her.27.20-24" in snap.assessments
        assert "cod.27.22-24" in snap.reference_points

    def test_handles_missing_assessment_files_gracefully(self, tmp_path):
        # Create a manifest referencing a stock with no assessment file —
        # load_snapshot should silently skip it.
        d = tmp_path / "ices_snapshots"
        d.mkdir()
        (d / "index.json").write_text(json.dumps({
            "model_species_to_ices_stocks": {"cod": ["nonexistent.stock"]},
            "units_by_stock": {"nonexistent.stock": "tonnes"},
        }))
        snap = load_snapshot(d)
        assert "nonexistent.stock" not in snap.assessments


class TestModelBiomassWindowMean:
    def test_takes_mean_over_trailing_window(self):
        # 10 years of biomass output: mean of last 5 = mean(6,7,8,9,10) = 8.0
        results = _fake_results({"cod": list(range(1, 11))})
        assert model_biomass_window_mean(results, "cod", window_years=5) == 8.0

    def test_window_clamped_to_total(self):
        # Window > total: take mean of all
        results = _fake_results({"cod": [10.0, 20.0, 30.0]})
        assert model_biomass_window_mean(results, "cod", window_years=10) == 20.0

    def test_raises_on_missing_species(self):
        results = _fake_results({"cod": [100.0]})
        with pytest.raises(ValueError, match="no biomass time series"):
            model_biomass_window_mean(results, "hake", window_years=5)


class TestCompareOutputsToIces:
    def test_in_range_species(self, tmp_path):
        # Cod ICES envelope from snapshot is [100, 140]. Model mean 120 → in range.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({
            "cod": [120.0] * 5,
            "herring": [1300.0] * 5,
            "flounder": [50.0] * 5,
            "perch": [10.0] * 5,
        })
        comps = compare_outputs_to_ices(
            results, snap, window_years=5, ices_window=range(2018, 2023)
        )
        cod = next(c for c in comps if c.species == "cod")
        assert cod.model_mean_tonnes == 120.0
        assert cod.ices_min_tonnes == 100.0
        assert cod.ices_max_tonnes == 140.0
        assert cod.in_range is True
        assert cod.magnitude_factor is not None
        # 120 / sqrt(100*140) ≈ 120 / 118.32 ≈ 1.014
        assert 1.0 < cod.magnitude_factor < 1.05

    def test_overshoot_species(self, tmp_path):
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"cod": [500.0] * 5})  # well above the [100, 140] envelope
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        cod = next(c for c in comps if c.species == "cod")
        assert cod.in_range is False
        assert cod.magnitude_factor is not None
        assert cod.magnitude_factor > 4.0  # roughly 500 / 118.3

    def test_index_only_species_has_no_envelope(self, tmp_path):
        # perch is linked only to an index-unit stock — should be excluded
        # from envelope, but model mean still recorded.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"perch": [5.0] * 5})
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        perch = next(c for c in comps if c.species == "perch")
        assert perch.model_mean_tonnes == 5.0
        assert perch.ices_min_tonnes is None
        assert perch.in_range is None
        assert "per.27.idx-only" in perch.excluded_index_stocks

    def test_no_linked_stocks_recorded_with_no_envelope(self, tmp_path):
        # flounder has empty stocks list in the manifest
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"flounder": [50.0] * 5})
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        fl = next(c for c in comps if c.species == "flounder")
        assert fl.model_mean_tonnes == 50.0
        assert fl.in_range is None
        assert fl.excluded_index_stocks == []

    def test_mixed_units_excludes_index_stocks_from_envelope(self, tmp_path):
        # herring has both tonnes-unit (her.27.20-24) and index-unit
        # (her.27.idx-only) stocks — only the tonnes-unit one feeds the envelope.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"herring": [1300.0] * 5})
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        her = next(c for c in comps if c.species == "herring")
        # Envelope from her.27.20-24 alone: [1000, 1500]
        assert her.ices_min_tonnes == 1000.0
        assert her.ices_max_tonnes == 1500.0
        assert her.in_range is True
        assert "her.27.idx-only" in her.excluded_index_stocks

    def test_missing_model_output_skips_species(self, tmp_path):
        # Snapshot includes cod, herring, flounder, perch — but results
        # only has cod. Other species should be silently skipped.
        snap = load_snapshot(_make_snapshot(tmp_path))
        results = _fake_results({"cod": [120.0] * 5})
        comps = compare_outputs_to_ices(results, snap, window_years=5)
        species_present = {c.species for c in comps}
        assert species_present == {"cod"}


class TestFormatMarkdownReport:
    def test_includes_headers_and_summary(self):
        comps = [
            SpeciesBiomassComparison(
                species="cod", model_mean_tonnes=120.0,
                ices_min_tonnes=100.0, ices_max_tonnes=140.0,
                in_range=True, magnitude_factor=1.014,
            ),
            SpeciesBiomassComparison(
                species="sprat", model_mean_tonnes=50.0,
                ices_min_tonnes=200.0, ices_max_tonnes=300.0,
                in_range=False, magnitude_factor=0.20,
            ),
            SpeciesBiomassComparison(
                species="flounder", model_mean_tonnes=10.0,  # no envelope
            ),
        ]
        report = format_markdown_report(comps, window_years=5)
        assert "# OSMOSE outputs vs ICES SSB envelope" in report
        assert "Model window: last 5 years" in report
        # cod in range
        assert "| cod | 120 | [100, 140] | ✓ | 1.01 |" in report
        # sprat out of range
        assert "| sprat | 50 | [200, 300] | ✗ | 0.20 |" in report
        # Summary count
        assert "1/2 species in ICES SSB envelope" in report
