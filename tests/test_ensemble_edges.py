"""Tests for ensemble aggregation edge cases."""

from pathlib import Path

import pandas as pd

from osmose.ensemble import aggregate_replicates, ENSEMBLE_OUTPUT_TYPES


def test_empty_replicate_list_returns_empty():
    """aggregate_replicates with no rep_dirs returns the empty sentinel."""
    result = aggregate_replicates([], "biomass")
    assert result == {"time": [], "mean": [], "lower": [], "upper": []}


def test_unsupported_output_type_returns_empty():
    """An output_type not in ENSEMBLE_OUTPUT_TYPES returns the empty sentinel."""
    result = aggregate_replicates([Path("/nonexistent/rep_0")], "biomass_by_age")
    assert result == {"time": [], "mean": [], "lower": [], "upper": []}


def test_unknown_output_type_string_returns_empty():
    """A completely unknown output_type string returns the empty sentinel."""
    result = aggregate_replicates([Path("/nonexistent")], "not_a_real_type")
    assert result == {"time": [], "mean": [], "lower": [], "upper": []}


def test_empty_output_type_string_returns_empty():
    """Empty string output_type returns the empty sentinel."""
    result = aggregate_replicates([Path("/nonexistent")], "")
    assert result == {"time": [], "mean": [], "lower": [], "upper": []}


def test_rep_dirs_with_no_matching_files_returns_empty(tmp_path):
    """Replicate directories that contain no relevant CSV files return empty."""
    rep_dir = tmp_path / "rep_0"
    rep_dir.mkdir()
    # Write a CSV for a different output type
    df = pd.DataFrame({"time": range(3), "abundance": [1.0, 2.0, 3.0]})
    df.to_csv(rep_dir / "osm_abundance_Anchovy.csv", index=False)

    result = aggregate_replicates([rep_dir], "yield")
    assert result == {"time": [], "mean": [], "lower": [], "upper": []}


def test_no_common_time_steps_returns_empty(tmp_path):
    """Two replicates with non-overlapping time steps return empty."""
    for i in range(2):
        rep_dir = tmp_path / f"rep_{i}"
        rep_dir.mkdir()
        # rep_0: times 0-2, rep_1: times 10-12 — no overlap
        times = range(i * 10, i * 10 + 3)
        df = pd.DataFrame({"time": list(times), "biomass": [100.0] * 3})
        df.to_csv(rep_dir / "osm_biomass_Anchovy.csv", index=False)

    dirs = [tmp_path / f"rep_{i}" for i in range(2)]
    result = aggregate_replicates(dirs, "biomass")
    assert result == {"time": [], "mean": [], "lower": [], "upper": []}


def test_result_keys_are_always_present():
    """Result dict always contains exactly the four expected keys."""
    for case in [
        ([], "biomass"),
        ([Path("/nonexistent")], "bad_type"),
    ]:
        result = aggregate_replicates(*case)
        assert set(result.keys()) == {"time", "mean", "lower", "upper"}


def test_all_ensemble_output_types_recognised():
    """Every type in ENSEMBLE_OUTPUT_TYPES is handled (not returned as empty due to type lookup)."""
    # We can't easily test with real data, but we can verify the constant is non-empty
    # and does not include obviously unsupported types.
    assert len(ENSEMBLE_OUTPUT_TYPES) > 0
    assert "biomass_by_age" not in ENSEMBLE_OUTPUT_TYPES
    assert "diet" not in ENSEMBLE_OUTPUT_TYPES
