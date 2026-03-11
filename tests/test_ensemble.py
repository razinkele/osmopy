"""Tests for osmose.ensemble — replicate aggregation."""

import numpy as np
import pandas as pd
import pytest

from osmose.ensemble import aggregate_replicates, ENSEMBLE_OUTPUT_TYPES


@pytest.fixture
def rep_dirs(tmp_path):
    """Create 3 fake replicate directories with biomass CSVs."""
    for i in range(3):
        rep_dir = tmp_path / f"rep_{i}"
        rep_dir.mkdir()
        for sp in ["Anchovy", "Sardine"]:
            df = pd.DataFrame(
                {
                    "time": range(5),
                    "biomass": np.random.rand(5) * 1000 + i * 100,
                }
            )
            df.to_csv(rep_dir / f"osm_biomass_{sp}.csv", index=False)
        # Also create abundance
        df = pd.DataFrame(
            {
                "time": range(5),
                "abundance": np.random.randint(100, 1000, 5),
            }
        )
        df.to_csv(rep_dir / "osm_abundance_Anchovy.csv", index=False)
    return [tmp_path / f"rep_{i}" for i in range(3)]


class TestAggregateReplicates:
    def test_basic_aggregation(self, rep_dirs):
        result = aggregate_replicates(rep_dirs, "biomass")
        assert "time" in result
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert len(result["time"]) == 5
        assert len(result["mean"]) == 5

    def test_mean_between_bounds(self, rep_dirs):
        result = aggregate_replicates(rep_dirs, "biomass")
        for i in range(len(result["time"])):
            assert result["lower"][i] <= result["mean"][i] <= result["upper"][i]

    def test_species_filter(self, rep_dirs):
        result = aggregate_replicates(rep_dirs, "biomass", species="Anchovy")
        assert len(result["time"]) == 5

    def test_empty_rep_dirs(self):
        result = aggregate_replicates([], "biomass")
        assert result["time"] == []
        assert result["mean"] == []

    def test_single_replicate(self, rep_dirs):
        result = aggregate_replicates(rep_dirs[:1], "biomass")
        assert len(result["time"]) == 5
        # With 1 rep, lower == mean == upper
        for i in range(5):
            assert result["lower"][i] == result["mean"][i] == result["upper"][i]

    def test_missing_output_in_some_reps(self, rep_dirs):
        # rep_0 has abundance, but delete from rep_1 and rep_2
        import os

        for i in [1, 2]:
            for f in (rep_dirs[i]).glob("osm_abundance_*.csv"):
                os.remove(f)
        # Should still work with just rep_0
        result = aggregate_replicates(rep_dirs, "abundance")
        assert len(result["time"]) > 0

    def test_different_time_lengths_uses_inner_join(self, tmp_path):
        """Replicates with different time ranges use inner join (shortest)."""
        for i in range(2):
            rep_dir = tmp_path / f"rep_{i}"
            rep_dir.mkdir()
            n_steps = 5 if i == 0 else 3  # rep_0 has 5, rep_1 has 3
            df = pd.DataFrame(
                {
                    "time": range(n_steps),
                    "biomass": [100.0] * n_steps,
                }
            )
            df.to_csv(rep_dir / "osm_biomass_Anchovy.csv", index=False)
        dirs = [tmp_path / f"rep_{i}" for i in range(2)]
        result = aggregate_replicates(dirs, "biomass")
        # Inner join: only time steps 0,1,2 are common
        assert len(result["time"]) == 3


class TestEnsembleOutputTypes:
    def test_1d_types_listed(self):
        assert "biomass" in ENSEMBLE_OUTPUT_TYPES
        assert "abundance" in ENSEMBLE_OUTPUT_TYPES
        assert "yield" in ENSEMBLE_OUTPUT_TYPES
        assert "trophic" in ENSEMBLE_OUTPUT_TYPES

    def test_2d_types_not_listed(self):
        assert "biomass_by_age" not in ENSEMBLE_OUTPUT_TYPES
        assert "diet" not in ENSEMBLE_OUTPUT_TYPES
