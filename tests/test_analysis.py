"""Tests for osmose.analysis — ensemble statistics and ecological indicators."""

import numpy as np
import pandas as pd
import pytest

from osmose.analysis import (
    ensemble_stats,
    mean_tl_catch,
    shannon_diversity,
    size_spectrum_slope,
    summary_table,
)


# ---------------------------------------------------------------------------
# ensemble_stats
# ---------------------------------------------------------------------------


class TestEnsembleStats:
    def test_basic_mean_std(self):
        df1 = pd.DataFrame({"time": [1, 2], "biomass": [10.0, 20.0]})
        df2 = pd.DataFrame({"time": [1, 2], "biomass": [12.0, 22.0]})
        result = ensemble_stats([df1, df2], value_col="biomass")
        assert list(result.columns) == ["time", "mean", "std", "ci_lower", "ci_upper"]
        assert len(result) == 2
        # time=1: mean=11, time=2: mean=21
        assert result.loc[result["time"] == 1, "mean"].iloc[0] == pytest.approx(11.0)
        assert result.loc[result["time"] == 2, "mean"].iloc[0] == pytest.approx(21.0)

    def test_ci_bounds(self):
        dfs = [pd.DataFrame({"time": [1], "biomass": [v]}) for v in [10.0, 12.0, 14.0, 16.0, 18.0]]
        result = ensemble_stats(dfs, value_col="biomass")
        row = result.iloc[0]
        assert row["ci_lower"] < row["mean"] < row["ci_upper"]

    def test_custom_group_cols(self):
        df1 = pd.DataFrame({"time": [1, 1], "species": ["A", "B"], "biomass": [10.0, 20.0]})
        df2 = pd.DataFrame({"time": [1, 1], "species": ["A", "B"], "biomass": [12.0, 22.0]})
        result = ensemble_stats([df1, df2], value_col="biomass", group_cols=["time", "species"])
        assert "species" in result.columns
        assert len(result) == 2

    def test_empty_list(self):
        result = ensemble_stats([], value_col="biomass")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_replicate(self):
        df = pd.DataFrame({"time": [1, 2], "biomass": [10.0, 20.0]})
        result = ensemble_stats([df], value_col="biomass")
        assert len(result) == 2
        assert result.loc[result["time"] == 1, "mean"].iloc[0] == pytest.approx(10.0)
        # std should be 0 (or NaN depending on ddof) for single replicate
        assert result.loc[result["time"] == 1, "std"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# summary_table
# ---------------------------------------------------------------------------


class TestSummaryTable:
    def test_basic(self):
        df1 = pd.DataFrame(
            {"time": [1, 2, 1, 2], "species": ["A", "A", "B", "B"], "val": [10, 20, 30, 40]}
        )
        df2 = pd.DataFrame(
            {"time": [1, 2, 1, 2], "species": ["A", "A", "B", "B"], "val": [12, 22, 32, 42]}
        )
        result = summary_table([df1, df2], value_col="val")
        assert list(result.columns) == ["species", "mean", "std", "min", "max", "median"]
        assert len(result) == 2
        a_row = result.loc[result["species"] == "A"].iloc[0]
        assert a_row["min"] == 10
        assert a_row["max"] == 22

    def test_empty_list(self):
        result = summary_table([], value_col="val")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# shannon_diversity
# ---------------------------------------------------------------------------


class TestShannonDiversity:
    def test_equal_biomass(self):
        """Equal biomass across species → max diversity for that number of species."""
        df = pd.DataFrame(
            {
                "time": [1, 1, 1],
                "species": ["A", "B", "C"],
                "biomass": [100.0, 100.0, 100.0],
            }
        )
        result = shannon_diversity(df)
        assert list(result.columns) == ["time", "shannon"]
        expected = -3 * (1 / 3) * np.log(1 / 3)
        assert result.iloc[0]["shannon"] == pytest.approx(expected, rel=1e-6)

    def test_single_species(self):
        """Single species → H = 0."""
        df = pd.DataFrame({"time": [1], "species": ["A"], "biomass": [100.0]})
        result = shannon_diversity(df)
        assert result.iloc[0]["shannon"] == pytest.approx(0.0)

    def test_multiple_timesteps(self):
        df = pd.DataFrame(
            {
                "time": [1, 1, 2, 2],
                "species": ["A", "B", "A", "B"],
                "biomass": [50.0, 50.0, 90.0, 10.0],
            }
        )
        result = shannon_diversity(df)
        assert len(result) == 2
        # time=1 is more diverse than time=2
        h1 = result.loc[result["time"] == 1, "shannon"].iloc[0]
        h2 = result.loc[result["time"] == 2, "shannon"].iloc[0]
        assert h1 > h2

    def test_zero_biomass_ignored(self):
        """Species with zero biomass should not cause errors."""
        df = pd.DataFrame(
            {
                "time": [1, 1, 1],
                "species": ["A", "B", "C"],
                "biomass": [100.0, 0.0, 0.0],
            }
        )
        result = shannon_diversity(df)
        assert result.iloc[0]["shannon"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mean_tl_catch
# ---------------------------------------------------------------------------


class TestMeanTLCatch:
    def test_basic(self):
        yield_df = pd.DataFrame(
            {
                "time": [1, 1, 2, 2],
                "species": ["A", "B", "A", "B"],
                "yield": [100.0, 200.0, 300.0, 100.0],
            }
        )
        tl_df = pd.DataFrame({"species": ["A", "B"], "tl": [3.0, 4.0]})
        result = mean_tl_catch(yield_df, tl_df)
        assert list(result.columns) == ["time", "mean_tl"]
        # time=1: (100*3 + 200*4) / 300 = 1100/300 = 3.667
        assert result.loc[result["time"] == 1, "mean_tl"].iloc[0] == pytest.approx(
            1100 / 300, rel=1e-4
        )
        # time=2: (300*3 + 100*4) / 400 = 1300/400 = 3.25
        assert result.loc[result["time"] == 2, "mean_tl"].iloc[0] == pytest.approx(
            1300 / 400, rel=1e-4
        )

    def test_zero_catch(self):
        """Zero total catch at a timestep should handle gracefully."""
        yield_df = pd.DataFrame({"time": [1, 1], "species": ["A", "B"], "yield": [0.0, 0.0]})
        tl_df = pd.DataFrame({"species": ["A", "B"], "tl": [3.0, 4.0]})
        result = mean_tl_catch(yield_df, tl_df)
        assert len(result) == 1
        # Should be NaN or 0 when no catch
        assert np.isnan(result.iloc[0]["mean_tl"]) or result.iloc[0]["mean_tl"] == 0.0


# ---------------------------------------------------------------------------
# size_spectrum_slope
# ---------------------------------------------------------------------------


class TestSizeSpectrumSlope:
    def test_perfect_linear(self):
        """Perfect power-law data should give exact slope and R^2 = 1."""
        slope_true = -1.5
        intercept_true = 10.0
        sizes = np.array([1, 2, 4, 8, 16, 32], dtype=float)
        abundances = 10**intercept_true * sizes**slope_true
        df = pd.DataFrame({"size": sizes, "abundance": abundances})
        slope, intercept, r_sq = size_spectrum_slope(df)
        assert slope == pytest.approx(slope_true, rel=1e-6)
        assert intercept == pytest.approx(intercept_true, rel=1e-6)
        assert r_sq == pytest.approx(1.0, abs=1e-10)

    def test_returns_tuple(self):
        df = pd.DataFrame({"size": [1, 2, 4], "abundance": [100, 50, 25]})
        result = size_spectrum_slope(df)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_negative_slope(self):
        """Typical size spectrum has negative slope."""
        df = pd.DataFrame({"size": [1, 10, 100], "abundance": [1000, 100, 10]})
        slope, _, _ = size_spectrum_slope(df)
        assert slope < 0

    def test_size_spectrum_slope_uses_log10(self):
        """Slope should be computed with log10 to match plotting convention."""
        # Known power law: abundance = 1000 * size^-2
        sizes = np.array([1, 10, 100, 1000], dtype=float)
        abundances = 1000 * sizes**-2.0

        df = pd.DataFrame({"size": sizes, "abundance": abundances})
        slope, intercept, r2 = size_spectrum_slope(df)

        # With log10: slope should be exactly -2.0
        assert abs(slope - (-2.0)) < 0.01
        assert r2 > 0.99
