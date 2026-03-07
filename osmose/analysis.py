"""Ensemble statistics and ecological indicators for OSMOSE outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def ensemble_stats(
    replicate_dfs: list[pd.DataFrame],
    value_col: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute mean, std, 95% CI across replicate DataFrames.

    Args:
        replicate_dfs: List of DataFrames, one per replicate.
        value_col: Column name holding the numeric values.
        group_cols: Columns to group by. Defaults to ``["time"]``.

    Returns:
        DataFrame with columns: ``<group_cols>``, mean, std, ci_lower, ci_upper.
    """
    if not replicate_dfs:
        return pd.DataFrame()

    if group_cols is None:
        group_cols = ["time"]

    combined = pd.concat(replicate_dfs, ignore_index=True)
    grouped = combined.groupby(group_cols, sort=True)[value_col]

    result = grouped.agg(["mean", "std"]).reset_index()
    result["std"] = result["std"].fillna(0.0)

    # 95% CI: mean +/- 1.96 * std / sqrt(n)
    n = len(replicate_dfs)
    se = result["std"] / np.sqrt(n)
    result["ci_lower"] = result["mean"] - 1.96 * se
    result["ci_upper"] = result["mean"] + 1.96 * se

    return result


def summary_table(
    replicate_dfs: list[pd.DataFrame],
    value_col: str,
) -> pd.DataFrame:
    """Per-species summary statistics across replicates.

    Args:
        replicate_dfs: List of DataFrames containing species and value columns.
        value_col: Column name holding the numeric values.

    Returns:
        DataFrame with columns: species, mean, std, min, max, median.
    """
    if not replicate_dfs:
        return pd.DataFrame()

    combined = pd.concat(replicate_dfs, ignore_index=True)
    result = (
        combined.groupby("species")[value_col]
        .agg(["mean", "std", "min", "max", "median"])
        .reset_index()
    )
    return result


def shannon_diversity(biomass_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Shannon-Wiener diversity index per timestep.

    H = -sum(p_i * ln(p_i)) where p_i is the proportion of species i.

    Args:
        biomass_df: DataFrame with columns: time, species, biomass.

    Returns:
        DataFrame with columns: time, shannon.
    """

    def _shannon(group: pd.DataFrame) -> float:
        biomass: NDArray[np.floating] = group["biomass"].values.astype(float)
        biomass = biomass[biomass > 0]
        total = biomass.sum()
        if total == 0:
            return 0.0
        p = biomass / total
        return float(-np.sum(p * np.log(p)))

    result = biomass_df.groupby("time").apply(_shannon, include_groups=False).reset_index()
    result.columns = ["time", "shannon"]
    return result


def mean_tl_catch(
    yield_df: pd.DataFrame,
    tl_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute weighted mean trophic level of catch per timestep.

    Args:
        yield_df: DataFrame with columns: time, species, yield.
        tl_df: DataFrame with columns: species, tl.

    Returns:
        DataFrame with columns: time, mean_tl.
    """
    merged = yield_df.merge(tl_df, on="species", how="left")

    def _weighted_tl(group: pd.DataFrame) -> float:
        total_yield = group["yield"].sum()
        if total_yield == 0:
            return float("nan")
        return float((group["yield"] * group["tl"]).sum() / total_yield)

    result = merged.groupby("time").apply(_weighted_tl, include_groups=False).reset_index()
    result.columns = ["time", "mean_tl"]
    return result


def size_spectrum_slope(
    spectrum_df: pd.DataFrame,
) -> tuple[float, float, float]:
    """Compute log-log regression slope of size spectrum.

    Args:
        spectrum_df: DataFrame with columns: size, abundance.

    Returns:
        Tuple of (slope, intercept, r_squared).
    """
    log_size = np.log(spectrum_df["size"].values.astype(float))
    log_abundance = np.log(spectrum_df["abundance"].values.astype(float))

    # Linear regression: log(abundance) = slope * log(size) + intercept
    coeffs = np.polyfit(log_size, log_abundance, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # R-squared
    predicted = slope * log_size + intercept
    ss_res = np.sum((log_abundance - predicted) ** 2)
    ss_tot = np.sum((log_abundance - np.mean(log_abundance)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return slope, intercept, r_squared
