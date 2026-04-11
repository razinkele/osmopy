# osmose/calibration/objectives.py
"""Objective functions for OSMOSE calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _timeseries_rmse(
    simulated: pd.DataFrame,
    observed: pd.DataFrame,
    value_col: str,
    species: str | None = None,
) -> float:
    """Generic RMSE for aligned time series with an optional species filter."""
    if species:
        simulated = simulated[simulated["species"] == species]  # type: ignore[assignment]
        observed = observed[observed["species"] == species]  # type: ignore[assignment]

    merge_cols = ["time"]
    if "species" in simulated.columns and "species" in observed.columns:
        merge_cols.append("species")
    merged = pd.merge(simulated, observed, on=merge_cols, suffixes=("_sim", "_obs"))
    if merged.empty:
        return float("inf")

    diff = merged[f"{value_col}_sim"] - merged[f"{value_col}_obs"]
    return float(np.sqrt(np.mean(diff**2)))


def biomass_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame, species: str | None = None
) -> float:
    """Root mean square error of biomass time series."""
    return _timeseries_rmse(simulated, observed, "biomass", species)


def abundance_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame, species: str | None = None
) -> float:
    """RMSE for abundance time series."""
    return _timeseries_rmse(simulated, observed, "abundance", species)


def diet_distance(simulated: pd.DataFrame, observed: pd.DataFrame) -> float:
    """Frobenius norm distance between diet composition matrices.

    Both DataFrames should be square matrices with predator rows and prey columns.
    """
    sim_vals = simulated.select_dtypes(include=[np.number]).values
    obs_vals = observed.select_dtypes(include=[np.number]).values

    if sim_vals.shape != obs_vals.shape:
        return float("inf")

    return float(np.linalg.norm(sim_vals - obs_vals, "fro"))


def yield_rmse(
    simulated: pd.DataFrame, observed: pd.DataFrame, species: str | None = None
) -> float:
    """RMSE for yield time series."""
    return _timeseries_rmse(simulated, observed, "yield", species)


def _binned_rmse(simulated: pd.DataFrame, observed: pd.DataFrame) -> float:
    """RMSE for 2D binned outputs (catch-at-size, size-at-age)."""
    merged = pd.merge(simulated, observed, on=["time", "bin"], suffixes=("_sim", "_obs"))
    if merged.empty:
        return float("inf")
    diff = merged["value_sim"] - merged["value_obs"]
    return float(np.sqrt(np.mean(diff**2)))


def catch_at_size_distance(simulated: pd.DataFrame, observed: pd.DataFrame) -> float:
    """RMSE between 2D catch-at-size outputs."""
    return _binned_rmse(simulated, observed)


def size_at_age_rmse(simulated: pd.DataFrame, observed: pd.DataFrame) -> float:
    """RMSE between 2D size-at-age outputs."""
    return _binned_rmse(simulated, observed)


def weighted_multi_objective(objectives: list[float], weights: list[float]) -> float:
    """Weighted dot product of objective values."""
    return float(np.dot(objectives, weights))


def normalized_rmse(simulated: np.ndarray, observed: np.ndarray) -> float:
    """RMSE normalized by the mean of observed values."""
    obs_mean = np.mean(observed)
    if obs_mean == 0:
        return float("inf")
    rmse = float(np.sqrt(np.mean((simulated - observed) ** 2)))
    return float(rmse / obs_mean)
