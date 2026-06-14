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
    """Generic RMSE for aligned time series with an optional species filter.

    When both frames contain a `species` column, rows are aligned on
    (time, species); otherwise on `time` alone. Asymmetric presence of the
    species column raises ValueError.
    """
    if species:
        simulated = simulated[simulated["species"] == species]  # type: ignore[assignment]
        observed = observed[observed["species"] == species]  # type: ignore[assignment]

    sim_has_species = "species" in simulated.columns
    obs_has_species = "species" in observed.columns
    if sim_has_species != obs_has_species:
        raise ValueError(
            "species column must be present in both simulated and observed, or in neither"
        )
    merge_cols = ["time", "species"] if sim_has_species else ["time"]
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


def _biomass_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the engine's WIDE biomass frame (a time column + one numeric column per species)
    to long ``[time, species, biomass]`` so ``biomass_rmse`` can merge it. Idempotent if the frame
    is already long. (OsmoseResults.biomass() returns wide — verify the exact columns against
    data/minimal during implementation; melt all numeric per-species columns, lowercase Time→time,
    drop any pre-existing non-value 'species' column.)
    """
    if "biomass" in df.columns and "time" in df.columns:
        return df
    time_col = "time" if "time" in df.columns else "Time"
    value_cols = [
        c
        for c in df.columns
        if c not in (time_col, "species") and pd.api.types.is_numeric_dtype(df[c])
    ]
    long = df.melt(
        id_vars=[time_col], value_vars=value_cols, var_name="species", value_name="biomass"
    )
    return long.rename(columns={time_col: "time"})


class BiomassRMSEObjective:
    """Picklable biomass-RMSE objective (wraps biomass_rmse; reshapes wide->long).

    Module-level (not a lambda) so it can cross a ProcessPoolExecutor boundary. The existing UI
    lambda fed the wide frame straight in and KeyError'd on real engine output — this fixes it.
    """

    def __init__(self, observed: pd.DataFrame, species: str | None = None):
        # Reshape the OBSERVED frame too (idempotent on already-long input) so a wide user CSV
        # doesn't merge-empty -> inf -> >50% abort. Both sides go through _biomass_long.
        self.observed = _biomass_long(observed) if observed is not None else observed
        self.species = species

    def __call__(self, results) -> float:
        return biomass_rmse(_biomass_long(results.biomass()), self.observed, self.species)


class DietDistanceObjective:
    """Picklable diet-distance objective (wraps `diet_distance`; holds the observed matrix)."""

    def __init__(self, observed: pd.DataFrame):
        self.observed = observed

    def __call__(self, results) -> float:
        return diet_distance(results.diet_matrix(), self.observed)
