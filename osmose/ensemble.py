"""Ensemble replicate aggregation for OSMOSE simulation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from osmose.results import OsmoseResults

# Output types suitable for ensemble aggregation (1D time series only).
ENSEMBLE_OUTPUT_TYPES = frozenset(
    {
        "biomass",
        "abundance",
        "yield",
        "mortality",
        "trophic",
        "yield_n",
        "mortality_rate",
    }
)

# Map output_type to the value column name in the resulting DataFrame.
_VALUE_COL: dict[str, str] = {
    "biomass": "biomass",
    "abundance": "abundance",
    "yield": "yield",
    "mortality": "mortality",
    "trophic": "meanTL",
    "yield_n": "yieldN",
    "mortality_rate": "mortalityRate",
}


def aggregate_replicates(
    rep_dirs: list[Path],
    output_type: str,
    species: str | None = None,
) -> dict[str, list]:
    """Aggregate replicate outputs into mean + 95% CI.

    Reads each replicate's output for the given type, aligns by inner join
    on time column, and computes mean + 2.5th/97.5th percentiles.

    Args:
        rep_dirs: Paths to replicate output directories (rep_0/, rep_1/, ...).
        output_type: One of ENSEMBLE_OUTPUT_TYPES (e.g., 'biomass').
        species: Optional species filter.

    Returns:
        {"time": [...], "mean": [...], "lower": [...], "upper": [...]}
    """
    empty: dict[str, list] = {"time": [], "mean": [], "lower": [], "upper": []}
    if not rep_dirs:
        return empty

    value_col = _VALUE_COL.get(output_type)
    if value_col is None:
        return empty

    # Collect per-replicate time series
    series_list: list[pd.DataFrame] = []
    for rep_dir in rep_dirs:
        res = OsmoseResults(rep_dir)
        df = res.export_dataframe(output_type, species=species)
        if df.empty or "time" not in df.columns:
            continue
        # Sum across species at each time step to get total
        if "species" in df.columns and value_col in df.columns:
            agg = df.groupby("time")[value_col].sum().reset_index()
        elif value_col in df.columns:
            agg = df[["time", value_col]].copy()
        else:
            # Try first numeric column that isn't time
            numeric_cols = df.select_dtypes(include="number").columns
            non_time = [c for c in numeric_cols if c != "time"]
            if not non_time:
                continue
            agg = df[["time", non_time[0]]].copy()
            agg = agg.rename(columns={non_time[0]: value_col})
        series_list.append(agg)

    if not series_list:
        return empty

    # Inner join on time (truncate to common time steps)
    common_times = set(series_list[0]["time"])
    for s in series_list[1:]:
        common_times &= set(s["time"])
    if not common_times:
        return empty

    sorted_times = sorted(common_times)

    # Build matrix: rows = time steps, cols = replicates
    matrix = np.empty((len(sorted_times), len(series_list)))
    for j, s in enumerate(series_list):
        s_indexed = s.set_index("time")
        for i, t in enumerate(sorted_times):
            matrix[i, j] = s_indexed.loc[t, value_col]

    mean = np.nanmean(matrix, axis=1)
    lower = np.nanpercentile(matrix, 2.5, axis=1)
    upper = np.nanpercentile(matrix, 97.5, axis=1)

    return {
        "time": [float(t) for t in sorted_times],
        "mean": mean.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }
