"""Ensemble statistics and ecological indicators for OSMOSE outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _require_columns(df: pd.DataFrame, *cols: str, context: str = "") -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {sorted(missing)}, got {sorted(df.columns)}")


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
    _require_columns(combined, *group_cols, value_col, context="ensemble_stats")
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
    _require_columns(combined, "species", value_col, context="summary_table")
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
    _require_columns(biomass_df, "time", "biomass", context="shannon_diversity")

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
    _require_columns(merged, "time", "species", "yield", "tl", context="mean_tl_catch")

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
    _require_columns(spectrum_df, "size", "abundance", context="size_spectrum_slope")
    positive = spectrum_df[(spectrum_df["size"] > 0) & (spectrum_df["abundance"] > 0)]
    if len(positive) < 2:
        raise ValueError("Need at least 2 positive size/abundance pairs for regression.")

    log_size = np.log10(np.asarray(positive["size"], dtype=float))
    log_abundance = np.log10(np.asarray(positive["abundance"], dtype=float))

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


_METRIC_ACCESSOR = {"biomass": "biomass", "yield": "yield_biomass", "abundance": "abundance"}
_NON_SPECIES_COLS = {"Time", "time", "species"}


def _trailing_window(df, time_col: str, window_years: int):
    """Rows whose time is within the last `window_years` (time is in YEARS).

    Filters by the time COLUMN, not by row count — so the window is correct
    regardless of how many rows-per-year the output was saved at (a row-count
    `tail` silently takes the last N *rows*, which is wrong for sub-annual output).
    """
    tmax = float(df[time_col].max())
    return df[df[time_col] > tmax - window_years]


def _per_species_window_mean(results, metric: str, window_years: int) -> dict[str, float]:
    """Per-species mean of `metric` over the trailing `window_years` of a run.

    Handles both output shapes:
    - WIDE (the disk default): `Time` + one column per species + a constant `species`
      column. Per-species values are the columns; mean each over the trailing window.
    - LONG: `time, species, value` with a real per-species `species` column; group by
      species and mean `value` over the trailing window per species.

    The window is selected by the time column (years), NOT by row count, so it is
    correct for sub-annual output cadences (recordfrequency.ndt < ndtPerYear).
    """
    if metric not in _METRIC_ACCESSOR:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(_METRIC_ACCESSOR)}")
    if window_years < 1:
        # window_years <= 0 empties the time filter → NaN means → corrupt sort + invalid JSON.
        raise ValueError(f"window_years must be >= 1, got {window_years}")
    df = getattr(results, _METRIC_ACCESSOR[metric])()
    if df is None or len(df) == 0:
        return {}
    cols = set(df.columns)
    # LONG iff a value column + a species column are present. (The WIDE global frame
    # has a `species` column too — but no `value` column — so this discriminates
    # correctly even for a single-species long frame, where a row-count heuristic fails.)
    is_long = "value" in cols and "species" in cols
    if is_long:
        time_col = "time" if "time" in cols else "Time"
        out: dict[str, float] = {}
        for sp, g in df.groupby("species"):
            win = _trailing_window(g.sort_values(time_col), time_col, window_years)
            out[str(sp)] = float(win["value"].mean())
        return out
    # WIDE: species are the non-Time/non-species columns
    time_col = "Time" if "Time" in cols else "time"
    species_cols = [c for c in df.columns if c not in _NON_SPECIES_COLS]
    win = _trailing_window(df, time_col, window_years)
    return {str(c): float(win[c].mean()) for c in species_cols}


@dataclass(frozen=True)
class SpeciesDelta:
    species: str
    baseline_mean: float
    variant_mean: float
    abs_delta: float
    pct_delta: float | None  # None when baseline_mean == 0
    from_zero: bool  # baseline_mean == 0 and variant_mean > 0


def run_delta(
    baseline,
    variant,
    *,
    metric: str = "biomass",
    window_years: int = 10,
    top_n: int | None = None,
) -> list[SpeciesDelta]:
    """Per-species delta of `metric` (windowed mean) between two runs, ranked by |% change|.

    Species set = union of both runs (a species absent from one contributes mean 0 there).
    pct_delta is None for a zero-baseline species (reported via from_zero + abs_delta).
    Sorted by |pct_delta| desc (from-zero species sort to the top); ties by |abs_delta| desc.
    """
    bmeans = _per_species_window_mean(baseline, metric, window_years)
    vmeans = _per_species_window_mean(variant, metric, window_years)
    species = sorted(set(bmeans) | set(vmeans))
    deltas: list[SpeciesDelta] = []
    for sp in species:
        b = bmeans.get(sp, 0.0)
        v = vmeans.get(sp, 0.0)
        abs_d = v - b
        pct = (abs_d / b) if b != 0 else None
        deltas.append(
            SpeciesDelta(
                species=sp,
                baseline_mean=b,
                variant_mean=v,
                abs_delta=abs_d,
                pct_delta=pct,
                from_zero=(b == 0.0 and v > 0.0),
            )
        )

    def _key(d: SpeciesDelta):
        # Genuine from-zero recoveries rank at top (inf). A 0->0 "dead" species also has
        # pct_delta None but is NOT a mover — it must rank LAST (0.0), not at the top.
        # Finite pct ranks by |pct|; ties by |abs|.
        if d.pct_delta is None:
            primary = float("inf") if d.from_zero else 0.0
        else:
            primary = abs(d.pct_delta)
        return (primary, abs(d.abs_delta))

    deltas.sort(key=_key, reverse=True)
    return deltas[:top_n] if top_n is not None else deltas


def format_delta_report(
    deltas: list[SpeciesDelta], *, metric: str = "biomass", window_years: int = 10
) -> str:
    """Markdown table of per-species deltas, ranked (as returned by run_delta)."""
    lines = [
        f"# OSMOSE run delta — {metric} (variant vs baseline)",
        "",
        f"Per-species mean {metric} over the last {window_years} years; ranked by |% change|. "
        "Δ% is undefined for a zero-baseline species (shown 'from 0').",
        "",
        "| species | baseline | variant | Δ | Δ% |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in deltas:
        if d.pct_delta is None:
            pct = "— (from 0)" if d.from_zero else "—"
        else:
            pct = f"{d.pct_delta * 100:+.1f}%"
        lines.append(
            f"| {d.species} | {d.baseline_mean:.3g} | {d.variant_mean:.3g} | "
            f"{d.abs_delta:+.3g} | {pct} |"
        )
    n_moved = sum(1 for d in deltas if d.abs_delta != 0.0)
    lines += ["", f"**Summary:** {n_moved} of {len(deltas)} species changed.", ""]
    return "\n".join(lines)
