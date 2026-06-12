"""Community size-spectrum diagnostics from OSMOSE *DistribBySize output.

Reads the community size-distribution file directly (the OsmoseResults *_by_size
accessors target a different, per-species layout), sums species per (Time, Size),
windows by Time, and derives community indicators: the size spectrum, its log-log
slope (reusing osmose.analysis.size_spectrum_slope), the Large-Fish Indicator, the
value-weighted mean size, and the modal (peak) bin.

This is a length-biomass (or length-abundance) spectrum over linear cm bins,
reported for trend/comparison — NOT the canonical Sheldon normalized-by-body-mass
exponent. The slope is sensitive to the small-bin cutoff; use `min_size_cm` to fit
the descending limb above the recruitment peak (see `peak_size_cm`).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from osmose.analysis import size_spectrum_slope
from osmose.results import _read_output_csv


def _read_community_by_size(output_dir: Path, output_type: str, prefix: str) -> pd.DataFrame:
    """Locate + read the {prefix}_{output_type}*.csv community file (wide Time,Size,<species>).

    rglob handles the Indicators/ subdir. Raises FileNotFoundError if absent;
    pandas.errors.EmptyDataError propagates for a 0-byte file.
    """
    matches = sorted(Path(output_dir).rglob(f"{prefix}_{output_type}*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No '{prefix}_{output_type}*.csv' under {output_dir}; the run did not persist "
            f"by-size output (enable the by-size flag and re-run)."
        )
    return _read_output_csv(matches[0])


def _community_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide Time,Size,<species> -> long time,size,value (species summed per (time,size))."""
    species_cols = [c for c in wide.columns if c not in ("Time", "Size")]
    out = pd.DataFrame(
        {
            "time": wide["Time"].astype(float),
            "size": wide["Size"].astype(float),
            "value": wide[species_cols].sum(axis=1).astype(float),
        }
    )
    return out


def _window_by_time(df: pd.DataFrame, time_col: str, window_years: int) -> pd.DataFrame:
    """Keep rows whose time is within the trailing `window_years` (by Time-years, not rows)."""
    if window_years < 1:
        raise ValueError("window_years must be >= 1")
    tmax = float(df[time_col].max())
    return cast(pd.DataFrame, df[df[time_col] > tmax - window_years])


def _infer_bin_width(edges: list[float]) -> float:
    """Median consecutive diff of sorted unique edges (handles the common equal-width case)."""
    uniq = sorted(set(edges))
    if len(uniq) < 2:
        return 1.0
    diffs = [b - a for a, b in zip(uniq[:-1], uniq[1:])]
    return float(statistics.median(diffs))


@dataclass(frozen=True)
class SizeSpectrum:
    metric: str
    bin_edges: list[float]
    bin_midpoints: list[float]
    values: list[float]
    peak_size_cm: float
    slope: float | None
    intercept: float | None
    r_squared: float | None
    n_bins_fit: int
    min_size_cm: float | None
    lfi: float
    lfi_threshold_cm: float
    mean_size_cm: float
    window_years: int
    n_timesteps_used: int
    note: str


def _large_fish_indicator(edges: list[float], values: list[float], threshold: float) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    large = sum(v for e, v in zip(edges, values) if e >= threshold)
    return float(large / total)


def _mean_size(midpoints: list[float], values: list[float]) -> float:
    total = sum(values)
    if total <= 0:
        return float("nan")
    return float(sum(m * v for m, v in zip(midpoints, values)) / total)


def _fit_slope(midpoints, values, min_size_cm):
    """Reuse analysis.size_spectrum_slope; apply min_size_cm filter; ValueError -> None.

    Note: min_size_cm is compared to bin MIDPOINTS (the `size` axis of the fit), not lower edges.
    """
    df = pd.DataFrame({"size": midpoints, "abundance": values})
    if min_size_cm is not None:
        df = cast(pd.DataFrame, df[df["size"] >= min_size_cm])
    n_fit = int(((df["size"] > 0) & (df["abundance"] > 0)).sum())
    try:
        slope, intercept, r2 = size_spectrum_slope(df)
        return slope, intercept, r2, n_fit
    except ValueError:
        return None, None, None, n_fit


def compute_size_spectrum(
    output_dir,
    *,
    metric: str = "biomass",
    prefix: str = "osm",
    window_years: int = 10,
    lfi_threshold_cm: float = 40.0,
    min_size_cm: float | None = None,
) -> SizeSpectrum:
    if metric not in ("biomass", "abundance"):
        raise ValueError("metric must be 'biomass' or 'abundance'")
    output_type = f"{metric}DistribBySize"
    wide = _read_community_by_size(Path(output_dir), output_type, prefix)
    long = _community_long(wide)

    notes: list[str] = []
    tmax = float(long["time"].max())
    tmin = float(long["time"].min())
    if tmax - tmin + 1 < window_years:
        notes.append(f"run spans < {window_years} yr; used the available {tmax - tmin + 1:.0f}.")
    windowed = _window_by_time(long, "time", window_years)
    n_steps = int(windowed["time"].nunique())

    per_bin = windowed.groupby("size")["value"].mean().sort_index()
    edges = [float(x) for x in per_bin.index]
    values = [float(x) for x in per_bin.values]
    width = _infer_bin_width(edges)
    midpoints = [e + width / 2.0 for e in edges]

    if values and max(values) > 0:
        peak = midpoints[max(range(len(values)), key=lambda i: values[i])]
    else:
        peak = float("nan")

    slope, intercept, r2, n_fit = _fit_slope(midpoints, values, min_size_cm)
    if n_fit < 2:
        notes.append("fewer than 2 positive bins in the fit window; slope undefined.")
    lfi = _large_fish_indicator(edges, values, lfi_threshold_cm)
    mean_size = _mean_size(midpoints, values)

    return SizeSpectrum(
        metric=metric,
        bin_edges=edges,
        bin_midpoints=midpoints,
        values=values,
        peak_size_cm=float(peak),
        slope=slope,
        intercept=intercept,
        r_squared=r2,
        n_bins_fit=n_fit,
        min_size_cm=min_size_cm,
        lfi=lfi,
        lfi_threshold_cm=lfi_threshold_cm,
        mean_size_cm=mean_size,
        window_years=window_years,
        n_timesteps_used=n_steps,
        note=" ".join(notes),
    )


def spectrum_plot_df(spec: SizeSpectrum) -> pd.DataFrame:
    """Build the {size, abundance} df that plotting.make_size_spectrum_plot expects."""
    return pd.DataFrame({"size": spec.bin_midpoints, "abundance": spec.values})


def size_spectrum_timeseries(
    output_dir,
    *,
    metric: str = "biomass",
    prefix: str = "osm",
    lfi_threshold_cm: float = 40.0,
    min_size_cm: float | None = None,
) -> pd.DataFrame:
    """Per-timestep community slope / LFI / mean size (for trend lines)."""
    wide = _read_community_by_size(Path(output_dir), f"{metric}DistribBySize", prefix)
    long = _community_long(wide)
    out_rows = []
    for t, g in long.groupby("time"):
        per_bin = g.groupby("size")["value"].sum().sort_index()
        edges = [float(x) for x in per_bin.index]
        values = [float(x) for x in per_bin.values]
        width = _infer_bin_width(edges)
        midpoints = [e + width / 2.0 for e in edges]
        slope, _intercept, _r2, _n = _fit_slope(midpoints, values, min_size_cm)
        out_rows.append(
            {
                "time": float(cast(float, t)),
                "slope": slope,
                "lfi": _large_fish_indicator(edges, values, lfi_threshold_cm),
                "mean_size_cm": _mean_size(midpoints, values),
            }
        )
    return pd.DataFrame(out_rows, columns=pd.Index(["time", "slope", "lfi", "mean_size_cm"]))


def format_size_spectrum_report(spec: SizeSpectrum) -> str:
    """Markdown summary of a SizeSpectrum (honest about the slope's interpretation)."""
    weighting = (
        "abundance-weighted mean length"
        if spec.metric == "abundance"
        else "biomass-weighted size centroid"
    )
    slope_txt = (
        f"{spec.slope:.3f} (intercept {spec.intercept:.3f}, R²={spec.r_squared:.3f}, "
        f"n_bins_fit={spec.n_bins_fit})"
        if spec.slope is not None
        else f"undefined (n_bins_fit={spec.n_bins_fit})"
    )
    cutoff = f"{spec.min_size_cm:.0f} cm" if spec.min_size_cm is not None else "none (all bins)"
    lines = [
        f"# OSMOSE community size spectrum — {spec.metric}",
        "",
        "A length–"
        + spec.metric
        + " spectrum over linear cm bins, reported for **trend/comparison** — "
        "not the canonical Sheldon normalized-by-body-mass exponent.",
        "",
        f"- Window: last {spec.window_years} yr ({spec.n_timesteps_used} timesteps)",
        f"- Spectrum slope: {slope_txt}",
        f"- Fit cutoff (min_size_cm): {cutoff}; peak (modal) bin midpoint: "
        f"{spec.peak_size_cm:.1f} cm",
        f"- Large-Fish Indicator (≥ {spec.lfi_threshold_cm:.0f} cm): {spec.lfi:.3f}",
        f"- Mean size ({weighting}): {spec.mean_size_cm:.2f} cm",
    ]
    if spec.note:
        lines += ["", f"_Note: {spec.note}_"]
    lines += ["", "| size (cm, midpoint) | value |", "|---|---|"]
    lines += [f"| {m:.1f} | {v:.6g} |" for m, v in zip(spec.bin_midpoints, spec.values)]
    return "\n".join(lines)
