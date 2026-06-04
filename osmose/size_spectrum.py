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
from pathlib import Path

import pandas as pd

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
    return df[df[time_col] > tmax - window_years]


def _infer_bin_width(edges: list[float]) -> float:
    """Median consecutive diff of sorted unique edges (handles the common equal-width case)."""
    uniq = sorted(set(edges))
    if len(uniq) < 2:
        return 1.0
    diffs = [b - a for a, b in zip(uniq[:-1], uniq[1:])]
    return float(statistics.median(diffs))
