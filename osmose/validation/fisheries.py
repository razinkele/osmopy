"""Fishing-vs-natural mortality (F/M) diagnostics for OSMOSE outputs.

Computes per-species F/M (realized fishing mortality vs natural mortality) from a
finished run — for all species, no ICES reference points. F is OSMOSE's Recruits-stage
instantaneous fishing mortality summed to annual; M = Mpred + Mstarv + Madd likewise.
F/M > 1 means fishing removes more than natural processes (an overexploitation signal).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_NATURAL_CAUSES = ("Mpred", "Mstarv", "Madd")


def annual_rate(per_step: pd.Series, steps_per_year: int, window_years: int) -> float:
    """Sum a per-saved-step rate within each year, then mean over the trailing window.

    A trailing partial year (len not a multiple of steps_per_year) is dropped so the
    window only averages complete years.
    """
    if steps_per_year < 1:
        raise ValueError(f"steps_per_year must be >= 1, got {steps_per_year}")
    vals = np.asarray(per_step, dtype=float)
    n_years = len(vals) // steps_per_year
    if n_years == 0:
        raise ValueError("mortality series shorter than one full year")
    annual = vals[: n_years * steps_per_year].reshape(n_years, steps_per_year).sum(axis=1)
    w = min(window_years, n_years)
    return float(annual[-w:].mean())


def read_mortality_recruits(path: Path) -> pd.DataFrame:
    """Read a `mortalityRate-{sp}` CSV into a (cause, stage) MultiIndex frame.

    The real file has a 1-line description preamble, a cause header row, a stage
    header row, and data rows with a trailing comma (one extra field). Skip the
    preamble, read the two header rows as a MultiIndex, drop the all-NaN trailing
    column the trailing comma produces.
    """
    df = pd.read_csv(path, skiprows=1, header=[0, 1])
    df = df.dropna(axis=1, how="all")
    return df
