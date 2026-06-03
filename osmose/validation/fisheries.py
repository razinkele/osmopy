"""Fishing-vs-natural mortality (F/M) diagnostics for OSMOSE outputs.

Computes per-species F/M (realized fishing mortality vs natural mortality) from a
finished run — for all species, no ICES reference points. F is OSMOSE's Recruits-stage
instantaneous fishing mortality summed to annual; M = Mpred + Mstarv + Madd likewise.
F/M > 1 means fishing removes more than natural processes (an overexploitation signal).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_NATURAL_CAUSES = ("Mpred", "Mstarv", "Madd")


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
