"""Community-level ecosystem-state diagnostics from OSMOSE output.

Adds the canonical Sheldon (body-mass) normalized biomass spectrum (NBSS) and a
suite of community indicators — Mean Trophic Level / Marine Trophic Index,
community totals + size diversity, and the Warwick Abundance-Biomass Comparison
(ABC) W-statistic — on top of the length spectrum in osmose.size_spectrum.

The Sheldon spectrum needs body MASS; OSMOSE writes by-LENGTH size classes, so we
convert per species via the config length-weight law W = a * L^b. Each unit fails
soft (records a note, returns degraded values) rather than raising past the
orchestrator. See docs/superpowers/specs/2026-06-17-community-size-spectrum-extension-design.md.
"""

from __future__ import annotations

import pandas as pd

from osmose.size_spectrum import _window_by_time

_META_COLS = {"Time", "time", "Size", "size", "species", "Simu", "simu"}


def _to_float(value) -> float | None:
    """float(value) or None if value is None / non-numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _species_columns(df: pd.DataFrame) -> list[str]:
    """Column names that are species (exclude Time/Size/species/Simu meta columns)."""
    return [c for c in df.columns if c not in _META_COLS]


def _per_species_window_mean(df: pd.DataFrame, window_years: int) -> dict[str, float]:
    """{species: trailing-window mean value} from a WIDE 1D OsmoseResults frame.

    `df` has a Time column + one column per species (+ a 'species' meta column).
    Empty/Time-less frame -> {}. Non-numeric cells coerce to NaN before the mean.
    """
    if df.empty or "Time" not in df.columns:
        return {}
    windowed = _window_by_time(df, "Time", window_years)
    out: dict[str, float] = {}
    for c in _species_columns(windowed):
        out[c] = float(pd.to_numeric(windowed[c], errors="coerce").mean())
    return out


def _species_lw_coeffs(config: dict) -> dict[str, tuple[float, float]]:
    """{species_name: (a, b)} from config length-weight keys.

    Skips a species whose name is missing or whose a/b is missing or non-positive
    (a non-positive coefficient can't define a usable W = a * L^b mapping).
    """
    out: dict[str, tuple[float, float]] = {}
    n = _to_float(config.get("simulation.nspecies"))
    if n is None:
        return out
    for i in range(int(n)):
        name = config.get(f"species.name.sp{i}")
        a = _to_float(config.get(f"species.length2weight.condition.factor.sp{i}"))
        b = _to_float(config.get(f"species.length2weight.allometric.power.sp{i}"))
        if name and a is not None and b is not None and a > 0 and b > 0:
            out[str(name)] = (a, b)
    return out
