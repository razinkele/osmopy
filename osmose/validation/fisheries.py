"""Fishing-vs-natural mortality (F/M) diagnostics for OSMOSE outputs.

Computes per-species F/M (realized fishing mortality vs natural mortality) from a
finished run — for all species, no ICES reference points. F and M are OSMOSE
instantaneous mortality rates computed on the exploited life stage(s) — those
carrying fishing mortality — so that natural mortality of unfished egg/adult stages
does not swamp the ratio. F is total annual fishing mortality, M = Mpred + Mstarv +
Madd on the same fished stage(s).
F/M > 1 means fishing removes more than natural processes (an overexploitation signal).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

_NATURAL_CAUSES = ("Mpred", "Mstarv", "Madd")
_STAGES = ("Eggs", "Pre-recruits", "Recruits")
_FISHED_TOL = 1e-9


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


def annual_by_year(values, time, *, how: str) -> dict[int, float]:
    """Aggregate a per-saved-step series to one value per ABSOLUTE simulation year.

    Groups by ``int(floor(time))`` so any output.recordfrequency.ndt works. ``how="sum"``
    for accumulating quantities (F), ``how="mean"`` for stock levels (SSB).
    """
    if how not in ("sum", "mean"):
        raise ValueError(f"how must be 'sum' or 'mean', got {how!r}")
    s = pd.Series(np.asarray(values, dtype=float))
    years = np.floor(np.asarray(time, dtype=float)).astype(int)
    grouped = s.groupby(years)
    agg = cast("pd.Series", grouped.sum() if how == "sum" else grouped.mean())
    return {int(cast(int, y)): float(cast(float, v)) for y, v in agg.items()}


def read_mortality(path: Path) -> pd.DataFrame:
    """Read a `mortalityRate-{sp}` CSV into a (cause, stage) MultiIndex frame.

    The real file has a 1-line description preamble, a cause header row, a stage
    header row, and data rows with a trailing comma (one extra field). Skip the
    preamble, read the two header rows as a MultiIndex, drop the all-NaN trailing
    column the trailing comma produces.
    """
    df = pd.read_csv(path, skiprows=1, header=[0, 1])
    df = df.dropna(axis=1, how="all")
    return df


@dataclass(frozen=True)
class MortalityBalance:
    species: str
    fishing_mortality: float
    natural_mortality: float
    f_over_m: float | None
    overexploited: bool


def _mortality_path(output_dir: Path, prefix: str, species: str) -> Path:
    return Path(output_dir) / "Mortality" / f"{prefix}_mortalityRate-{species}_Simu0.csv"


def discover_species(output_dir: Path, prefix: str) -> list[str]:
    """Species names with a mortalityRate file in {output_dir}/Mortality."""
    mdir = Path(output_dir) / "Mortality"
    stem = f"{prefix}_mortalityRate-"
    out = []
    for p in sorted(mdir.glob(f"{stem}*_Simu0.csv")):
        out.append(p.name[len(stem) :].rsplit("_Simu0.csv", 1)[0])
    return out


def compute_mortality_balance(
    output_dir: Path,
    *,
    prefix: str,
    species_list: list[str] | None = None,
    steps_per_year: int,
    window_years: int = 10,
) -> list[MortalityBalance]:
    """Per-species F/M from the mortalityRate outputs. steps_per_year is REQUIRED
    (config-derived by the caller; never inferred from row counts)."""
    species = species_list if species_list is not None else discover_species(output_dir, prefix)
    out: list[MortalityBalance] = []
    for sp in species:
        path = _mortality_path(output_dir, prefix, sp)
        if not path.exists():
            print(f"WARN: no mortality file for {sp!r} at {path}", file=sys.stderr)
            continue
        try:
            df = read_mortality(path)
            # per-stage windowed annual rates
            f_by_stage = {
                s: annual_rate(cast(pd.Series, df[("F", s)]), steps_per_year, window_years)
                for s in _STAGES
                if ("F", s) in df.columns
            }
            m_by_stage = {
                s: annual_rate(
                    cast(pd.Series, sum(df[(c, s)] for c in _NATURAL_CAUSES)),
                    steps_per_year,
                    window_years,
                )
                for s in _STAGES
                if all((c, s) in df.columns for c in _NATURAL_CAUSES)
            }
        except (KeyError, ValueError, pd.errors.ParserError) as e:
            print(f"WARN: skipping {sp!r}: {e}", file=sys.stderr)
            continue
        fished = [s for s, fv in f_by_stage.items() if fv > _FISHED_TOL]
        f = sum(f_by_stage.get(s, 0.0) for s in fished)
        if fished:
            # natural mortality on the exploited stage(s)
            m = sum(m_by_stage.get(s, 0.0) for s in fished)
        else:
            # unfished: F=0; report M over the post-egg stock for a defined denominator
            m = sum(m_by_stage.get(s, 0.0) for s in ("Pre-recruits", "Recruits"))
        f_over_m = (f / m) if m > 0 else None
        out.append(
            MortalityBalance(
                species=sp,
                fishing_mortality=f,
                natural_mortality=m,
                f_over_m=f_over_m,
                overexploited=(f_over_m is not None and f_over_m > 1.0),
            )
        )
    return out


def format_mortality_report(balances: list[MortalityBalance], *, window_years: int = 10) -> str:
    """Markdown table of per-species F/M (fishing vs natural mortality)."""
    lines = [
        "# OSMOSE fishing-vs-natural mortality (F/M)",
        "",
        f"Model window: last {window_years} years. F and M are computed on the "
        "**exploited life stage(s)** — those carrying fishing mortality (so natural "
        "mortality of unfished egg/adult stages doesn't swamp the ratio). F = total "
        "annual fishing mortality; M = annual natural mortality (Mpred+Mstarv+Madd) on "
        "the same fished stage(s). F/M > 1 means fishing exceeds natural mortality for "
        "the exploited cohort.",
        "",
        "| species | F | M | F/M | overexploited |",
        "|---|---:|---:|---:|:---:|",
    ]
    n_over = 0
    for b in balances:
        fm = f"{b.f_over_m:.2f}" if b.f_over_m is not None else "—"
        over = "✓" if b.overexploited else "—"
        if b.overexploited:
            n_over += 1
        lines.append(
            f"| {b.species} | {b.fishing_mortality:.3f} | {b.natural_mortality:.3f} | {fm} | {over} |"
        )
    lines += ["", f"**Summary:** {n_over} overexploited (F/M > 1) of {len(balances)} species.", ""]
    return "\n".join(lines)
