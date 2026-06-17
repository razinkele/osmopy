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

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from osmose.analysis import size_spectrum_slope
from osmose.results import OsmoseResults
from osmose.size_spectrum import _infer_bin_width, _read_community_by_size, _window_by_time

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
        out[c] = float(cast(pd.Series, pd.to_numeric(windowed[c], errors="coerce")).mean())
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


@dataclass(frozen=True)
class SheldonSpectrum:
    metric: str
    mass_bin_edges: list[float]
    mass_bin_midpoints: list[float]
    nbss_values: list[float]
    slope: float | None
    intercept: float | None
    r_squared: float | None
    n_bins_fit: int
    size_diversity: float
    total_biomass: float
    total_abundance: float
    mean_body_mass: float
    window_years: int
    n_timesteps_used: int
    dropped_species: list[str]
    note: str


def _shannon_evenness(values: list[float]) -> float:
    """Shannon evenness H/ln(S) over positive shares; NaN if fewer than 2 positive bins."""
    positive = [v for v in values if v > 0]
    if len(positive) < 2:
        return float("nan")
    total = sum(positive)
    shares = [v / total for v in positive]
    h = -sum(p * math.log(p) for p in shares)
    return float(h / math.log(len(positive)))


def compute_sheldon_spectrum(
    output_dir,
    config: dict,
    *,
    metric: str = "biomass",
    prefix: str = "osm",
    window_years: int = 10,
) -> SheldonSpectrum:
    """Canonical Sheldon NBSS over equal log2 (octave) body-mass bins + derived metrics.

    Reads the per-species {metric}DistribBySize file (does NOT sum species), converts
    each species' length midpoints to mass via config W = a*L^b, bins biomass into
    octaves, normalizes by bin width, and log-log fits the slope. Totals come from the
    1D biomass/abundance outputs (read non-strict; absent -> 0/NaN). Raises
    FileNotFoundError if the by-size file is missing; ValueError on a bad metric.
    """
    if metric not in ("biomass", "abundance"):
        raise ValueError("metric must be 'biomass' or 'abundance'")
    coeffs = _species_lw_coeffs(config or {})
    wide = _read_community_by_size(Path(output_dir), f"{metric}DistribBySize", prefix)
    windowed = _window_by_time(wide, "Time", window_years)
    n_steps = int(cast(pd.Series, windowed["Time"]).nunique())

    sp_cols = _species_columns(windowed)
    sizes = sorted({float(s) for s in windowed["Size"].unique()})
    width_len = _infer_bin_width(sizes)
    per_size = cast(pd.DataFrame, windowed.groupby("Size")[sp_cols].mean())

    notes: list[str] = []
    dropped: list[str] = []
    masses: list[float] = []
    vals: list[float] = []
    for sp in sp_cols:
        if sp not in coeffs:
            dropped.append(sp)
            continue
        a, b = coeffs[sp]
        sp_series = cast(pd.Series, per_size[sp])
        for size_lo, value in zip(sp_series.index.tolist(), sp_series.tolist()):
            mid_len = float(size_lo) + width_len / 2.0
            mass = a * mid_len**b
            v = float(value)
            if mass > 0 and v > 0:
                masses.append(mass)
                vals.append(v)
    if dropped:
        notes.append(f"dropped {len(dropped)} species without usable a,b: {', '.join(dropped)}.")

    edges: list[float] = []
    midpoints: list[float] = []
    nbss: list[float] = []
    bin_biomass: list[
        float
    ] = []  # raw per-octave biomass (for size diversity, NOT width-normalized)
    if masses:
        w_ref = min(masses)
        binned: dict[int, float] = defaultdict(float)
        for m, v in zip(masses, vals):
            k = int(math.floor(math.log2(m / w_ref)))
            binned[k] += v
        for k in sorted(binned):
            lower = w_ref * 2.0**k
            edges.append(lower)
            midpoints.append(w_ref * 2.0 ** (k + 0.5))
            nbss.append(binned[k] / lower)  # octave linear width == lower edge value
            bin_biomass.append(binned[k])
    else:
        notes.append("no positive mass bins (no species with usable a,b and data).")

    slope = intercept = r2 = None
    n_fit = sum(1 for m, v in zip(midpoints, nbss) if m > 0 and v > 0)
    if n_fit >= 2:
        try:
            slope, intercept, r2 = size_spectrum_slope(
                pd.DataFrame({"size": midpoints, "abundance": nbss})
            )
        except ValueError:
            notes.append("NBSS slope fit failed.")
    else:
        notes.append("fewer than 2 positive mass bins; NBSS slope undefined.")

    # Size diversity is evenness over RAW per-octave biomass shares (spec §Unit 1.5), NOT the
    # width-normalized NBSS density — the latter would depend on the arbitrary w_ref alignment.
    size_diversity = _shannon_evenness(bin_biomass)

    res = OsmoseResults(Path(output_dir), prefix=prefix, strict=False)
    bm = _per_species_window_mean(res.biomass(), window_years)
    ab = _per_species_window_mean(res.abundance(), window_years)
    total_biomass = float(sum(bm.values()))
    total_abundance = float(sum(ab.values()))
    mean_body_mass = total_biomass / total_abundance if total_abundance > 0 else float("nan")

    return SheldonSpectrum(
        metric=metric,
        mass_bin_edges=edges,
        mass_bin_midpoints=midpoints,
        nbss_values=nbss,
        slope=slope,
        intercept=intercept,
        r_squared=r2,
        n_bins_fit=n_fit,
        size_diversity=size_diversity,
        total_biomass=total_biomass,
        total_abundance=total_abundance,
        mean_body_mass=mean_body_mass,
        window_years=window_years,
        n_timesteps_used=n_steps,
        dropped_species=dropped,
        note=" ".join(notes),
    )


@dataclass(frozen=True)
class TrophicIndicators:
    mtl: float
    mti: float
    mti_tl_cutoff: float
    n_species: int
    n_species_above_cutoff: int
    window_years: int
    note: str


def compute_trophic_indicators(
    output_dir,
    *,
    prefix: str = "osm",
    window_years: int = 10,
    mti_tl_cutoff: float = 3.25,
) -> TrophicIndicators:
    """Biomass-weighted community Mean Trophic Level + Marine Trophic Index.

    MTL = biomass-weighted mean of per-species mean TL. MTI = same but only over
    species with mean TL >= mti_tl_cutoff (Pauly & Watson; default 3.25), a
    biomass-weighted standing-stock analogue of the catch-based index. meanTL is
    required (strict read raises FileNotFoundError if absent); biomass is the weight
    (read non-strict; if absent, equal weights with a note).
    """
    res = OsmoseResults(Path(output_dir), prefix=prefix, strict=True)
    tl = _per_species_window_mean(res.mean_trophic_level(), window_years)
    res_soft = OsmoseResults(Path(output_dir), prefix=prefix, strict=False)
    bm = _per_species_window_mean(res_soft.biomass(), window_years)

    species = [s for s in tl if not math.isnan(tl[s])]
    if not species:
        return TrophicIndicators(
            float("nan"),
            float("nan"),
            mti_tl_cutoff,
            0,
            0,
            window_years,
            "no usable meanTL values.",
        )

    note = ""
    weights = {s: bm.get(s, 0.0) for s in species}
    if sum(weights.values()) <= 0:
        weights = {s: 1.0 for s in species}
        note = "no biomass output; trophic indices use equal weights."

    wsum = sum(weights.values())
    mtl = sum(tl[s] * weights[s] for s in species) / wsum
    above = [s for s in species if tl[s] >= mti_tl_cutoff]
    if above:
        wabove = sum(weights[s] for s in above)
        mti = sum(tl[s] * weights[s] for s in above) / wabove
    else:
        mti = float("nan")

    return TrophicIndicators(
        float(mtl), float(mti), mti_tl_cutoff, len(species), len(above), window_years, note
    )


@dataclass(frozen=True)
class ABCResult:
    w_statistic: float
    ranks: list[int]
    cum_biomass_pct: list[float]
    cum_abundance_pct: list[float]
    n_species: int
    window_years: int
    note: str


def compute_abc(output_dir, *, prefix: str = "osm", window_years: int = 10) -> ABCResult:
    """Warwick Abundance-Biomass Comparison W-statistic + cumulative dominance curves.

    Ranks species separately by biomass and by abundance (descending), builds the two
    cumulative %-dominance curves, and computes W = sum(Bi - Ai) / (50*(S-1)) over the
    curves (Warwick 1986). W > 0 => biomass-dominated (undisturbed); W < 0 => disturbed.
    Both 1D outputs are required (strict read raises FileNotFoundError if absent).
    """
    res = OsmoseResults(Path(output_dir), prefix=prefix, strict=True)
    bm = _per_species_window_mean(res.biomass(), window_years)
    ab = _per_species_window_mean(res.abundance(), window_years)
    species = sorted(set(bm) & set(ab))
    n = len(species)
    if n < 2:
        return ABCResult(
            float("nan"), [], [], [], n, window_years, "need >= 2 species for ABC; W undefined."
        )

    b_sorted = sorted((bm[s] for s in species), reverse=True)
    a_sorted = sorted((ab[s] for s in species), reverse=True)
    bt, at = sum(b_sorted), sum(a_sorted)
    if bt <= 0 or at <= 0:
        return ABCResult(
            float("nan"),
            [],
            [],
            [],
            n,
            window_years,
            "zero total biomass or abundance; W undefined.",
        )

    cum_b: list[float] = []
    cum_a: list[float] = []
    cb = ca = 0.0
    for bv, av in zip(b_sorted, a_sorted):
        cb += bv
        ca += av
        cum_b.append(100.0 * cb / bt)
        cum_a.append(100.0 * ca / at)
    w = sum(b - a for b, a in zip(cum_b, cum_a)) / (50.0 * (n - 1))
    return ABCResult(float(w), list(range(1, n + 1)), cum_b, cum_a, n, window_years, "")
