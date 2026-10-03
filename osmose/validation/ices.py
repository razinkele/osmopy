"""ICES Stock Assessment Graph (SAG) snapshot validator for OSMOSE outputs.

Reads frozen ICES SAG JSON snapshots (produced by `_pull_ices_snapshots.py`
or fetched live via the ICES MCP server) and compares model run outputs
against per-species SSB envelopes.

Snapshot layout (matches ``data/baltic/reference/ices_snapshots/``)::

    <snapshot_dir>/
        index.json                          # manifest: model_species_to_ices_stocks, units_by_stock
        <stock>.assessment.json             # list of {year, ssb, f, ...} dicts
        <stock>.reference_points.json       # {blim, bpa, fmsy, msy_btrigger, ...}

The validator:

1. Loads the snapshot manifest + per-stock assessments.
2. Computes the model's mean biomass per species over a configurable
   window (e.g. last 5 years of the run).
3. Computes the ICES SSB envelope (min, max) over a configurable
   window of historical SAG data, summed across tonnes-unit stocks
   linked to the species.
4. Reports per-species: in-range (model_mean ∈ [ices_min, ices_max]),
   magnitude factor (model_mean / ices_geomean), excluded index-unit
   stocks (which can't be summed with tonnes-unit stocks).

Index-unit stocks are excluded from the envelope sum because relative
indices and tonnes can't be combined. This matches the existing
`scripts/validate_baltic_vs_ices_sag.py` convention.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from osmose.results import OsmoseResults


@dataclass
class IcesSnapshot:
    """Loaded ICES SAG snapshot bundle for a model.

    Attributes
    ----------
    manifest:
        Index dict from index.json. Keys include `model_species_to_ices_stocks`
        (species → list of stock keys), `units_by_stock` (stock key →
        "tonnes" / "index"), `advice_year_by_stock` (stock key → int).
    assessments:
        Stock key → list of {year, ssb, f, ...} dicts (the SAG time series).
    reference_points:
        Stock key → dict of reference points (blim, bpa, fmsy, msy_btrigger).
    snapshot_dir:
        Path the snapshot was loaded from (for traceability).
    """

    manifest: dict
    assessments: dict[str, list[dict]]
    reference_points: dict[str, dict]
    snapshot_dir: Path


@dataclass
class SpeciesBiomassComparison:
    """Result of comparing one species' model mean biomass to its ICES envelope.

    Attributes
    ----------
    species:
        Model species name.
    model_mean_tonnes:
        Mean model biomass over the configured window, in tonnes.
    ices_min_tonnes / ices_max_tonnes:
        ICES SSB envelope min / max over the configured window, summed
        across tonnes-unit stocks linked to the species. None if no
        tonnes-unit stocks linked or no full-coverage years in window.
    in_range:
        True iff `ices_min <= model_mean <= ices_max`. None if envelope
        unavailable.
    magnitude_factor:
        `model_mean / sqrt(ices_min * ices_max)` (geometric mean of the
        ICES envelope). >1 = model overshoots, <1 = undershoots. None
        if envelope unavailable.
    excluded_index_stocks:
        Index-unit stocks linked to this species that were excluded
        from the envelope sum (logged so the report is honest about
        what was compared).
    """

    species: str
    model_mean_tonnes: float
    ices_min_tonnes: float | None = None
    ices_max_tonnes: float | None = None
    in_range: bool | None = None
    magnitude_factor: float | None = None
    excluded_index_stocks: list[str] = field(default_factory=list)


def load_snapshot(snapshot_dir: Path) -> IcesSnapshot:
    """Load an ICES SAG snapshot bundle from disk."""
    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / "index.json").read_text())
    assessments: dict[str, list[dict]] = {}
    reference_points: dict[str, dict] = {}
    for stocks in manifest.get("model_species_to_ices_stocks", {}).values():
        for stock in stocks:
            apath = snapshot_dir / f"{stock}.assessment.json"
            rpath = snapshot_dir / f"{stock}.reference_points.json"
            if apath.exists():
                assessments[stock] = json.loads(apath.read_text())
            if rpath.exists():
                reference_points[stock] = json.loads(rpath.read_text())
    return IcesSnapshot(
        manifest=manifest,
        assessments=assessments,
        reference_points=reference_points,
        snapshot_dir=snapshot_dir,
    )


def _series_by_year(assessment: list[dict], field_name: str) -> dict[int, float]:
    """Extract {year: value} for `field_name` from a flat ICES SAG assessment.

    Drops missing values (None / empty string) silently — ICES convention
    for "not reported." Logs uncoercible values to stderr to surface schema
    drift rather than mask it.
    """
    out: dict[int, float] = {}
    for row in assessment or []:
        y = row.get("year")
        v = row.get(field_name)
        if y is None or v is None or v == "":
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            print(
                f"WARN: uncoercible {field_name}[{y}] value {v!r}; dropped",
                file=sys.stderr,
            )
            continue
        if math.isnan(fv):
            continue
        out[int(y)] = fv
    return out


def _ices_ssb_envelope(
    snapshot: IcesSnapshot,
    stocks: list[str],
    window: range,
) -> tuple[float | None, float | None]:
    """Return (min, max) of summed SSB across stocks per year over the window.

    Only years with full coverage (all stocks reporting) contribute.
    Returns (None, None) if no year satisfies full coverage. Caller
    must pre-filter to tonnes-unit stocks only.
    """
    if not stocks:
        return None, None
    per_stock_series = [_series_by_year(snapshot.assessments.get(s, []), "ssb") for s in stocks]
    full_coverage_years = [y for y in window if all(y in s for s in per_stock_series)]
    if not full_coverage_years:
        return None, None
    yearly_totals = [sum(s[y] for s in per_stock_series) for y in full_coverage_years]
    return min(yearly_totals), max(yearly_totals)


def model_biomass_window_mean(
    results: OsmoseResults,
    species: str,
    window_years: int = 5,
) -> float:
    """Mean model biomass for `species` over the last `window_years` of the run.

    Reads the `biomass` output (long-form DataFrame with `time` + `value`
    columns), filters to the species, takes the trailing window, and
    averages. Returns model biomass in TONNES — the same unit OSMOSE
    writes biomass outputs in (per `output.biomass.unit`).

    Some configs (e.g. eec/BoB) instead emit a WIDE cross-species frame from
    `results.biomass()` — species are columns, and the frame's own `species`
    column is the literal string `"all"` — so `results.biomass(species=X)`
    returns 0 rows for any real species name. In that case, fall back to the
    unfiltered wide frame and pull out the species' column directly.

    Raises
    ------
    KeyError if `species` has no biomass output in the results dir.
    ValueError if the biomass time series is empty.
    """
    df = results.biomass(species=species)
    if df is None or len(df) == 0 or "value" not in getattr(df, "columns", []):
        wide = results.biomass()
        if wide is not None and species in getattr(wide, "columns", []):
            if "Time" in wide.columns:
                # Time is in fractional YEARS, not row index — some configs
                # (e.g. eec/BoB) write multiple rows per year (e.g. 24), so a
                # row-count window would average a few weeks, not
                # `window_years` calendar years. Filter by Time value instead.
                wide = wide.sort_values("Time")
                # .to_numpy() before the reduction: the .loc[...] slice types as
                # `Series | Any | Unknown`, and float(Series) is a type error under
                # pandas-stubs — going through numpy yields an unambiguous scalar.
                tmax = float(wide["Time"].to_numpy().max())
                tail = wide.loc[wide["Time"] > tmax - window_years, species]
            else:
                tail = wide[species]
            if len(tail) == 0:
                raise ValueError(f"empty biomass window for {species!r}")
            return float(tail.to_numpy().mean())
        raise ValueError(f"no biomass time series for {species!r} in {results.output_dir}")

    if "time" in df.columns:
        df = df.sort_values("time")

    n_total = len(df)
    n_window = min(window_years, n_total)
    if n_window <= 0:
        raise ValueError(f"empty biomass window for {species!r}")

    tail = df.iloc[-n_window:]
    return float(tail["value"].mean())


def compare_outputs_to_ices(
    results: OsmoseResults,
    snapshot: IcesSnapshot,
    *,
    window_years: int = 5,
    ices_window: range = range(2018, 2023),
) -> list[SpeciesBiomassComparison]:
    """Compare model biomass to ICES SSB envelopes per species.

    Parameters
    ----------
    results:
        Loaded OsmoseResults from a finished simulation.
    snapshot:
        Loaded IcesSnapshot bundle.
    window_years:
        Number of trailing simulation years to average for the model mean.
    ices_window:
        Range of historical years to compute the ICES envelope over.

    Returns
    -------
    One SpeciesBiomassComparison per species in
    `snapshot.manifest["model_species_to_ices_stocks"]`.
    """
    out: list[SpeciesBiomassComparison] = []
    units = snapshot.manifest.get("units_by_stock", {})
    for species, stocks in snapshot.manifest.get("model_species_to_ices_stocks", {}).items():
        try:
            model_mean = model_biomass_window_mean(results, species, window_years=window_years)
        except (KeyError, ValueError) as e:
            print(
                f"WARN: skipping {species!r} — model output missing or empty: {e}",
                file=sys.stderr,
            )
            continue

        if not stocks:
            out.append(SpeciesBiomassComparison(species=species, model_mean_tonnes=model_mean))
            continue

        tonnes_stocks = [s for s in stocks if units.get(s) == "tonnes"]
        index_stocks = [s for s in stocks if units.get(s) == "index"]

        if not tonnes_stocks:
            out.append(
                SpeciesBiomassComparison(
                    species=species,
                    model_mean_tonnes=model_mean,
                    excluded_index_stocks=index_stocks,
                )
            )
            continue

        ices_min, ices_max = _ices_ssb_envelope(snapshot, tonnes_stocks, ices_window)
        if ices_min is None or ices_max is None:
            out.append(
                SpeciesBiomassComparison(
                    species=species,
                    model_mean_tonnes=model_mean,
                    excluded_index_stocks=index_stocks,
                )
            )
            continue

        in_range = ices_min <= model_mean <= ices_max
        # geometric mean of the envelope — symmetric on log scale, robust
        # to envelope width.
        ices_geomean = math.sqrt(ices_min * ices_max) if ices_min > 0 and ices_max > 0 else None
        magnitude_factor = (model_mean / ices_geomean) if ices_geomean else None

        out.append(
            SpeciesBiomassComparison(
                species=species,
                model_mean_tonnes=model_mean,
                ices_min_tonnes=ices_min,
                ices_max_tonnes=ices_max,
                in_range=in_range,
                magnitude_factor=magnitude_factor,
                excluded_index_stocks=index_stocks,
            )
        )
    return out


def format_markdown_report(
    comparisons: list[SpeciesBiomassComparison],
    *,
    snapshot_dir: Path | None = None,
    window_years: int = 5,
    ices_window: range = range(2018, 2023),
    m2_comparisons: list[PredationM2Comparison] | None = None,
) -> str:
    """Format comparison results as a markdown report (plus the SMS M2 section when given)."""
    lines = [
        "# OSMOSE outputs vs ICES SSB envelope — Validation Report",
        "",
    ]
    if snapshot_dir is not None:
        lines.append(f"Snapshot: `{snapshot_dir}`")
    lines += [
        f"Model window: last {window_years} years of run",
        f"ICES window: {ices_window.start}-{ices_window.stop - 1}",
        "",
        "| species | model mean (t) | ICES envelope (t) | in range | magnitude × | excluded (index-unit) |",
        "|---|---:|---:|:---:|---:|---|",
    ]
    n_in_range = 0
    n_with_envelope = 0
    for c in comparisons:
        model = f"{c.model_mean_tonnes:,.0f}"
        if c.ices_min_tonnes is None:
            envelope = "—"
            in_range = "—"
            magnitude = "—"
        else:
            envelope = f"[{c.ices_min_tonnes:,.0f}, {c.ices_max_tonnes:,.0f}]"
            in_range = "✓" if c.in_range else "✗"
            magnitude = f"{c.magnitude_factor:.2f}" if c.magnitude_factor is not None else "—"
            n_with_envelope += 1
            if c.in_range:
                n_in_range += 1
        excluded = ", ".join(f"`{s}`" for s in c.excluded_index_stocks) or "—"
        lines.append(
            f"| {c.species} | {model} | {envelope} | {in_range} | {magnitude} | {excluded} |"
        )
    lines += [
        "",
        f"**Summary:** {n_in_range}/{n_with_envelope} species in ICES SSB envelope.",
        "",
    ]
    if m2_comparisons is not None:
        lines += format_m2_section(
            m2_comparisons, window_years=window_years, ices_window=ices_window
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WGSAM SMS cod-predation mortality (M2) — issue #136
# ---------------------------------------------------------------------------
#
# The ICES WGSAM Eastern Baltic Sea SMS key run estimates M2, the annual instantaneous
# predation mortality on herring (her.27.25-2932) and sprat (spr.27.22-32) BY COD, at age.
# OSMOSE's `mortalityRate` Predation cause lumps every predator (both cods, percids, the
# background seal and cormorant), so the comparable model quantity is cod-ATTRIBUTED
# predation mortality: annual tonnes of the prey eaten by the cod stocks (the
# `predatorPressure` output, per-step mean tonnes per predator x prey) over the prey's mean
# biomass. That is a BIOMASS-basis rate; SMS M2 is a numbers-basis rate at age, so the SMS
# side is weighted across ages by biomass at age from the same key run. The two bases differ
# only through the age structure and are stated in the report. Report-only — no gate.

SMS_M2_DEFAULT_PREDATORS: tuple[str, ...] = ("cod_west", "cod_east")


@dataclass
class SmsM2Snapshot:
    """Loaded WGSAM SMS M2 bundle (one key-run scenario, sentinel rows dropped)."""

    m2: pd.DataFrame  # Year, scenario, Species, variable, Age, value (>= 0)
    weights: pd.DataFrame  # Year, Species, Age, N, west, BIO (quarter-1 stock at age)
    meta: dict  # index.json['sms_m2']
    snapshot_dir: Path


def load_sms_m2(snapshot_dir: Path, scenario: str | None = None) -> SmsM2Snapshot:
    """Load the SMS M2 snapshot described by ``index.json['sms_m2']``.

    Rows with ``value < 0`` are dropped (defensive: the annual file carries none, but the
    quarterly ``summary.out`` it is derived from marks the projection year with ``-1``).
    ``scenario`` defaults to the manifest's ``scenario_used``.
    """
    snapshot_dir = Path(snapshot_dir)
    meta = json.loads((snapshot_dir / "index.json").read_text()).get("sms_m2")
    if not meta:
        raise KeyError(f"index.json in {snapshot_dir} has no 'sms_m2' block")
    scenario = scenario or meta["scenario_used"]
    m2 = pd.read_csv(snapshot_dir / meta["files"]["m2_annual"])
    m2 = pd.DataFrame(m2[(m2["scenario"] == scenario) & (m2["value"] >= 0)]).reset_index(drop=True)
    if m2.empty:
        raise ValueError(f"no M2 rows for scenario {scenario!r} in {meta['files']['m2_annual']}")
    weights = pd.read_csv(snapshot_dir / meta["files"]["weights"], comment="#")
    return SmsM2Snapshot(m2=m2, weights=weights, meta=meta, snapshot_dir=snapshot_dir)


def sms_m2_weighted_by_year(
    snap: SmsM2Snapshot, sms_species: str, *, basis: str = "biomass"
) -> dict[int, float]:
    """Per-year M2 aggregated over ages: Σ M2(a)·w(a) / Σ w(a) with w = BIO (``basis="biomass"``)
    or N (``basis="numbers"``) at age from the key run's quarter-1 stock. Ages with zero weight
    (age 0 in quarter 1) drop out, so the biomass basis is effectively ages 1+."""
    if basis not in ("biomass", "numbers"):
        raise ValueError(f"basis must be 'biomass' or 'numbers', got {basis!r}")
    wcol = "BIO" if basis == "biomass" else "N"
    m2 = pd.DataFrame(snap.m2[snap.m2["Species"] == sms_species])[["Year", "Age", "value"]]
    w = pd.DataFrame(snap.weights[snap.weights["Species"] == sms_species])[["Year", "Age", wcol]]
    merged = m2.merge(w, on=["Year", "Age"], how="inner")
    merged = pd.DataFrame(merged[merged[wcol] > 0])
    out: dict[int, float] = {}
    for year in sorted(int(y) for y in merged["Year"].unique()):
        g = merged[merged["Year"] == year]
        w_arr = np.asarray(g[wcol], dtype=float)
        total = float(w_arr.sum())
        if total > 0:
            out[year] = float(np.dot(np.asarray(g["value"], dtype=float), w_arr) / total)
    return out


@dataclass
class ModelCodM2:
    """Cod-attributed predation mortality on one prey over the model's trailing window."""

    mean: float
    min: float
    max: float
    n_years: int
    by_predator: dict[str, float]  # mean annual M2 contributed by each cod stock
    predators: tuple[str, ...]


def _trailing_years(frame: pd.DataFrame, window_years: int) -> pd.DataFrame:
    """Rows of a Time-indexed frame inside the last ``window_years`` calendar years."""
    frame = frame.sort_values("Time")
    tmax = float(frame["Time"].to_numpy().max())
    # 1e-9 guards the float subtraction: with yearly rows at k + 23/24, tmax - window can
    # land a hair below the boundary row and pull in one year too many.
    return pd.DataFrame(frame[frame["Time"] > tmax - window_years + 1e-9])


def model_cod_attributed_m2(
    results: OsmoseResults,
    prey: str,
    *,
    predators: tuple[str, ...] = SMS_M2_DEFAULT_PREDATORS,
    window_years: int = 5,
    n_dt_per_year: int = 24,
) -> ModelCodM2:
    """Annual tonnes of ``prey`` eaten by ``predators`` over the prey's mean biomass, per year
    of the trailing window (biomass-basis M2), summarised as mean/min/max.

    ``predatorPressure`` rows are per-STEP mean tonnes over each recording window (Java's
    convention), so a yearly row times ``n_dt_per_year`` is the year's eaten tonnage. The
    biomass denominator is the ``biomass`` output, which applies ``output.cutoff.age`` — for
    the Baltic that excludes young-of-year, consistent with the quarter-1 weighting that makes
    the SMS side effectively ages 1+.
    """
    pressure = results.predator_pressure()
    missing = [p for p in predators if p not in pressure.columns]
    if missing:
        raise KeyError(
            f"predator column(s) {missing} not in predatorPressure; have "
            f"{[c for c in pressure.columns if c not in ('Time', 'Prey', 'species')]}"
        )
    rows = pd.DataFrame(pressure[pressure["Prey"] == prey])
    if rows.empty:
        raise KeyError(f"prey {prey!r} has no predatorPressure rows")
    bio = results.biomass()
    if bio is None or prey not in getattr(bio, "columns", []):
        raise KeyError(f"no biomass column for prey {prey!r}")
    rows = _trailing_years(rows, window_years)
    # The prey's own name is ALSO a predator column in predatorPressure (every focal species
    # is), so the biomass denominator travels under a private name — merging on the species
    # name silently read the (near-zero) cannibalism column instead.
    bio = _trailing_years(
        pd.DataFrame(bio[["Time", prey]]).rename(columns={prey: "_prey_biomass"}), window_years
    )
    # Align per record: both outputs carry one row per recording window at the same Time.
    merged = rows.merge(bio, on="Time", how="inner")
    if merged.empty:
        raise ValueError(
            f"no overlapping Time rows between predatorPressure and biomass for {prey!r}"
        )
    # Yearly Baltic stamps are k + 23/24; per-step stamps end at exact integers for the
    # last step of a year, and plain floor assigns both correctly.
    year = np.floor(merged["Time"].to_numpy()).astype(int)
    merged = merged.assign(_year=year)
    denom = pd.Series(merged.groupby("_year")["_prey_biomass"].mean())
    per_pred: dict[str, float] = {}
    total = pd.Series(0.0, index=denom.index)
    for p in predators:
        # per-step mean -> annual tonnes
        eaten = pd.Series(merged.groupby("_year")[p].mean()) * n_dt_per_year
        rate = (eaten / denom).replace([np.inf, -np.inf], np.nan).dropna()
        per_pred[p] = float(rate.mean()) if len(rate) else float("nan")
        total = total.add(rate, fill_value=0.0)
    total = total.dropna()
    return ModelCodM2(
        mean=float(total.mean()),
        min=float(total.min()),
        max=float(total.max()),
        n_years=len(total),
        by_predator=per_pred,
        predators=tuple(predators),
    )


def model_predation_rate_by_stage(
    results: OsmoseResults, species: str, *, window_years: int = 5
) -> dict[str, float]:
    """Window-mean of the model's TOTAL predation rate (all predators) per life stage from the
    ``mortalityRate`` output — context for the non-cod residual. Returns {} when the output is
    absent or carries no stage split."""
    try:
        df = results.mortality_rate(species)
    except (KeyError, FileNotFoundError, ValueError):
        return {}
    if df is None or len(df) == 0:
        return {}
    cols = df.columns
    if not isinstance(cols, pd.MultiIndex):
        return {}
    time_col = next((c for c in cols if str(c[0]).lower() == "time"), None)
    if time_col is None:
        return {}
    frame = pd.DataFrame({"Time": df[time_col].to_numpy()})
    stages: dict[str, float] = {}
    for cause, stage in cols:
        if str(cause).lower() in ("predation", "mpred") and stage in ("Juvenil", "Adult"):
            frame[stage] = df[(cause, stage)].to_numpy()
    if len(frame.columns) == 1:
        return {}
    tail = _trailing_years(frame, window_years)
    for stage in [c for c in tail.columns if c != "Time"]:
        stages[str(stage)] = float(np.asarray(tail[stage], dtype=float).mean())
    return stages


@dataclass
class PredationM2Comparison:
    """One prey stock: SMS cod-predation M2 beside the model's cod-attributed rate."""

    model_species: str
    sms_species: str
    sms_stock: str
    sms_m2_mean: float | None
    sms_m2_min: float | None
    sms_m2_max: float | None
    sms_n_years: int
    model_cod_m2_mean: float | None = None
    model_cod_m2_min: float | None = None
    model_cod_m2_max: float | None = None
    model_n_years: int = 0
    model_by_predator: dict[str, float] = field(default_factory=dict)
    model_predation_rate_by_stage: dict[str, float] = field(default_factory=dict)
    ratio: float | None = None  # model cod-attributed / SMS
    note: str = ""


def compare_predation_m2_to_sms(
    results: OsmoseResults,
    snap: SmsM2Snapshot,
    *,
    predators: tuple[str, ...] = SMS_M2_DEFAULT_PREDATORS,
    window_years: int = 5,
    ices_window: range = range(2018, 2023),
    n_dt_per_year: int = 24,
    basis: str = "biomass",
) -> list[PredationM2Comparison]:
    """Per SMS prey species: biomass-weighted SMS M2 over ``ices_window`` beside the model's
    cod-attributed M2 over its trailing ``window_years``. Model-side failures (missing output,
    predator column) are reported in ``note`` rather than raised, so the SMS side always prints."""
    species_map = snap.meta.get("sms_species_to_model_species", {})
    stock_map = snap.meta.get("sms_species_to_ices_stocks", {})
    out: list[PredationM2Comparison] = []
    for sms_sp in sorted(species_map):
        model_sp = species_map[sms_sp]
        by_year = sms_m2_weighted_by_year(snap, sms_sp, basis=basis)
        vals = [by_year[y] for y in ices_window if y in by_year]
        row = PredationM2Comparison(
            model_species=model_sp,
            sms_species=sms_sp,
            sms_stock=stock_map.get(sms_sp, "?"),
            sms_m2_mean=float(np.mean(vals)) if vals else None,
            sms_m2_min=float(min(vals)) if vals else None,
            sms_m2_max=float(max(vals)) if vals else None,
            sms_n_years=len(vals),
        )
        try:
            m = model_cod_attributed_m2(
                results,
                model_sp,
                predators=predators,
                window_years=window_years,
                n_dt_per_year=n_dt_per_year,
            )
        except (KeyError, ValueError, FileNotFoundError, OSError) as e:
            row.note = f"model side unavailable: {e}"
            print(f"WARN: SMS M2 {model_sp!r}: {e}", file=sys.stderr)
            out.append(row)
            continue
        row.model_cod_m2_mean, row.model_cod_m2_min, row.model_cod_m2_max = m.mean, m.min, m.max
        row.model_n_years = m.n_years
        row.model_by_predator = m.by_predator
        row.model_predation_rate_by_stage = model_predation_rate_by_stage(
            results, model_sp, window_years=window_years
        )
        if row.sms_m2_mean and row.sms_m2_mean > 0:
            row.ratio = m.mean / row.sms_m2_mean
        out.append(row)
    return out


def format_m2_section(
    rows: list[PredationM2Comparison], *, window_years: int, ices_window: range
) -> list[str]:
    """Markdown lines for the SMS M2 comparison (appended by ``format_markdown_report``)."""
    lines = [
        "## WGSAM SMS cod-predation mortality (M2) — report-only, not gated",
        "",
        (
            "SMS M2 = annual instantaneous predation mortality BY COD on the prey stock, at age, "
            "weighted here across ages by biomass at age from the same key run (effectively ages 1+). "
            "Model = cod-ATTRIBUTED M2 on a biomass basis: annual tonnes of the prey eaten by the cod "
            "stocks (predatorPressure) over the prey's mean biomass (young-of-year excluded by "
            "output.cutoff.age). The two rates are NOT directly equivalent: they differ through age "
            "structure (biomass basis vs numbers at age), through DOMAIN (SMS: ICES SD 25-32 excl. "
            "Gulf of Riga; the model grid also holds the western basin) and through the PREY POOL "
            "each rate is taken on (the model's stock sizes vs the assessed stocks — see the SSB "
            "table). The model's TOTAL predation rate (all predators, per stage, from "
            "mortalityRate) is shown beside it as context; it is a stage rate on a different "
            "basis, so no share is computed from the two."
        ),
        "",
        (
            f"Model window: last {window_years} years of run · "
            f"SMS window: {ices_window.start}-{ices_window.stop - 1}"
        ),
        "",
        (
            "| prey | SMS stock | SMS M2 mean [min, max] | model cod M2 mean [min, max] | model/SMS | "
            "by cod stock | model total predation (Juvenil / Adult) | note |"
        ),
        "|---|---|---:|---:|---:|---|---|---|",
    ]

    def _rng(mean, lo, hi):
        if mean is None:
            return "—"
        return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"

    for r in rows:
        by_pred = ", ".join(f"{k} {v:.3f}" for k, v in r.model_by_predator.items()) or "—"
        st = r.model_predation_rate_by_stage
        total = (
            f"{st.get('Juvenil', float('nan')):.3f} / {st.get('Adult', float('nan')):.3f}"
            if st
            else "—"
        )
        lines.append(
            f"| {r.model_species} | `{r.sms_stock}` | "
            f"{_rng(r.sms_m2_mean, r.sms_m2_min, r.sms_m2_max)} | "
            f"{_rng(r.model_cod_m2_mean, r.model_cod_m2_min, r.model_cod_m2_max)} | "
            f"{'—' if r.ratio is None else f'{r.ratio:.2f}'} | {by_pred} | {total} | "
            f"{r.note or '—'} |"
        )
    lines.append("")
    return lines
