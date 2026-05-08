"""ICES Stock Assessment Graph (SAG) snapshot validator for OSMOSE outputs.

Reads frozen ICES SAG JSON snapshots (produced by `_pull_ices_snapshots.py`
or fetched live via the ICES MCP server) and compares model run outputs
against per-species SSB envelopes.

Snapshot layout (matches `data/baltic/reference/ices_snapshots/`):

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

    Raises
    ------
    KeyError if `species` has no biomass output in the results dir.
    ValueError if the biomass time series is empty.
    """
    df = results.biomass(species=species)
    if df is None or len(df) == 0:
        raise ValueError(f"no biomass time series for {species!r} in {results.output_dir}")

    if "value" not in df.columns:
        raise ValueError(
            f"biomass DataFrame for {species!r} missing 'value' column "
            f"(got {list(df.columns)})"
        )

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
) -> str:
    """Format comparison results as a markdown report."""
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
    return "\n".join(lines)
