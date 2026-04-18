#!/usr/bin/env python3
"""Baltic OSMOSE calibration vs ICES SAG snapshots validator.

Reads frozen ICES snapshots from data/baltic/reference/ices_snapshots/ and
compares:
  1. Fishing mortality rates (model vs SSB-weighted ICES F, 2018-2022).
  2. Biomass targets (model envelope vs ICES SSB envelope, 2018-2022) —
     only for stocks whose SSB is reported in tonnes. Index-scale
     stocks (see index.json units_by_stock) are excluded from the
     envelope sum because tonnes and indices cannot be combined.
  3. Reference points (Fmsy/Blim/Bpa/MSYBtrigger — reported, not
     enforced).

Usage:
    .venv/bin/python scripts/validate_baltic_vs_ices_sag.py            # run + print
    .venv/bin/python scripts/validate_baltic_vs_ices_sag.py --report   # + write markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "baltic" / "reference" / "ices_snapshots"
TARGETS_CSV = PROJECT_ROOT / "data" / "baltic" / "reference" / "biomass_targets.csv"
FISHING_CSV = PROJECT_ROOT / "data" / "baltic" / "baltic_param-fishing.csv"
REPORT_MD = PROJECT_ROOT / "docs" / "baltic_ices_validation_2026-04-18.md"

WINDOW_YEARS = range(2018, 2023)  # 2018..2022 inclusive
F_TOLERANCE = (0.5, 1.5)  # model F must land within [0.5x, 1.5x] of ICES F


@dataclass
class FComparison:
    species: str
    model_f: float
    ices_f_weighted: float | None
    in_tolerance: bool | None


@dataclass
class BiomassComparison:
    species: str
    model_lower: float
    model_upper: float
    ices_min_ssb: float | None
    ices_max_ssb: float | None
    envelopes_overlap: bool | None
    excluded_index_stocks: list[str]


def _load_manifest() -> dict:
    return json.loads((SNAPSHOT_DIR / "index.json").read_text())


def _load_assessment(stock_key: str) -> list[dict]:
    return json.loads((SNAPSHOT_DIR / f"{stock_key}.assessment.json").read_text())


def _load_reference_points(stock_key: str) -> dict:
    return json.loads((SNAPSHOT_DIR / f"{stock_key}.reference_points.json").read_text())


def _series_by_year(assessment: list[dict], field: str) -> dict[int, float]:
    """Extract {year: value} for a field from a flat ICES SAG assessment.

    Snapshots are MCP-equivalent: list of dicts with lowercase string-valued
    keys. Empty strings and NaN are dropped.
    """
    out: dict[int, float] = {}
    for row in assessment or []:
        y = row.get("year")
        v = row.get(field)
        if y is None or v is None or v == "":
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        out[int(y)] = fv
    return out


def _ssb_weighted_f(stocks: list[str]) -> float | None:
    """SSB-weighted mean F across linked stocks, window 2018-2022.

    Uses SSB as weight regardless of unit — index vs tonnes cancels out within
    a single stock, and the weighting is only approximate when stocks with
    different units are combined. Flag such cases in the report.
    """
    num = 0.0
    den = 0.0
    for stock in stocks:
        assessment = _load_assessment(stock)
        f_series = _series_by_year(assessment, "f")
        ssb_series = _series_by_year(assessment, "ssb")
        for y in WINDOW_YEARS:
            if y in f_series and y in ssb_series:
                num += f_series[y] * ssb_series[y]
                den += ssb_series[y]
    return num / den if den > 0 else None


def _ices_ssb_envelope(
    stocks: list[str],
) -> tuple[float | None, float | None]:
    """Sum SSB across tonnes-unit stocks per year, return (min, max) across window.

    Only years with full coverage (all stocks reporting) contribute. Returns
    `(None, None)` if no year satisfies full coverage. Caller must pre-filter
    the list to tonnes-unit stocks only.
    """
    per_stock_series: list[dict[int, float]] = [
        _series_by_year(_load_assessment(stock), "ssb") for stock in stocks
    ]
    if not per_stock_series:
        return None, None
    full_coverage_years = [y for y in WINDOW_YEARS if all(y in s for s in per_stock_series)]
    if not full_coverage_years:
        return None, None
    yearly_totals = [sum(s[y] for s in per_stock_series) for y in full_coverage_years]
    return min(yearly_totals), max(yearly_totals)


def _parse_model_fishing_rates() -> dict[int, float]:
    rates: dict[int, float] = {}
    with FISHING_CSV.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("fisheries.rate.base.fsh"):
                continue
            key, _, value = line.partition(";")
            idx = int(key.rsplit(".fsh", 1)[1])
            rates[idx] = float(value.strip())
    return rates


def _parse_model_targets() -> list[dict]:
    with TARGETS_CSV.open() as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    return list(reader)


# Model species → fsh index (from baltic_param-fishing.csv fisheries.name.fshN rows).
_SPECIES_FSH_INDEX = {
    "cod": 0,
    "herring": 1,
    "sprat": 2,
    "flounder": 3,
    "perch": 4,
    "pikeperch": 5,
    "smelt": 6,
    "stickleback": 7,
}


def _compare_f_rates(manifest: dict, model_fsh: dict[int, float]) -> list[FComparison]:
    out: list[FComparison] = []
    for species, stocks in manifest["model_species_to_ices_stocks"].items():
        idx = _SPECIES_FSH_INDEX.get(species)
        if idx is None:
            continue
        model_f = model_fsh.get(idx)
        if model_f is None:
            continue
        if not stocks:
            out.append(FComparison(species, model_f, None, None))
            continue
        ices_f = _ssb_weighted_f(stocks)
        if ices_f is None:
            out.append(FComparison(species, model_f, None, None))
            continue
        lo, hi = F_TOLERANCE
        in_tol = lo * ices_f <= model_f <= hi * ices_f
        out.append(FComparison(species, model_f, ices_f, in_tol))
    return out


def _compare_biomass(manifest: dict, targets: list[dict]) -> list[BiomassComparison]:
    out: list[BiomassComparison] = []
    by_species = {row["species"]: row for row in targets}
    units = manifest.get("units_by_stock", {})
    for species, stocks in manifest["model_species_to_ices_stocks"].items():
        row = by_species.get(species)
        if row is None:
            continue
        lower = float(row["lower_tonnes"])
        upper = float(row["upper_tonnes"])
        if not stocks:
            out.append(BiomassComparison(species, lower, upper, None, None, None, []))
            continue
        tonnes_stocks = [s for s in stocks if units.get(s) == "tonnes"]
        index_stocks = [s for s in stocks if units.get(s) == "index"]
        if not tonnes_stocks:
            out.append(BiomassComparison(species, lower, upper, None, None, None, index_stocks))
            continue
        ices_min, ices_max = _ices_ssb_envelope(tonnes_stocks)
        if ices_min is None or ices_max is None:
            out.append(BiomassComparison(species, lower, upper, None, None, None, index_stocks))
            continue
        overlap = not (upper < ices_min or lower > ices_max)
        out.append(
            BiomassComparison(species, lower, upper, ices_min, ices_max, overlap, index_stocks)
        )
    return out


def _collect_reference_points(manifest: dict) -> dict[str, dict]:
    rp: dict[str, dict] = {}
    for stocks in manifest["model_species_to_ices_stocks"].values():
        for stock in stocks:
            rp[stock] = _load_reference_points(stock)
    return rp


def run(*, write_report: bool = True) -> dict:
    manifest = _load_manifest()
    model_fsh = _parse_model_fishing_rates()
    targets = _parse_model_targets()

    report = {
        "f_rates": [f.__dict__ for f in _compare_f_rates(manifest, model_fsh)],
        "biomass_envelopes": [b.__dict__ for b in _compare_biomass(manifest, targets)],
        "reference_points": _collect_reference_points(manifest),
        "units_by_stock": manifest.get("units_by_stock", {}),
        "advice_year_by_stock": manifest.get("advice_year_by_stock", {}),
    }

    if write_report:
        _write_markdown_report(report)
    return report


def _write_markdown_report(report: dict) -> None:
    lines = [
        "# Baltic OSMOSE vs ICES SAG (2024 advice) — Validation Report",
        "",
        f"Generated by `{Path(__file__).name}` from snapshots in "
        f"`{SNAPSHOT_DIR.relative_to(PROJECT_ROOT)}/`.",
        "",
        "## Fishing Mortality Rates (2018-2022 window)",
        "",
        "| species | model F | ICES F (SSB-weighted) | in [0.5x, 1.5x] tolerance? |",
        "|---|---:|---:|:---:|",
    ]
    for r in report["f_rates"]:
        f_model = f"{r['model_f']:.3f}"
        f_ices = "—" if r["ices_f_weighted"] is None else f"{r['ices_f_weighted']:.3f}"
        flag = "—" if r["in_tolerance"] is None else ("✓" if r["in_tolerance"] else "✗")
        lines.append(f"| {r['species']} | {f_model} | {f_ices} | {flag} |")
    lines += [
        "",
        "## Biomass Envelope Overlap (SSB 2018-2022, summed across tonnes-unit stocks)",
        "",
        "Stocks reported in relative-index units (see `units_by_stock`) are "
        "excluded from the envelope sum — tonnes and indices cannot be combined.",
        "",
        "| species | model [lower, upper] t | ICES SSB [min, max] t | overlap? | excluded (index-unit) |",
        "|---|---:|---:|:---:|---|",
    ]
    for r in report["biomass_envelopes"]:
        mdl = f"[{r['model_lower']:.0f}, {r['model_upper']:.0f}]"
        ices = (
            "—"
            if r["ices_min_ssb"] is None
            else f"[{r['ices_min_ssb']:.0f}, {r['ices_max_ssb']:.0f}]"
        )
        flag = "—" if r["envelopes_overlap"] is None else ("✓" if r["envelopes_overlap"] else "✗")
        excl = ", ".join(f"`{s}`" for s in r["excluded_index_stocks"]) or "—"
        lines.append(f"| {r['species']} | {mdl} | {ices} | {flag} | {excl} |")
    lines += [
        "",
        "## Reference Points (per ICES stock)",
        "",
        "Values for index-unit stocks (cod.27.24-32, her.27.25-2932, fle.27.2223) "
        "are in the same relative scale as their SSB, not tonnes.",
        "",
        "| stock | unit | Blim | Bpa | Fmsy | MSYBtrigger |",
        "|---|:---:|---:|---:|---:|---:|",
    ]
    units = report.get("units_by_stock", {})
    for stock, rp in sorted(report["reference_points"].items()):
        unit = units.get(stock, "?")
        cells = [_format_rp_cell(rp, k) for k in ("blim", "bpa", "fmsy", "msy_btrigger")]
        lines.append(f"| `{stock}` | {unit} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n")


def _format_rp_cell(rp: dict, key: str) -> str:
    v = rp.get(key) if isinstance(rp, dict) else None
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="store_true",
        help=f"Write markdown report to {REPORT_MD.relative_to(PROJECT_ROOT)}",
    )
    args = parser.parse_args()
    report = run(write_report=args.report)
    manifest = _load_manifest()
    assessed = {s for s, stocks in manifest["model_species_to_ices_stocks"].items() if stocks}

    for r in report["f_rates"]:
        ices_f = "—" if r["ices_f_weighted"] is None else f"{r['ices_f_weighted']:.3f}"
        print(f"F[{r['species']}]: model={r['model_f']:.3f} ices={ices_f} tol={r['in_tolerance']}")
        if r["species"] in assessed and r["ices_f_weighted"] is None:
            print(
                f"  WARN: {r['species']} has linked ICES stocks but no F in window "
                f"{WINDOW_YEARS.start}-{WINDOW_YEARS.stop - 1} — check snapshots.",
                file=sys.stderr,
            )
    for r in report["biomass_envelopes"]:
        ices = (
            "[—]"
            if r["ices_min_ssb"] is None
            else f"[{r['ices_min_ssb']:.0f},{r['ices_max_ssb']:.0f}]"
        )
        print(
            f"B[{r['species']}]: overlap={r['envelopes_overlap']} "
            f"model=[{r['model_lower']:.0f},{r['model_upper']:.0f}] ices={ices} "
            f"excluded={r['excluded_index_stocks']}"
        )
    failures = [r for r in report["f_rates"] if r["in_tolerance"] is False]
    failures += [r for r in report["biomass_envelopes"] if r["envelopes_overlap"] is False]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
