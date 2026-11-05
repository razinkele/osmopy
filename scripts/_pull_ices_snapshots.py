#!/usr/bin/env python3
"""One-shot helper: pull ICES SAG snapshots for Baltic stocks into the repo.

Hits the ICES SAG REST API directly (same endpoints as `ices-mcp-server` wraps),
captures `StockSizeUnits` per stock, and writes MCP-equivalent flat JSON (lowercase
keys) so the validator can read them unchanged.

Run once; the snapshots are committed and then this script is no longer needed
until the next advice-year refresh (see ices_snapshots/README.md § How to Refresh).
``--only STOCK ...`` pulls just those stocks and merges them into index.json, leaving every
other snapshot as committed (SAG revises series, so a full re-pull moves other targets too).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import httpx

SAG_BASE = "https://sag.ices.dk/SAG_API/api"
SNAPSHOT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "baltic"
    / "reference"
    / "ices_snapshots"
)

# Nine stocks have a 2026 assessment in SAG (checked 2026-11-06). cod.27.22-24 stays on its
# 2022 assessment, the last with absolute SSB and F (2025 is a harvest-rate ratio), and
# fle.27.2223 on 2024, its latest (no 2025 or 2026 entry).
STOCKS = [
    ("cod.27.24-32", 2026),
    ("cod.27.22-24", 2022),
    ("her.27.25-2932", 2026),
    ("her.27.28", 2026),
    ("her.27.3031", 2026),
    ("her.27.20-24", 2026),
    ("spr.27.22-32", 2026),
    ("fle.27.2223", 2024),
    # Baltic flounder is four ICES stocks; fle.27.2223 alone is ~5% of the catch (2026-11).
    ("bzq.27.2425", 2026),
    ("bzq.27.2628", 2026),
    ("bwp.27.2729-32", 2026),
]


def _resolve_key(client: httpx.Client, stock_key: str, year: int) -> int:
    r = client.get(f"{SAG_BASE}/StockList", params={"year": str(year)})
    r.raise_for_status()
    for entry in r.json():
        if entry.get("StockKeyLabel") == stock_key:
            return entry["AssessmentKey"]
    raise LookupError(f"{stock_key} not in {year}")


def _flatten_assessment(raw: list[dict]) -> list[dict]:
    """MCP-equivalent lowercase-key flattening (sag.py:89-102)."""
    return [
        {
            "year": e.get("Year"),
            "ssb": e.get("StockSize"),
            "recruitment": e.get("Recruitment"),
            "f": e.get("FishingPressure"),
            "catches": e.get("Catches"),
            "landings": e.get("Landings"),
            "discards": e.get("Discards"),
            "low_ssb": e.get("Low_StockSize"),
            "high_ssb": e.get("High_StockSize"),
        }
        for e in raw
    ]


def _flatten_ref_points(raw: dict) -> dict:
    result = {
        "flim": raw.get("Flim"),
        "fpa": raw.get("Fpa"),
        "fmsy": raw.get("FMSY"),
        "blim": raw.get("Blim"),
        "bpa": raw.get("Bpa"),
        "msy_btrigger": raw.get("MSYBtrigger"),
        "f_age_range": raw.get("FAge"),
        "recruitment_age": raw.get("RecruitmentAge"),
    }
    null_f = [k for k in ("flim", "fpa", "fmsy") if result[k] is None]
    if null_f:
        result["note"] = f"{', '.join(k.upper() for k in null_f)} not defined for this stock"
    return result


# Stocks whose SAG unit label is wrong AND that have no Blim to correct it, keyed by
# (stock, assessment year). Each entry needs the evidence; the guard stays strict for the rest.
UNIT_OVERRIDES = {
    # 2026: labelled "SSB, tonnes" but reads 113-207 against 2.7-20 kt of catch a year,
    # impossible as tonnes; no Blim, no F values, "FMSY proxy" 1. Its 2024 entry was a
    # tonnes/h biomass index.
    ("bzq.27.2425", 2026): "index",
}


def _unit_for(raw_assessment: list[dict], blim: str | None) -> str:
    """Classify SSB unit as 'tonnes' or 'index'.

    ICES SAG's per-row StockSizeUnits field is unreliable for data-limited stocks
    (reports "tonnes" even when values are clearly biomass indices in the 0.1-10
    range). Use Blim magnitude as the ground truth: index-scale assessments have
    Blim < 100 (relative to a reference point), tonnes-scale have Blim > 1000.

    Without a Blim the units field is used, asymmetrically: anything but "tonnes" (e.g.
    "tonnes/h", "Kilograms per gillnet station") is an index, but "tonnes" itself is NOT
    trusted to mean tonnes (fle.27.2223 says "tonnes" for B/Bmsy), so it stays unknown.
    """
    if blim is None or blim == "":
        units = (raw_assessment[0].get("StockSizeUnits") or "").strip().lower()
        return "index" if units and units != "tonnes" else "unknown"
    try:
        b = float(blim)
    except (TypeError, ValueError):
        return "unknown"
    return "index" if b < 100 else "tonnes"


def _f_type(stock_key: str, year: int, raw_assessment: list[dict]) -> dict[str, str]:
    """SAG's FishingPressureDescription/Units, verbatim.

    A blank description raises when the stock publishes F values: the validator reads anything
    but "F" as relative, so a blank would silently strip an absolute stock of its verdict. A
    stock with no F values at all (bzq.27.2425) has no verdict to lose; its blank is recorded."""
    first = raw_assessment[0]
    f_desc = (first.get("FishingPressureDescription") or "").strip()
    has_f = any(r.get("FishingPressure") not in (None, "") for r in raw_assessment)
    if not f_desc and has_f:
        raise RuntimeError(f"{stock_key} @ {year}: blank FishingPressureDescription in SAG payload")
    return {"description": f_desc, "units": (first.get("FishingPressureUnits") or "").strip()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pull ICES SAG snapshots for Baltic stocks.")
    parser.add_argument("--only", nargs="+", metavar="STOCK", help="pull only these stocks")
    args = parser.parse_args(argv)
    stocks = STOCKS
    if args.only:
        known = dict(STOCKS)
        unknown = sorted(set(args.only) - set(known))
        if unknown:
            parser.error(f"not in STOCKS: {', '.join(unknown)}")
        stocks = [(s, known[s]) for s in args.only]

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    units_by_stock: dict[str, str | None] = {}
    year_by_stock: dict[str, int] = {}
    f_type_by_stock: dict[str, dict[str, str]] = {}

    with httpx.Client(timeout=60.0) as client:
        for stock_key, year in stocks:
            print(f"pulling {stock_key} @ {year} ...", flush=True)
            ak = _resolve_key(client, stock_key, year)

            r_a = client.get(f"{SAG_BASE}/StockDownload", params={"assessmentKey": str(ak)})
            r_a.raise_for_status()
            raw_assessment = r_a.json()
            if not raw_assessment:
                # ICES returns 200 + [] for some category-3/4 stocks even when
                # list_stocks claims an advice exists. Fail loudly rather than
                # commit an empty snapshot that silently poisons downstream
                # F-rate and envelope comparisons.
                raise RuntimeError(f"{stock_key} @ {year}: empty assessment payload from ICES SAG")

            r_rp = client.get(
                f"{SAG_BASE}/FishStockReferencePoints",
                params={"assessmentKey": str(ak)},
            )
            r_rp.raise_for_status()
            raw_rp = r_rp.json()
            rp_entry = raw_rp[0] if isinstance(raw_rp, list) and raw_rp else raw_rp
            if not isinstance(rp_entry, dict) or not rp_entry:
                raise RuntimeError(
                    f"{stock_key} @ {year}: empty reference-points payload from ICES SAG"
                )

            flat_a = _flatten_assessment(raw_assessment)
            flat_rp = _flatten_ref_points(rp_entry)

            (SNAPSHOT_DIR / f"{stock_key}.assessment.json").write_text(
                json.dumps(flat_a, indent=2) + "\n"
            )
            (SNAPSHOT_DIR / f"{stock_key}.reference_points.json").write_text(
                json.dumps(flat_rp, indent=2) + "\n"
            )

            unit = UNIT_OVERRIDES.get((stock_key, year)) or _unit_for(
                raw_assessment, flat_rp.get("blim")
            )
            if unit == "unknown":
                # Every snapshotted stock must be classifiable — either the Blim
                # is missing from the ref-points payload (surprising, worth
                # seeing) or the magnitude falls in the borderline zone. Don't
                # silently write "unknown" into index.json; the validator
                # treats unknown-unit stocks as absent, which would be
                # indistinguishable from "stock not assessed".
                raise RuntimeError(
                    f"{stock_key} @ {year}: cannot classify SSB unit "
                    f"(Blim={flat_rp.get('blim')!r}) — inspect raw payload"
                )
            units_by_stock[stock_key] = unit
            year_by_stock[stock_key] = year
            # Verbatim, so the validator can tell absolute F from F/FMSY-type ratios. This
            # is the only field that says so — FMSY=1 is a hint, not a guarantee.
            f_type_by_stock[stock_key] = _f_type(stock_key, year, raw_assessment)

    # Merge units + per-stock advice years + F types into index.json. A full pull rewrites every
    # stock's entry; --only leaves the other stocks' entries as committed.
    index_path = SNAPSHOT_DIR / "index.json"
    index = json.loads(index_path.read_text())
    for key, pulled in (
        ("units_by_stock", units_by_stock),
        ("advice_year_by_stock", year_by_stock),
        ("f_type_by_stock", f_type_by_stock),
    ):
        index[key] = {**index.get(key, {}), **pulled} if args.only else pulled
    # Drop fle.27.24-32 (never existed in SAG) — already absent from manifest.
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print("done; units:", units_by_stock)


if __name__ == "__main__":
    main()
