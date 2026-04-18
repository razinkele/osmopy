#!/usr/bin/env python3
"""One-shot helper: pull ICES SAG snapshots for Baltic stocks into the repo.

Hits the ICES SAG REST API directly (same endpoints as `ices-mcp-server` wraps),
captures `StockSizeUnits` per stock, and writes MCP-equivalent flat JSON (lowercase
keys) so the validator can read them unchanged.

Run once; the snapshots are committed and then this script is no longer needed
until the next advice-year refresh (see ices_snapshots/README.md § How to Refresh).
"""

from __future__ import annotations

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

# 7 stocks have a 2024 advice; cod.27.22-24 is category-3 in 2024 (no SSB/F
# time series), so fall back to its last full 2022 assessment.
STOCKS = [
    ("cod.27.24-32", 2024),
    ("cod.27.22-24", 2022),
    ("her.27.25-2932", 2024),
    ("her.27.28", 2024),
    ("her.27.3031", 2024),
    ("her.27.20-24", 2024),
    ("spr.27.22-32", 2024),
    ("fle.27.2223", 2024),
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


def _unit_for(raw_assessment: list[dict], blim: str | None) -> str:
    """Classify SSB unit as 'tonnes' or 'index'.

    ICES SAG's per-row StockSizeUnits field is unreliable for data-limited stocks
    (reports "tonnes" even when values are clearly biomass indices in the 0.1-10
    range). Use Blim magnitude as the ground truth: index-scale assessments have
    Blim < 100 (relative to a reference point), tonnes-scale have Blim > 1000.
    """
    if blim is None or blim == "":
        return "unknown"
    try:
        b = float(blim)
    except (TypeError, ValueError):
        return "unknown"
    return "index" if b < 100 else "tonnes"


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    units_by_stock: dict[str, str | None] = {}
    year_by_stock: dict[str, int] = {}

    with httpx.Client(timeout=60.0) as client:
        for stock_key, year in STOCKS:
            print(f"pulling {stock_key} @ {year} ...", flush=True)
            ak = _resolve_key(client, stock_key, year)

            r_a = client.get(f"{SAG_BASE}/StockDownload", params={"assessmentKey": str(ak)})
            r_a.raise_for_status()
            raw_assessment = r_a.json()

            r_rp = client.get(
                f"{SAG_BASE}/FishStockReferencePoints",
                params={"assessmentKey": str(ak)},
            )
            r_rp.raise_for_status()
            raw_rp = r_rp.json()
            rp_entry = raw_rp[0] if isinstance(raw_rp, list) and raw_rp else raw_rp

            flat_a = _flatten_assessment(raw_assessment)
            flat_rp = _flatten_ref_points(rp_entry)

            (SNAPSHOT_DIR / f"{stock_key}.assessment.json").write_text(
                json.dumps(flat_a, indent=2) + "\n"
            )
            (SNAPSHOT_DIR / f"{stock_key}.reference_points.json").write_text(
                json.dumps(flat_rp, indent=2) + "\n"
            )

            units_by_stock[stock_key] = _unit_for(raw_assessment, flat_rp.get("blim"))
            year_by_stock[stock_key] = year

    # Rewrite index.json with units + per-stock advice-year overrides.
    index_path = SNAPSHOT_DIR / "index.json"
    index = json.loads(index_path.read_text())
    index["units_by_stock"] = units_by_stock
    index["advice_year_by_stock"] = year_by_stock
    # Drop fle.27.24-32 (never existed in SAG) — already absent from manifest.
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print("done; units:", units_by_stock)


if __name__ == "__main__":
    main()
