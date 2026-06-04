#!/usr/bin/env python3
"""One-shot helper: pull HELCOM HOLAS-3 fish snapshots for the Baltic into the repo.

Hits the HELCOM ArcGIS REST services directly (the same services the
`helcom-mcp-server` wraps), filters to the OSMOSE Baltic model domain
(ICES SD 22-32 for commercial fish; Baltic coastal sub-basins for coastal
fish), and writes flat JSON snapshots the diagnostic reads.

This is a DIAGNOSTIC reference, not a quantitative validator. HOLAS-3 fish
exposes only a 0-1 quality ratio (BQR / EQR) per subdivision / sub-basin
split into two coarse buckets (DEM = demersal, PEL = pelagic) - no
per-species biomass target. See docs/baltic_holas3_diagnostic_2026-06-04.md
for why an ICES-style per-species HOLAS-3 validator is not possible.

Run once; the snapshots are committed and this script is no longer needed
until the next HOLAS assessment (see helcom_snapshots/README.md).

CORRECTNESS NOTES (learned from review against the live service):
  - The commercial layer 434 keys SD by `Area_Full`, and the ICES Division
    letter VARIES: SD 22 is `27.3.c.22`, SD 23 is `27.3.b.23`, SD 24-32 are
    `27.3.d.*`. A naive `Area_Full LIKE '27.3.d.%'` silently drops SD 22 & 23
    (the western-Baltic cod/flounder grounds). Filter on `27.3.%` then keep
    SD in 22..32 in code; SD 28 appears as two records (28.1 / 28.2).
  - Layer 417 (NOT 433) carries the coastal `EQR` + `Status`; 433 has `BQR`.
  - `Confidence` was 0 for every commercial record in this vintage - dropped.
  - GES boundary 0.6 is an external HELCOM convention, not a service field.
"""

from __future__ import annotations

import json
import pathlib

import httpx

ARCGIS_BASE = "https://maps.helcom.fi/arcgis/rest/services"
SERVICE = "MADS/Indicators_and_assessments/MapServer"
COMMERCIAL_LAYER = 434
COASTAL_LAYER = 417

SNAPSHOT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "baltic"
    / "reference"
    / "helcom_snapshots"
)

# SD in the OSMOSE Baltic model domain (SD 21 Kattegat excluded).
DOMAIN_SUBDIVISIONS = {str(sd) for sd in range(22, 33)}

# Coastal sub-basins inside the model domain (Kattegat excluded as SD 21).
EXCLUDED_COASTAL_SUBBASINS = {"Kattegat"}


def _query(client: httpx.Client, layer: int, out_fields: str, where: str = "1=1") -> list[dict]:
    url = f"{ARCGIS_BASE}/{SERVICE}/{layer}/query"
    r = client.get(
        url,
        params={
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "false",
            "f": "json",
        },
    )
    r.raise_for_status()
    return [feat["attributes"] for feat in r.json().get("features", [])]


def _pull_commercial(client: httpx.Client) -> list[dict]:
    rows = _query(
        client,
        COMMERCIAL_LAYER,
        out_fields="Area_Full,SubDivisio,DEM,PEL,BQR",
        where="Area_Full LIKE '27.3.%'",
    )
    out = []
    for a in rows:
        sd = str(a.get("SubDivisio", "")).strip()
        if sd not in DOMAIN_SUBDIVISIONS:
            continue  # drops SD 21 (Kattegat), keeps 22-32 incl. 28.1/28.2
        out.append(
            {
                "sub_division": sd,
                "area_full": a.get("Area_Full"),
                "dem": a.get("DEM"),
                "pel": a.get("PEL"),
                "bqr": a.get("BQR"),
            }
        )
    out.sort(key=lambda r: r["area_full"])
    return out


def _pull_coastal(client: httpx.Client) -> list[dict]:
    rows = _query(
        client,
        COASTAL_LAYER,
        out_fields="level_2,country,EQR,Status",
        where="1=1",
    )
    out = []
    for a in rows:
        eqr_raw = str(a.get("EQR", "")).strip()
        sub_basin = str(a.get("level_2", "")).strip()
        if not eqr_raw or eqr_raw == " ":
            continue  # drop blank / unassessed water bodies
        if sub_basin in EXCLUDED_COASTAL_SUBBASINS:
            continue
        out.append(
            {
                "sub_basin": sub_basin,
                "country": str(a.get("country", "")).strip(),
                "eqr": float(eqr_raw),
                "status": str(a.get("Status", "")).strip(),
            }
        )
    out.sort(key=lambda r: (r["sub_basin"], r["country"]))
    return out


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0) as client:
        commercial = _pull_commercial(client)
        coastal = _pull_coastal(client)

    (SNAPSHOT_DIR / "commercial_fish.json").write_text(
        json.dumps(commercial, indent=2, ensure_ascii=False) + "\n"
    )
    (SNAPSHOT_DIR / "coastal_fish.json").write_text(
        json.dumps(coastal, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {len(commercial)} commercial + {len(coastal)} coastal records to {SNAPSHOT_DIR}")
    print("NOTE: index.json (manifest + domain_summary) is maintained by hand - refresh its")
    print("domain_summary means/fractions if the pulled values change.")


if __name__ == "__main__":
    main()
