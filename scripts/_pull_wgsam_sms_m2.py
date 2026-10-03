#!/usr/bin/env python3
"""One-shot helper: snapshot the ICES WGSAM Eastern Baltic SMS key-run predation mortality.

Pulls two artefacts from the public WGSAM repository (``ices-eg/wg_WGSAM``,
``Baltic-2025-keyRun/``) at a PINNED commit and writes them, plus a provenance block in
``index.json``, into ``data/baltic/reference/ices_snapshots/``:

- ``wgsam_sms_baltic_2025.m2_annual.csv`` — verbatim ``M2_annu_.csv``: annual cod-predation
  mortality M2 by Year × Species (Herring, Sprat) × Age (0-8), for BOTH the 2022 and the 2025
  key runs (``scenario`` column). The final projection year carries a ``-1`` sentinel.
- ``wgsam_sms_baltic_2025.weights.csv`` — derived from ``summary.out`` (quarter 1 rows for
  herring and sprat): ``Year, Species, Age, N, west, BIO`` — stock numbers, mean weight and
  biomass at age, used to weight M2 across ages (biomass basis) for comparison with the
  whole-population rate OSMOSE reports.

Run once; the snapshots are committed. Re-run only to move to a newer key run (bump COMMIT
and the folder name), then update ``index.json['sms_m2']`` by hand if the layout changed.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys

import httpx
import pandas as pd

REPO = "ices-eg/wg_WGSAM"
FOLDER = "Baltic-2025-keyRun"
COMMIT = "f690d4ff"  # 2025-10-08 "2025 baltic keyrun" (Morten Vinther, DTU Aqua)
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{FOLDER}"
SNAPSHOT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "baltic"
    / "reference"
    / "ices_snapshots"
)
SPECIES_N = {2: "Herring", 3: "Sprat"}  # SMS species_names.in: 1 Cod, 2 Herring, 3 Sprat


def main() -> int:
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        m2_text = client.get(f"{RAW}/M2_annu_.csv").raise_for_status().text
        summary_text = client.get(f"{RAW}/summary.out").raise_for_status().text

    m2_path = SNAPSHOT_DIR / "wgsam_sms_baltic_2025.m2_annual.csv"
    m2_path.write_text(m2_text)

    summary = pd.read_csv(io.StringIO(summary_text), sep=r"\s+")
    q1 = pd.DataFrame(
        summary[(summary["Quarter"] == 1) & (summary["Species.n"].isin(SPECIES_N))]
    ).copy()
    q1["Species"] = pd.Series(q1["Species.n"]).map(SPECIES_N)
    weights = pd.DataFrame(q1[["Year", "Species", "Age", "N", "west", "BIO"]]).sort_values(
        ["Species", "Year", "Age"]
    )
    w_path = SNAPSHOT_DIR / "wgsam_sms_baltic_2025.weights.csv"
    with w_path.open("w") as f:
        f.write(
            f"# Derived from {REPO}@{COMMIT} {FOLDER}/summary.out, quarter-1 rows for herring and "
            "sprat: stock numbers N (thousands), mean weight in the stock west (kg) and biomass "
            "BIO (tonnes) at age. Age 0 has N = 0 in quarter 1 (recruits enter later), so a "
            "BIO-weighted mean over ages is effectively ages 1+.\n"
        )
        weights.to_csv(f, index=False)

    index_path = SNAPSHOT_DIR / "index.json"
    index = json.loads(index_path.read_text())
    index["sms_m2"] = {
        "source": "ICES WGSAM Eastern Baltic Sea SMS key run 2025 (GitHub artefact; the WGSAM 2025 "
        "report was not yet in the ICES library when snapshotted)",
        "repo": REPO,
        "folder": FOLDER,
        "commit": COMMIT,
        "commit_date": "2025-10-08",
        "author": "Morten Vinther (DTU Aqua), stock assessor per the Eastern Baltic SMS stock annex",
        "files": {
            "m2_annual": m2_path.name,
            "weights": w_path.name,
        },
        "scenario_used": "2025 key run",
        "scenarios_present": sorted(pd.read_csv(m2_path)["scenario"].unique().tolist()),
        "sentinel": "-1 marks the final (projection) year; drop rows with value < 0",
        "predator": "Cod only — treated as an external predator with numbers/size from the "
        "ICES SS3 assessment (stock annex: keyruns since 2019)",
        "area": "ICES Subdivisions 25-32 excluding the Gulf of Riga (stock annex Summary)",
        "sms_species_to_ices_stocks": {
            "Herring": "her.27.25-2932",
            "Sprat": "spr.27.22-32",
        },
        "sms_species_to_model_species": {"Herring": "herring", "Sprat": "sprat"},
        "units": "M2 = annual instantaneous cod-predation mortality (per year) at age",
    }
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {m2_path.name}, {w_path.name}; index.json['sms_m2'] updated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
