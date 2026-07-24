#!/usr/bin/env python
"""Expand the predation-accessibility matrix for cod_west + cod_east (Phase 1 Task 4).

The matrix is name-labeled (rows=prey, cols=predators; the engine resolves
sp_idx->row by name). This renames cod->cod_west and inserts cod_east (position 8,
after stickleback, matching its sp index), then hand-adjusts cod_east's diet from
eastern-Baltic-cod feeding literature:

  * more SPRAT-dependent (the dominant central/eastern pelagic prey; sprat-cod
    spatial overlap drives eastern condition),
  * more BENTHOS-dependent when prey-limited,
  * less HERRING and less coastal fish (perch/pikeperch/stickleback/smelt) — those
    are western/coastal, outside the deep-basin eastern range.

cod_west and cod_east are spatially separated (SD22-24 vs deep basins), so neither
preys on the other; each keeps within-stock cannibalism (0.05).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MATRIX = Path(__file__).resolve().parent.parent / "data" / "baltic" / "predation-accessibility.csv"

SPECIES_ORDER = [
    "cod_west", "herring", "sprat", "flounder", "perch", "pikeperch", "smelt",
    "stickleback", "cod_east", "Diatoms", "Dinoflagellates", "Microzooplankton",
    "Mesozooplankton", "Macrozooplankton", "Benthos",
]

# cod_east as PREDATOR (its column): eastern-diet adjustments vs the inherited
# western cod diet. Prey -> accessibility.
COD_EAST_DIET = {
    "sprat": 0.5,        # up from 0.4 — dominant eastern pelagic prey
    "herring": 0.3,      # down from 0.4 — more western/coastal
    "Benthos": 0.7,      # up from 0.6 — benthos-dependent when prey-limited
    "smelt": 0.5,        # down from 0.6 — coastal
    "perch": 0.1,        # down from 0.15 — coastal
    "pikeperch": 0.05,   # down from 0.1  — coastal
    "stickleback": 0.2,  # down from 0.3  — coastal
}


def expand() -> None:
    df = pd.read_csv(MATRIX, sep=";", index_col=0)
    header = df.index.name  # "v Prey / Predator >"

    df = df.rename(index={"cod": "cod_west"}, columns={"cod": "cod_west"})

    # cod_east inherits cod_west as both prey (row) and predator (column)
    df.loc["cod_east"] = df.loc["cod_west"]
    df["cod_east"] = df["cod_west"]

    # Spatial separation: no cross-predation; keep within-stock cannibalism
    df.loc["cod_west", "cod_east"] = 0.0
    df.loc["cod_east", "cod_west"] = 0.0
    df.loc["cod_east", "cod_east"] = 0.05

    # Eastern diet adjustments (cod_east predator column)
    for prey, acc in COD_EAST_DIET.items():
        df.loc[prey, "cod_east"] = acc

    df = df.reindex(index=SPECIES_ORDER, columns=SPECIES_ORDER)
    df.index.name = header

    df.to_csv(MATRIX, sep=";", float_format="%g")
    print(f"expanded matrix to {df.shape[0]}x{df.shape[1]} (cod_west + cod_east)")


if __name__ == "__main__":
    expand()
