#!/usr/bin/env python
"""Apply the percid missing-removals config edits (Tasks 2-3, data side).

On the aggregate 8-species baseline (cormorant = sp15):
  * Lever A: fixed elevated percid F (perch fsh4 = 0.40, pikeperch fsh5 = 0.50) —
    total commercial + recreational coastal removal (percid_removal_provenance.md).
  * Lever B: cormorant predation — biomass multiplier + physiological ingestion,
    and a tunable Cormorant predator column in the accessibility matrix shaped
    onto percids without over-cropping the forage fish.

Code-side edits (apply_calibration routing, calibrate_baltic free params) are made
separately.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
CONFIG = _HERE.parent / "data" / "baltic"

spec = importlib.util.spec_from_file_location("apply_calibration", _HERE / "apply_calibration.py")
apply_calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_calibration)
set_key = apply_calibration.set_key

# Cormorant predator-column accessibility (prey -> coefficient). Shaped toward
# percids (perch 0.6, pikeperch 0.4) while damping the far-more-abundant forage
# fish (herring/sprat 0.15) so the cormorant does not spend its ration on clupeids.
# Both aggregate ("cod") and disaggregated ("cod_west"/"cod_east") row names carry
# the low cormorant-on-cod accessibility, so this works on either config.
CORMORANT_COL = {
    "perch": 0.6, "pikeperch": 0.4, "smelt": 0.25, "stickleback": 0.15,
    "herring": 0.15, "sprat": 0.15, "flounder": 0.1,
    "cod": 0.05, "cod_west": 0.05, "cod_east": 0.05,
}


def _cormorant_sp(bg: Path) -> int:
    """Cormorant sp index (sp15 aggregate baseline, sp16 disaggregated master)."""
    for line in bg.read_text().splitlines():
        s = line.strip()
        if s.lower().startswith("species.name.sp") and s.split(";")[-1].strip() == "Cormorant":
            return int(s.split(";")[0].strip().lower().removeprefix("species.name.sp"))
    raise SystemExit("Cormorant species not found in background config")


def apply() -> None:
    fishing = CONFIG / "baltic_param-fishing.csv"
    set_key(fishing, "fisheries.rate.base.fsh4", "0.40")   # perch — total coastal F
    set_key(fishing, "fisheries.rate.base.fsh5", "0.50")   # pikeperch — total coastal F

    bg = CONFIG / "baltic_param-background.csv"
    c = _cormorant_sp(bg)
    set_key(bg, f"predation.ingestion.rate.max.sp{c}", "70.0")   # ~physiological (was 40)
    set_key(bg, f"species.biomass.multiplier.sp{c}", "2.0")       # count-based standing-stock anchor

    matrix = CONFIG / "predation-accessibility.csv"
    df = pd.read_csv(matrix, sep=";", index_col=0)
    header = df.index.name
    df["Cormorant"] = [CORMORANT_COL.get(str(i), 0.0) for i in df.index]
    df.index.name = header
    df.to_csv(matrix, sep=";", float_format="%g")
    print(f"percid removals applied: fsh4=0.40 fsh5=0.50; cormorant sp{c} mult=2.0 "
          f"ingest=70; matrix -> {df.shape[0]}x{df.shape[1]} (+Cormorant column)")


if __name__ == "__main__":
    apply()
