#!/usr/bin/env python
"""Reconstruct an APPROXIMATE phase13_results.json for the Baltic calibration.

The original PR #50 phase-13 result JSON is gone: ``data/baltic/calibration_results``
is gitignored, and the transient output was deleted. We rebuild it approximately,
which is scientifically defensible because phase 13 *warm-started* mortality and
fishing from phase 12 and then "concentrated on the 15 SR dimensions" (7 ssb_half +
8 Shepherd beta). So:

  * mortality + fishing (24 params)  ~= the committed ``phase12_results.json`` values
  * the 16 SR params (8 ssb_half + 8 shape-beta) come *exactly* from the committed
    doc ``docs/baltic_shepherd_calibration_2026-05-30.md`` ("Proper 40-year
    calibration" section), with cod sp0 ssb_half pinned at its fixed 120 kt Bpa.

This produces ``data/baltic/calibration_results/phase13_results.json``, the frozen
base that phase 14 (predator functional-response K tuning) freezes on top of.

The OUTPUT JSON is a gitignored runtime artifact; THIS SCRIPT is the committed,
reproducible source of truth. Re-run it any time the JSON needs regenerating:

    .venv/bin/python scripts/reconstruct_phase13_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "baltic" / "calibration_results"
PHASE12_FILE = RESULTS_DIR / "phase12_results.json"
PHASE13_FILE = RESULTS_DIR / "phase13_results.json"

# 16 SR params from docs/baltic_shepherd_calibration_2026-05-30.md
# ("Proper 40-year calibration"). cod sp0 ssb_half is the fixed 120 kt Bpa.
SR_SSBHALF = {
    "stock.recruitment.ssbhalf.sp0": 120000,  # cod (fixed Bpa)
    "stock.recruitment.ssbhalf.sp1": 98000,  # herring
    "stock.recruitment.ssbhalf.sp2": 193000,  # sprat
    "stock.recruitment.ssbhalf.sp3": 6900,  # flounder
    "stock.recruitment.ssbhalf.sp4": 41000,  # perch
    "stock.recruitment.ssbhalf.sp5": 6000,  # pikeperch
    "stock.recruitment.ssbhalf.sp6": 36000,  # smelt
    "stock.recruitment.ssbhalf.sp7": 230000,  # stickleback
}
SR_SHAPE_BETA = {
    "stock.recruitment.shape.sp0": 1.88,  # cod
    "stock.recruitment.shape.sp1": 0.76,  # herring
    "stock.recruitment.shape.sp2": 0.75,  # sprat
    "stock.recruitment.shape.sp3": 1.80,  # flounder
    "stock.recruitment.shape.sp4": 1.60,  # perch
    "stock.recruitment.shape.sp5": 0.50,  # pikeperch
    "stock.recruitment.shape.sp6": 2.56,  # smelt
    "stock.recruitment.shape.sp7": 1.79,  # stickleback
}

_NOTE = (
    "APPROXIMATE reconstruction. The original PR #50 phase-13 result JSON was a "
    "gitignored transient that has been deleted. Mortality + fishing (24 params) are "
    "inherited verbatim from phase12_results.json (phase 13 warm-started them and held "
    "them effectively fixed while concentrating on the 15 SR dimensions). The 16 SR "
    "params (8 ssb_half + 8 Shepherd shape-beta) are the documented values from "
    "docs/baltic_shepherd_calibration_2026-05-30.md ('Proper 40-year calibration'), "
    "with cod sp0 ssb_half pinned at the fixed 120 kt Bpa. The exact per-eval transient "
    "is lost; this is the reproducible best reconstruction. Regenerate with "
    "scripts/reconstruct_phase13_results.py."
)


def build_parameters() -> dict[str, float]:
    """Assemble the reconstructed phase-13 ``parameters`` dict (40 entries)."""
    if not PHASE12_FILE.exists():
        raise FileNotFoundError(
            f"Cannot reconstruct phase 13: {PHASE12_FILE} not found. "
            "Phase 12 mortality + fishing params are required."
        )
    with open(PHASE12_FILE) as f:
        p12 = json.load(f)
    p12_params = p12.get("parameters", {})
    if len(p12_params) != 24:
        raise ValueError(
            f"Expected 24 mortality+fishing params in phase12_results.json, got {len(p12_params)}."
        )

    params: dict[str, float] = {}
    params.update({k.lower(): float(v) for k, v in p12_params.items()})
    params.update({k: float(v) for k, v in SR_SSBHALF.items()})
    params.update({k: float(v) for k, v in SR_SHAPE_BETA.items()})
    return params


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    parameters = build_parameters()
    result = {
        "phase": "13",
        "_note": _NOTE,
        "_reconstructed": True,
        "_source_phase12": str(PHASE12_FILE.name),
        "_source_doc": "docs/baltic_shepherd_calibration_2026-05-30.md",
        "parameters": parameters,
    }
    with open(PHASE13_FILE, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    n_sr = len(SR_SSBHALF) + len(SR_SHAPE_BETA)
    n_mort_fish = len(parameters) - n_sr
    print(f"Wrote {PHASE13_FILE}")
    print(
        f"  {len(parameters)} parameters "
        f"({n_mort_fish} mortality+fishing from phase 12 + {n_sr} SR from doc)"
    )


if __name__ == "__main__":
    main()
