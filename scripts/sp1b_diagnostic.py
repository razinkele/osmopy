#!/usr/bin/env python
"""SP1b diagnostic: records the recalibrated rate, achieved rel-err, and the cod overshoot
ratio SP1-on-recalibrated vs SP1-off (measured, not gated — does mean-neutral spatial
egg-survival damp the boom/bust?)."""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

from osmose.calibration.larva_recal import RECAL_RATE, mean_cod, sp1_on_config, with_determinism
from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
FIELD = ROOT / "data" / "baltic" / "forcing" / "baltic_rv_field.nc"


def _overshoot(cfg) -> float:
    b = PythonEngine().run_in_memory(cfg, seed=0).biomass()["cod"].to_numpy()[3:15]
    b = b[np.isfinite(b) & (b > 0)]
    return float(b.max() / b.mean()) if b.size and b.mean() > 0 else float("nan")


def main() -> int:
    numba.set_num_threads(1)
    base = dict(OsmoseConfigReader().read(CONFIG))
    base["simulation.time.nyear"] = "15"
    off = with_determinism(base)

    baseline = mean_cod(off)
    over_off = _overshoot(off)
    if RECAL_RATE is None:
        lines = [
            "# SP1b recalibration diagnostic",
            "",
            "RESULT: INFEASIBLE — mean-neutrality not achievable via the cod larva rate alone.",
            f"SP1-off baseline mean cod = {baseline:.1f}; overshoot(off) = {over_off:.2f}",
            "RECAL_RATE = None. See the solve grid in the recalibrate_sp1b commit.",
        ]
    else:
        on = sp1_on_config(base, FIELD)
        mean_on = mean_cod(on)
        over_on = _overshoot(on)
        lines = [
            "# SP1b recalibration diagnostic",
            "",
            f"RECAL_RATE = {RECAL_RATE:.4f}  (cod larval mortality, resolved per-cohort; d0=15.0)",
            f"mean cod: off={baseline:.1f}  on_recal={mean_on:.1f}  "
            f"rel_err={abs(mean_on / baseline - 1):.3f}  (target <= 0.02)",
            "",
            "## Overshoot (max/mean over years 3-14) — measured, NOT gated",
            f"off={over_off:.2f}  on_recal={over_on:.2f}  "
            f"ratio={over_on / over_off:.2f}  "
            f"({'damps' if over_on < over_off else 'does not damp'} the boom/bust)",
        ]
    print("\n".join(lines))
    out = ROOT / "docs" / "diagnostics" / "sp1b_recalibration.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
