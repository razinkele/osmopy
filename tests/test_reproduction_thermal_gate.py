"""Gate is inert by default (bit-identical) and, when on, deterministically
reduces percid recruitment. Uses the bundled Baltic config and the real engine
API (same as scripts/baltic_recruitment_ceiling_diagnostic.py)."""
from pathlib import Path

import numpy as np

from osmose.config import OsmoseConfigReader
from osmose.engine import PythonEngine

BALTIC = sorted(Path("data/baltic").glob("*all-parameters*.csv"))[0]
THERMAL = Path("tests/data/percid_thermal_ok.csv").resolve()
DET = {"movement.randomseed.fixed": "true", "stochastic.mortality.randomseed.fixed": "true"}


def _series(overrides):
    base = dict(OsmoseConfigReader().read(str(BALTIC)))
    base.update(DET)
    base["simulation.time.nyear"] = "6"  # shrink the in-suite Baltic run (RV-gate test does the same)
    base.update(overrides)
    return PythonEngine().run_in_memory(base, seed=0).biomass()


def test_gate_off_is_bit_identical_to_baseline():
    base = _series({})
    off = _series({"reproduction.thermal.gate.enabled": "false"})
    np.testing.assert_array_equal(base.to_numpy(), off.to_numpy())


def _rel_change(off, on, sp):
    a, b = off[sp].to_numpy(), on[sp].to_numpy()
    d = float(np.abs(a).sum())
    return float(np.abs(b - a).sum()) / d if d else 0.0


def test_gate_on_targets_percids_and_reduces_perch():
    off = _series({})
    on = _series({
        "reproduction.thermal.gate.enabled": "true",
        "reproduction.thermal.gate.series.file": str(THERMAL),
        "reproduction.thermal.gate.mode": "thermal_cap",
        "reproduction.thermal.gate.species.enabled.sp4": "true",
        "reproduction.thermal.gate.species.enabled.sp5": "true",
    })
    # The gate fired on the intended species: both percids move materially.
    assert _rel_change(off, on, "perch") > 0.02
    assert _rel_change(off, on, "pikeperch") > 0.02
    # ...and the effect is PERCID-TARGETED, not a global artifact: the directly
    # gated percids move far more than the weakly-coupled clupeids (the far-field
    # control). Cod is deliberately NOT the yardstick — it is strongly trophically
    # coupled to both percids and legitimately shifts as much (review finding 3);
    # herring/sprat are the honest control.
    clupeid = max(_rel_change(off, on, "herring"), _rel_change(off, on, "sprat"))
    assert _rel_change(off, on, "perch") > clupeid
    assert _rel_change(off, on, "pikeperch") > clupeid
    # Physical direction: the mean-reducing thermal_cap lowers the strongest-signal
    # percid's (perch) recruitment, so its mean biomass drops.
    assert on["perch"].to_numpy().mean() < off["perch"].to_numpy().mean()
