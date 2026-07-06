# tests/test_bob_440_smoke.py
from pathlib import Path
import numpy as np
import pytest
from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine

ROOT = Path(__file__).resolve().parents[1]
BOB = ROOT / "data" / "examples" / "osm_all-parameters.csv"

@pytest.mark.skipif(not BOB.exists(), reason="no BoB config")
def test_bob_runs_on_python_engine():
    raw = dict(OsmoseConfigReader().read(str(BOB)))
    raw["simulation.time.nyear"] = "3"  # pin: do NOT inherit nyear;50
    res = PythonEngine().run_in_memory(raw, seed=42)
    bio = res.biomass()
    assert bio is not None and len(bio) > 0
    vals = bio[[c for c in bio.columns if c not in ("Time", "species")]].to_numpy(dtype=float)
    assert np.isfinite(vals).any() and np.nansum(vals) > 0
