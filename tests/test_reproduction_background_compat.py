"""Regression test: reproduction.py must handle configs with background species.

Before 2026-04-25, the broadcast `config.sex_ratio * config.relative_fecundity
* ssb` failed with a shape mismatch when the config had any
`species.type.spN=background` entries. `_merge_focal_background` in
osmose/engine/config.py pads sex_ratio + relative_fecundity to length
n_focal+n_bkg, but ssb is built only for the n_focal focal species.

The fix slices the per-species arrays to focal-only inside reproduction.py.
This test exercises the path by running a 1-year Baltic sim with the two
background species (grey seal, cormorant) activated. If reproduction.py
ever drops the slicing, this test fails.
"""
from pathlib import Path
import tempfile

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BALTIC_CONFIG = PROJECT_ROOT / "data" / "baltic" / "baltic_all-parameters.csv"
BALTIC_BG_CSV = PROJECT_ROOT / "data" / "baltic" / "baltic_param-background.csv"
BALTIC_BG_NC = PROJECT_ROOT / "data" / "baltic" / "baltic_predator_biomass.nc"


@pytest.mark.skipif(
    not (BALTIC_BG_CSV.exists() and BALTIC_BG_NC.exists()),
    reason="background species artifacts not present",
)
def test_baltic_runs_with_background_species():
    from osmose.config.reader import OsmoseConfigReader
    from osmose.engine import PythonEngine
    from osmose.engine.background import parse_background_species

    cfg = OsmoseConfigReader().read(BALTIC_CONFIG)
    cfg["_osmose.config.dir"] = str((PROJECT_ROOT / "data" / "baltic").resolve())

    bg = parse_background_species(cfg, n_focal=8, n_dt_per_year=24)
    assert len(bg) == 2, f"expected 2 background species, found {len(bg)}"
    names = {sp.name for sp in bg}
    assert names == {"GreySeal", "Cormorant"}, f"unexpected names: {names}"

    cfg["simulation.time.nyear"] = "1"
    cfg["output.spatial.enabled"] = "false"
    cfg["output.recordfrequency.ndt"] = "24"

    with tempfile.TemporaryDirectory(prefix="osmose_bg_test_") as tmp:
        out = Path(tmp) / "output"
        result = PythonEngine().run(cfg, out, seed=0)
        assert result.returncode == 0, "engine returned non-zero exit code"
