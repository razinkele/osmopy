import shutil
from pathlib import Path
import pytest
from scripts.migrate_bundled_to_440 import convert_config, _collect_param_files

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "examples_433_orig"  # migrate a copy of the ORIGINAL


def _all_keys(master: Path) -> dict[str, str]:
    keys = {}
    for f in _collect_param_files(master):
        for ln in f.read_text().splitlines():
            s = ln.strip()
            if s and not s.startswith("#") and (";" in s):
                k, v = s.split(";", 1)
                keys[k.strip()] = v.strip()
    return keys


@pytest.mark.skipif(not SRC.exists(), reason="need Task 3 snapshot")
def test_bob_converts_to_fully_native(tmp_path):
    cfg = tmp_path / "examples"
    shutil.copytree(SRC, cfg)
    convert_config(cfg)
    keys = _all_keys(cfg / "osm_all-parameters.csv")
    assert keys["osmose.version"] == "4.4.1"
    # species.tl renamed to species.trophic.level (engine species.type path reads the latter)
    assert "species.tl.sp10" not in keys
    assert keys["species.trophic.level.sp10"] == "2.0"
    assert keys["species.trophic.level.sp11"] == "2.5"
    # per-species forcing path added, pointing at the 24-step file
    assert keys["species.file.sp8"] == "ltl/roms_n2p2z2d2_biscay_24step.nc"
    assert keys["species.file.sp13"] == "ltl/roms_n2p2z2d2_biscay_24step.nc"
    # ALL ltl.* keys dropped (a single leftover ltl.name.rscN re-routes the Python engine)
    assert not any(k.startswith("ltl.") for k in keys)
    # species.biomass.* NOT baked on disk (emitted at Java-stage time)
    assert not any(k.startswith("species.biomass.") for k in keys)
