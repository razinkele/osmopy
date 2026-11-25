"""output.file.prefix: both engines name every output file with it, as Java does; readers find it.

Before 2026-11-25 this engine always wrote ``osm_*`` and listed the key as unimplemented (#123).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osmose.config.reader import OsmoseConfigReader
from osmose.engine import PythonEngine
from osmose.engine.config import _output_file_prefix
from osmose.results import OsmoseResults, detect_prefix

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", ["", "  "])
def test_unset_is_osm(value):
    assert _output_file_prefix({"output.file.prefix": value}) == "osm"
    assert _output_file_prefix({}) == "osm"


@pytest.mark.parametrize("value", ["baltic", "eec", "run_2.v1", "A-b"])
def test_plain_names_are_kept(value):
    assert _output_file_prefix({"output.file.prefix": value}) == value


@pytest.mark.parametrize(
    "value", ["../x", "a/b", "a\\b", ".hidden", "-x", "a..b", "x" * 65, "a b", "a;b"]
)
def test_anything_that_is_not_a_plain_name_is_refused(value):
    """It becomes part of file names in the run directory and configs can be uploaded."""
    with pytest.raises(ValueError, match="output.file.prefix"):
        _output_file_prefix({"output.file.prefix": value})


def test_detect_prefix(tmp_path):
    assert detect_prefix(tmp_path) == "osm"  # nothing written yet
    assert detect_prefix(None) == "osm"  # in-memory
    (tmp_path / "baltic_biomass_Simu0.csv").write_text("x")
    assert detect_prefix(tmp_path) == "baltic"
    (tmp_path / "osm_biomass_Simu0.csv").write_text("x")
    with pytest.raises(ValueError, match="several runs"):
        detect_prefix(tmp_path)  # ambiguous: refuse rather than read another run's files


def test_a_run_writes_its_prefix_and_the_reader_finds_it(tmp_path):
    cfg = OsmoseConfigReader().read(ROOT / "data/baltic/baltic_all-parameters.csv")
    cfg["simulation.time.nyear"] = "1"
    assert cfg["output.file.prefix"] == "baltic"
    PythonEngine().run(cfg, tmp_path, seed=3)
    names = [p.name for p in tmp_path.rglob("*.csv")]
    assert "baltic_biomass_Simu0.csv" in names
    assert not [n for n in names if n.startswith("osm_")]  # every file carries the prefix
    res = OsmoseResults(tmp_path, strict=False)
    assert res.prefix == "baltic"
    assert len(res.biomass()) > 0  # the default reader found and read the prefixed files
